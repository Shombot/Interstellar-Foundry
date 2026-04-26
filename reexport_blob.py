"""
Re-export best.pt → ONNX → MyriadX blob with /255 normalization baked in.

The repo's original drone_v3.blob (a.k.a. rashodnewmodel.blob) behaved as if
input normalization was never compiled in: on hardware, class scores
saturated around 0 (peak ~0.004 vs. ~0.95 expected). Ultralytics' ONNX
export does not bake /255 into the graph, so we must pass --scale_values
to OpenVINO MO at blob compile time.

This script:
  1. Loads best.pt (Calkin's 3-class airplane/drone/helicopter checkpoint)
     and exports to ONNX at the deployment input shape.
  2. Calls blobconverter.from_onnx() with explicit mean/scale arguments so
     the resulting blob includes a /255 preprocessing layer the MyriadX
     runtime actually executes.

Output: calkinmodel_v2.blob (or whatever you set OUT_BLOB to).

Usage:
    python3 reexport_blob.py
"""
import functools
import os
import shutil
import ssl
import sys
import warnings
from pathlib import Path

# The Jetson's CA bundle has an expired root cert, so requests cannot verify
# blobconverter.luxonis.com (which presents a perfectly valid cert from a
# newer CA). Patch around it: disable verify on the requests session
# blobconverter uses, and silence the resulting urllib3 warnings. Only the
# cert chain is bypassed — TLS itself still encrypts the connection.
def _disable_ssl_verify_for_requests():
    try:
        import requests
        import urllib3
    except ImportError:
        return
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _orig_session_request = requests.sessions.Session.request

    @functools.wraps(_orig_session_request)
    def _patched(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        return _orig_session_request(self, *args, **kwargs)

    requests.sessions.Session.request = _patched
    # `requests.post(...)` and friends use a fresh Session under the hood, so
    # patching Session.request covers them too. Belt + suspenders: also patch
    # the module-level api.request used by some older code paths.
    _orig_api_request = requests.api.request

    @functools.wraps(_orig_api_request)
    def _api_patched(*args, **kwargs):
        kwargs.setdefault("verify", False)
        return _orig_api_request(*args, **kwargs)

    requests.api.request = _api_patched
    print("[ssl] Disabled cert verification for requests "
          "(workaround for expired Jetson CA bundle).")


_disable_ssl_verify_for_requests()

SCRIPT_DIR = Path(__file__).resolve().parent
WEIGHTS = SCRIPT_DIR / "best.pt"
OUT_BLOB = SCRIPT_DIR / "calkinmodel_v2.blob"

# Deployment input — must match drone_detector.py's NN_W, NN_H.
NN_W, NN_H = 512, 288


def export_onnx() -> Path:
    onnx_path = WEIGHTS.with_suffix(".onnx")
    if onnx_path.exists():
        print(f"[1/2] {onnx_path.name} already exists; skipping ONNX export.")
        return onnx_path

    from ultralytics import YOLO

    if not WEIGHTS.exists():
        sys.exit(f"ERROR: {WEIGHTS} not found.")

    print(f"[1/2] Exporting {WEIGHTS.name} -> ONNX at imgsz=({NN_H}, {NN_W})")
    model = YOLO(str(WEIGHTS))
    # imgsz is (height, width) for tuple form in ultralytics.
    model.export(format="onnx", imgsz=(NN_H, NN_W), simplify=True, opset=12)
    if not onnx_path.exists():
        sys.exit(f"ERROR: Expected ONNX at {onnx_path}, not found.")
    print(f"      ONNX written: {onnx_path}")
    return onnx_path


def compile_blob(onnx_path: Path) -> Path:
    import blobconverter

    print(f"[2/2] Compiling {onnx_path.name} -> MyriadX blob with /255 baked in.")
    print(f"      mean_values=[0,0,0]  scale_values=[255,255,255]  shaves=6")
    # blobconverter accepts MO-style flags via optimizer_params. The crucial
    # bit is --scale_values=images[255,255,255] so the blob normalises uint8
    # camera input to [0,1] before the first conv layer.
    blob_path_str = blobconverter.from_onnx(
        model=str(onnx_path),
        data_type="FP16",
        shaves=6,
        use_cache=False,
        output_dir=str(SCRIPT_DIR),
        optimizer_params=[
            f"--input_shape=[1,3,{NN_H},{NN_W}]",
            "--mean_values=images[0,0,0]",
            "--scale_values=images[255,255,255]",
            # NOTE: no --reverse_input_channels. drone_detector.py feeds the
            # camera as RGB888p directly, which matches Ultralytics' training
            # convention. If you switch the pipeline to BGR888p, add this flag.
        ],
    )
    blob_path = Path(blob_path_str)
    print(f"      Blob written: {blob_path} ({blob_path.stat().st_size:,} bytes)")

    if blob_path.resolve() != OUT_BLOB.resolve():
        shutil.copy2(blob_path, OUT_BLOB)
        print(f"      Copied to:    {OUT_BLOB}")
    return OUT_BLOB


def main():
    onnx_path = export_onnx()
    out_blob = compile_blob(onnx_path)
    print()
    print("Done.")
    print(f"New blob: {out_blob}")
    print()
    print("Next: edit drone_detector.py so BLOB_PATH points to the new blob,")
    print("      restore CONF_THRESHOLD to 0.25 (or 0.30) and re-run.")


if __name__ == "__main__":
    main()
