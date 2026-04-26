"""
Live Drone Detection Dashboard: Oak-D S2 camera + TI IWR6843AOPEVM mmWave radar.

Same dashboard as oldradar.py — left half is the Oak-D RGB feed with detection
overlays, right half is the TI radar PPI plot — but the YOLO model is Calkin's
fine-tuned 3-class blob (calkinmodel_v2.blob) running on the Myriad X VPU
instead of the generic 80-class yolov6-nano from the depthai zoo.

Inference uses dai.node.NeuralNetwork (raw output) + manual YOLOv8 parsing on
host, mirroring the pipeline that was validated on rashod-testing. The
DetectionNetwork built-in parser path produced 0 detections for this blob.
"""
import os
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import serial
except ImportError:
    serial = None

import depthai as dai

# ---------- Camera / detector config ----------
BLOB_PATH = Path(__file__).parent / "calkinmodel_v2.blob"
# Display + stereo depth size. 16:9 to match NN_W:NN_H — keeps both outputs
# from the same center-cropped 16:9 strip of the IMX378 sensor, so bboxes
# from NN coords map cleanly to display coords (no aspect distortion / no
# hidden top-bottom crop that hides the drone from the NN).
MAIN_W, MAIN_H = 640, 360
NN_W, NN_H = 512, 288             # NN input size — must match the compiled blob
# calkinmodel_v2.blob = Calkin's 3-class best.pt (airplane/drone/helicopter)
# recompiled with /255 normalization baked into MO. Empirically this checkpoint
# detects drones much more reliably than the single-class drone_v3.pt — that
# model was undertrained for the deployment scene. Drone is class index 1.
NUM_CLASSES = 3
CLASS_NAMES = ["airplane", "drone", "helicopter"]
# Accept all three classes as targets — Calkin's training set is small enough
# that close-range / hand-held views often classify the drone as airplane or
# helicopter. Set DRONE_LABEL back to 1 if you only want strict drone hits.
DRONE_LABEL = None
CONF_THRESHOLD = 0.50      # real targets score 0.7+; 0.5 kills the 0.3–0.5 noise
IOU_THRESHOLD = 0.45
FPS_TARGET = 30
TRACK_TTL_SEC = 0.25           # short — one-frame false positives die fast
TRACK_IOU_MATCH = 0.25          # min IoU (against predicted bbox) to re-associate
TRACK_MAX_EXTRAP_SEC = 0.15     # minimal motion extrapolation: no sliding ghosts
TRACK_VEL_EMA = 0.5            # EMA weight on newest velocity estimate

# ---------- TI IWR6843AOPEVM config ----------
# XDS-110 on the EVM enumerates two UARTs; Linux typically assigns
# /dev/ttyUSB0 (CLI) and /dev/ttyUSB1 (Data).
RADAR_CLI_PORT = os.environ.get("RADAR_CLI_PORT", "/dev/ttyUSB0")
RADAR_DATA_PORT = os.environ.get("RADAR_DATA_PORT", "/dev/ttyUSB1")
RADAR_CLI_BAUD = 115200
RADAR_DATA_BAUD = 921600
RADAR_CONFIG_FILE = os.environ.get(
    "RADAR_CONFIG_FILE",
    os.path.expanduser("~/iwr6843aop_drone.cfg"),
)
RADAR_MAX_RANGE_M = 15.0        # horizon of the PPI plot
RADAR_PANEL_W = 480             # pixels — width of the radar side panel
# TI mmWave demo wire format
MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"
HEADER_LEN = 40                 # 8-byte magic + 8 × uint32
TLV_DETECTED_POINTS = 1         # payload: N × (x, y, z, velocity) float32
TLV_SIDE_INFO = 7               # payload: N × (snr, noise) uint16 — per-point SNR


class TiMmwRadarReader(threading.Thread):
    """Streams TLV frames from a TI IWR6843AOP demo and exposes the latest cloud."""

    def __init__(self):
        super().__init__(daemon=True)
        self._cli = None
        self._data = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._points = np.empty((0, 5), dtype=np.float32)  # x, y, z, v, snr
        self._frames = 0
        self._last_frame_t = 0.0
        self.status = "init"                                # init / live / stale / error / nodev

    def stop(self):
        self._stop_evt.set()

    def _open(self):
        if serial is None:
            print("pyserial not installed — radar disabled. `pip install pyserial`")
            self.status = "nodev"
            return False
        try:
            self._data = serial.Serial(RADAR_DATA_PORT, RADAR_DATA_BAUD, timeout=0.2)
            self._cli = serial.Serial(RADAR_CLI_PORT, RADAR_CLI_BAUD, timeout=0.2)
        except serial.SerialException as e:
            print(f"Radar serial open failed ({e}); continuing without radar")
            self.status = "nodev"
            return False
        self._send_config()
        return True

    def _send_config(self):
        """Push the TI CLI script line-by-line over the control UART."""
        if not os.path.exists(RADAR_CONFIG_FILE):
            print(
                f"Radar config '{RADAR_CONFIG_FILE}' not found — assuming the "
                "sensor is already running. Set RADAR_CONFIG_FILE to override."
            )
            return
        print(f"Sending radar config from {RADAR_CONFIG_FILE}")
        with open(RADAR_CONFIG_FILE) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("%"):
                    continue
                self._cli.write((line + "\n").encode())
                # The demo CLI echoes back each command; small pause keeps
                # the command FIFO from overflowing on slower firmwares.
                time.sleep(0.02)
                self._cli.reset_input_buffer()

    def _read_exact(self, n, buf):
        """Top up `buf` until it has at least n bytes, or return None on stop."""
        while len(buf) < n:
            if self._stop_evt.is_set():
                return None
            chunk = self._data.read(n - len(buf))
            if not chunk:
                return buf  # caller will re-check length
            buf += chunk
        return buf

    def _read_frame(self, buf):
        """Parse one TLV frame out of the running buffer. Returns (points, new_buf)."""
        # Resync to magic word.
        idx = buf.find(MAGIC_WORD)
        while idx < 0 and not self._stop_evt.is_set():
            buf += self._data.read(256)
            if len(buf) > 8192:
                buf = buf[-1024:]                 # don't grow unbounded on garbage
            idx = buf.find(MAGIC_WORD)
        if self._stop_evt.is_set():
            return None, buf
        buf = buf[idx:]

        buf = self._read_exact(HEADER_LEN, buf)
        if buf is None or len(buf) < HEADER_LEN:
            return None, buf or b""

        # Header (after magic) is 8 uint32: version, totalPacketLen, platform,
        # frameNumber, timeCpuCycles, numDetectedObj, numTLVs, subFrameNumber.
        # Only total_len and numTLVs drive parsing.
        header = struct.unpack_from("<8I", buf, 8)
        total_len = header[1]
        num_tlvs = header[6]

        if total_len < HEADER_LEN or total_len > 65536:
            return None, buf[1:]                  # corrupt; drop one byte and re-sync

        buf = self._read_exact(total_len, buf)
        if buf is None or len(buf) < total_len:
            return None, buf or b""

        xyzv = None                               # TLV 1 payload, shape (N, 4)
        snr = None                                # TLV 7 payload, shape (N,) in 0.1 dB
        off = HEADER_LEN
        for _ in range(num_tlvs):
            if off + 8 > total_len:
                break
            t_type, t_len = struct.unpack_from("<2I", buf, off)
            off += 8
            if off + t_len > total_len:
                break
            if t_type == TLV_DETECTED_POINTS and t_len % 16 == 0:
                n = t_len // 16
                xyzv = np.frombuffer(
                    buf, dtype=np.float32, count=n * 4, offset=off
                ).reshape(-1, 4).copy()
            elif t_type == TLV_SIDE_INFO and t_len % 4 == 0:
                # Each record is (snr_uint16, noise_uint16), SNR in 0.1 dB units.
                m = t_len // 4
                side = np.frombuffer(
                    buf, dtype=np.uint16, count=m * 2, offset=off
                ).reshape(-1, 2)
                snr = side[:, 0].astype(np.float32) * 0.1
            off += t_len

        if xyzv is None:
            points = np.empty((0, 5), dtype=np.float32)
        else:
            if snr is None or len(snr) != len(xyzv):
                snr_col = np.zeros((len(xyzv), 1), dtype=np.float32)
            else:
                snr_col = snr.reshape(-1, 1)
            points = np.hstack([xyzv, snr_col]).astype(np.float32)
        return points, buf[total_len:]

    def run(self):
        if not self._open():
            return
        self.status = "live"
        rolling = b""
        while not self._stop_evt.is_set():
            try:
                pts, rolling = self._read_frame(rolling)
            except Exception as e:
                print(f"Radar read error: {e}")
                self.status = "error"
                time.sleep(0.25)
                continue
            if pts is None:
                continue
            with self._lock:
                self._points = pts
                self._frames += 1
                self._last_frame_t = time.monotonic()

    def snapshot(self):
        with self._lock:
            pts = self._points.copy()
            frames = self._frames
            last_t = self._last_frame_t
        age = time.monotonic() - last_t if last_t else float("inf")
        if self.status in ("nodev", "error"):
            status = self.status
        else:
            status = "live" if age < 1.5 else "stale"
        return pts, frames, status


_NN_INTROSPECT_DONE = False
_NN_PARSE_DIAG_COUNT = 0
_NN_RUNNING_MAX_SCORE = 0.0
_NN_LAST_DIAG_T = 0.0


def _get_raw_tensor(nn_data):
    """DepthAI 3.x removed getFirstLayerFp16(); raw output now comes from
    getTensor(name). Walk through whatever introspection methods this dai
    build exposes so we don't hard-code a single API."""
    global _NN_INTROSPECT_DONE

    # First-call diagnostic: log every layer name + shape so we can confirm
    # what the blob is actually emitting (yolov8 single-output vs. yolov6r2
    # multi-head, FP16 vs FP32, etc.).
    if not _NN_INTROSPECT_DONE:
        _NN_INTROSPECT_DONE = True
        for getter in ("getAllLayerNames", "getAllLayers"):
            if hasattr(nn_data, getter):
                try:
                    layers = getattr(nn_data, getter)()
                    print(f"[NN introspect] {getter}() -> {layers}")
                    break
                except Exception as e:
                    print(f"[NN introspect] {getter}() failed: {e}")

    # Preferred path on DepthAI 3.x: getTensor(name) returns a numpy array.
    if hasattr(nn_data, "getTensor"):
        # Try common ONNX/Ultralytics YOLOv8 output names first.
        for name in ("output0", "output", "outputs"):
            try:
                arr = nn_data.getTensor(name)
                if arr is not None:
                    return np.array(arr)
            except Exception:
                pass
        # Fall back to whatever layer name the blob actually exposes.
        for getter in ("getAllLayerNames", "getAllLayers"):
            if hasattr(nn_data, getter):
                try:
                    layers = getattr(nn_data, getter)()
                    if layers:
                        first = layers[0]
                        # getAllLayers() may return objects with .name; getAllLayerNames() returns strings.
                        name = first if isinstance(first, str) else getattr(first, "name", None)
                        if name:
                            return np.array(nn_data.getTensor(name))
                except Exception:
                    pass

    # Legacy DepthAI 2.x path.
    if hasattr(nn_data, "getFirstLayerFp16"):
        return np.array(nn_data.getFirstLayerFp16())

    raise AttributeError(
        "NNData has no recognised method to extract raw tensor — "
        f"available attrs include: {[a for a in dir(nn_data) if not a.startswith('_')][:20]}"
    )


def parse_yolov8_raw(nn_data, conf_thresh=CONF_THRESHOLD, iou_thresh=IOU_THRESHOLD):
    """Parse raw YOLOv8 NN output into a list of detection dicts.

    For nc=1: output shape (1, 5, N) where 5 = 4 bbox + 1 class score.
    Returns list of dicts with normalized xmin/ymin/xmax/ymax + confidence + label.
    Mirrors the validated parser on rashod-testing, but uses DepthAI 3.x
    getTensor() instead of the removed getFirstLayerFp16().
    """
    raw = _get_raw_tensor(nn_data).astype(np.float32, copy=False).ravel()
    output = raw.reshape(NUM_CLASSES + 4, -1).T
    boxes = output[:, :4]
    scores = output[:, 4:]

    # Diagnostic: log distribution. First few parses print full detail;
    # after that, print a running max-score every ~1s so we can see if the
    # model ever produces real detections (vs. always-zero) — the previous
    # diagnostic only fired on startup before the camera + scene were warm.
    global _NN_PARSE_DIAG_COUNT, _NN_RUNNING_MAX_SCORE, _NN_LAST_DIAG_T
    s = scores.ravel()
    cur_max = float(s.max()) if s.size else 0.0
    _NN_RUNNING_MAX_SCORE = max(_NN_RUNNING_MAX_SCORE, cur_max)
    _NN_PARSE_DIAG_COUNT += 1
    if _NN_PARSE_DIAG_COUNT <= 3:
        b = boxes
        print(
            f"[NN parse #{_NN_PARSE_DIAG_COUNT}] raw.size={raw.size} "
            f"output.shape={output.shape} | "
            f"raw min={raw.min():.6g} max={raw.max():.6g} | "
            f"score min={s.min():.6g} max={s.max():.6g} mean={s.mean():.6g} | "
            f"xywh max={b.max(axis=0)} mean={b.mean(axis=0)}"
        )
    else:
        now = time.monotonic()
        if now - _NN_LAST_DIAG_T > 1.0:
            _NN_LAST_DIAG_T = now
            print(
                f"[NN running] max_score_seen={_NN_RUNNING_MAX_SCORE:.4f} "
                f"this_frame_max={cur_max:.4f} "
                f"frames={_NN_PARSE_DIAG_COUNT}"
            )

    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    mask = confidences > conf_thresh
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    if len(boxes) == 0:
        return []

    # xywh → xyxy, normalized by NN input size
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (x - w / 2) / NN_W
    y1 = (y - h / 2) / NN_H
    x2 = (x + w / 2) / NN_W
    y2 = (y + h / 2) / NN_H

    indices = cv2.dnn.NMSBoxes(
        bboxes=list(zip(x1, y1, x2 - x1, y2 - y1)),
        scores=confidences.tolist(),
        score_threshold=conf_thresh,
        nms_threshold=iou_thresh,
    )
    if len(indices) == 0:
        return []

    detections = []
    for i in np.array(indices).flatten():
        detections.append({
            "xmin": float(np.clip(x1[i], 0, 1)),
            "ymin": float(np.clip(y1[i], 0, 1)),
            "xmax": float(np.clip(x2[i], 0, 1)),
            "ymax": float(np.clip(y2[i], 0, 1)),
            "confidence": float(confidences[i]),
            "label": int(class_ids[i]),
        })
    return detections


def _iou(b1, b2):
    """IoU of two (x1, y1, x2, y2) axis-aligned boxes."""
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _predict_bbox(tr, now):
    """Bbox with linear motion extrapolation from last seen time, bounded by
    TRACK_MAX_EXTRAP_SEC so stale velocity can't drag the box off-screen."""
    age = max(0.0, now - tr.get("last_seen", now))
    age = min(age, TRACK_MAX_EXTRAP_SEC)
    vx, vy = tr.get("vx", 0.0), tr.get("vy", 0.0)
    dx, dy = int(vx * age), int(vy * age)
    x1, y1, x2, y2 = tr["bbox"]
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def _sample_depth_m(depth_mm, bbox, frame_w, frame_h, win=5):
    """Median depth (m) in a small window at the bbox center; 0 if invalid."""
    if depth_mm is None or frame_w <= 0 or frame_h <= 0:
        return 0.0
    dh, dw = depth_mm.shape
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2 / frame_w * dw)
    cy = int((y1 + y2) / 2 / frame_h * dh)
    x0 = max(0, cx - win); x_end = min(dw, cx + win + 1)
    y0 = max(0, cy - win); y_end = min(dh, cy + win + 1)
    roi = depth_mm[y0:y_end, x0:x_end]
    valid = roi[roi > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid)) / 1000.0


def draw_radar_panel(height, points, status, frames, max_range=RADAR_MAX_RANGE_M):
    """Render a top-down PPI view matching the camera frame's height."""
    w = RADAR_PANEL_W
    panel = np.full((height, w, 3), 16, dtype=np.uint8)
    cx = w // 2
    cy = height - 24
    draw_h = cy - 30                              # leave room for title / footer
    ppm = draw_h / max_range                      # pixels per meter

    # Range rings + azimuth graticule
    for frac in (0.25, 0.5, 0.75, 1.0):
        r_px = int(frac * max_range * ppm)
        cv2.circle(panel, (cx, cy), r_px, (40, 80, 40), 1)
        cv2.putText(
            panel, f"{frac * max_range:.0f}m",
            (cx + 4, cy - r_px + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 160, 0), 1,
        )
    for ang_deg in (-60, -30, 0, 30, 60):
        ang = np.radians(ang_deg)
        ex = int(cx + np.sin(ang) * max_range * ppm)
        ey = int(cy - np.cos(ang) * max_range * ppm)
        cv2.line(panel, (cx, cy), (ex, ey), (30, 60, 30), 1)

    # Header
    status_color = {
        "live": (0, 255, 255), "stale": (40, 140, 200),
        "nodev": (120, 120, 120), "error": (0, 0, 255), "init": (120, 120, 120),
    }[status]
    cv2.putText(panel, "mmWave RADAR (top-down)", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(panel, f"[{status.upper()}] frames={frames}", (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

    if status == "nodev":
        cv2.putText(panel, "radar not connected", (10, cy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
        return panel

    # Draw every in-range return in green; remember the strongest one (highest
    # SNR) so we can overlay a red ring + readout for it at the end.
    strongest = None                              # (snr, r, x, y, z, v)
    any_in_range = False
    for x, y, z, v, snr in points:
        if y <= 0 or y > max_range:
            continue
        px = int(cx + x * ppm)
        py = int(cy - y * ppm)
        if not (0 <= px < w and 0 <= py < height):
            continue
        any_in_range = True
        cv2.circle(panel, (px, py), 3, (0, 180, 0), -1)
        if strongest is None or snr > strongest[0]:
            strongest = (float(snr), float(np.hypot(x, y)),
                         float(x), float(y), float(z), float(v))

    if strongest is not None:
        snr_val, r, x, y, z, v = strongest
        px = int(cx + x * ppm)
        py = int(cy - y * ppm)
        cv2.circle(panel, (px, py), 10, (0, 0, 255), 2)
        az_deg = float(np.degrees(np.arctan2(x, y)))
        cv2.putText(
            panel,
            f"TGT  r={r:.2f}m  az={az_deg:+.1f}deg  v={v:+.2f}m/s  "
            f"z={z:+.2f}m  SNR={snr_val:.1f}dB",
            (10, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1,
        )
    elif not any_in_range:
        cv2.putText(panel, "no returns in range", (10, cy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

    return panel


def main():
    if not BLOB_PATH.exists():
        raise FileNotFoundError(f"Model blob not found: {BLOB_PATH}")

    radar = TiMmwRadarReader()
    radar.start()

    print(f"Opening Oak-D S2 and loading blob '{BLOB_PATH.name}'...")
    with dai.Pipeline() as pipeline:
        # Single RGB output at NN input size — wired directly to the NN.
        # We display nn.passthrough below, so detections and the displayed
        # frame come from the same NN execution (no async drift between
        # bbox positions and what the user sees).
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        nn_in_out = cam.requestOutput(
            (NN_W, NN_H), dai.ImgFrame.Type.RGB888p, fps=FPS_TARGET
        )

        # Stereo depth aligned to CAM_A at MAIN_W×MAIN_H so bbox coords map
        # directly into the depth map without rescaling.
        left_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        left_out = left_cam.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8, fps=FPS_TARGET)
        right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        right_out = right_cam.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8, fps=FPS_TARGET)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(MAIN_W, MAIN_H)
        left_out.link(stereo.left)
        right_out.link(stereo.right)

        # Raw NN — host parses output via parse_yolov8_raw().
        # Camera RGB888p output is wired straight in; no ImageManip in
        # the inference path so the blob actually receives RGB planes.
        nn = pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(str(BLOB_PATH))
        nn.setNumInferenceThreads(2)
        nn_in_out.link(nn.input)

        # Display from nn.passthrough — the exact frame the NN ran on, so
        # bboxes (also from this NN execution) overlay perfectly. Blocking
        # get() on both pairs them up automatically.
        img_queue = nn.passthrough.createOutputQueue(maxSize=1, blocking=False)
        depth_queue = stereo.depth.createOutputQueue(maxSize=1, blocking=False)
        nn_queue = nn.out.createOutputQueue(maxSize=1, blocking=False)

        pipeline.start()
        print(f"Detection running. conf>{CONF_THRESHOLD} iou>{IOU_THRESHOLD}. "
              f"Press 'q' to quit.")

        win_name = "Drone Dashboard — Oak-D + TI mmWave (Rashod model)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, MAIN_W + RADAR_PANEL_W, MAIN_H)

        last_t = time.monotonic()
        fps_ema = 0.0
        shape_logged = False
        tracks = []                               # [{bbox, cls, conf, last_seen, dist_m}]
        next_track_id = 0
        latest_depth_mm = None                    # last uint16 depth map seen

        try:
            while pipeline.isRunning():
                # Blocking get on both NN passthrough and NN output — the two
                # streams emit one message per inference, so blocking pairs
                # them. Bboxes therefore overlay the exact frame they were
                # computed on (no async drift, no "ghost" trailing boxes).
                img_msg = img_queue.get()
                nn_msg = nn_queue.get()
                if img_msg is None or nn_msg is None:
                    continue

                # depthai's getCvFrame() returns a BGR-arranged numpy array
                # for cv2 compatibility, even when the source ImgFrame is
                # RGB888p. So no explicit cvtColor is needed — extra flip
                # was double-swapping channels and turning skin blue/purple.
                frame = img_msg.getCvFrame()
                h, w = frame.shape[:2]

                try:
                    latest_dets = parse_yolov8_raw(nn_msg)
                except Exception as e:
                    print(f"YOLO parse error: {e}")
                    latest_dets = []

                # Depth is independent of the NN; pull non-blocking and
                # cache the most recent map.
                depth_msg = depth_queue.tryGet()
                if depth_msg is not None:
                    latest_depth_mm = depth_msg.getFrame()
                if not shape_logged:
                    print(f"Camera frame shape: {frame.shape} (h={h}, w={w})")
                    shape_logged = True

                now = time.monotonic()

                # 3-class blob (airplane / drone / helicopter). When
                # DRONE_LABEL is None we accept every class — the close-range
                # drone view often gets labelled airplane / helicopter, so
                # filtering strictly to label==1 was hiding real targets.
                frame_dets = []
                for det in latest_dets:
                    if DRONE_LABEL is not None and det["label"] != DRONE_LABEL:
                        continue
                    cls_idx = int(det["label"])
                    cls_name = (CLASS_NAMES[cls_idx]
                                if 0 <= cls_idx < len(CLASS_NAMES)
                                else f"cls{cls_idx}")
                    bbox = (
                        int(det["xmin"] * w), int(det["ymin"] * h),
                        int(det["xmax"] * w), int(det["ymax"] * h),
                    )
                    dist_m = _sample_depth_m(latest_depth_mm, bbox, w, h)
                    frame_dets.append((bbox, cls_name, det["confidence"], dist_m))

                # Match each detection against each track's PREDICTED bbox (so a
                # moving drone that YOLO briefly lost can still re-associate even
                # though its stored bbox is stale).
                used_track_idxs = set()
                for bbox, cls_name, conf, dist_m in frame_dets:
                    best_i, best_iou = -1, TRACK_IOU_MATCH
                    for i, tr in enumerate(tracks):
                        if i in used_track_idxs:
                            continue
                        score = _iou(bbox, _predict_bbox(tr, now))
                        if score > best_iou:
                            best_iou, best_i = score, i
                    if best_i >= 0:
                        tr = tracks[best_i]
                        dt_box = now - tr["last_seen"]
                        if 0 < dt_box < 1.0:
                            ocx = (tr["bbox"][0] + tr["bbox"][2]) / 2
                            ocy = (tr["bbox"][1] + tr["bbox"][3]) / 2
                            ncx = (bbox[0] + bbox[2]) / 2
                            ncy = (bbox[1] + bbox[3]) / 2
                            new_vx = (ncx - ocx) / dt_box
                            new_vy = (ncy - ocy) / dt_box
                            tr["vx"] = TRACK_VEL_EMA * new_vx + (1 - TRACK_VEL_EMA) * tr.get("vx", 0.0)
                            tr["vy"] = TRACK_VEL_EMA * new_vy + (1 - TRACK_VEL_EMA) * tr.get("vy", 0.0)
                        tr.update(bbox=bbox, cls=cls_name, conf=conf,
                                  last_seen=now, dist_m=dist_m)
                        used_track_idxs.add(best_i)
                    else:
                        tracks.append({
                            "id": next_track_id, "bbox": bbox, "cls": cls_name,
                            "conf": conf, "last_seen": now, "dist_m": dist_m,
                            "vx": 0.0, "vy": 0.0,
                        })
                        next_track_id += 1

                # Drop expired tracks, draw the survivors in red. HOLD boxes use
                # the predicted (motion-extrapolated) position so the lock roughly
                # follows the drone while YOLO is between hits.
                tracks = [t for t in tracks if now - t["last_seen"] <= TRACK_TTL_SEC]
                best_det = None
                for tr in tracks:
                    fresh = (now - tr["last_seen"]) < 0.05
                    x1, y1, x2, y2 = tr["bbox"] if fresh else _predict_bbox(tr, now)
                    tr["draw_bbox"] = (x1, y1, x2, y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    tag = "LOCK" if fresh else "HOLD"
                    dist = tr.get("dist_m", 0.0)
                    dist_txt = f" d={dist:.2f}m" if dist > 0 else ""
                    label = f"{tr['cls'].upper()} {tag} {tr['conf']:.2f}{dist_txt}"
                    (tw, th), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        frame,
                        (x1, y1 - th - baseline - 4),
                        (x1 + tw + 4, y1),
                        (0, 0, 255), -1,
                    )
                    cv2.putText(
                        frame, label, (x1 + 2, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    )
                    if best_det is None or tr["conf"] > best_det["conf"]:
                        best_det = tr

                dt = now - last_t
                last_t = now
                if dt > 0:
                    inst = 1.0 / dt
                    fps_ema = inst if fps_ema == 0 else (0.9 * fps_ema + 0.1 * inst)

                cv2.putText(frame, f"FPS: {fps_ema:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if best_det is not None:
                    bx1, by1, bx2, by2 = best_det.get("draw_bbox", best_det["bbox"])
                    cxn = (bx1 + bx2) / (2 * w)
                    cyn = (by1 + by2) / (2 * h)
                    dist = best_det.get("dist_m", 0.0)
                    dist_txt = f"{dist:.2f} m" if dist > 0 else "-- m"

                    # Prominent camera-distance banner, top right of the frame.
                    banner = f"CAM DIST: {dist_txt}"
                    (bw, bh), bl = cv2.getTextSize(
                        banner, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                    )
                    bx = w - bw - 12
                    by = 30
                    cv2.rectangle(frame, (bx - 6, by - bh - 4),
                                  (bx + bw + 6, by + bl + 4), (0, 0, 0), -1)
                    cv2.putText(frame, banner, (bx, by),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    # Bottom-of-frame detail line for the top track.
                    cv2.putText(
                        frame,
                        f"CAM tgt DRONE  dist={dist_txt}  "
                        f"px=({cxn*w:.0f},{cyn*h:.0f})  conf={best_det['conf']:.2f}",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                    )

                pts, rframes, rstatus = radar.snapshot()
                radar_panel = draw_radar_panel(h, pts, rstatus, rframes)
                dashboard = np.hstack([frame, radar_panel])

                cv2.imshow(win_name, dashboard)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cv2.destroyAllWindows()
            radar.stop()
            radar.join(timeout=1.0)


if __name__ == "__main__":
    main()
