"""
Live Drone Detection on Jetson Orin Nano GPU
Uses the Oak-D S2 purely as a camera source; inference runs on the Jetson's
native CUDA GPU via Ultralytics YOLO against the fine-tuned best.pt weights.
"""
import time
from pathlib import Path

import cv2
import depthai as dai
import torch
from ultralytics import YOLO

# ---------- Configuration ----------
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "best.pt"
NN_INPUT_SIZE = (512, 288)        # (w, h) fed to the model
CLASS_NAMES = ["drone"]
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
FPS_TARGET = 30


def build_camera_pipeline():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_out = cam.requestOutput(
        size=NN_INPUT_SIZE,
        type=dai.ImgFrame.Type.BGR888p,
        fps=FPS_TARGET,
    )
    frame_queue = cam_out.createOutputQueue()
    return pipeline, frame_queue


def main():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available — cannot run on Jetson GPU.")

    device = "cuda:0"
    print(f"Loading '{WEIGHTS_PATH.name}' onto {torch.cuda.get_device_name(0)}...")
    model = YOLO(str(WEIGHTS_PATH))
    model.to(device)
    # Warm up the GPU so the first real frame isn't slow.
    model.predict(
        source=torch.zeros(1, 3, NN_INPUT_SIZE[1], NN_INPUT_SIZE[0], device=device),
        verbose=False,
    )

    pipeline, frame_queue = build_camera_pipeline()
    with pipeline:
        pipeline.start()
        print("Detection running. Press 'q' to quit.")

        last_t = time.monotonic()
        fps_ema = 0.0

        while pipeline.isRunning():
            img = frame_queue.get()
            if img is None:
                continue

            frame = img.getCvFrame()
            h, w = frame.shape[:2]

            results = model.predict(
                source=frame,
                imgsz=(NN_INPUT_SIZE[1], NN_INPUT_SIZE[0]),
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=device,
                verbose=False,
            )

            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                cls_name = (
                    CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                )
                conf = float(box.conf.item())
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{cls_name} {conf:.2f}"
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    frame,
                    (x1, y1 - th - baseline - 4),
                    (x1 + tw + 4, y1),
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                )

            now = time.monotonic()
            dt = now - last_t
            last_t = now
            if dt > 0:
                inst = 1.0 / dt
                fps_ema = inst if fps_ema == 0 else (0.9 * fps_ema + 0.1 * inst)

            cv2.putText(
                frame,
                f"FPS: {fps_ema:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Drone Detection (Jetson GPU, Oak-D camera)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
