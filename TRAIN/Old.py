"""
Live Drone Detection on Oak-D S2
Runs YOLOv6-nano on the camera's on-board Myriad X VPU. The Jetson only
draws boxes, so FPS is limited by the camera/NN, not the host CPU.

Model: YOLOv6-nano (COCO) pulled from the DepthAI zoo on first run and cached.
The DEPTHAI_ZOO_CACHE_PATH env var controls where the cache lives.
"""
import os
import time
from pathlib import Path

import cv2
import depthai as dai

# ---------- Configuration ----------
MODEL_SLUG = "yolov6-nano"        # DepthAI zoo slug (RVC2 build, COCO-trained)
CONF_THRESHOLD = 0.25             # Minimum confidence to display a detection
FPS_TARGET = 30                   # Requested camera/NN fps

# Class names that represent a drone. COCO has "airplane" (class 4) as the
# closest analog. Leave ACCEPT_CLASSES = None to show every class.
ACCEPT_CLASSES = None  # e.g. {"airplane", "drone", "uav", "quadcopter"}

# Make sure the zoo cache dir is writable before touching depthai.
os.environ.setdefault(
    "DEPTHAI_ZOO_CACHE_PATH",
    os.path.expanduser("~/.depthai_cache"),
)
os.makedirs(os.environ["DEPTHAI_ZOO_CACHE_PATH"], exist_ok=True)


def main():
    print(f"Opening Oak-D S2 and loading model '{MODEL_SLUG}' from zoo...")
    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

        model_desc = dai.NNModelDescription(MODEL_SLUG, platform="RVC2")
        nn = pipeline.create(dai.node.DetectionNetwork).build(
            cam, model_desc, fps=FPS_TARGET
        )
        nn.setConfidenceThreshold(CONF_THRESHOLD)
        class_names = nn.getClasses() or []

        det_queue = nn.out.createOutputQueue()
        frame_queue = nn.passthrough.createOutputQueue()

        pipeline.start()
        print("Detection running. Press 'q' to quit.")

        last_t = time.monotonic()
        fps_ema = 0.0

        while pipeline.isRunning():
            img = frame_queue.get()
            dets = det_queue.get()
            if img is None or dets is None:
                continue

            frame = img.getCvFrame()
            h, w = frame.shape[:2]

            for det in dets.detections:
                cls_id = int(det.label)
                cls_name = (
                    class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                ).lower()
                if ACCEPT_CLASSES is not None and cls_name not in ACCEPT_CLASSES:
                    continue
                
                want = ["airplane", "bird", "surfboard", "cell phone", "bird", "mouse", "snowboard", "skateboard", "remote"]

                if cls_name not in want:
                    continue
                if cls_name == "person":
                    continue

                x1 = int(det.xmin * w)
                y1 = int(det.ymin * h)
                x2 = int(det.xmax * w)
                y2 = int(det.ymax * h)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{cls_name} {det.confidence:.2f}"
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

            cv2.imshow("Drone Detection (Oak-D on-device YOLOv6n)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
