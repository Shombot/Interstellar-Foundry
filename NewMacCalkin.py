"""
Live Drone Detection on Oak-D S2 (VPU-only)
Jetson port of the Mac script: fine-tuned YOLO drone detector runs on the
Oak-D's on-board Myriad X VPU via the precompiled OpenVINO blob. The Jetson
only receives frames + detections and draws overlays.
"""
import time
from pathlib import Path

import cv2
import depthai as dai

# ---------- Configuration ----------
BLOB_PATH = Path(__file__).parent / "calkinnewmodel1.blob"
NN_INPUT_SIZE = (512, 288)        # (w, h) — must match the compiled blob
NUM_CLASSES = 3
CLASS_NAMES = ["airplane", "drone", "helicopter"]
CONF_THRESHOLD = 0.5              # Matches Mac script
IOU_THRESHOLD = 0.45              # Matches Mac script
FPS_TARGET = 30

DRONE_CLASS_NAMES = {"airplane", "drone", "uav", "quadcopter"}


def main():
    if not BLOB_PATH.exists():
        raise FileNotFoundError(f"Model blob not found: {BLOB_PATH}")

    print(f"Opening Oak-D S2 and loading blob '{BLOB_PATH.name}'...")
    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam_out = cam.requestOutput(
            size=NN_INPUT_SIZE,
            type=dai.ImgFrame.Type.BGR888p,
            fps=FPS_TARGET,
        )

        nn = pipeline.create(dai.node.DetectionNetwork)
        nn.setBlobPath(str(BLOB_PATH))
        nn.setConfidenceThreshold(CONF_THRESHOLD)
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)

        parser = nn.detectionParser
        parser.setNNFamily(dai.DetectionNetworkType.YOLO)
        parser.setSubtype("yolov8")
        parser.setNumClasses(NUM_CLASSES)
        parser.setCoordinateSize(4)
        parser.setAnchors([])
        parser.setAnchorMasks({})
        parser.setIouThreshold(IOU_THRESHOLD)
        parser.setClasses(CLASS_NAMES)

        cam_out.link(nn.input)

        det_queue = nn.out.createOutputQueue()
        frame_queue = nn.passthrough.createOutputQueue()

        pipeline.start()
        print("Starting detection. Press 'q' to quit.")

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
                    CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                ).lower()

                conf = float(det.confidence)
                x1 = int(det.xmin * w)
                y1 = int(det.ymin * h)
                x2 = int(det.xmax * w)
                y2 = int(det.ymax * h)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"DRONE ({cls_name}) {conf:.2f}"
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

            cv2.imshow("Drone Detection (Oak-D VPU)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
