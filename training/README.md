# Drone Fine-Tuning Pipeline

Single-class YOLOv8n drone detector (**nc=1**, class 0 = drone)
for the OAK Myriad X VPU.

## Directory structure

```
training/
├── configs/
│   └── dataset.yaml          # single-class config (nc=1, drone)
├── datasets/
│   └── drone_detect/
│       ├── images/
│       │   ├── raw/           # put raw images here for --from-local
│       │   ├── train/         # populated by prepare_dataset.py
│       │   └── val/
│       └── labels/
│           ├── train/         # YOLO format: class x_center y_center w h
│           └── val/
├── runs/                      # training outputs (created during training)
├── prepare_dataset.py         # dataset download / split / negatives tool
├── finetune.py                # two-phase training script
├── export_blob.sh             # PT -> ONNX -> OpenVINO -> MyriadX blob
└── README.md
```

## Quick start

### 1. Get drone images

**Option A -- Roboflow (easiest):**
Find a drone detection dataset on Roboflow Universe, export as YOLOv8 format, then:
```bash
python3 training/prepare_dataset.py --roboflow-dir path/to/export
```

**Option B -- Local images:**
Place drone images (.jpg/.png) and YOLO-format .txt labels in
`training/datasets/drone_detect/images/raw/`, then:
```bash
python3 training/prepare_dataset.py --from-local
```

Label format (one .txt per image, same filename):
```
0 0.5 0.4 0.1 0.08
```
= class 0 (drone), center_x, center_y, width, height (all normalized 0-1)

### 2. Add negative samples

Add images with NO drones to reduce false positives:
```bash
python3 training/prepare_dataset.py --negatives-dir path/to/background_images
```
This creates empty .txt label files (tells YOLO "nothing here").
Aim for ~10-20% of the dataset being negatives.

### 3. Train

```bash
# On Google Colab (recommended):
python3 training/finetune.py --epochs 200 --batch 16 --imgsz 640 --device 0

# On Mac (Apple Silicon):
python3 training/finetune.py --epochs 200 --batch 8 --device mps

# CPU only (slow):
python3 training/finetune.py --epochs 50 --batch 8 --device cpu
```

### 4. Export to OAK blob

```bash
bash training/export_blob.sh training/runs/drone_phase2/weights/best.pt
```

### 5. Deploy

Copy the new `.blob` file to `yolov6n_model/`. The `DroneScorer` in
`drone_detector.py` automatically detects the single-class model and
handles class 0 = drone.

## Requirements

```
pip install ultralytics onnx onnxruntime openvino-dev blobconverter
```
