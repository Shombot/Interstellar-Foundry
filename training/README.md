# Drone Fine-Tuning Pipeline

Fine-tune YOLOv8n (COCO pretrained) to add a **drone** class (class 80)
for the OAK Myriad X VPU.

## Directory structure

```
training/
├── configs/
│   └── dataset.yaml          # 81-class config (80 COCO + drone)
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
├── prepare_dataset.py         # dataset download / split tool
├── finetune.py                # two-phase training script
├── export_blob.sh             # PT → ONNX → OpenVINO → MyriadX blob
└── README.md
```

## Quick start

### 1. Get drone images

**Option A — Roboflow (easiest):**
Find a drone detection dataset on Roboflow Universe, export as YOLOv8 format, then:
```bash
python3 training/prepare_dataset.py --roboflow-dir path/to/export
```

**Option B — Local images:**
Place drone images (.jpg/.png) and YOLO-format .txt labels in
`training/datasets/drone_detect/images/raw/`, then:
```bash
python3 training/prepare_dataset.py --from-local
```

Label format (one .txt per image, same filename):
```
80 0.5 0.4 0.1 0.08
```
= class 80 (drone), center_x, center_y, width, height (all normalized 0-1)

### 2. Train

```bash
# On a machine with a GPU:
python3 training/finetune.py --epochs 100 --batch 16 --device 0

# On Mac (Apple Silicon):
python3 training/finetune.py --epochs 100 --batch 8 --device mps

# CPU only (slow):
python3 training/finetune.py --epochs 50 --batch 8 --device cpu
```

### 3. Export to OAK blob

```bash
bash training/export_blob.sh training/runs/drone_phase2/weights/best.pt
```

### 4. Deploy

Copy the new `.blob` file to `yolov6n_model/` and update `config.json`
to list 81 classes. The `DroneScorer` in `drone_detector.py` already
handles class 80 = drone.

## Requirements

```
pip install ultralytics onnx onnxruntime openvino-dev blobconverter
```
