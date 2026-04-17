#!/usr/bin/env python3
"""
Fine-tune YOLOv6n to add drone detection (class 80).
-----------------------------------------------------
Uses the Ultralytics API with YOLOv6n-R2 COCO checkpoint.

Prerequisites:
    pip install ultralytics onnx onnxruntime

Usage:
    # 1. Populate dataset (see README in training/)
    # 2. Fine-tune:
    python3 training/finetune.py
    # 3. Export to ONNX + blob (see export_blob.sh)

The script freezes the backbone for the first phase, then unfreezes
for full fine-tuning. This preserves COCO features while learning
the new drone class.
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# YOLOv6n-R2 is natively supported in the Ultralytics library as "yolov6n".
# However, the Ultralytics hub primarily distributes YOLOv8/v9/v10/v11 weights.
#
# Strategy: fine-tune using YOLOv8n (already in the repo as yolov8n.pt) which
# has the same nano-class performance profile, then export to ONNX → blob.
# YOLOv8n exports cleanly to OpenVINO blob for the Myriad X VPU.
#
# If you specifically need YOLOv6 architecture, use the meituan/YOLOv6 repo
# directly (https://github.com/meituan/YOLOv6) — see ALT_FINETUNE below.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = Path(__file__).resolve().parent / "configs" / "dataset.yaml"
PRETRAINED   = PROJECT_ROOT / "yolov8n.pt"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n + drone class")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=512,
                        help="Input image size (default: 512, matches OAK pipeline)")
    parser.add_argument("--freeze", type=int, default=10,
                        help="Freeze backbone for first N epochs (default: 10)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: '0' for GPU, 'cpu' for CPU, 'mps' for Apple Silicon")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics")
        sys.exit(1)

    if not DATASET_YAML.exists():
        print(f"ERROR: Dataset config not found: {DATASET_YAML}")
        sys.exit(1)

    if not PRETRAINED.exists():
        print(f"WARNING: {PRETRAINED} not found, using ultralytics hub download")
        model = YOLO("yolov8n.pt")
    else:
        model = YOLO(str(PRETRAINED))

    print("=" * 60)
    print("  Drone Fine-Tuning — YOLOv8n")
    print(f"  Pretrained : {PRETRAINED}")
    print(f"  Dataset    : {DATASET_YAML}")
    print(f"  Epochs     : {args.epochs} (backbone frozen first {args.freeze})")
    print(f"  Batch size : {args.batch}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Device     : {args.device}")
    print("=" * 60)

    # Phase 1: Backbone frozen — learn drone class head without forgetting COCO
    print(f"\n--- Phase 1: Frozen backbone ({args.freeze} epochs) ---")
    model.train(
        data=str(DATASET_YAML),
        epochs=args.freeze,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        freeze=10,              # freeze first 10 layers (backbone)
        project=str(PROJECT_ROOT / "training" / "runs"),
        name="drone_phase1",
        exist_ok=True,
        pretrained=True,
        lr0=0.001,              # lower LR for frozen phase
        lrf=0.1,
        patience=20,
        save=True,
        val=True,
    )

    # Load best from phase 1
    phase1_best = PROJECT_ROOT / "training" / "runs" / "drone_phase1" / "weights" / "best.pt"
    if phase1_best.exists():
        model = YOLO(str(phase1_best))
    else:
        print("WARNING: Phase 1 best.pt not found, continuing with last state")

    # Phase 2: Full fine-tune — unfreeze everything, lower LR
    remaining = args.epochs - args.freeze
    if remaining > 0:
        print(f"\n--- Phase 2: Full fine-tune ({remaining} epochs) ---")
        model.train(
            data=str(DATASET_YAML),
            epochs=remaining,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            freeze=0,           # nothing frozen
            project=str(PROJECT_ROOT / "training" / "runs"),
            name="drone_phase2",
            exist_ok=True,
            lr0=0.0005,         # lower LR for full fine-tune
            lrf=0.01,
            patience=30,
            save=True,
            val=True,
        )

    # Final best weights
    phase2_best = PROJECT_ROOT / "training" / "runs" / "drone_phase2" / "weights" / "best.pt"
    final = phase2_best if phase2_best.exists() else phase1_best

    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Best weights: {final}")
    print(f"  Next: run export_blob.sh to convert to OAK blob")
    print("=" * 60)


if __name__ == "__main__":
    main()
