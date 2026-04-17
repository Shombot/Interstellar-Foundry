#!/usr/bin/env python3
"""
Prepare drone detection dataset for fine-tuning.
-------------------------------------------------
Downloads drone images from public datasets and converts annotations
to YOLO format (class x_center y_center width height, normalized).

Supported sources:
  1. Local images: place .jpg/.png in datasets/drone_detect/images/raw/
     with matching .txt YOLO labels, and this script splits train/val.
  2. Roboflow export: download a YOLO-format dataset from Roboflow and
     point --roboflow-dir at it.

Usage:
    # From local raw images (80/20 split):
    python3 training/prepare_dataset.py --from-local

    # From a Roboflow YOLO-format export:
    python3 training/prepare_dataset.py --roboflow-dir path/to/roboflow_export

    # Download from Roboflow API (requires API key):
    python3 training/prepare_dataset.py --roboflow-api KEY --roboflow-workspace WS --roboflow-project PROJ
"""

import argparse
import shutil
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "datasets" / "drone_detect"
DRONE_CLASS_ID = 80  # class index for drone


def split_local(raw_dir: Path, val_ratio=0.2):
    """Split raw images + labels into train/val."""
    images = sorted(raw_dir.glob("*.jpg")) + sorted(raw_dir.glob("*.png"))
    if not images:
        print(f"No images found in {raw_dir}")
        print("Place your drone images (.jpg/.png) and YOLO .txt labels there.")
        return

    random.shuffle(images)
    split_idx = int(len(images) * (1 - val_ratio))
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for subset, img_list in [("train", train_imgs), ("val", val_imgs)]:
        img_dst = DATASET_DIR / "images" / subset
        lbl_dst = DATASET_DIR / "labels" / subset
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img_path in img_list:
            # Copy image
            shutil.copy2(img_path, img_dst / img_path.name)

            # Copy label (same name, .txt extension)
            lbl_path = img_path.with_suffix(".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_dst / lbl_path.name)
            else:
                # Create empty label (negative sample — no drones)
                (lbl_dst / lbl_path.name).touch()

    print(f"Split {len(images)} images → {len(train_imgs)} train / {len(val_imgs)} val")


def import_roboflow_dir(rf_dir: Path):
    """Import a Roboflow YOLO-format export."""
    for subset in ["train", "val"]:
        rf_subset = rf_dir / subset
        if not rf_subset.exists():
            # Try "valid" instead of "val" (Roboflow convention)
            rf_subset = rf_dir / ("valid" if subset == "val" else subset)

        if not rf_subset.exists():
            print(f"WARNING: {rf_subset} not found, skipping")
            continue

        img_src = rf_subset / "images"
        lbl_src = rf_subset / "labels"
        img_dst = DATASET_DIR / "images" / subset
        lbl_dst = DATASET_DIR / "labels" / subset
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        count = 0
        for img in sorted(img_src.glob("*.*")):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                shutil.copy2(img, img_dst / img.name)
                lbl = lbl_src / img.with_suffix(".txt").name
                if lbl.exists():
                    # Remap class IDs: Roboflow exports drone as class 0
                    # We need it as class 80 in our 81-class setup
                    remap_labels(lbl, lbl_dst / lbl.name)
                count += 1
        print(f"Imported {count} images to {subset}/")


def remap_labels(src: Path, dst: Path):
    """
    Remap Roboflow label class IDs to our 81-class scheme.
    If the source dataset is drone-only (class 0 = drone),
    remap 0 → 80.
    """
    lines = src.read_text().strip().split("\n")
    remapped = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        cls_id = int(parts[0])
        # Roboflow single-class export: 0 = drone → 80
        if cls_id == 0:
            parts[0] = str(DRONE_CLASS_ID)
        remapped.append(" ".join(parts))
    dst.write_text("\n".join(remapped) + "\n")


def download_roboflow_api(api_key, workspace, project, version=1):
    """Download dataset via Roboflow Python API."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: pip install roboflow")
        return

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ds = proj.version(version).download("yolov8", location=str(SCRIPT_DIR / "roboflow_dl"))
    print(f"Downloaded to {ds.location}")
    import_roboflow_dir(Path(ds.location))


def main():
    parser = argparse.ArgumentParser(description="Prepare drone detection dataset")
    parser.add_argument("--from-local", action="store_true",
                        help="Split local raw images in datasets/drone_detect/images/raw/")
    parser.add_argument("--roboflow-dir", type=str, default=None,
                        help="Path to Roboflow YOLO-format export directory")
    parser.add_argument("--roboflow-api", type=str, default=None,
                        help="Roboflow API key (for direct download)")
    parser.add_argument("--roboflow-workspace", type=str, default=None)
    parser.add_argument("--roboflow-project", type=str, default=None)
    parser.add_argument("--roboflow-version", type=int, default=1)
    args = parser.parse_args()

    if args.from_local:
        raw_dir = DATASET_DIR / "images" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        split_local(raw_dir)
    elif args.roboflow_dir:
        import_roboflow_dir(Path(args.roboflow_dir))
    elif args.roboflow_api:
        download_roboflow_api(args.roboflow_api, args.roboflow_workspace,
                              args.roboflow_project, args.roboflow_version)
    else:
        print("Specify a source: --from-local, --roboflow-dir, or --roboflow-api")
        print("Run with --help for details.")


if __name__ == "__main__":
    main()
