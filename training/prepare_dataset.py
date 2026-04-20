#!/usr/bin/env python3
"""
Prepare drone detection dataset for fine-tuning.
-------------------------------------------------
Single-class drone detector (nc=1, class 0 = drone).

Supported sources:
  1. Local images: place .jpg/.png in datasets/drone_detect/images/raw/
     with matching .txt YOLO labels, and this script splits train/val.
  2. Roboflow export: download a YOLO-format dataset from Roboflow and
     point --roboflow-dir at it.
  3. Negative samples: images with no drones (empty labels) to reduce
     false positives. Use --negatives-dir.

Usage:
    # From local raw images (80/20 split):
    python3 training/prepare_dataset.py --from-local

    # From a Roboflow YOLO-format export:
    python3 training/prepare_dataset.py --roboflow-dir path/to/roboflow_export

    # Add negative samples (images with no drones):
    python3 training/prepare_dataset.py --negatives-dir path/to/background_images

    # Download from Roboflow API (requires API key):
    python3 training/prepare_dataset.py --roboflow-api KEY --roboflow-workspace WS --roboflow-project PROJ
"""

import argparse
import shutil
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "datasets" / "drone_detect"
DRONE_CLASS_ID = 0  # single-class: class 0 = drone


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
            shutil.copy2(img_path, img_dst / img_path.name)
            lbl_path = img_path.with_suffix(".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_dst / lbl_path.name)
            else:
                # Create empty label (negative sample — no drones)
                (lbl_dst / lbl_path.name).touch()

    print(f"Split {len(images)} images -> {len(train_imgs)} train / {len(val_imgs)} val")


def import_roboflow_dir(rf_dir: Path, drone_class_id=0):
    """Import a Roboflow YOLO-format export.
    For multi-class datasets (e.g. drone-vs-bird), keeps only drone annotations
    (matching drone_class_id) and treats images with only non-drone objects as
    negative samples (empty labels)."""
    for subset in ["train", "val"]:
        rf_subset = rf_dir / subset
        if not rf_subset.exists():
            # Try "valid" or "test" (Roboflow conventions)
            for alt in ["valid", "test"]:
                rf_subset = rf_dir / (alt if subset == "val" else subset)
                if rf_subset.exists():
                    break

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
        neg_count = 0
        for img in sorted(img_src.glob("*.*")):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            shutil.copy2(img, img_dst / img.name)
            lbl = lbl_src / img.with_suffix(".txt").name
            dst_lbl = lbl_dst / img.with_suffix(".txt").name
            if lbl.exists():
                # Filter: keep only lines where class == drone_class_id,
                # remap to class 0 for our nc=1 setup
                drone_lines = []
                for line in lbl.read_text().strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    if int(parts[0]) == drone_class_id:
                        parts[0] = "0"
                        drone_lines.append(" ".join(parts))
                if drone_lines:
                    dst_lbl.write_text("\n".join(drone_lines) + "\n")
                else:
                    # Image has objects but no drones — negative sample
                    dst_lbl.touch()
                    neg_count += 1
            else:
                # No label file — negative sample
                dst_lbl.touch()
                neg_count += 1
            count += 1
        print(f"Imported {count} images to {subset}/ ({neg_count} negatives)")


def add_negatives(neg_dir: Path, val_ratio=0.2):
    """Add negative samples (images with no drones) to the dataset.
    Creates empty .txt label files for each image, telling YOLO
    'nothing to detect here'."""
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        images.extend(neg_dir.glob(ext))

    if not images:
        print(f"No images found in {neg_dir}")
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
            dst_name = f"neg_{img_path.name}"
            shutil.copy2(img_path, img_dst / dst_name)
            # Empty label = no objects in this image
            (lbl_dst / Path(dst_name).with_suffix(".txt").name).touch()

    print(f"Added {len(images)} negative samples -> {len(train_imgs)} train / {len(val_imgs)} val")


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
    parser = argparse.ArgumentParser(description="Prepare drone detection dataset (nc=1)")
    parser.add_argument("--from-local", action="store_true",
                        help="Split local raw images in datasets/drone_detect/images/raw/")
    parser.add_argument("--roboflow-dir", type=str, default=None,
                        help="Path to Roboflow YOLO-format export directory")
    parser.add_argument("--drone-class", type=int, default=0,
                        help="Class ID for 'drone' in the source dataset (default: 0)")
    parser.add_argument("--negatives-dir", type=str, default=None,
                        help="Path to directory of background images (no drones)")
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
        import_roboflow_dir(Path(args.roboflow_dir), drone_class_id=args.drone_class)
    elif args.negatives_dir:
        add_negatives(Path(args.negatives_dir))
    elif args.roboflow_api:
        download_roboflow_api(args.roboflow_api, args.roboflow_workspace,
                              args.roboflow_project, args.roboflow_version)
    else:
        print("Specify a source: --from-local, --roboflow-dir, --negatives-dir, or --roboflow-api")
        print("Run with --help for details.")


if __name__ == "__main__":
    main()
