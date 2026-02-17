#!/usr/bin/env python3
"""
Prepare the CCMT classification dataset for YOLOv8 training.

Downloads PlantVillage via tensorflow_datasets, extracts Tomato and Maize classes,
and organizes images into YOLO classification folder structure. Optionally includes
local cocoa data from training/data/Cocoa/.

Output:
  training/datasets/ccmt_cls/
    train/
      cocoa__black_pod/
      cocoa__healthy/
      maize__rust/
      maize__healthy/
      tomato__early_blight/
      tomato__healthy/
    val/
      (same class folders)
"""

import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow_datasets as tfds

SEED = 42
VAL_RATIO = 0.2

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "datasets" / "ccmt_cls"
COCOA_DIR = SCRIPT_DIR / "data" / "Cocoa"

# PlantVillage label substrings → YOLO folder names
PV_LABEL_MAP = {
    ("corn", "common_rust"): "maize__rust",
    ("corn", "healthy"): "maize__healthy",
    ("tomato", "early_blight"): "tomato__early_blight",
    ("tomato", "healthy"): "tomato__healthy",
}

ALL_CLASSES = [
    "cocoa__black_pod",
    "cocoa__healthy",
    "maize__rust",
    "maize__healthy",
    "tomato__early_blight",
    "tomato__healthy",
]


def match_pv_label(name: str) -> str | None:
    """Match a PlantVillage label name to a YOLO folder name."""
    nl = name.lower()
    for (keyword, condition), folder in PV_LABEL_MAP.items():
        if keyword in nl and condition in nl:
            return folder
    return None


def create_class_dirs():
    """Create all class directories for train and val splits."""
    for split in ("train", "val"):
        for cls in ALL_CLASSES:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)


def extract_plantvillage():
    """Download PlantVillage and extract relevant classes as image files."""
    print("Loading PlantVillage dataset...")
    ds, info = tfds.load("plant_village", split="train", with_info=True, as_supervised=False)
    label_names = info.features["label"].names

    # Build index → folder mapping
    idx_to_folder = {}
    for i, name in enumerate(label_names):
        folder = match_pv_label(name)
        if folder:
            idx_to_folder[i] = folder
            print(f"  {name} -> {folder}")

    # Collect images by class
    images_by_class: dict[str, list[tuple[np.ndarray, int]]] = {
        folder: [] for folder in idx_to_folder.values()
    }

    print("Extracting images...")
    count = 0
    for example in tqdm(ds, desc="Scanning PlantVillage"):
        label_idx = int(example["label"].numpy())
        if label_idx in idx_to_folder:
            folder = idx_to_folder[label_idx]
            images_by_class[folder].append((example["image"].numpy(), count))
            count += 1

    print(f"Extracted {count} images from PlantVillage")
    return images_by_class


def save_split(images: list[tuple[np.ndarray, int]], class_name: str):
    """Split images 80/20 and save to train/val directories."""
    if not images:
        return

    train_imgs, val_imgs = train_test_split(
        images, test_size=VAL_RATIO, random_state=SEED
    )

    for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
        out_dir = OUTPUT_DIR / split_name / class_name
        for img_array, idx in split_imgs:
            img = Image.fromarray(img_array)
            img.save(out_dir / f"{class_name}_{idx:05d}.jpg", quality=95)

    print(f"  {class_name}: {len(train_imgs)} train, {len(val_imgs)} val")


def copy_cocoa_data():
    """Copy cocoa images from local data directory if available."""
    cocoa_mapping = {
        "cocoa_black_pod": "cocoa__black_pod",
        "healthy": "cocoa__healthy",
    }

    found_any = False
    for src_subdir, cls_name in cocoa_mapping.items():
        src_dir = COCOA_DIR / src_subdir
        if not src_dir.exists():
            continue

        images = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            images.extend(list(src_dir.glob(ext)))

        if not images:
            continue

        found_any = True
        train_imgs, val_imgs = train_test_split(
            images, test_size=VAL_RATIO, random_state=SEED
        )

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
            out_dir = OUTPUT_DIR / split_name / cls_name
            for i, img_path in enumerate(split_imgs):
                shutil.copy2(img_path, out_dir / f"{cls_name}_{i:05d}{img_path.suffix}")

        print(f"  {cls_name}: {len(train_imgs)} train, {len(val_imgs)} val (from local data)")

    if not found_any:
        print("  No cocoa data found in training/data/Cocoa/")
        print("  Empty cocoa folders created (YOLOv8 will see 6 classes)")


def main():
    np.random.seed(SEED)

    # Clean previous output
    if OUTPUT_DIR.exists():
        print(f"Removing existing {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)

    create_class_dirs()

    # Extract PlantVillage data
    images_by_class = extract_plantvillage()
    print("\nSaving images with train/val split...")
    for class_name, images in images_by_class.items():
        save_split(images, class_name)

    # Copy cocoa data if available
    print("\nChecking for cocoa data...")
    copy_cocoa_data()

    # Summary
    print("\nDataset ready:")
    for split in ("train", "val"):
        split_dir = OUTPUT_DIR / split
        total = 0
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                n = len(list(cls_dir.glob("*")))
                total += n
        print(f"  {split}: {total} images")

    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
