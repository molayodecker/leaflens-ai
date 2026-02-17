#!/usr/bin/env python3
"""
Train a YOLOv8 nano classification model on the CCMT dataset and export to TFLite.

Expects the dataset at training/datasets/ccmt_cls/ in YOLO classification format
(created by prepare_dataset.py).

Output:
  - app/src/main/assets/model.tflite
  - app/src/main/assets/labels.txt
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "datasets" / "ccmt_cls"
ASSETS_DIR = SCRIPT_DIR.parent / "app" / "src" / "main" / "assets"

# Training config
MODEL = "yolov8n-cls.pt"
IMGSZ = 224
EPOCHS = 30
BATCH = 64
PROJECT = SCRIPT_DIR / "runs"
NAME = "ccmt_cls"

# Folder name → Android label mapping
# Folder: cocoa__black_pod → Label: Cocoa:cocoa_black_pod
# The rule: split on __ to get (crop, condition).
#   - Crop label = capitalized crop name
#   - If condition is "healthy", disease = "healthy"
#   - Otherwise, disease = "{crop}_{condition}"
FOLDER_TO_LABEL = {
    "cocoa__black_pod": "Cocoa:cocoa_black_pod",
    "cocoa__healthy": "Cocoa:healthy",
    "maize__rust": "Maize:maize_rust",
    "maize__healthy": "Maize:healthy",
    "tomato__early_blight": "Tomato:tomato_early_blight",
    "tomato__healthy": "Tomato:healthy",
}


def generate_labels(model_names: list[str]) -> list[str]:
    """Map YOLOv8's class names (folder names) to Android label format.

    Args:
        model_names: Class names from the trained model (sorted folder names).

    Returns:
        List of labels in "Crop:disease" format matching the model's class order.
    """
    labels = []
    for name in model_names:
        label = FOLDER_TO_LABEL.get(name)
        if label is None:
            # Fallback: convert folder__condition to Folder:condition
            parts = name.split("__", 1)
            if len(parts) == 2:
                crop, condition = parts
                if condition == "healthy":
                    label = f"{crop.capitalize()}:healthy"
                else:
                    label = f"{crop.capitalize()}:{crop}_{condition}"
            else:
                label = name
        labels.append(label)
    return labels


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_DIR}\n"
            "Run prepare_dataset.py first to create the dataset."
        )

    # Train
    print(f"Training YOLOv8 classifier on {DATASET_DIR}")
    model = YOLO(MODEL)
    results = model.train(
        data=str(DATASET_DIR),
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        project=str(PROJECT),
        name=NAME,
        exist_ok=True,
        seed=42,
    )

    # Locate best weights
    best_pt = PROJECT / NAME / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Best weights not found at {best_pt}")

    print(f"\nBest weights: {best_pt}")

    # Export to TFLite
    print("Exporting to TFLite...")
    best_model = YOLO(str(best_pt))
    export_path = best_model.export(format="tflite", imgsz=IMGSZ)
    print(f"Exported: {export_path}")

    # Find the exported .tflite file
    export_dir = Path(export_path)
    if export_dir.is_dir():
        tflite_files = list(export_dir.glob("*.tflite"))
        if not tflite_files:
            raise FileNotFoundError(f"No .tflite file found in {export_dir}")
        tflite_path = tflite_files[0]
    else:
        tflite_path = export_dir

    # Generate labels from model's class order
    class_names = best_model.names  # dict: {0: 'cocoa__black_pod', 1: 'cocoa__healthy', ...}
    ordered_names = [class_names[i] for i in sorted(class_names.keys())]
    labels = generate_labels(ordered_names)

    print(f"\nModel classes ({len(ordered_names)}):")
    for i, (name, label) in enumerate(zip(ordered_names, labels)):
        print(f"  {i}: {name} -> {label}")

    # Copy to Android assets
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    dst_model = ASSETS_DIR / "model.tflite"
    shutil.copy2(tflite_path, dst_model)
    print(f"\nCopied model to {dst_model}")

    dst_labels = ASSETS_DIR / "labels.txt"
    dst_labels.write_text("\n".join(labels), encoding="utf-8")
    print(f"Wrote labels to {dst_labels}")

    print("\nDone. Rebuild the Android app to use the new model.")


if __name__ == "__main__":
    main()
