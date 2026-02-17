# LeafLens Model Training

Training pipeline for the plant disease classifier used by the LeafLens Android app.

## Quick Start

```bash
cd training
pip install -r requirements.txt
python prepare_dataset.py
python train_yolov8_cls.py
```

1. `prepare_dataset.py` downloads PlantVillage, extracts relevant classes, and organizes images into YOLO classification folder format.
2. `train_yolov8_cls.py` trains a YOLOv8n classifier, exports to TFLite, and copies `model.tflite` + `labels.txt` to `../app/src/main/assets/`.

## Data Sources

### PlantVillage (auto-downloaded)

The dataset prep script uses [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/plant_village) for:

| LeafLens Label | PlantVillage Class | Dataset Folder |
|----------------|-------------------|----------------|
| Tomato:tomato_early_blight | Tomato___Early_blight | tomato__early_blight/ |
| Tomato:healthy | Tomato___healthy | tomato__healthy/ |
| Maize:maize_rust | Corn_(maize)___Common_rust | maize__rust/ |
| Maize:healthy | Corn_(maize)___healthy | maize__healthy/ |

### Cocoa (optional custom data)

For cocoa black pod and healthy leaves, add your own images:

```
training/data/
  Cocoa/
    cocoa_black_pod/    # images of diseased cocoa leaves
    healthy/            # images of healthy cocoa leaves
```

Place JPEG/PNG images in these folders. If no cocoa data is found, empty class folders are created so the model trains with 6 classes (cocoa predictions will be low-confidence).

## Dataset Structure

After running `prepare_dataset.py`:

```
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
```

## Model Specs

- **Architecture:** YOLOv8n-cls (nano classification)
- **Input:** 224x224 RGB, normalized [0, 1]
- **Output:** 6-way softmax
- **Format:** TFLite (float32)
- **Labels:** See `app/src/main/assets/labels.txt`
- **Training:** 30 epochs, batch 64, on PlantVillage + optional cocoa data

## Output

- `model.tflite` → copied to `app/src/main/assets/`
- `labels.txt` → copied to `app/src/main/assets/`
- `runs/ccmt_cls/` → training logs and weights

## Legacy

The previous MobileNetV2 training script is preserved as `train_mobilenet.py` for reference.
