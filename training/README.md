# LeafLens Model Training

Training pipeline for the plant disease classifier used by the LeafLens Android app.

## Quick Start

```bash
cd training
pip install -r requirements.txt
python train.py
```

This downloads the PlantVillage dataset, trains a MobileNetV2 model, and exports `model.tflite` to `../app/src/main/assets/`.

## Data Sources

### PlantVillage (auto-downloaded)

The script uses [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/plant_village) for:

| LeafLens Label | PlantVillage Class |
|----------------|-------------------|
| Tomato:tomato_early_blight | Tomato___Early_blight |
| Tomato:healthy | Tomato___healthy |
| Maize:maize_rust | Corn_(maize)___Common_rust |
| Maize:healthy | Corn_(maize)___healthy |

### Cocoa (optional custom data)

For cocoa black pod and healthy leaves, add your own images:

```
training/data/
├── Cocoa/
│   ├── cocoa_black_pod/    # images of diseased cocoa leaves
│   └── healthy/            # images of healthy cocoa leaves
```

Place JPEG/PNG images in these folders. If no cocoa data is found, the model trains on 4 classes (tomato + maize only) and cocoa predictions will be low-confidence.

## Model Specs

- **Architecture:** MobileNetV2 (transfer learning from ImageNet)
- **Input:** 224×224 RGB, normalized [0, 1]
- **Output:** 6-way softmax (or 4-way if no cocoa data)
- **Format:** TFLite (float32)
- **Labels:** See `app/src/main/assets/labels.txt`

## Output

- `model.tflite` → copied to `app/src/main/assets/`
- `labels.txt` → copied to `app/src/main/assets/` (if labels change)
- `checkpoints/` → saved Keras weights (optional)
- `training_log.png` → loss/accuracy curves (optional)
