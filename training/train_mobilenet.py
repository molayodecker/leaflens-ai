#!/usr/bin/env python3
"""
Train LeafLens plant disease classifier and export to TFLite.

Uses PlantVillage (Tomato, Maize) and optional local cocoa data.
Output: model.tflite + labels.txt in app/src/main/assets/
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
SEED = 42

# LeafLens labels (order must match model output)
LABELS = [
    "Cocoa:cocoa_black_pod",
    "Cocoa:healthy",
    "Maize:maize_rust",
    "Maize:healthy",
    "Tomato:tomato_early_blight",
    "Tomato:healthy",
]

SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR.parent / "app" / "src" / "main" / "assets"
DATA_DIR = SCRIPT_DIR / "data"
COCOA_DIR = DATA_DIR / "Cocoa"


def preprocess(img, label):
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, [INPUT_SIZE, INPUT_SIZE])
    img = img / 255.0
    return img, label


def build_model(num_classes: int):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    x = tf.keras.layers.Dense(128, activation="relu")(base.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=base.input, outputs=out)


def export_tflite(model, labels: list[str], path: Path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tflite_model)
    labels_path = path.parent / "labels.txt"
    labels_path.write_text("\n".join(labels), encoding="utf-8")
    print(f"Exported {path} and {labels_path}")


def main():
    import sys
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    print("Loading PlantVillage...", flush=True)
    sys.stdout.flush()
    ds_pv, info = tfds.load("plant_village", split="train", with_info=True, as_supervised=False)
    label_names = info.features["label"].names

    # Map PV labels to LeafLens indices (0–5)
    print(f"PlantVillage label names: {label_names}")
    pv_to_leaf = {}
    for i, name in enumerate(label_names):
        nl = name.lower()
        if "tomato" in nl and "early" in nl and "blight" in nl:
            pv_to_leaf[i] = 4
        elif "tomato" in nl and "healthy" in nl:
            pv_to_leaf[i] = 5
        elif ("maize" in nl or "corn" in nl) and ("rust" in nl or "common_rust" in nl):
            pv_to_leaf[i] = 2
        elif ("maize" in nl or "corn" in nl) and "healthy" in nl:
            pv_to_leaf[i] = 3
    print(f"Matched labels: {pv_to_leaf}")

    valid = set(pv_to_leaf.keys())
    valid_t = tf.constant(list(valid), dtype=tf.int64)
    mapping = [0] * len(label_names)
    for k, v in pv_to_leaf.items():
        mapping[k] = v

    def filter_fn(x):
        return tf.reduce_any(tf.equal(x["label"], valid_t))

    def map_fn(x):
        leaf_idx = tf.gather(tf.constant(mapping), x["label"])
        return (x["image"], tf.cast(leaf_idx, tf.int32))

    ds_pv = ds_pv.filter(filter_fn).map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    print(f"PlantVillage: {len(pv_to_leaf)} classes (Tomato Early blight/healthy, Maize rust/healthy)")

    # Optional cocoa
    cocoa_images, cocoa_labels = [], []
    for subdir, leaf_idx in [("cocoa_black_pod", 0), ("healthy", 1)]:
        folder = COCOA_DIR / subdir
        if folder.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for fp in folder.glob(ext):
                    cocoa_images.append(str(fp))
                    cocoa_labels.append(leaf_idx)

    num_classes = 6
    if cocoa_images:
        def load_img(path):
            raw = tf.io.read_file(path)
            return tf.io.decode_image(raw, channels=3)

        ds_cocoa = tf.data.Dataset.from_tensor_slices((cocoa_images, cocoa_labels))
        ds_cocoa = ds_cocoa.map(
            lambda p, l: (load_img(p), l),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds_all = ds_pv.concatenate(ds_cocoa)
        print(f"Cocoa: {len(cocoa_images)} samples added")
    else:
        ds_all = ds_pv
        print("No cocoa data found. Training on 4 classes (Tomato + Maize only).")
        # Use 6-class output; cocoa logits will stay near zero
        num_classes = 6

    # Preprocess, shuffle, and split ~80% train / 20% val
    ds_all = ds_all.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_all = ds_all.cache()
    ds_all = ds_all.shuffle(5000, seed=SEED)

    def is_train(idx, xy):
        return idx % 5 != 0

    def is_val(idx, xy):
        return idx % 5 == 0

    def drop_idx(idx, xy):
        return xy

    ds_train = (
        ds_all.enumerate()
        .filter(is_train)
        .map(drop_idx)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    ds_val = (
        ds_all.enumerate()
        .filter(is_val)
        .map(drop_idx)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    print("Train/val split: 80% / 20%")

    model = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Training...")
    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=3,
                restore_best_weights=True,
            )
        ],
    )

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    tflite_path = ASSETS_DIR / "model.tflite"
    export_tflite(model, LABELS, tflite_path)

    print("Done. Rebuild the Android app to use the real classifier.")


if __name__ == "__main__":
    main()
