"""
Plant Disease Detection — Training Pipeline
============================================
Architecture: MobileNetV2 Transfer Learning
Dataset: PlantVillage (38 Classes)
Reference: https://www.tensorflow.org/datasets/catalog/plant_village

Usage:
  python train.py --data_dir /path/to/plantvillage --epochs 15 --batch_size 32
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def build_model(num_classes: int = 38, input_shape=(224, 224, 3)):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2

    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="mobilenetv2_plant_disease")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def train(data_dir: str, output_model_path: str, epochs: int = 15, batch_size: int = 32, seed: int = DEFAULT_SEED):
    import tensorflow as tf
    set_seed(seed)

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    print(f"Loading PlantVillage dataset from {data_dir}...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes.")

    model, base_model = build_model(num_classes=num_classes)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
    ]

    print("Stage 1: Training classifier head...")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    print("Stage 2: Fine-tuning top layers of base model...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max(5, epochs // 2),
        callbacks=callbacks,
    )

    out_file = Path(output_model_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_file))
    print(f"Model saved to {out_file}")

    # Also save class labels
    labels_file = out_file.parent / "trained_labels.json"
    with open(labels_file, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Labels saved to {labels_file}")


def main():
    parser = argparse.ArgumentParser(description="Train Plant Disease Detection Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to PlantVillage images directory")
    parser.add_argument("--output_model", type=str, default="model/krishi_model.h5", help="Path to save output .h5 model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_model_path=args.output_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
