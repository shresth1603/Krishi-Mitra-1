"""
Plant Disease Detection — Model Evaluation
===========================================
Calculates accuracy, precision, recall, F1-score, and confusion matrix
on a held-out test dataset using a trained Keras model.
"""

import argparse
import json
from pathlib import Path
import numpy as np

def evaluate_model(model_path: str, test_data_dir: str, labels_path: str = None):
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    print(f"Loading test dataset from {test_data_dir}...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_data_dir,
        image_size=(224, 224),
        batch_size=32,
        shuffle=False,
    )
    class_names = test_ds.class_names

    y_true = []
    y_pred = []

    print("Running inference on test dataset...")
    for images, labels in test_ds:
        preds = model.predict(images / 255.0, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    results = {
        "accuracy": float(acc),
        "f1_score_weighted": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": cr,
    }

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {acc:.4%}")
    print(f"Weighted F1: {f1:.4f}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate plant disease detection model")
    parser.add_argument("--model", type=str, default="model/krishi_model.h5", help="Path to .h5 model")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test images directory")
    args = parser.parse_args()

    evaluate_model(args.model, args.test_dir)

if __name__ == "__main__":
    main()
