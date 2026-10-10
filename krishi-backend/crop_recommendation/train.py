"""
Crop Recommendation — Training Script
======================================
Dataset: Kaggle crop-recommendation-dataset (CC BY 3.0 IGO)
         https://www.kaggle.com/datasets/siddharthss/crop-recommendation-dataset

Inputs (7 features):
  N, P, K, temperature, humidity, ph, rainfall

Output: recommended crop label (22 classes)

Run:
  python krishi-backend/crop_recommendation/train.py \
      --data krishi-backend/datasets/crop_recommendation/Crop_recommendation.csv \
      --out  krishi-backend/models/crop_recommendation

This script saves:
  models/crop_recommendation/crop_rf_model.joblib   — trained RandomForest
  models/crop_recommendation/crop_scaler.joblib     — StandardScaler
  models/crop_recommendation/crop_label_encoder.joblib  — LabelEncoder
  models/crop_recommendation/metadata.json          — training config + metrics
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 42
FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
LABEL_COL = "label"


def load_and_validate(csv_path: Path) -> pd.DataFrame:
    """Load CSV and run basic sanity checks."""
    df = pd.read_csv(csv_path)

    missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    nulls = df[FEATURE_COLS + [LABEL_COL]].isnull().sum()
    if nulls.any():
        print(f"WARNING — null values detected:\n{nulls[nulls > 0]}")
        df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
        print(f"Dropped nulls → {len(df)} rows remain.")

    dups = df.duplicated().sum()
    if dups:
        print(f"WARNING — {dups} duplicate rows found and removed.")
        df = df.drop_duplicates()

    print(f"Dataset shape: {df.shape}")
    print(f"Classes ({df[LABEL_COL].nunique()}): {sorted(df[LABEL_COL].unique().tolist())}")
    print(f"Class distribution (min/max rows):")
    vc = df[LABEL_COL].value_counts()
    print(f"  min={vc.min()} ({vc.idxmin()})  max={vc.max()} ({vc.idxmax()})")

    # Detect suspicious values
    for col in FEATURE_COLS:
        low, high = df[col].quantile(0.01), df[col].quantile(0.99)
        out = ((df[col] < low) | (df[col] > high)).sum()
        if out > 0:
            print(f"  Note: {out} potential outliers in '{col}' (outside 1st–99th percentile)")

    return df


def build_and_evaluate(df: pd.DataFrame, out_dir: Path) -> dict:
    """Train, evaluate, save model and metadata. Returns metrics dict."""
    X = df[FEATURE_COLS].values.astype(np.float32)
    y_raw = df[LABEL_COL].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Stratified split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # ── Baseline: Decision Tree ────────────────────────────────────────────────
    print("\n--- Baseline: Decision Tree ---")
    dt = DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=None)
    dt.fit(X_train_sc, y_train)
    dt_pred = dt.predict(X_test_sc)
    dt_acc = accuracy_score(y_test, dt_pred)
    print(f"Decision Tree accuracy: {dt_acc:.4f}")

    # ── Primary: Random Forest ─────────────────────────────────────────────────
    print("\n--- Primary: Random Forest (n=200) ---")
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train_sc, y_train)
    rf_pred = rf.predict(X_test_sc)
    rf_proba = rf.predict_proba(X_test_sc)

    acc  = accuracy_score(y_test, rf_pred)
    prec = precision_score(y_test, rf_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, rf_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, rf_pred, average="weighted", zero_division=0)

    print(f"Random Forest — Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")

    # 5-fold cross-validation on full data
    print("\nRunning 5-fold cross-validation on full dataset …")
    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
        scaler.fit_transform(X),
        y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        scoring="accuracy",
    )
    # Re-fit scaler on train only for final model
    scaler.fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    rf.fit(X_train_sc, y_train)

    print(f"CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, rf_pred, target_names=le.classes_, zero_division=0))

    # ── Feature importance ─────────────────────────────────────────────────────
    importances = dict(zip(FEATURE_COLS, rf.feature_importances_.tolist()))

    # ── Save artifacts ─────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf,     out_dir / "crop_rf_model.joblib")
    joblib.dump(scaler, out_dir / "crop_scaler.joblib")
    joblib.dump(le,     out_dir / "crop_label_encoder.joblib")

    metadata = {
        "model_type":         "RandomForestClassifier",
        "n_estimators":       200,
        "random_seed":        RANDOM_SEED,
        "features":           FEATURE_COLS,
        "classes":            le.classes_.tolist(),
        "n_classes":          int(len(le.classes_)),
        "train_samples":      int(len(X_train)),
        "test_samples":       int(len(X_test)),
        "dataset_source":     "Kaggle: siddharthss/crop-recommendation-dataset (CC BY 3.0 IGO)",
        "dataset_rows":       int(len(df)),
        "metrics": {
            "test_accuracy":          round(float(acc), 4),
            "test_precision_weighted": round(float(prec), 4),
            "test_recall_weighted":    round(float(rec), 4),
            "test_f1_weighted":        round(float(f1), 4),
            "baseline_dt_accuracy":    round(float(dt_acc), 4),
            "cv_5fold_mean":           round(float(cv_scores.mean()), 4),
            "cv_5fold_std":            round(float(cv_scores.std()), 4),
        },
        "feature_importance": {k: round(v, 4) for k, v in sorted(importances.items(), key=lambda x: -x[1])},
        "limitations": [
            "Prototype dataset — Indian regional soil and climate diversity not fully represented.",
            "Recommendations are based on average lab-measured parameters.",
            "Predictions should be validated with local agronomists before acting.",
            "Dataset license: CC BY 3.0 IGO — suitable for educational and prototyping use.",
        ],
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel artifacts saved to: {out_dir}")
    for fpath in out_dir.iterdir():
        size_kb = fpath.stat().st_size // 1024
        print(f"  {fpath.name}  ({size_kb} KB)")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train crop recommendation model")
    parser.add_argument(
        "--data", type=Path,
        default=Path("krishi-backend/datasets/crop_recommendation/Crop_recommendation.csv"),
        help="Path to Crop_recommendation.csv",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("krishi-backend/models/crop_recommendation"),
        help="Output directory for model artifacts",
    )
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("Krishi Mitra — Crop Recommendation Training")
    print("=" * 60)

    df = load_and_validate(args.data)
    metrics = build_and_evaluate(df, args.out)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Test Accuracy : {metrics['metrics']['test_accuracy']:.4f}")
    print(f"  Test F1       : {metrics['metrics']['test_f1_weighted']:.4f}")
    print(f"  CV Accuracy   : {metrics['metrics']['cv_5fold_mean']:.4f} ± {metrics['metrics']['cv_5fold_std']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
