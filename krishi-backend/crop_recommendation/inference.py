"""
Crop Recommendation — Inference Module
=======================================
Loads pre-trained RandomForest model and provides prediction function.
Called by main.py /api/crop-recommendation endpoint.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

log = logging.getLogger("krishi.crop_rec")

# Default model directory (relative to krishi-backend/)
_DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models" / "crop_recommendation"

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Feature ranges for validation (from dataset statistics + agronomic knowledge)
FEATURE_RANGES = {
    "N":           (0, 200,   "Nitrogen (kg/ha)"),
    "P":           (0, 200,   "Phosphorus (kg/ha)"),
    "K":           (0, 250,   "Potassium (kg/ha)"),
    "temperature": (0, 55,    "Temperature (°C)"),
    "humidity":    (0, 100,   "Humidity (%)"),
    "ph":          (0, 14,    "Soil pH"),
    "rainfall":    (0, 4000,  "Rainfall (mm/year)"),
}

_model   = None
_scaler  = None
_encoder = None
_metadata: Optional[dict] = None


def _load_artifacts(model_dir: Optional[Path] = None) -> None:
    """Load model, scaler and encoder from disk (once)."""
    global _model, _scaler, _encoder, _metadata
    if _model is not None:
        return

    d = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR

    model_path   = d / "crop_rf_model.joblib"
    scaler_path  = d / "crop_scaler.joblib"
    encoder_path = d / "crop_label_encoder.joblib"
    meta_path    = d / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Crop recommendation model not found at {model_path}. "
            "Run: python krishi-backend/crop_recommendation/train.py"
        )

    log.info(f"Loading crop recommendation model from {d} …")
    _model   = joblib.load(model_path)
    _scaler  = joblib.load(scaler_path)
    _encoder = joblib.load(encoder_path)

    if meta_path.exists():
        with open(meta_path) as f:
            _metadata = json.load(f)

    log.info(f"Crop recommendation model loaded ✓  ({_metadata['n_classes'] if _metadata else '?'} classes)")


def validate_inputs(inputs: dict) -> list[str]:
    """Return list of validation errors; empty list means OK."""
    errors = []
    for feat, (lo, hi, label) in FEATURE_RANGES.items():
        val = inputs.get(feat)
        if val is None:
            errors.append(f"Missing required field: '{feat}' ({label})")
        else:
            try:
                v = float(val)
                if v < lo or v > hi:
                    errors.append(f"'{feat}' = {v} is outside the valid range [{lo}, {hi}] ({label})")
            except (TypeError, ValueError):
                errors.append(f"'{feat}' must be a number, got: {repr(val)}")
    return errors


def predict_crop(inputs: dict, top_k: int = 3) -> dict:
    """
    Predict recommended crop from soil/climate inputs.

    Parameters
    ----------
    inputs : dict with keys N, P, K, temperature, humidity, ph, rainfall
    top_k  : number of top alternatives to return (max = n_classes)

    Returns
    -------
    dict with keys:
        recommended_crop  : str
        confidence        : float (0-100)
        top_alternatives  : list of {crop, confidence}
        feature_values    : dict of validated input values
        model_version     : str
        limitations       : list[str]
    """
    _load_artifacts()

    errors = validate_inputs(inputs)
    if errors:
        raise ValueError(f"Input validation failed: {'; '.join(errors)}")

    X = np.array([[float(inputs[f]) for f in FEATURE_COLS]], dtype=np.float32)
    X_sc = _scaler.transform(X)

    probas = _model.predict_proba(X_sc)[0]
    top_indices = np.argsort(probas)[::-1][:top_k]

    recommended = _encoder.classes_[top_indices[0]]
    confidence  = round(float(probas[top_indices[0]]) * 100, 1)

    alternatives = [
        {"crop": _encoder.classes_[i], "confidence": round(float(probas[i]) * 100, 1)}
        for i in top_indices[1:]
        if probas[i] > 0.01
    ]

    meta = _metadata or {}
    return {
        "recommended_crop":  recommended,
        "confidence":        confidence,
        "top_alternatives":  alternatives,
        "feature_values": {f: float(inputs[f]) for f in FEATURE_COLS},
        "model_version":     meta.get("model_type", "RandomForestClassifier"),
        "train_accuracy":    meta.get("metrics", {}).get("test_accuracy"),
        "dataset_source":    meta.get("dataset_source", "unknown"),
        "limitations":       meta.get("limitations", []),
    }


def get_model_info() -> dict:
    """Return model metadata without loading model if not already loaded."""
    meta_path = _DEFAULT_MODEL_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {"status": "model not trained"}
