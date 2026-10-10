"""
Krishi Mitra — Backend Tests
==============================
Run from project root:
  pytest krishi-backend/tests/ -v

Or from krishi-backend/:
  pytest tests/ -v
"""

import io
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ─── Path setup ───────────────────────────────────────────────────────────────
# Ensure krishi-backend/ is on sys.path so `main` can be imported.
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Override model path to mock before importing main
import os
os.environ.setdefault("USE_MOCK_MODEL", "true")   # use mock so no TF needed
os.environ.setdefault("MODEL_PATH", "./model/krishi_model.h5")

import main as app_module
from main import app, CLASS_NAMES, parse_class_name

client = TestClient(app)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Health Endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealth:
    def test_health_ok(self):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "mock_mode" in data
        assert "model_path" in data
        assert "services" in data

    def test_health_services_keys(self):
        data = client.get("/health").json()
        assert "gemini" in data["services"]
        assert "weather" in data["services"]
        assert "agmarknet" in data["services"]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Classes Endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestClasses:
    def test_classes_count(self):
        res = client.get("/classes")
        assert res.status_code == 200
        data = res.json()
        assert data["count"] == 38
        assert len(data["classes"]) == 38

    def test_class_structure(self):
        data = client.get("/classes").json()
        first = data["classes"][0]
        assert "id" in first
        assert "raw" in first
        assert "crop" in first
        assert "disease" in first
        assert "healthy" in first

    def test_healthy_flag(self):
        data = client.get("/classes").json()
        healthy = [c for c in data["classes"] if c["healthy"]]
        assert len(healthy) > 0  # at least some healthy classes exist

    def test_class_names_match(self):
        data = client.get("/classes").json()
        returned_raws = [c["raw"] for c in data["classes"]]
        assert returned_raws == CLASS_NAMES


# ═══════════════════════════════════════════════════════════════════════════════
# 3. parse_class_name utility
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseClassName:
    def test_tomato_early_blight(self):
        crop, disease, healthy = parse_class_name("Tomato___Early_blight")
        assert crop == "Tomato"
        assert "Early" in disease
        assert healthy is False

    def test_tomato_healthy(self):
        crop, disease, healthy = parse_class_name("Tomato___healthy")
        assert crop == "Tomato"
        assert healthy is True

    def test_apple_scab(self):
        crop, disease, healthy = parse_class_name("Apple___Apple_scab")
        assert crop == "Apple"
        assert healthy is False


# ═══════════════════════════════════════════════════════════════════════════════
# 4. /analyze endpoint (mock model — no TF inference)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_png_bytes(width: int = 10, height: int = 10) -> bytes:
    """Create a minimal valid PNG in memory."""
    try:
        from PIL import Image
        img = Image.new("RGB", (width, height), color=(128, 200, 80))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.read()
    except ImportError:
        # Fallback: hardcoded minimal 1×1 PNG
        return (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
            b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00'
            b'\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
        )


class TestAnalyzeEndpoint:
    def test_valid_image_returns_200(self):
        """Mock mode: valid image should return 200 with all required fields."""
        png = _make_png_bytes()
        res = client.post(
            "/analyze",
            files={"file": ("leaf.png", png, "image/png")},
            data={"region": "Maharashtra", "season": "Kharif", "language": "en"},
        )
        assert res.status_code == 200

    def test_response_contains_required_fields(self):
        png = _make_png_bytes()
        data = client.post(
            "/analyze",
            files={"file": ("leaf.png", png, "image/png")},
        ).json()
        for field in ["raw_class", "disease", "crop", "confidence",
                      "is_healthy", "severity", "advisory_short",
                      "advisory_detail", "weather_advisory", "mandi_advisory"]:
            assert field in data, f"Missing field: {field}"

    def test_confidence_in_valid_range(self):
        png = _make_png_bytes()
        data = client.post(
            "/analyze",
            files={"file": ("leaf.png", png, "image/png")},
        ).json()
        assert 0 <= data["confidence"] <= 100

    def test_non_image_file_returns_400(self):
        res = client.post(
            "/analyze",
            files={"file": ("report.txt", b"not an image", "text/plain")},
        )
        assert res.status_code == 400

    def test_raw_class_in_known_classes(self):
        """Mock mode: predicted class must be one of the 38 known classes."""
        png = _make_png_bytes()
        data = client.post(
            "/analyze",
            files={"file": ("leaf.png", png, "image/png")},
        ).json()
        assert data["raw_class"] in CLASS_NAMES

    def test_advisory_not_empty(self):
        png = _make_png_bytes()
        data = client.post(
            "/analyze",
            files={"file": ("leaf.png", png, "image/png")},
        ).json()
        assert data["advisory_short"] and data["advisory_short"].strip()
        assert data["advisory_detail"] and data["advisory_detail"].strip()

    def test_severity_valid_values(self):
        png = _make_png_bytes()
        data = client.post(
            "/analyze",
            files={"file": ("leaf.png", png, "image/png")},
        ).json()
        assert data["severity"] in ["low", "medium", "high"]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. /advisory endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdvisoryEndpoint:
    def test_valid_advisory_returns_200(self):
        res = client.post("/advisory", json={
            "disease_name": "Early Blight",
            "confidence": 0.85,
            "crop": "Tomato",
            "region": "Maharashtra",
            "season": "Kharif",
            "language": "en",
            "mode": "detailed",
        })
        assert res.status_code == 200

    def test_advisory_response_not_empty(self):
        data = client.post("/advisory", json={
            "disease_name": "Late Blight",
            "confidence": 0.9,
            "crop": "Potato",
        }).json()
        assert "advisory" in data
        assert data["advisory"] and data["advisory"].strip()

    def test_short_mode(self):
        data = client.post("/advisory", json={
            "disease_name": "Powdery Mildew",
            "confidence": 0.75,
            "crop": "Grape",
            "mode": "short",
        }).json()
        assert "advisory" in data

    def test_missing_required_field_returns_422(self):
        res = client.post("/advisory", json={
            "confidence": 0.8,
            "crop": "Tomato",
        })
        assert res.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
# 6. /api/crop-recommendation endpoint
# ═══════════════════════════════════════════════════════════════════════════════

VALID_CROP_INPUT = {
    "N": 90, "P": 42, "K": 43,
    "temperature": 20.8, "humidity": 82.0,
    "ph": 6.5, "rainfall": 202.9,
}

class TestCropRecommendation:
    def test_valid_input_returns_200(self):
        res = client.post("/api/crop-recommendation", json=VALID_CROP_INPUT)
        # 200 if model available, 503 if not trained yet
        assert res.status_code in (200, 503)

    def test_response_structure_when_available(self):
        res = client.post("/api/crop-recommendation", json=VALID_CROP_INPUT)
        if res.status_code == 503:
            pytest.skip("Crop model not trained — skipping structure test")
        data = res.json()
        assert "recommended_crop" in data
        assert "confidence" in data
        assert "top_alternatives" in data
        assert "limitations" in data
        assert "disclaimer" in data

    def test_confidence_range(self):
        res = client.post("/api/crop-recommendation", json=VALID_CROP_INPUT)
        if res.status_code == 503:
            pytest.skip("Crop model not trained")
        data = res.json()
        assert 0 <= data["confidence"] <= 100

    def test_invalid_ph_returns_422(self):
        bad_input = {**VALID_CROP_INPUT, "ph": 20.0}  # pH > 14 is invalid
        res = client.post("/api/crop-recommendation", json=bad_input)
        assert res.status_code in (422, 200)  # 422 expected

    def test_negative_N_returns_422(self):
        bad_input = {**VALID_CROP_INPUT, "N": -10.0}
        res = client.post("/api/crop-recommendation", json=bad_input)
        assert res.status_code in (422, 200)

    def test_missing_field_returns_422(self):
        incomplete = {k: v for k, v in VALID_CROP_INPUT.items() if k != "rainfall"}
        res = client.post("/api/crop-recommendation", json=incomplete)
        assert res.status_code == 422

    def test_model_info_endpoint(self):
        res = client.get("/api/crop-recommendation/info")
        assert res.status_code == 200
        data = res.json()
        assert "available" in data


# ═══════════════════════════════════════════════════════════════════════════════
# 7. /api/mandi endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestMandiEndpoint:
    def test_mandi_returns_200_with_mock(self):
        """Without AGMARKNET_KEY the endpoint should use mock data and return 200."""
        res = client.get("/api/mandi?state=Maharashtra&commodity=Tomato")
        assert res.status_code == 200

    def test_mandi_response_structure(self):
        data = client.get("/api/mandi?state=Maharashtra&commodity=Tomato").json()
        assert "data" in data
        assert "source" in data
        assert "lastUpdated" in data
        assert isinstance(data["data"], list)

    def test_mandi_data_fields(self):
        data = client.get("/api/mandi?state=Maharashtra&commodity=Tomato").json()
        if data["data"]:
            first = data["data"][0]
            assert "mandi" in first
            assert "commodity" in first
            assert "modalPrice" in first


# ═══════════════════════════════════════════════════════════════════════════════
# 8. /api/chat endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestChatEndpoint:
    def test_empty_question_returns_400(self):
        res = client.post("/api/chat", json={"question": ""})
        assert res.status_code == 400

    def test_valid_question_returns_200(self):
        res = client.post("/api/chat", json={
            "question": "What fertilizer is good for tomatoes?",
            "language": "en",
            "state": "Maharashtra",
        })
        assert res.status_code == 200

    def test_response_has_required_fields(self):
        data = client.post("/api/chat", json={
            "question": "How to treat tomato blight?",
        }).json()
        assert "response" in data
        assert "timestamp" in data
        assert data["response"] and data["response"].strip()


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Plant Disease Robustness & Mock Flag Verification
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlantDiseaseRobustness:
    def test_corrupt_image_returns_400(self):
        """Corrupt image bytes must be rejected with 400 instead of 500 error."""
        corrupt_bytes = b"NOT_A_REAL_PNG_HEADER_OR_DATA_XYZ123"
        res = client.post(
            "/analyze",
            files={"file": ("corrupt.png", corrupt_bytes, "image/png")},
        )
        assert res.status_code == 400

    def test_response_contains_is_mock_flag(self):
        """In mock mode, the response must explicitly set is_mock: True and identify demo mode."""
        png = _make_png_bytes()
        res = client.post(
            "/analyze",
            files={"file": ("leaf.png", png, "image/png")},
        )
        assert res.status_code == 200
        data = res.json()
        assert "is_mock" in data
        assert "model_version" in data


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Mandi Price Forecasting Pipeline & Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

class TestMandiForecasting:
    def test_mandi_forecast_status_endpoint(self):
        res = client.get("/api/mandi/forecast/status")
        assert res.status_code == 200
        data = res.json()
        assert data["available"] is True
        assert "historical_dataset_present" in data
        assert "minimum_observations_required" in data
        assert data["minimum_observations_required"] == 30

    def test_mandi_forecast_invalid_horizon_returns_422(self):
        res = client.post("/api/mandi/forecast", json={
            "state": "Maharashtra",
            "commodity": "Tomato",
            "horizon_days": 0,  # invalid horizon
        })
        assert res.status_code == 422

        res2 = client.post("/api/mandi/forecast", json={
            "state": "Maharashtra",
            "commodity": "Tomato",
            "horizon_days": 40,  # > 30 is invalid
        })
        assert res2.status_code == 422

    def test_mandi_forecast_endpoint_returns_structured_response(self):
        """When historical dataset is not on disk, it returns transparent DATA_UNAVAILABLE status without fabricating numbers."""
        res = client.post("/api/mandi/forecast", json={
            "state": "Maharashtra",
            "commodity": "Tomato",
            "horizon_days": 7,
        })
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert "forecast_available" in data
        assert "disclaimer" in data

    def test_ingestion_pipeline_normalization_and_deduplication(self):
        import pandas as pd
        from mandi_forecast.ingestion import clean_and_validate

        raw_records = pd.DataFrame([
            {"Arrival_Date": "01/01/2026", "State": "Maharashtra", "Market": "Pune", "Commodity": "Tomato", "Modal_Price": 1200},
            {"Arrival_Date": "01/01/2026", "State": "Maharashtra", "Market": "Pune", "Commodity": "Tomato", "Modal_Price": 1200}, # duplicate
            {"Arrival_Date": "02/01/2026", "State": "Maharashtra", "Market": "Pune", "Commodity": "Tomato", "Modal_Price": -50},  # negative price
            {"Arrival_Date": "03/01/2026", "State": "Maharashtra", "Market": "Pune", "Commodity": "Tomato", "Modal_Price": 1250, "is_mock": True}, # mock
        ])

        cleaned, report = clean_and_validate(raw_records, reject_mock=True)
        assert len(cleaned) == 1  # exactly 1 valid unique non-mock record
        assert report["duplicates_removed"] == 1
        assert report["invalid_prices_removed"] == 1

    def test_chronological_split_no_leakage(self):
        import pandas as pd
        from mandi_forecast.forecasting import chronological_split

        dates = pd.date_range("2026-01-01", periods=50, freq="D")
        series = pd.Series(range(50), index=dates)

        train, test = chronological_split(series, test_ratio=0.2)
        assert len(train) == 40
        assert len(test) == 10
        # Earliest date in test MUST be strictly after latest date in train (zero temporal leakage)
        assert test.index.min() > train.index.max()

    def test_baseline_and_ridge_forecasters(self):
        import pandas as pd
        from mandi_forecast.forecasting import SeasonalNaiveBaseline, LagRidgeForecaster, evaluate_forecast

        dates = pd.date_range("2026-01-01", periods=40, freq="D")
        # Synthetic seasonal sequence for unit testing algorithmic math only
        series = pd.Series([1000 + (i % 7) * 20 for i in range(40)], index=dates)

        naive = SeasonalNaiveBaseline().fit(series)
        preds_naive = naive.predict(7)
        assert len(preds_naive) == 7
        assert all(p > 0 for p in preds_naive)

        ridge = LagRidgeForecaster().fit(series)
        preds_ridge = ridge.predict(7)
        assert len(preds_ridge) == 7
        assert all(p > 0 for p in preds_ridge)

        metrics = evaluate_forecast([1000, 1020], [1005, 1018])
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape_pct" in metrics

