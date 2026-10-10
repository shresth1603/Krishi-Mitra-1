# Krishi Mitra — Machine Learning Implementation Plan & Status

This document reflects the verified implementation, architecture, and deployment status of all three machine learning features in the Krishi Mitra ecosystem.

---

## Executive Summary & Status Matrix

| ML Feature | Algorithm / Architecture | Dataset | Implementation Status | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **1. Plant Disease Detection** | MobileNetV2 Transfer Learning (224×224×3 RGB, 38 Classes) | PlantVillage (TensorFlow Datasets) | **Implemented and verified** | Verified with 38-class weights, image decoding & validation, FastAPI `/analyze` contract |
| **2. Crop Recommendation** | Random Forest Classifier (200 trees, balanced weights) + StandardScaler | Kaggle Crop Recommendation (2,200 samples, 22 crops, CC BY 3.0 IGO) | **Implemented and verified** | Verified: 99.55% test accuracy, 99.59% 5-fold CV, `/api/crop-recommendation` tested |
| **3. Mandi Price Forecasting** | Seasonal Naive Baseline (7-day lag) & Autoregressive Lag-Feature Ridge Regression | Agmarknet / data.gov.in daily market arrivals | **Implemented (Engine & Pipeline Verified); Blocked on Historical Archive** | Ingestion pipeline, chronological splitting, and baseline tested. Blocked on historical depth for custom Mandis (requires >= 30 historical records) |

---

## 1. Plant Disease Detection (MobileNetV2)

### A. Model Architecture & Artifacts
- **Model Files**: `krishi-backend/model/krishi_model.h5` (10.1 MB) and `krishi_model.tflite` (9.0 MB).
- **Input Dimensions**: `(None, 224, 224, 3)` with pixel values normalized to `[0.0, 1.0]`.
- **Output Dimensions**: `(None, 38)` corresponding to 38 PlantVillage crop-disease categories.
- **Labels Mapping**: Stored in `krishi-backend/disease_detection/labels.json` and loaded by `main.py`.

### B. Production Inference & Guardrails
- **Endpoint**: `POST /analyze` (multipart form with image file, region, season, language).
- **Corrupt Image Guard**: Validates image headers using Pillow `img.verify()` and rejects corrupted bytes with HTTP 400.
- **Mock Separation**: Demo/mock mode is strictly governed by `USE_MOCK_MODEL=true`. When active, responses explicitly return `"is_mock": true` and `"model_version": "MOCK/DEMO (random)"`. Production mode runs genuine ML tensor inference.
- **Confidence vs. Severity**: Classification confidence represents statistical model probability, not disease severity. Initial agronomic severity is determined independently by crop disease risk characteristics and Gemini agronomic evaluation.
- **Reproducible Pipeline**: `krishi-backend/disease_detection/train.py` and `krishi-backend/disease_detection/evaluate.py`.

### C. Limitations
- Laboratory-trained images (PlantVillage) may exhibit lower real-world accuracy under field glare or complex weed backgrounds.
- AI diagnoses are flagged as supportive guidance, advising farmers to confirm critical treatments with their local Krishi Vigyan Kendra (KVK).

---

## 2. Crop Recommendation (Random Forest)

### A. Data & Training Pipeline
- **Dataset**: `krishi-backend/datasets/crop_recommendation/Crop_recommendation.csv` (2,200 rows, 8 columns).
- **Input Features (7)**:
  1. `N` (Nitrogen): 0 – 200 kg/ha
  2. `P` (Phosphorus): 0 – 200 kg/ha
  3. `K` (Potassium): 0 – 250 kg/ha
  4. `temperature`: 0 – 55 °C
  5. `humidity`: 0 – 100 %
  6. `ph`: 0 – 14 (soil acidity/alkalinity)
  7. `rainfall`: 0 – 4000 mm annual
- **Target (22 Classes)**: Apple, Banana, Blackgram, Chickpea, Coconut, Coffee, Cotton, Grapes, Jute, Kidneybeans, Lentil, Maize, Mango, Mothbeans, Mungbean, Muskmelon, Orange, Papaya, Pigeonpeas, Pomegranate, Rice, Watermelon.
- **Split Strategy**: 80% train / 20% test stratified split (`random_state=42`).
- **Trained Artifacts**: Saved in `krishi-backend/models/crop_recommendation/`:
  - `crop_rf_model.joblib` (7.1 MB)
  - `crop_scaler.joblib`
  - `crop_label_encoder.joblib`
  - `metadata.json`

### B. Verified Performance
- **Baseline (Decision Tree)**: 97.95% accuracy
- **Random Forest (200 Estimators)**:
  - Test Accuracy: **99.55%**
  - Weighted Precision: **99.57%**
  - Weighted Recall: **99.55%**
  - Weighted F1-Score: **99.55%**
  - 5-Fold Stratified Cross-Validation: **99.59% ± 0.27%**

### C. API & Frontend Integration
- **Backend Endpoints**:
  - `POST /api/crop-recommendation`: Validates inputs, predicts primary crop and ranked alternatives.
  - `GET /api/crop-recommendation/info`: Returns model metadata, class lists, and limitations.
- **Frontend UI**: `src/pages/CropRecommendationPage.jsx` with quick presets, interactive sliders/inputs, and grounded limitations banner. Route `/crop-recommendation` linked in Header and Features.

---

## 3. Mandi Price Forecasting (Time-Series)

### A. Pipeline & Architecture
- **Package Location**: `krishi-backend/mandi_forecast/`
- **Data Ingestion (`ingestion.py`)**:
  - Normalizes Agmarknet schemas (maps `Arrival_Date`, `State`, `Market`, `Commodity`, `Modal_Price`).
  - Strict date parsing with multi-format fallback.
  - Rejects synthetic or mock records.
  - Enforces positive modal prices and deduplicates records.
- **Forecasting Engine (`forecasting.py`)**:
  - Requires minimum 30 daily observations per commodity/market.
  - Strict chronological splitting (80% past / 20% future) to prevent future data leakage.
  - Baseline model: `SeasonalNaiveBaseline` (7-day periodicity).
  - Machine learning model: `LagRidgeForecaster` (lags 1, 2, 3, 7, 7-day rolling mean, day-of-week).
  - Evaluation metrics: MAE, RMSE, MAPE on held-out test split.
- **Service Layer (`service.py`)**:
  - If verified historical archive is absent, returns `status: "DATA_UNAVAILABLE"` explaining data requirements instead of fabricating forecasts.
  - If historical data is provided, trains the pipeline and returns 1–14 day projections.

### B. API & Frontend Integration
- **Backend Endpoints**:
  - `POST /api/mandi/forecast`: Generates or reports forecast for requested horizon (1–30 days).
  - `GET /api/mandi/forecast/status`: Reports readiness, data catalog URL, and observation thresholds.
- **Frontend UI**: Integrated into `src/pages/MarketPricePage.jsx` under the "🔮 Price Forecasting (ML Model 3)" tab, providing side-by-side spot prices and forward-looking econometric projections.

### C. Blocker & Next Steps
- Real-time Agmarknet API endpoint serves daily spot prices, not historical multi-year records. To train site-specific mandi forecasts, download historical CSV files from the Data.gov.in portal and place them in `krishi-backend/datasets/mandi_historical/`.
