# 🌱 Krishi Mitra (कृषि मित्र) — ML-Powered Agricultural Platform

Krishi Mitra is an integrated agricultural intelligence platform designed to assist Indian farmers with machine-learning-driven crop diagnostics, data-backed crop suitability recommendations, real-time APMC mandi prices, econometric price forecasting, hyperlocal weather risk alerts, and government agricultural scheme discovery.

---

## 🌟 Key Features

1. **🔬 Real-Time Plant Disease Detection (ML Model 1)**
   - Pre-trained MobileNetV2 deep learning model trained on the 38-class PlantVillage dataset.
   - Accepts leaf photographs (JPEG/PNG), normalizes to 224×224×3, and provides genuine class predictions.
   - Integrated with Gemini AI for localized, actionable 48-hour chemical and organic recovery advisories.
   - Built-in corrupted image detection and strict separation of statistical confidence from agronomic severity.

2. **🌱 Intelligent Crop Recommendation (ML Model 2)**
   - Random Forest Classifier (200 trees, balanced class weights, StandardScaler).
   - Validated on 2,200 soil and agro-climatic records across 22 major Indian crops.
   - **Performance**: 99.55% test accuracy, 99.59% ± 0.27% 5-fold cross-validation.
   - Analyzes Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, Soil pH, and Rainfall.
   - Provides ranked crop alternatives with probabilities and grounded agricultural disclaimers.

3. **📊 Mandi Spot Prices & Econometric Forecasting (ML Model 3)**
   - Real-time spot price integration across 2,700+ mandis via the official Agmarknet / data.gov.in resource.
   - Time-series price forecasting pipeline utilizing Seasonal-Naive baselines and Autoregressive Lag-Feature Ridge regression.
   - Strict chronological splitting (no future data leakage) and MAE/RMSE evaluation metrics.
   - Strict anti-fabrication policy: reports data requirements clearly when historical depth is below 30 days.

4. **🌾 Farmer AI Chat & Multilingual Advisory**
   - Context-aware chatbot with state weather, season, and soil context.
   - Multilingual support for Hindi, Malayalam, Bengali, Telugu, Marathi, and English.
   - Voice input (Speech Recognition) and Text-to-Speech audio readout for accessibility.

5. **🌤️ Weather & Agricultural Advisories**
   - Hyperlocal current conditions and 5-day forecasts.
   - Fungal disease risk indicators, spray window warnings, and irrigation advisories.

6. **🏛️ Government Schemes Directory**
   - Interactive database of 30+ central and state schemes (PM-KISAN, PMFBY, etc.) with eligibility filters.

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python**: Version 3.11 – 3.13
- **Node.js**: Version 18.x or 20.x
- **Git**

### 1. Backend Setup (FastAPI + ML)

```bash
# Navigate to backend directory
cd krishi-backend

# (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
copy env.example .env   # On Windows
# cp env.example .env   # On Linux/macOS
```

#### Required Environment Variables (`krishi-backend/.env`)
```ini
# Gemini API Key (Get at https://aistudio.google.com/app/apikey)
GEMINI_API_KEY=your_gemini_api_key_here

# OpenWeatherMap API Key (Get at https://openweathermap.org/api)
WEATHER_API_KEY=your_openweather_api_key_here

# Agmarknet / data.gov.in Key (Optional — fallback mock data used if unset)
AGMARKNET_KEY=your_agmarknet_key_here

# Plant Disease Model Path
MODEL_PATH=./model/krishi_model.h5
USE_MOCK_MODEL=false

PORT=8000
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
```

#### Start Backend Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
The interactive Swagger API documentation will be available at `http://127.0.0.1:8000/docs`.

---

### 2. Frontend Setup (React + Vite)

```bash
# From repository root
npm install

# Start development server
npm run dev
```
The application will launch at `http://localhost:5173/`.

---

## 🧠 Machine Learning Pipelines & Commands

### Model 1: Plant Disease Detection (MobileNetV2)
- **Model Files**: `krishi-backend/model/krishi_model.h5` (10.1 MB) and `krishi_model.tflite` (9.0 MB).
- **Labels Mapping**: `krishi-backend/disease_detection/labels.json` (38 classes).
- **Train New Model**:
  ```bash
  python krishi-backend/disease_detection/train.py --data_dir /path/to/PlantVillage --epochs 15
  ```
- **Evaluate Model**:
  ```bash
  python krishi-backend/disease_detection/evaluate.py --model krishi-backend/model/krishi_model.h5 --test_dir /path/to/test_data
  ```

### Model 2: Crop Recommendation (Random Forest)
- **Dataset**: `krishi-backend/datasets/crop_recommendation/Crop_recommendation.csv` (Kaggle, CC BY 3.0 IGO).
- **Trained Artifacts**: `krishi-backend/models/crop_recommendation/`
- **Train & Evaluate**:
  ```bash
  python krishi-backend/crop_recommendation/train.py
  ```
  *Output: 99.55% accuracy, saves model, scaler, label encoder, and metadata.json.*

### Model 3: Mandi Price Forecasting (Time-Series)
- **Module**: `krishi-backend/mandi_forecast/`
- **Ingestion & Validation**:
  Place historical daily APMC records in `krishi-backend/datasets/mandi_historical/`.
- **Pipeline Architecture**:
  - Ingestion: `mandi_forecast/ingestion.py`
  - Engine: `mandi_forecast/forecasting.py` (Seasonal-Naive & Lag-Ridge)
  - Service: `mandi_forecast/service.py`

---

## 🧪 Testing & Verification

### Run Backend Tests (Pytest)
```bash
python -m pytest krishi-backend/tests/test_backend.py -v
```
*Current test suite: **41/41 passing** covering health, class parsing, plant disease image decoding, corrupt image rejection, mock flag reporting, crop recommendation validation & inference, mandi proxy, mandi forecasting time-series split & models, and chat proxy.*

### Run Frontend Verification
```bash
# Linting
npm run lint

# Production Build
npm run build
```

---

## ⚠️ Known Limitations & Responsible AI Disclaimers

1. **Plant Disease Detection**: PlantVillage training images are photographed in controlled/laboratory settings with plain backgrounds. Real-world field accuracy may vary under complex natural canopy backgrounds or direct sun glare. All predictions are supportive guidance and should be verified by a local agricultural extension officer or Krishi Vigyan Kendra (KVK).
2. **Crop Recommendation**: Model recommendations represent generalized agronomic profiles. Farmers must consider localized microclimate, soil drainage, seed variety, and water availability before planting.
3. **Mandi Price Forecasts**: Mandi price forecasts are econometric statistical models and not guaranteed selling prices. Spot prices vary according to daily arrivals, weather disruptions, and local market demand.
