# 🌱 Krishi AI — Complete System & ML Project Structure

This document represents the actual file tree and architecture of the Krishi Mitra platform (React 18 + Vite frontend & FastAPI machine learning backend).

---

## 📁 Repository Structure

```
Krishi-Mitra-1/
├── krishi-backend/                         # FastAPI Machine Learning Backend
│   ├── crop_recommendation/                # ML Model 2: Crop Recommendation
│   │   ├── __init__.py
│   │   ├── inference.py                    # Inference function, validation & metadata loader
│   │   └── train.py                        # Reproducible Random Forest training pipeline
│   ├── disease_detection/                  # ML Model 1: Plant Disease Detection
│   │   ├── __init__.py
│   │   ├── evaluate.py                     # Evaluation script (accuracy, F1, confusion matrix)
│   │   ├── labels.json                     # 38 PlantVillage class mappings
│   │   ├── metadata.json                   # Architecture & field validation limitations
│   │   └── train.py                        # MobileNetV2 transfer learning pipeline
│   ├── mandi_forecast/                     # ML Model 3: Mandi Price Forecasting
│   │   ├── __init__.py
│   │   ├── forecasting.py                  # Chronological splitting, Seasonal Naive & Lag Ridge
│   │   ├── ingestion.py                    # Schema normalization, date parsing & deduplication
│   │   └── service.py                      # Forecast coordinator & grounded status reporting
│   ├── datasets/                           # Machine Learning Datasets (Ignored in Git)
│   │   ├── crop_recommendation/
│   │   │   └── Crop_recommendation.csv     # 2,200 rows, 22 crops, CC BY 3.0 IGO
│   │   └── mandi_historical/               # Target folder for verified Agmarknet archives
│   ├── model/                              # Pre-trained Plant Disease Model Artifacts
│   │   ├── krishi_model.h5                 # Keras HDF5 model (224×224×3 -> 38 classes)
│   │   └── krishi_model.tflite             # Mobile/Edge optimized TFLite binary
│   ├── models/                             # Trained Scikit-Learn Model Artifacts (Ignored in Git)
│   │   └── crop_recommendation/
│   │       ├── crop_label_encoder.joblib
│   │       ├── crop_rf_model.joblib        # Fitted Random Forest Classifier (200 trees)
│   │       ├── crop_scaler.joblib          # Fitted StandardScaler
│   │       └── metadata.json               # Training metrics (99.55% accuracy) & config
│   ├── tests/                              # Comprehensive Backend Test Suite
│   │   ├── __init__.py
│   │   └── test_backend.py                 # 41 pytest tests covering all endpoints & ML logic
│   ├── env.example                         # Example environment variables with documentation
│   ├── main.py                             # FastAPI server & route orchestration
│   └── requirements.txt                    # Python dependencies
│
├── src/                                    # React + Vite Frontend
│   ├── components/                         # UI Components
│   │   ├── DiseaseResultCard.jsx           # Renders ML plant disease & advisory reports
│   │   ├── FarmerHelper.jsx                # Interactive farmer prompt suggestion bubbles
│   │   ├── Features.jsx                    # Homepage service cards & routing
│   │   ├── Footer.jsx                      # App footer
│   │   ├── Header.jsx                      # Navigation header with auth & route links
│   │   ├── Hero.jsx                        # Homepage banner
│   │   ├── HowItWorks.jsx                  # 3-step farmer user guide
│   │   ├── MobileBottomNav.jsx             # Responsive mobile navigation bar
│   │   ├── ProtectedRoute.jsx              # Authentication route guard
│   │   ├── QueryInterface.jsx              # AI agricultural query input
│   │   ├── ResponseFormatter.jsx           # Markdown & structured advisory formatter
│   │   ├── ResponseSection.jsx             # Sample query responses
│   │   ├── SkeletonLoader.jsx              # Shimmer loader for async states
│   │   ├── TopStrip.jsx                    # Language switcher & government branding
│   │   └── WeatherWidget.jsx               # Real-time weather display
│   ├── context/                            # Global State Management
│   │   ├── AuthContext.jsx                 # User authentication & session state
│   │   └── LanguageContext.jsx             # Multilingual localization (English, Hindi, Malayalam)
│   ├── hooks/                              # Custom React Hooks
│   │   ├── useDiseaseScan.js               # Multi-stage image upload, inference & advisory hook
│   │   ├── useMarketPrices.js              # Real-time mandi price fetching hook
│   │   └── useWeather.js                   # Weather data fetching hook
│   ├── pages/                              # Page Views
│   │   ├── AdvisoriesPage.jsx              # Agronomic season advisories
│   │   ├── ChatPage.jsx                    # AI Farmer Assistant chat & leaf scanner
│   │   ├── ContactPage.jsx                 # Expert & KVK contact directory
│   │   ├── CropRecommendationPage.jsx      # ML Model 2: Soil & climate crop recommender
│   │   ├── HomePage.jsx                    # Landing page
│   │   ├── LoginPage.jsx                   # Farmer authentication & signup
│   │   ├── MarketPricePage.jsx             # Real-time spot prices & ML Model 3 price forecasting
│   │   ├── ProfilePage.jsx                 # User profile & saved farm details
│   │   ├── SchemesPage.jsx                 # Central & state government schemes directory
│   │   └── WeatherPage.jsx                 # 5-day weather forecast & agricultural risk alerts
│   ├── services/                           # API Service Layer
│   │   ├── cropService.js                  # Crop recommendation ML endpoints
│   │   ├── diseaseService.js               # Leaf disease detection backend API
│   │   ├── geminiService.js                # AI chatbot API proxy
│   │   ├── marketService.js                # Spot prices & econometric forecasting API
│   │   ├── voiceService.js                 # Speech-to-text & text-to-speech helpers
│   │   └── weatherService.js               # Weather API proxy
│   ├── styles/                             # Component CSS files
│   ├── utils/                              # Utility helpers
│   ├── App.css                             # Global styles & design system tokens
│   ├── App.jsx                             # Main routing setup
│   ├── index.css                           # Reset & typography
│   └── main.jsx                            # React entry point
│
├── .gitignore                              # Comprehensive version control ignore rules
├── AGENTS.md                               # Project rules & guidelines
├── eslint.config.js                        # Flat ESLint configuration
├── ML_IMPLEMENTATION_PLAN.md               # ML architecture, benchmarks & verification matrix
├── package.json                            # Frontend Node dependencies & scripts
├── PROJECT_STRUCTURE.md                    # Actual project structure documentation (this file)
├── README.md                               # Setup, run, training & API documentation
├── vercel.json                             # Deployment configuration
└── vite.config.js                          # Vite build configuration
```

---

## 🔌 API Endpoints Summary

| Endpoint | Method | Feature | Inputs |
| :--- | :--- | :--- | :--- |
| `/analyze` | `POST` | ML Disease Detection + Gemini Advisory | Leaf image file (JPEG/PNG), region, season, language |
| `/advisory` | `POST` | Agronomic Advisory Refresh | Disease name, confidence, crop, region, season |
| `/api/crop-recommendation` | `POST` | ML Crop Recommendation (Random Forest) | N, P, K, temperature, humidity, pH, rainfall |
| `/api/crop-recommendation/info` | `GET` | Crop Model Metadata & Classes | None |
| `/api/mandi` | `GET` | Live Spot Mandi Prices (Agmarknet) | state, commodity, limit |
| `/api/mandi/forecast` | `POST` | Mandi Price Forecasting (Time-Series) | state, commodity, market, horizon_days (1–30) |
| `/api/mandi/forecast/status` | `GET` | Mandi Forecast Ingestion Status | None |
| `/api/chat` | `POST` | Gemini Multilingual Chat Proxy | question, language, state, weather_summary |
| `/api/weather` | `GET` | Weather & Advisory Proxy | lat, lon OR city |
| `/classes` | `GET` | 38 Plant Disease Categories | None |
| `/health` | `GET` | System & Model Health Status | None |
