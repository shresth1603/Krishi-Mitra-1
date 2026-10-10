# Krishi Mitra — ML Audit, Dataset Plan & Antigravity Prompt

**Prepared:** 10 October 2026  
**Repository:** https://github.com/ak6862219-cloud/Krishi-Mitra-1

## 1. Audit of the existing project

### Features already present in the repository

- **Frontend:** React + Vite, React Router, auth/language contexts, pages for login, market prices, weather, chat, profile, schemes, advisories and contact.
- **Farmer chat:** `src/pages/ChatPage.jsx` calls the FastAPI backend through `src/services/geminiService.js`. Voice input/output is also implemented in the chat UI.
- **Image-diagnosis flow:** `src/hooks/useDiseaseScan.js` calls a disease service; the backend exposes `POST /analyze` and `POST /advisory`.
- **Backend:** `krishi-backend/main.py` is a FastAPI service. It exposes `/health`, `/classes`, `/analyze`, `/advisory`, `/api/chat`, `/api/weather` and `/api/mandi`.
- **Weather:** a backend OpenWeatherMap proxy and frontend weather service already exist.
- **Mandi prices:** a backend AGMARKNET/data.gov.in proxy already exists. It uses an API key when configured and falls back to mock values when the key is absent or a request fails.
- **Existing ML interface:** the backend already contains image preprocessing and model-inference code for 38 hard-coded PlantVillage-style classes.

### Critical gaps to verify/fix before calling the project “real ML”

1. `krishi-backend/main.py` expects a model file at `./model/krishi_model.h5` by default (`MODEL_PATH`). Its `predict_disease()` function uses randomly selected classes and random confidence scores when `USE_MOCK_MODEL=true`. **A random mock result must never be presented to a farmer as a real diagnosis.**
2. The backend source contains inference logic, but no training pipeline is visible in `main.py`. Check whether a separate training script and trained model exist locally.
3. The repository `.gitignore` ignores `*.h5` and `*.tflite`. Therefore, verify that the model file actually exists in the local backend folder or has a documented download/release process; do not assume it was committed to Git.
4. The model's class list is 38 PlantVillage-style labels. Training data must use exactly the same label mapping and order, or the model can return the wrong crop/disease name.
5. The code currently derives a “severity” estimate from model confidence. Confidence means how certain the classifier is about its class, not how severely a plant is infected. Keep those concepts separate.
6. `README.md` at the repository root is still the default React/Vite template. `PROJECT_STRUCTURE.md` is useful, but it also documents mocked APIs and a simpler structure than the current repository. Update both after testing the real implementation.
7. The current mandi route intentionally has mock fallback data. The UI must clearly label fallback/demo data rather than imply it is live government data.

Useful source files to inspect first:

- [Backend main.py](https://github.com/ak6862219-cloud/Krishi-Mitra-1/blob/main/krishi-backend/main.py)
- [Backend requirements.txt](https://github.com/ak6862219-cloud/Krishi-Mitra-1/blob/main/krishi-backend/requirements.txt)
- [Repository .gitignore](https://github.com/ak6862219-cloud/Krishi-Mitra-1/blob/main/.gitignore)
- [Frontend App.jsx](https://github.com/ak6862219-cloud/Krishi-Mitra-1/blob/main/src/App.jsx)
- [Frontend ChatPage.jsx](https://github.com/ak6862219-cloud/Krishi-Mitra-1/blob/main/src/pages/ChatPage.jsx)
- [Frontend useDiseaseScan.js](https://github.com/ak6862219-cloud/Krishi-Mitra-1/blob/main/src/hooks/useDiseaseScan.js)

## 2. Recommended datasets

### A. Plant disease detection — first ML task

**Dataset:** [PlantVillage in TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/plant_village) / [Mendeley Data source](https://data.mendeley.com/datasets/tywbtsjrjv/1)

- Around 54,000+ leaf images across 38 crop-health/disease categories.
- Best starting point because the backend already defines 38 PlantVillage-style class names.
- Use this to train a reproducible baseline model, then evaluate it separately on real field photos. Images captured in controlled conditions do not guarantee accuracy on messy field photos.
- Do not put the entire dataset in `src/` or commit it to Git.

**More India-field-focused option:** [ICAR rice and maize crop disease/pest image dataset on AIKosh](https://aikosh.indiaai.gov.in/home/datasets/details/crop_disease_and_insect_pest_image_dataset_for_rice_and_maize.html)

- This dataset describes real-field rice and maize images and disease/pest categories; its listed version is about 926 MB and carries an ICAR Data Use License.
- It is more locally relevant for rice/maize, but its categories do **not** match the current 38 PlantVillage classes. Do not plug it into the existing model unchanged. Either create a distinct rice/maize model or update the class mapping, API and UI together.

**Training guidance:** use transfer learning (for example, MobileNetV2 or EfficientNet), consistent RGB/224×224 preprocessing, stratified train/validation/test splits, class-wise precision/recall, a confusion matrix, and a separate field-photo evaluation set. Store the trained model's class mapping beside the model artifact. Only return a disease when input and confidence checks pass; otherwise ask for a clearer photo or advise consultation with a KVK/agricultural expert.

### B. Crop recommendation — best additional tabular ML feature

**Prototype dataset:** [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/siddharthss/crop-recommendation-dataset)

Typical columns include soil N, P, K, temperature, humidity, pH, rainfall and a crop label. This is useful for a student prototype and a first multi-class classifier, but it is not a substitute for locally validated agronomic recommendations. Document the dataset provenance and limitations.

**India-specific soil reference:** [Government Soil Health Card Portal](https://soilhealth.dac.gov.in/) and the [NIC description of the portal](https://www.nic.gov.in/project/soil-health-card-portal/). Use real soil-test values where possible; do not invent NPK or pH values. Validate crop/fertilizer advice with official local agricultural guidance.

Suggested first model: Random Forest or a gradient-boosted tree with a reproducible training/evaluation script. Save both the model and exact feature order. Add a new backend endpoint such as `POST /api/crop-recommendation`; do not overload `/analyze`, which is currently the leaf-image diagnosis endpoint.

### C. Mandi price trends / later price forecasting

**Government data:** [Current daily mandi prices from AGMARKNET on data.gov.in](https://www.data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi)

- Includes daily wholesale minimum, maximum and modal prices for commodities/markets.
- Your backend already attempts the data.gov.in AGMARKNET API. First make that API reliable and clearly show source and data date. Keep mock fallback clearly marked.
- Only after collecting and cleaning sufficient historical records should you add forecasting (e.g., a simple seasonal baseline before trying more complex models). Return a prediction range and historical context, not a guaranteed future price.

### D. Yield estimation — later phase

**Government data:** [District-wise, season-wise crop production statistics](https://www.data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0)

This dataset describes crop area and production by district, crop, season and year. It can support a carefully designed yield-analysis prototype, but useful predictive performance requires sound joins with location/time/weather/soil features and careful handling of missing records and changed reporting practices.

## 3. Recommended build order

1. **Audit and run the current system:** confirm `/health`, API-key configuration, model file path, CORS, and whether mock mode is enabled. Never silently pass random mock diagnoses off as ML.
2. **Finish the existing plant-disease ML path:** download PlantVillage, create a training pipeline, save a model plus class-map metadata, load it from the existing FastAPI path, and test `/analyze` with known examples and unseen field photos.
3. **Add crop recommendation as a separate ML feature:** dataset cleaning, training, evaluation, saved model, FastAPI endpoint, form/UI, loading/error states and tests.
4. **Make data provenance visible:** show whether weather/mandi values came from a real API, cache, or demo fallback, with an “updated at” timestamp.
5. **Add trusted advisory grounding:** source advice from ICAR/KVK/agricultural-university material; do not let the chatbot invent pesticide dosages, chemical treatments, prices, yield, or savings. Show a source or uncertainty message when advice is uncertain.
6. **Only then consider price forecasting and yield estimation.**

Suggested structure (adapt to the files that already exist; do not duplicate existing modules):

```text
krishi-backend/
  main.py
  requirements.txt
  model/                  # model artifacts; not raw data
  ml/
    train_disease_model.py
    evaluate_disease_model.py
    train_crop_recommendation.py
    evaluate_crop_recommendation.py
    class_names.json
  data/                   # downloaded datasets; excluded from Git
  tests/
  ML_DATASETS.md
  .env.example
```

Use a Python environment supported by the selected TensorFlow version. Keep dataset/model downloads out of the frontend. Do not commit `.env`, API keys, or large raw datasets. If model files are too large for Git, document how to download them or use an appropriate artifact store/Git LFS.

## 4. Copy-paste prompt for Antigravity

> You are modifying my existing repository: https://github.com/ak6862219-cloud/Krishi-Mitra-1. First inspect the current codebase completely, including `src/`, `krishi-backend/`, `package.json`, `krishi-backend/requirements.txt`, `.gitignore`, and existing services/hooks. Treat the actual code as the source of truth; `README.md` is currently a default Vite template and `PROJECT_STRUCTURE.md` may be outdated.
>
> **Do not rewrite the project from scratch. Do not replace the current UI or remove working features.** Make minimal, incremental, tested changes and reuse existing routing, contexts, hooks, UI styles, FastAPI app, and API services.
>
> ### Phase 1 — make existing plant-disease ML real
> - Inspect `krishi-backend/main.py`. It already has a 38-class PlantVillage-style class list, image preprocessing, `predict_disease()`, and `POST /analyze`. It expects `MODEL_PATH` (default `./model/krishi_model.h5`). The current `USE_MOCK_MODEL=true` path chooses a random class/confidence. Never represent this random output as a real prediction.
> - Check whether a real model and training script exist locally. If they do not, create a reproducible training/evaluation pipeline using the PlantVillage dataset, but do not claim that training or accuracy is complete unless the dataset was actually available and the training/evaluation ran successfully.
> - Keep train/validation/test separation, consistent preprocessing, deterministic class ordering, a saved class-name mapping, evaluation metrics (overall and per-class precision/recall, confusion matrix), and a field-image test plan. Ensure model output indices match the class mapping returned by `/classes`.
> - Make inference load the model safely. If the model is missing, return a clear service-unavailable response. If demo/mock mode is deliberately enabled, label the result as DEMO/MOCK in the API and UI and never show it as a diagnosis. Prefer an explicit low-confidence/unknown response instead of forcing a disease prediction.
> - Do not infer disease severity from classifier confidence. Keep `confidence` and `severity` separate. Do not create fake clinical/agronomic certainty or unsupported economic-loss estimates.
>
> ### Phase 2 — add crop recommendation as a separate ML feature
> - Use the Kaggle Crop Recommendation Dataset with N, P, K, temperature, humidity, pH and rainfall as input features and crop as the target. Document its source, licensing/usage constraints and limitations. Do not present this prototype dataset as official agronomic ground truth.
> - Create a training script, evaluation script, saved model, exact feature metadata and tests. Start with Random Forest or another sensible tabular baseline; compare metrics and use a stratified test split.
> - Add a separate `POST /api/crop-recommendation` FastAPI endpoint with schema validation, useful validation errors and no hard-coded prediction. Add a frontend form/UI that calls the endpoint through an existing service layer, handles loading/error states, and displays the input assumptions and model limitations. Keep it separate from the `/analyze` image endpoint.
> - Never invent soil-test measurements. Allow farmers to enter measured values and state clearly when values are estimated. Advise users to confirm high-impact planting/fertilizer decisions with local KVK/agricultural experts.
>
> ### Data and security rules
> - Do not commit raw datasets, `.env`, API keys, or private credentials. Respect the existing `.gitignore`; document model download/artifact handling because `*.h5` and `*.tflite` are ignored.
> - Keep secrets server-side. Do not call secret APIs directly from React.
> - Preserve the existing AGMARKNET/data.gov.in and weather proxies. The mandi endpoint can use mock fallback data, so expose clear source/status metadata and never label mock prices as live.
> - For agricultural treatment guidance, ground responses in trusted ICAR/KVK/agricultural-university sources. Do not invent pesticide products/dosages or guaranteed yields/profits. Show uncertainty and escalation guidance where appropriate.
>
> ### Required deliverables
> 1. A short audit report listing files reused, files changed and existing features preserved.
> 2. `ML_DATASETS.md` with direct dataset URLs, license/usage notes, expected columns/classes, download instructions and limitations.
> 3. Training/evaluation scripts, model metadata/class map, a clear model download/save path, and updated backend dependencies.
> 4. API and frontend integration for the two separate flows: image disease detection and tabular crop recommendation.
> 5. Tests for class-index mapping, invalid input, missing model, mock-mode labeling, and crop-recommendation request validation.
> 6. Updated root `README.md` and `PROJECT_STRUCTURE.md` based on the real repository, including setup commands and environment-variable documentation.
> 7. Run lint/build/tests where available. At the end, report what actually passed, what needs a dataset/API key/model file, and what remains incomplete. Never claim a model has been trained, an API is live, or a test passes unless you verified it.

## 5. Source links

- PlantVillage: https://www.tensorflow.org/datasets/catalog/plant_village
- ICAR rice/maize disease and pest image dataset: https://aikosh.indiaai.gov.in/home/datasets/details/crop_disease_and_insect_pest_image_dataset_for_rice_and_maize.html
- Crop recommendation dataset: https://www.kaggle.com/datasets/siddharthss/crop-recommendation-dataset
- AGMARKNET / mandi prices: https://www.data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi
- District- and season-wise production: https://www.data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0
- Soil Health Card Portal: https://soilhealth.dac.gov.in/
- TensorFlow installation compatibility: https://www.tensorflow.org/install
