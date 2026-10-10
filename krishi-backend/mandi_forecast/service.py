"""
Mandi Forecast Service
======================
High-level service handling historical data checking, pipeline execution,
and structured API response generation.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import datetime

from .ingestion import clean_and_validate
from .forecasting import (
    prepare_univariate_series,
    chronological_split,
    MIN_OBSERVATIONS_FOR_FORECAST,
    SeasonalNaiveBaseline,
    LagRidgeForecaster,
    evaluate_forecast,
)

HISTORICAL_DATA_DIR = Path(__file__).parent.parent / "datasets" / "mandi_historical"


def find_available_dataset() -> Optional[Path]:
    """Search for historical CSV files in the mandi_historical directory."""
    if not HISTORICAL_DATA_DIR.exists():
        return None
    csvs = list(HISTORICAL_DATA_DIR.glob("*.csv"))
    return csvs[0] if csvs else None


def get_forecast(
    state: str,
    commodity: str,
    market: Optional[str] = None,
    horizon_days: int = 7,
) -> Dict[str, Any]:
    """
    Generate or report forecast status for a given commodity and location.
    Strictly avoids fabricating predictions when historical records are unavailable.
    """
    dataset_path = find_available_dataset()

    if dataset_path is None:
        return {
            "status": "DATA_UNAVAILABLE",
            "state": state,
            "commodity": commodity,
            "market": market or "All Markets",
            "forecast_available": False,
            "message": (
                "Verified historical time-series data is not currently loaded on disk. "
                "Agmarknet daily API returns live spot prices, which cannot be used for "
                "autoregressive time-series forecasting without sufficient historical depth (>= 30 days). "
                "To enable forecasting for this commodity, place historical daily records in "
                "krishi-backend/datasets/mandi_historical/."
            ),
            "required_schema": ["Arrival_Date", "State", "Market", "Commodity", "Modal_Price"],
            "disclaimer": "Krishi Mitra strictly prohibits fabricating price forecasts without verified historical data.",
        }

    try:
        raw_df = pd.read_csv(dataset_path)
        cleaned_df, report = clean_and_validate(raw_df, reject_mock=True)
    except Exception as e:
        return {
            "status": "INGESTION_ERROR",
            "forecast_available": False,
            "error": f"Failed to ingest dataset: {e}",
        }

    # If market not specified, pick the most frequent market for this state/commodity
    sub = cleaned_df[
        (cleaned_df["state"].str.lower() == state.lower()) &
        (cleaned_df["commodity"].str.lower() == commodity.lower())
    ]

    if len(sub) == 0:
        return {
            "status": "NO_MATCHING_RECORDS",
            "forecast_available": False,
            "state": state,
            "commodity": commodity,
            "message": f"No historical records found for '{commodity}' in '{state}'.",
        }

    target_market = market or sub["market"].value_counts().index[0]
    series = prepare_univariate_series(cleaned_df, target_market, commodity)

    if len(series) < MIN_OBSERVATIONS_FOR_FORECAST:
        return {
            "status": "INSUFFICIENT_OBSERVATIONS",
            "forecast_available": False,
            "state": state,
            "commodity": commodity,
            "market": target_market,
            "available_observations": len(series),
            "required_observations": MIN_OBSERVATIONS_FOR_FORECAST,
            "message": (
                f"Found only {len(series)} observations for {commodity} at {target_market}. "
                f"At least {MIN_OBSERVATIONS_FOR_FORECAST} observations are required to prevent overfit forecasts."
            ),
        }

    # Chronological train/test split
    train_s, test_s = chronological_split(series, test_ratio=0.2)

    # Fit baseline & ML models
    naive = SeasonalNaiveBaseline().fit(train_s)
    ml_model = LagRidgeForecaster().fit(train_s)

    # Evaluate on held-out test split
    naive_test_preds = naive.predict(len(test_s))
    ml_test_preds = ml_model.predict(len(test_s))

    naive_metrics = evaluate_forecast(test_s.values, naive_test_preds)
    ml_metrics = evaluate_forecast(test_s.values, ml_test_preds)

    # Pick best model based on MAE
    chosen_model = ml_model if ml_metrics["mae"] <= naive_metrics["mae"] else naive
    chosen_name = "LagRidgeRegression" if chosen_model == ml_model else "SeasonalNaiveBaseline"
    chosen_metrics = ml_metrics if chosen_model == ml_model else naive_metrics

    # Re-fit on full series to forecast future horizon
    chosen_model.fit(series)
    future_preds = chosen_model.predict(horizon_days)

    last_date = series.index[-1]
    forecast_dates = [
        (last_date + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(horizon_days)
    ]

    forecast_items = [
        {"date": d, "predicted_modal_price": p, "unit": "₹/Quintal"}
        for d, p in zip(forecast_dates, future_preds)
    ]

    return {
        "status": "SUCCESS",
        "forecast_available": True,
        "state": state,
        "commodity": commodity,
        "market": target_market,
        "last_observed_date": last_date.strftime("%Y-%m-%d"),
        "last_observed_price": round(float(series.iloc[-1]), 2),
        "forecast_horizon_days": horizon_days,
        "forecasts": forecast_items,
        "model_used": chosen_name,
        "evaluation_metrics_held_out": chosen_metrics,
        "baseline_naive_metrics": naive_metrics,
        "data_provenance": {
            "source": dataset_path.name,
            "total_observations": len(series),
        },
        "disclaimer": (
            "Price forecasts are statistical estimates based on historical trends. "
            "Actual mandi prices depend on spot weather, daily arrivals, and transport conditions. "
            "Not a guaranteed selling price."
        ),
    }
