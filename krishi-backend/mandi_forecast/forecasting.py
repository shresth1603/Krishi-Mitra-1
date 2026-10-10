"""
Mandi Price Forecasting Engine
==============================
Provides chronological time-series splitting, feature engineering (lags, rolling stats),
baseline seasonal-naive modeling, and machine learning time-series regression.
Never shuffles time series randomly to strictly prevent data leakage.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

MIN_OBSERVATIONS_FOR_FORECAST = 30


def prepare_univariate_series(df: pd.DataFrame, market: str, commodity: str) -> pd.Series:
    """Filter to a single market and commodity and resample to daily frequency."""
    sub = df[(df["market"].str.lower() == market.lower()) & (df["commodity"].str.lower() == commodity.lower())].copy()
    if len(sub) == 0:
        return pd.Series(dtype=float)

    sub = sub.sort_values(by="date")
    # Group by date in case multiple arrivals occurred on the same day
    daily = sub.groupby("date")["modal_price"].mean()
    # Daily frequency reindex and forward-fill short gaps (up to 7 days)
    full_idx = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="D")
    daily = daily.reindex(full_idx)
    daily = daily.ffill(limit=7)
    daily = daily.dropna()
    return daily


def chronological_split(series: pd.Series, test_ratio: float = 0.2) -> Tuple[pd.Series, pd.Series]:
    """Strict chronological split. Earliest (1-test_ratio) for train, latest test_ratio for test."""
    n = len(series)
    split_idx = int(n * (1.0 - test_ratio))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    return train, test


def create_lag_features(series: pd.Series, max_lags: int = 7) -> Tuple[pd.DataFrame, pd.Series]:
    """Create tabular lag features (lag-1, lag-2, lag-3, lag-7, 7d rolling mean)."""
    df = pd.DataFrame({"target": series})
    for lag in [1, 2, 3, 7]:
        if lag <= max_lags:
            df[f"lag_{lag}"] = df["target"].shift(lag)
    df["rolling_mean_7"] = df["target"].shift(1).rolling(window=7, min_periods=3).mean()
    df["day_of_week"] = df.index.dayofweek

    df = df.dropna()
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


class SeasonalNaiveBaseline:
    """Seasonal Naive baseline: predicts the value from 7 days ago (or last available)."""
    def __init__(self, season_period: int = 7):
        self.season_period = season_period
        self.history = None

    def fit(self, train_series: pd.Series):
        self.history = train_series.copy()
        return self

    def predict(self, horizon: int) -> List[float]:
        if self.history is None or len(self.history) == 0:
            raise ValueError("Model is not fitted.")
        vals = self.history.values
        preds = []
        for h in range(horizon):
            idx = -self.season_period + (h % self.season_period)
            val = vals[idx] if len(vals) >= self.season_period else vals[-1]
            preds.append(float(val))
        return preds


class LagRidgeForecaster:
    """Auto-regressive linear ridge regression with rolling and calendar features."""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.last_known_series = None

    def fit(self, train_series: pd.Series):
        X, y = create_lag_features(train_series)
        if len(X) < 10:
            raise ValueError("Not enough samples after lag feature creation.")
        self.model.fit(X, y)
        self.last_known_series = train_series.copy()
        return self

    def predict(self, horizon: int) -> List[float]:
        """Recursive multi-step forecast."""
        curr = list(self.last_known_series.values)
        curr_dates = list(self.last_known_series.index)
        preds = []

        for h in range(horizon):
            next_date = curr_dates[-1] + pd.Timedelta(days=1)
            # Build feature row for step
            lag_1 = curr[-1]
            lag_2 = curr[-2] if len(curr) >= 2 else curr[-1]
            lag_3 = curr[-3] if len(curr) >= 3 else curr[-1]
            lag_7 = curr[-7] if len(curr) >= 7 else curr[-1]
            roll_7 = np.mean(curr[-7:])
            dow = next_date.dayofweek

            x_step = pd.DataFrame([{
                "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3, "lag_7": lag_7,
                "rolling_mean_7": roll_7, "day_of_week": dow
            }])

            p = float(self.model.predict(x_step)[0])
            p = max(1.0, p)  # enforce positive price
            preds.append(round(p, 2))

            curr.append(p)
            curr_dates.append(next_date)

        return preds


def evaluate_forecast(actuals: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """Calculate MAE, RMSE, and MAPE."""
    actuals = np.array(actuals, dtype=float)
    predictions = np.array(predictions, dtype=float)
    mae = mean_absolute_error(actuals, predictions)
    rmse = root_mean_squared_error(actuals, predictions)
    # Avoid zero division
    denom = np.where(actuals == 0, 1.0, actuals)
    mape = float(np.mean(np.abs((actuals - predictions) / denom)) * 100.0)

    return {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "mape_pct": round(mape, 2),
    }
