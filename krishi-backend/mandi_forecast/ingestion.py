"""
Mandi Historical Data Ingestion & Cleaning Pipeline
===================================================
Handles ingestion, schema normalization, deduplication, chronological
sorting, and validation of agricultural market arrival/price records.
Official Source Target: Agmarknet / data.gov.in
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

log = logging.getLogger("krishi.mandi_ingest")

REQUIRED_COLUMNS = ["date", "state", "commodity", "modal_price"]
COLUMN_MAPPINGS = {
    "arrival_date": "date",
    "arrival date": "date",
    "Arrival_Date": "date",
    "Arrival Date": "date",
    "Date": "date",
    "date": "date",
    "State": "state",
    "state": "state",
    "District": "district",
    "district": "district",
    "Market": "market",
    "market": "market",
    "Mandi": "market",
    "Commodity": "commodity",
    "commodity": "commodity",
    "Variety": "variety",
    "variety": "variety",
    "Min_Price": "min_price",
    "min_price": "min_price",
    "Min Price": "min_price",
    "Max_Price": "max_price",
    "max_price": "max_price",
    "Max Price": "max_price",
    "Modal_Price": "modal_price",
    "modal_price": "modal_price",
    "Modal Price": "modal_price",
}


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize arbitrary Agmarknet/CSV column names to standard internal schema."""
    df = df.copy()
    rename_dict = {}
    for col in df.columns:
        if col in COLUMN_MAPPINGS:
            rename_dict[col] = COLUMN_MAPPINGS[col]
        elif col.strip() in COLUMN_MAPPINGS:
            rename_dict[col] = COLUMN_MAPPINGS[col.strip()]

    df = df.rename(columns=rename_dict)
    return df


def parse_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Parse mixed date formats common in Indian government records."""
    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found.")

    # Try common formats: %d/%m/%Y, %d-%m-%Y, %Y-%m-%d
    df[date_col] = pd.to_datetime(df[date_col], format="mixed", dayfirst=True, errors="coerce")
    invalid_dates = df[date_col].isnull().sum()
    if invalid_dates > 0:
        log.warning(f"Dropped {invalid_dates} rows with unparseable dates.")
        df = df.dropna(subset=[date_col])

    return df


def clean_and_validate(
    df: pd.DataFrame,
    reject_mock: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Clean, validate, deduplicate, and chronologically sort mandi data.

    Returns:
        (cleaned_df, report_dict)
    """
    raw_count = len(df)
    report = {
        "raw_records": raw_count,
        "duplicates_removed": 0,
        "invalid_prices_removed": 0,
        "valid_records": 0,
        "start_date": None,
        "end_date": None,
    }

    df = normalize_schema(df)

    # Check for synthetic/mock indicator columns or flags
    if reject_mock and "is_mock" in df.columns:
        mock_mask = df["is_mock"] == True
        if mock_mask.any():
            log.warning(f"Rejected {mock_mask.sum()} mock records from ingestion.")
            df = df[~mock_mask]

    # Validate required columns
    for req in REQUIRED_COLUMNS:
        if req not in df.columns:
            raise ValueError(f"Required column '{req}' is missing after normalization.")

    df = parse_dates(df, "date")

    # Clean text columns
    for str_col in ["state", "district", "market", "commodity", "variety"]:
        if str_col in df.columns:
            df[str_col] = df[str_col].astype(str).str.strip().str.title()

    # Numeric conversion for prices
    price_cols = [c for c in ["min_price", "max_price", "modal_price"] if c in df.columns]
    for pcol in price_cols:
        df[pcol] = pd.to_numeric(df[pcol], errors="coerce")

    # Filter invalid/negative/zero modal prices
    valid_price_mask = (df["modal_price"] > 0) & (df["modal_price"] < 1_000_000)
    report["invalid_prices_removed"] = int((~valid_price_mask).sum())
    df = df[valid_price_mask]

    # Deduplicate on date + market + commodity + variety
    dedup_cols = ["date", "market", "commodity"]
    if "variety" in df.columns:
        dedup_cols.append("variety")
    before_dedup = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep="last")
    report["duplicates_removed"] = before_dedup - len(df)

    # Chronological sort
    df = df.sort_values(by="date").reset_index(drop=True)

    report["valid_records"] = len(df)
    if len(df) > 0:
        report["start_date"] = df["date"].min().strftime("%Y-%m-%d")
        report["end_date"] = df["date"].max().strftime("%Y-%m-%d")

    return df, report
