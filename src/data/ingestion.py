"""
IDEWS — Data Ingestion Module
Downloads the UCI Credit Card Default dataset and prepares it for feature engineering.

Why UCI Credit Card Default?
- 30,000 real credit card accounts from a Taiwanese bank (2005)
- 25 features covering demographics, payment history, bill amounts
- Binary target: default payment next month (1=default, 0=no default)
- Industry-standard benchmark for credit risk ML research
- Publicly available, zero licensing cost — perfect for a portfolio project
"""

import os
import io
import logging
from pathlib import Path

import pandas as pd
import requests
from loguru import logger


# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00350/default%20of%20credit%20card%20clients.xls"
)
RAW_DIR  = Path("data/raw")
PROC_DIR = Path("data/processed")

COLUMN_RENAME = {
    "default payment next month": "default_payment_next_month",
    "PAY_0": "PAY_0",  # Keep as-is; some versions label this PAY_1
}


# ----------------------------------------------------------------
# Public API
# ----------------------------------------------------------------
def load_dataset(force_download: bool = False) -> pd.DataFrame:
    """
    Load the UCI Credit Card Default dataset.

    Downloads on first call; subsequent calls read from local cache.

    Args:
        force_download: Re-download even if local file exists.

    Returns:
        Raw DataFrame with 30,000 rows and 25 feature columns.
    """
    csv_path = RAW_DIR / "credit_card_default.csv"

    if csv_path.exists() and not force_download:
        logger.info(f"Loading dataset from cache: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        logger.info("Downloading UCI Credit Card Default dataset ...")
        df = _download_dataset()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"Dataset saved to {csv_path}")

    logger.info(f"Dataset loaded: {df.shape[0]:,} rows · {df.shape[1]} columns")
    return df


def get_feature_target_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features (X) and target (y).

    Returns:
        X: Feature DataFrame (24 raw features)
        y: Target Series (binary: 1=default, 0=no default)
    """
    target_col = "default_payment_next_month"

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target_col, "ID"], errors="ignore")
    y = df[target_col]

    default_rate = y.mean() * 100
    logger.info(
        f"Feature/target split: X={X.shape}, y={y.shape} "
        f"(default rate: {default_rate:.1f}%)"
    )
    return X, y


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Return a quick summary of the dataset for reporting and audit.
    """
    y = df.get("default_payment_next_month", pd.Series())
    return {
        "n_records":          len(df),
        "n_features":         df.shape[1] - 1,  # Exclude target
        "default_rate_pct":   round(y.mean() * 100, 2) if len(y) > 0 else None,
        "missing_values":     int(df.isnull().sum().sum()),
        "duplicate_rows":     int(df.duplicated().sum()),
        "age_range":          f"{df['AGE'].min()} – {df['AGE'].max()}" if "AGE" in df.columns else "N/A",
        "credit_limit_range": f"{df['LIMIT_BAL'].min():,} – {df['LIMIT_BAL'].max():,}" if "LIMIT_BAL" in df.columns else "N/A",
    }


# ----------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------
def _download_dataset() -> pd.DataFrame:
    """Download the UCI XLS file and convert to DataFrame."""
    try:
        response = requests.get(DATA_URL, timeout=60)
        response.raise_for_status()
        df = pd.read_excel(io.BytesIO(response.content), header=1)
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        raise RuntimeError(
            "Could not download the UCI dataset. "
            "Check your internet connection or download manually from: "
            "https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients"
        ) from e

    # Normalise column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"default_payment_next_month": "default_payment_next_month"})

    # Drop the ID column (not predictive)
    df = df.drop(columns=["ID"], errors="ignore")

    logger.info(f"Downloaded: {df.shape[0]:,} rows · {df.shape[1]} columns")
    return df
