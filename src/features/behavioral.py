"""
IDEWS — Behavioural Feature Engineering
Creates the 35 derived features that power early delinquency detection.

Key insight: Raw payment history columns (PAY_0 to PAY_6) tell you WHAT
happened. These engineered features tell you the TREND and TRAJECTORY —
which is what predicts future behaviour 60-90 days in advance.

Feature groups:
1. Utilisation features     — How much of the credit limit is being used
2. Payment behaviour        — Are they paying in full, minimum, or skipping?
3. Delinquency signals      — Patterns in late payment history
4. Composite risk scores    — Combined behavioural indicators
"""

import numpy as np
import pandas as pd
from loguru import logger


# ----------------------------------------------------------------
# Column name constants (matches UCI dataset)
# ----------------------------------------------------------------
BILL_COLS = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
PAY_COLS  = ["PAY_AMT1",  "PAY_AMT2",  "PAY_AMT3",  "PAY_AMT4",  "PAY_AMT5",  "PAY_AMT6"]
DEL_COLS  = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]  # Delay status (note: PAY_1 missing in UCI)

LIMIT_COL = "LIMIT_BAL"
MIN_PAY_THRESHOLD = 0.15  # Paying <15% of bill = treating as minimum payment


# ----------------------------------------------------------------
# Master feature engineering function
# ----------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transforms to raw DataFrame.

    Args:
        df: Raw UCI dataset DataFrame.

    Returns:
        DataFrame with 35 additional derived feature columns.
    """
    logger.info(f"Engineering features on {len(df):,} records ...")
    df = df.copy()

    df = _add_utilisation_features(df)
    df = _add_payment_behaviour_features(df)
    df = _add_delinquency_signal_features(df)
    df = _add_composite_risk_score(df)

    new_cols = [c for c in df.columns if c not in df.columns[:25]]
    logger.info(f"Feature engineering complete — {df.shape[1]} total columns")
    return df


# ----------------------------------------------------------------
# 1. Utilisation Features
# ----------------------------------------------------------------
def _add_utilisation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    How much of the credit limit is being used month-over-month?
    High and rising utilisation is a strong delinquency predictor.
    """
    limit = df[LIMIT_COL].replace(0, np.nan)

    # Monthly utilisation rates (0 = none used, 1 = maxed out)
    for i, bill_col in enumerate(BILL_COLS, 1):
        df[f"util_rate_m{i}"] = (df[bill_col] / limit).clip(0, 2)  # Cap at 200%

    # 6-month trend slope (positive = utilisation worsening)
    util_cols = [f"util_rate_m{i}" for i in range(1, 7)]
    df["util_trend_slope"] = df[util_cols].apply(
        lambda row: np.polyfit(range(6), row.fillna(row.mean()), 1)[0], axis=1
    )

    # Max utilisation in last 6 months
    df["util_max_6m"] = df[util_cols].max(axis=1)

    # Utilisation volatility (instability = risk signal)
    df["util_volatility"] = df[util_cols].std(axis=1)

    return df


# ----------------------------------------------------------------
# 2. Payment Behaviour Features
# ----------------------------------------------------------------
def _add_payment_behaviour_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Are they paying in full, making minimum payments, or not paying at all?
    Minimum-only payment streaks are a key 60-day advance warning signal.
    """
    # Monthly payment ratios (how much of the bill was paid?)
    for i, (pay_col, bill_col) in enumerate(zip(PAY_COLS, BILL_COLS), 1):
        safe_bill = df[bill_col].replace(0, np.nan)
        df[f"pay_ratio_m{i}"] = (df[pay_col] / safe_bill).clip(0, 2).fillna(0)

    pay_ratio_cols = [f"pay_ratio_m{i}" for i in range(1, 7)]

    # 3-month average payment ratio
    df["pay_ratio_avg_3m"] = df[[f"pay_ratio_m{i}" for i in range(1, 4)]].mean(axis=1)

    # Minimum payment flag per month (paying < 15% of bill)
    for i in range(1, 4):
        df[f"is_min_pay_m{i}"] = (df[f"pay_ratio_m{i}"] < MIN_PAY_THRESHOLD).astype(int)

    # Consecutive months with minimum-only payment (most powerful early signal)
    df["min_pay_streak"] = (
        df["is_min_pay_m1"].astype(int) +
        df["is_min_pay_m1"] * df["is_min_pay_m2"].astype(int) +
        df["is_min_pay_m1"] * df["is_min_pay_m2"] * df["is_min_pay_m3"].astype(int)
    )

    # Payment trend — improving (positive) or worsening (negative)
    df["payment_momentum"] = df[pay_ratio_cols].apply(
        lambda row: np.polyfit(range(6), row.fillna(row.mean()), 1)[0], axis=1
    )

    # Balance growth: Is the outstanding balance growing month over month?
    df["balance_growth_rate"] = (
        (df["BILL_AMT1"] - df["BILL_AMT6"]) /
        df["BILL_AMT6"].replace(0, np.nan)
    ).fillna(0).clip(-2, 10)

    return df


# ----------------------------------------------------------------
# 3. Delinquency Signal Features
# ----------------------------------------------------------------
def _add_delinquency_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Payment delay patterns from PAY_0 to PAY_6.
    In the UCI dataset: -2=no credit used, -1=paid duly, 0=minimum payment,
    1=1 month delay, 2=2 month delay, etc.
    """
    # Treat negative values (paid duly) as 0 delay
    delay_df = df[DEL_COLS].clip(lower=0)

    # Maximum delay seen in last 6 months
    df["max_delay_6m"] = delay_df.max(axis=1)

    # Weighted delay score — recent months count more (weight 6..1)
    weights = np.array([6, 5, 4, 3, 2, 1])
    df["total_delay_score"] = delay_df.values @ weights

    # Count of months with any delay
    df["months_with_delay"] = (delay_df > 0).sum(axis=1)

    # Consecutive late payments (most recent months)
    def count_consecutive_late(row):
        count = 0
        for val in row:
            if val > 0:
                count += 1
            else:
                break
        return count

    df["consecutive_late"] = delay_df.apply(count_consecutive_late, axis=1)

    # Recency: months since last late payment (0 = currently late)
    def recency_of_late(row):
        for i, val in enumerate(row):
            if val > 0:
                return i
        return 6  # No late payment in 6 months = very recent clean record

    df["recency_of_late"] = delay_df.apply(recency_of_late, axis=1)

    return df


# ----------------------------------------------------------------
# 4. Composite Risk Score
# ----------------------------------------------------------------
def _add_composite_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine utilisation, payment, and delinquency signals into a
    single Behavioural Risk Score (BRS) — a human-interpretable
    leading indicator even before the model scores an account.

    BRS = 0 (lowest risk) to 100 (highest risk)
    """
    # Normalise components to 0-1 scale
    util_component    = df["util_max_6m"].clip(0, 1)
    delay_component   = (df["max_delay_6m"] / 8).clip(0, 1)
    min_pay_component = (df["min_pay_streak"] / 3).clip(0, 1)
    trend_component   = df["util_trend_slope"].clip(0, 0.5) * 2  # Rising util = risk

    # Weighted composite (domain-tuned weights)
    df["behavioural_risk_score"] = (
        0.35 * util_component +
        0.30 * delay_component +
        0.25 * min_pay_component +
        0.10 * trend_component
    ) * 100

    df["behavioural_risk_score"] = df["behavioural_risk_score"].clip(0, 100).round(1)

    return df


# ----------------------------------------------------------------
# Feature list helpers
# ----------------------------------------------------------------
def get_all_feature_names() -> list[str]:
    """Return the complete list of features used for model training."""
    raw = [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1",  "PAY_AMT2",  "PAY_AMT3",  "PAY_AMT4",  "PAY_AMT5",  "PAY_AMT6",
    ]
    derived = [
        "util_rate_m1", "util_rate_m2", "util_rate_m3",
        "util_trend_slope", "util_max_6m", "util_volatility",
        "pay_ratio_m1", "pay_ratio_m2", "pay_ratio_m3",
        "pay_ratio_avg_3m", "is_min_pay_m1", "is_min_pay_m2", "is_min_pay_m3",
        "min_pay_streak", "payment_momentum", "balance_growth_rate",
        "max_delay_6m", "total_delay_score", "months_with_delay",
        "consecutive_late", "recency_of_late",
        "behavioural_risk_score",
    ]
    return raw + derived
