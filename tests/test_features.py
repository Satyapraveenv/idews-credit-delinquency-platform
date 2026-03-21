"""
IDEWS — Feature Engineering Tests
Verifies that engineered features are computed correctly.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.behavioral import engineer_features, get_all_feature_names


@pytest.fixture
def sample_raw_data():
    """Minimal valid UCI-format record for testing."""
    return pd.DataFrame([{
        "LIMIT_BAL": 150000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 90000, "BILL_AMT2": 85000, "BILL_AMT3": 80000,
        "BILL_AMT4": 75000, "BILL_AMT5": 70000, "BILL_AMT6": 65000,
        "PAY_AMT1": 5000, "PAY_AMT2": 4500, "PAY_AMT3": 4000,
        "PAY_AMT4": 4200, "PAY_AMT5": 3800, "PAY_AMT6": 4100,
        "default_payment_next_month": 0,
    }])


def test_engineer_features_adds_derived_columns(sample_raw_data):
    result = engineer_features(sample_raw_data)
    assert "util_rate_m1" in result.columns
    assert "min_pay_streak" in result.columns
    assert "behavioural_risk_score" in result.columns
    assert "total_delay_score" in result.columns


def test_utilisation_rate_correct(sample_raw_data):
    result = engineer_features(sample_raw_data)
    expected = 90000 / 150000
    assert abs(result["util_rate_m1"].iloc[0] - expected) < 0.001


def test_brs_in_valid_range(sample_raw_data):
    result = engineer_features(sample_raw_data)
    brs = result["behavioural_risk_score"].iloc[0]
    assert 0 <= brs <= 100, f"BRS out of range: {brs}"


def test_min_pay_streak_zero_for_full_payer(sample_raw_data):
    """A customer paying 100% each month should have streak of 0."""
    df = sample_raw_data.copy()
    for col in ["BILL_AMT1","BILL_AMT2","BILL_AMT3"]:
        df[col.replace("BILL","PAY")] = df[col]  # Pay in full
    result = engineer_features(df)
    assert result["min_pay_streak"].iloc[0] == 0


def test_feature_names_list_is_consistent():
    features = get_all_feature_names()
    assert len(features) > 30, "Expected at least 30 features"
    assert "LIMIT_BAL" in features
    assert "behavioural_risk_score" in features
    assert len(features) == len(set(features)), "Duplicate feature names detected"


def test_high_delay_increases_brs(sample_raw_data):
    """Account with late payments should have higher BRS than clean account."""
    clean = engineer_features(sample_raw_data)

    late_df = sample_raw_data.copy()
    late_df["PAY_0"] = 3  # 3 months delay
    late_df["PAY_2"] = 2
    late_df["PAY_3"] = 1
    late = engineer_features(late_df)

    assert late["behavioural_risk_score"].iloc[0] > clean["behavioural_risk_score"].iloc[0]
