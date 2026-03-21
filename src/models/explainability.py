"""
IDEWS — SHAP Explainability Module
Makes every risk score transparent and auditable — a non-negotiable
requirement for production credit risk models under SR 11-7 and ECOA.

What this module produces:
1. Feature importance ranking (what matters most overall)
2. Per-account explanation (why THIS customer scored high/low)
3. Adverse action reason codes (regulator-ready plain-English explanations)
4. SHAP waterfall and summary plots for governance documentation
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from loguru import logger


# ----------------------------------------------------------------
# Plain-English reason code templates (ECOA / Reg B compliant)
# ----------------------------------------------------------------
REASON_CODE_TEMPLATES = {
    "util_rate_m1":          "High credit utilisation in the most recent billing cycle",
    "util_max_6m":           "Consistently high credit utilisation over the past 6 months",
    "util_trend_slope":      "Steadily increasing credit utilisation trend",
    "min_pay_streak":        "Multiple consecutive months of minimum-only payments",
    "pay_ratio_avg_3m":      "Below-average payment amount relative to outstanding balance",
    "max_delay_6m":          "Late payment history in the past 6 months",
    "total_delay_score":     "Accumulated payment delay history across multiple months",
    "consecutive_late":      "Pattern of consecutive late payments",
    "PAY_0":                 "Current month repayment status indicates payment delay",
    "PAY_2":                 "Recent month repayment status indicates payment delay",
    "balance_growth_rate":   "Outstanding balance has been increasing over recent months",
    "behavioural_risk_score":"Overall behavioural risk indicators are elevated",
    "LIMIT_BAL":             "Credit limit relative to current outstanding balances",
    "payment_momentum":      "Deteriorating payment behaviour trend over recent months",
    "months_with_delay":     "Multiple months with payment delays recorded",
}

DEFAULT_REASON = "Account payment behaviour and credit utilisation patterns"


# ----------------------------------------------------------------
# Public API
# ----------------------------------------------------------------
def compute_shap_values(
    model,
    X: pd.DataFrame,
    feature_names: list[str],
    n_samples: int = 500,
) -> tuple:
    """
    Compute SHAP values for the test set.

    Args:
        model: Trained XGBoost model.
        X: Feature DataFrame (test set or batch to explain).
        feature_names: List of feature names.
        n_samples: Number of samples for SHAP background dataset.

    Returns:
        (shap_values array, shap.TreeExplainer)
    """
    logger.info(f"Computing SHAP values for {len(X):,} accounts ...")

    # TreeExplainer is fastest for XGBoost — no sampling approximation needed
    explainer = shap.TreeExplainer(model)

    # Use a subsample for speed (full set can be slow at 30K)
    sample = X.sample(min(n_samples, len(X)), random_state=42)
    shap_values = explainer.shap_values(sample)

    logger.info(f"SHAP values computed. Shape: {shap_values.shape}")
    return shap_values, explainer


def explain_single_account(
    explainer,
    account_features: pd.DataFrame,
    feature_names: list[str],
    top_n: int = 5,
) -> dict:
    """
    Generate a full explanation for a single account's risk score.

    Args:
        explainer: Fitted shap.TreeExplainer
        account_features: Single-row DataFrame with feature values
        feature_names: Feature name list
        top_n: Number of top contributing factors to return

    Returns:
        Dict with risk score, top factors, and adverse action codes
    """
    shap_vals = explainer.shap_values(account_features)[0]

    # Pair features with SHAP values and sort by absolute impact
    contributions = sorted(
        zip(feature_names, shap_vals, account_features.values[0]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    top_factors = []
    for feature, shap_val, raw_val in contributions[:top_n]:
        direction = "increases" if shap_val > 0 else "decreases"
        top_factors.append({
            "feature":          feature,
            "shap_value":       round(float(shap_val), 4),
            "feature_value":    round(float(raw_val), 4),
            "direction":        direction,
            "reason_code":      REASON_CODE_TEMPLATES.get(feature, DEFAULT_REASON),
        })

    # Adverse action codes for regulatory compliance (top 4 risk-increasing factors)
    adverse_codes = [
        f["reason_code"]
        for f in top_factors
        if f["direction"] == "increases"
    ][:4]

    return {
        "top_risk_factors":   top_factors,
        "adverse_action_codes": adverse_codes,
        "n_features_analysed": len(feature_names),
    }


def get_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Return a DataFrame of features ranked by mean absolute SHAP value.
    This is the model-level explanation for governance documentation.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":         feature_names[:len(mean_abs_shap)],
        "mean_abs_shap":   mean_abs_shap,
        "rank":            range(1, len(mean_abs_shap) + 1),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance_df["rank"] = range(1, len(importance_df) + 1)
    return importance_df.head(top_n)


def save_shap_plots(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: list[str],
    output_dir: str = "reports/explainability",
) -> list[str]:
    """
    Generate and save SHAP visualisation plots for governance documentation.

    Produces:
    1. Summary bar chart (global feature importance)
    2. Beeswarm plot (impact distribution per feature)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_paths = []

    # 1. Summary bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X, feature_names=feature_names,
        plot_type="bar", show=False, max_display=15
    )
    plt.title("IDEWS — Feature Importance (Mean |SHAP|)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    bar_path = f"{output_dir}/shap_importance_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    saved_paths.append(bar_path)
    logger.info(f"Saved SHAP bar chart: {bar_path}")

    # 2. Beeswarm plot (shows both direction and magnitude)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X, feature_names=feature_names,
        plot_type="dot", show=False, max_display=15
    )
    plt.title("IDEWS — SHAP Value Distribution by Feature", fontsize=14, fontweight="bold")
    plt.tight_layout()
    dot_path = f"{output_dir}/shap_beeswarm.png"
    plt.savefig(dot_path, dpi=150, bbox_inches="tight")
    plt.close()
    saved_paths.append(dot_path)
    logger.info(f"Saved SHAP beeswarm plot: {dot_path}")

    return saved_paths
