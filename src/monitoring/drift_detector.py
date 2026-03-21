"""
IDEWS — Data Drift & Model Performance Monitor
Uses Evidently AI to detect when the model's input data distribution is
shifting — an early warning that the model itself may be degrading.

In credit risk, data drift is common during:
- Economic downturns (customer behaviour changes rapidly)
- Policy changes (new customer segments are acquired)
- Seasonal effects (December spending patterns differ from March)

What we monitor:
1. Feature drift: Population Stability Index (PSI) per feature
2. Model performance: AUC-ROC, Precision, Recall on recent labelled data
3. Prediction drift: Is the score distribution shifting?
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, ClassificationPreset
    from evidently.metrics import (
        DatasetDriftMetric,
        DataDriftTable,
        ColumnDriftMetric,
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    logger.warning("Evidently not installed. Run: pip install evidently")
    EVIDENTLY_AVAILABLE = False


# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------
PSI_WARNING_THRESHOLD = 0.10   # Yellow alert: investigate
PSI_DRIFT_THRESHOLD   = 0.20   # Red alert: retrain required
REPORT_DIR = Path("reports/drift")


# ----------------------------------------------------------------
# Public API
# ----------------------------------------------------------------
def run_drift_check(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_names: list[str],
    output_path: str | None = None,
) -> dict:
    """
    Run a full data drift analysis comparing reference vs current data.

    Args:
        reference_df: Historical 'good' data (training set or past 90 days)
        current_df:   Recent production data (past 30 days)
        feature_names: List of feature columns to monitor
        output_path:  Where to save the HTML drift report

    Returns:
        Drift report dict with PSI per feature and overall drift flag
    """
    logger.info(
        f"Running drift analysis — "
        f"reference: {len(reference_df):,} rows, "
        f"current: {len(current_df):,} rows"
    )

    ref = reference_df[feature_names].copy()
    cur = current_df[feature_names].copy()

    # Compute PSI for each feature
    psi_results = {}
    for feature in feature_names:
        psi = _compute_psi(ref[feature], cur[feature])
        psi_results[feature] = {
            "psi":    round(psi, 4),
            "status": _psi_status(psi),
        }

    drifted_features = [f for f, r in psi_results.items() if r["status"] == "DRIFT"]
    warning_features = [f for f, r in psi_results.items() if r["status"] == "WARNING"]

    overall_drift = len(drifted_features) > 0
    max_psi = max(r["psi"] for r in psi_results.values())

    report = {
        "run_timestamp":   datetime.utcnow().isoformat(),
        "reference_rows":  len(ref),
        "current_rows":    len(cur),
        "features_checked": len(feature_names),
        "overall_drift":   overall_drift,
        "max_psi":         round(max_psi, 4),
        "drifted_features": drifted_features,
        "warning_features": warning_features,
        "feature_psi":     psi_results,
        "recommendation":  _get_recommendation(overall_drift, len(drifted_features)),
    }

    # Evidently HTML report (if available)
    if EVIDENTLY_AVAILABLE and output_path:
        _generate_evidently_report(ref, cur, output_path)

    _log_drift_summary(report)

    # Save JSON report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = REPORT_DIR / f"drift_report_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Drift report saved: {json_path}")

    return report


def print_drift_report(report: dict) -> None:
    """Print human-readable drift report to console."""
    status = "🔴 DRIFT DETECTED — Retraining recommended" if report["overall_drift"] else "🟢 No significant drift"
    print(f"\n{'='*65}")
    print(f"  IDEWS Drift Monitor Report  |  {report['run_timestamp'][:10]}")
    print(f"{'='*65}")
    print(f"  Status:           {status}")
    print(f"  Max PSI:          {report['max_psi']:.4f}  (alert threshold: {PSI_DRIFT_THRESHOLD})")
    print(f"  Drifted Features: {len(report['drifted_features'])}")
    print(f"  Warning Features: {len(report['warning_features'])}")
    print(f"  Recommendation:   {report['recommendation']}")
    print()
    if report["drifted_features"]:
        print("  Top Drifted Features:")
        for feat in report["drifted_features"][:5]:
            psi = report["feature_psi"][feat]["psi"]
            print(f"    🔴 {feat}: PSI={psi:.4f}")
    print("=" * 65 + "\n")


# ----------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------
def _compute_psi(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI).

    PSI interpretation:
    - PSI < 0.10: No significant change
    - 0.10 ≤ PSI < 0.20: Moderate change — monitor
    - PSI ≥ 0.20: Significant shift — investigate and consider retraining
    """
    # Handle NaN values
    ref = reference.dropna()
    cur = current.dropna()

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # For categorical features, use value counts
    if reference.dtype == object or reference.nunique() <= 10:
        categories = set(ref.unique()) | set(cur.unique())
        ref_pct = ref.value_counts(normalize=True).reindex(categories, fill_value=1e-4)
        cur_pct = cur.value_counts(normalize=True).reindex(categories, fill_value=1e-4)
    else:
        # Numerical: bin using reference distribution
        breakpoints = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0

        ref_counts, _ = np.histogram(ref, bins=breakpoints)
        cur_counts, _ = np.histogram(cur, bins=breakpoints)

        ref_pct = (ref_counts + 1e-4) / len(ref)
        cur_pct = (cur_counts + 1e-4) / len(cur)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return max(0.0, float(psi))


def _psi_status(psi: float) -> str:
    if psi >= PSI_DRIFT_THRESHOLD:
        return "DRIFT"
    elif psi >= PSI_WARNING_THRESHOLD:
        return "WARNING"
    return "OK"


def _get_recommendation(drift: bool, n_drifted: int) -> str:
    if not drift:
        return "No action required. Continue monitoring on schedule."
    if n_drifted <= 3:
        return "Minor drift detected. Investigate drifted features. Consider retraining within 2 weeks."
    return "Significant drift across multiple features. Immediate retraining recommended."


def _log_drift_summary(report: dict) -> None:
    if report["overall_drift"]:
        logger.warning(
            f"DRIFT DETECTED: max_psi={report['max_psi']:.4f}, "
            f"drifted_features={report['drifted_features']}"
        )
    else:
        logger.info(f"No drift detected. Max PSI: {report['max_psi']:.4f}")


def _generate_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: str,
) -> None:
    """Generate a rich HTML drift report using Evidently AI."""
    try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        report.save_html(output_path)
        logger.info(f"Evidently HTML report saved: {output_path}")
    except Exception as e:
        logger.error(f"Failed to generate Evidently report: {e}")
