"""
IDEWS — Model Evaluation Module
Computes banking-grade model performance metrics.

Why these metrics matter:
- AUC-ROC: How well the model ranks good vs bad accounts (0.5=random, 1.0=perfect)
- KS Statistic: The maximum separation between good/bad cumulative distributions
  (banking gold standard — regulators expect KS > 0.35 for a production model)
- Gini Coefficient: = 2 * AUC - 1 (commonly reported in IFRS 9 model validation)
- Precision/Recall: Business trade-off between false alarms and missed defaults
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# ----------------------------------------------------------------
# Public API
# ----------------------------------------------------------------
def compute_all_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict:
    """
    Compute full suite of banking credit risk model metrics.

    Args:
        y_true: True binary labels (1=default, 0=no default)
        y_pred: Binary predictions at operating threshold
        y_pred_proba: Predicted probability of default (0 to 1)

    Returns:
        Dict of metric name → float value
    """
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    gini    = 2 * auc_roc - 1
    ks      = _compute_ks_statistic(y_true, y_pred_proba)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "auc_roc":         round(auc_roc, 4),
        "gini":            round(gini, 4),
        "ks_statistic":    round(ks, 4),
        "f1_score":        round(f1, 4),
        "precision":       round(prec, 4),
        "recall":          round(rec, 4),
        "true_positives":  int(tp),
        "false_positives": int(fp),
        "true_negatives":  int(tn),
        "false_negatives": int(fn),
        "false_positive_rate": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
        "false_negative_rate": round(fn / (fn + tp) if (fn + tp) > 0 else 0, 4),
    }


def print_metrics_report(metrics: dict) -> None:
    """Print a formatted banking-grade metrics report."""
    print("\n" + "=" * 60)
    print("  IDEWS Model Performance Report")
    print("=" * 60)
    print(f"  AUC-ROC        : {metrics['auc_roc']:.4f}  {'✅' if metrics['auc_roc'] >= 0.80 else '⚠️'}")
    print(f"  Gini           : {metrics['gini']:.4f}")
    print(f"  KS Statistic   : {metrics['ks_statistic']:.4f}  {'✅' if metrics['ks_statistic'] >= 0.35 else '⚠️'}")
    print(f"  F1 Score       : {metrics['f1_score']:.4f}")
    print(f"  Precision      : {metrics['precision']:.4f}")
    print(f"  Recall         : {metrics['recall']:.4f}")
    print(f"  False Pos Rate : {metrics['false_positive_rate']:.4f}  {'✅' if metrics['false_positive_rate'] <= 0.08 else '⚠️'}")
    print(f"  False Neg Rate : {metrics['false_negative_rate']:.4f}")
    print("-" * 60)
    print(f"  True  Positives (caught defaults)  : {metrics['true_positives']:,}")
    print(f"  False Positives (false alarms)     : {metrics['false_positives']:,}")
    print(f"  True  Negatives (correct clears)   : {metrics['true_negatives']:,}")
    print(f"  False Negatives (missed defaults)  : {metrics['false_negatives']:,}")
    print("=" * 60 + "\n")


# ----------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------
def _compute_ks_statistic(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic: maximum separation between the
    cumulative distribution of good and bad account scores.

    Banking regulators typically require KS > 0.35 for a production model.
    """
    df = pd.DataFrame({"y_true": y_true, "score": y_pred_proba})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    n_bad  = (df["y_true"] == 1).sum()
    n_good = (df["y_true"] == 0).sum()

    df["cum_bad"]  = (df["y_true"] == 1).cumsum() / n_bad
    df["cum_good"] = (df["y_true"] == 0).cumsum() / n_good
    df["ks"]       = (df["cum_bad"] - df["cum_good"]).abs()

    return df["ks"].max()
