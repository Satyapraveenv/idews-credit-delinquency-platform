"""
IDEWS — Model Training Pipeline
Trains the XGBoost credit delinquency prediction model with full MLflow tracking.

Run this script to:
1. Load and validate data
2. Engineer 35 behavioural features
3. Train XGBoost with optimal hyperparameters
4. Evaluate with banking-grade metrics (AUC-ROC, KS, Gini)
5. Log everything to MLflow for reproducibility and governance
6. Save model artifacts for API deployment

Usage:
    python -m src.models.train
    python -m src.models.train --config config/model_config.yaml
"""

import argparse
import time
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
import xgboost as xgb
from loguru import logger
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src.data.ingestion import load_dataset, get_feature_target_split
from src.data.validation import validate_dataset
from src.features.behavioral import engineer_features, get_all_feature_names
from src.models.evaluate import compute_all_metrics, print_metrics_report
from src.models.explainability import compute_shap_values, save_shap_plots


# ----------------------------------------------------------------
# Main training entrypoint
# ----------------------------------------------------------------
def train(config_path: str = "config/model_config.yaml") -> dict:
    """
    End-to-end model training pipeline.

    Args:
        config_path: Path to model configuration YAML.

    Returns:
        Dictionary with trained model, metrics, and artifact paths.
    """
    config = _load_config(config_path)
    model_cfg = config["model"]

    # Setup MLflow
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    with mlflow.start_run(run_name=f"xgboost_v{model_cfg['version']}") as run:
        logger.info(f"MLflow run started: {run.info.run_id}")
        mlflow.log_params(model_cfg["params"])
        mlflow.log_param("model_version", model_cfg["version"])

        # ── Step 1: Load & validate data ──────────────────────────
        logger.info("Step 1/6: Loading dataset ...")
        df = load_dataset()
        report = validate_dataset(df, raise_on_failure=True)
        mlflow.log_metric("data_validation_checks", len(report["checks"]))
        mlflow.log_metric("data_validation_failures", report["n_failures"])

        # ── Step 2: Feature engineering ───────────────────────────
        logger.info("Step 2/6: Engineering features ...")
        df_featured = engineer_features(df)
        feature_names = get_all_feature_names()

        X = df_featured[feature_names]
        y = df_featured["default_payment_next_month"]

        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("n_records", len(X))
        mlflow.log_metric("default_rate_pct", round(y.mean() * 100, 2))

        # ── Step 3: Train/test split ───────────────────────────────
        logger.info("Step 3/6: Splitting data (80/20 stratified) ...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=model_cfg["train_test_split"],
            random_state=model_cfg["random_state"],
            stratify=y
        )

        # SMOTE: Handle class imbalance (22% default rate → balance to improve recall)
        logger.info("Applying SMOTE to handle class imbalance ...")
        smote = SMOTE(random_state=model_cfg["random_state"])
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        logger.info(
            f"After SMOTE: {len(X_train_bal):,} training samples "
            f"({y_train_bal.mean():.1%} default rate)"
        )

        # Validation split from training set
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_bal, y_train_bal,
            test_size=0.1,
            random_state=model_cfg["random_state"],
            stratify=y_train_bal
        )

        # ── Step 4: Train XGBoost ──────────────────────────────────
        logger.info("Step 4/6: Training XGBoost model ...")
        start = time.time()

        model = xgb.XGBClassifier(
            **model_cfg["params"],
            eval_metric="auc",
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=50
        )

        train_time_s = round(time.time() - start, 2)
        mlflow.log_metric("training_time_seconds", train_time_s)
        logger.info(f"Training complete in {train_time_s}s")

        # ── Step 5: Evaluate ──────────────────────────────────────
        logger.info("Step 5/6: Evaluating model ...")
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred       = (y_pred_proba >= model_cfg["thresholds"]["operating_threshold"]).astype(int)

        metrics = compute_all_metrics(y_test, y_pred, y_pred_proba)
        print_metrics_report(metrics)

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, round(value, 4))

        # Check against performance SLAs
        _check_performance_thresholds(metrics, model_cfg["thresholds"])

        # ── Step 6: SHAP explainability ───────────────────────────
        logger.info("Step 6/6: Computing SHAP explanations ...")
        shap_values, explainer = compute_shap_values(model, X_test, feature_names)
        plot_paths = save_shap_plots(shap_values, X_test, feature_names)
        for path in plot_paths:
            mlflow.log_artifact(path)

        # ── Log model to MLflow registry ──────────────────────────
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=config["mlflow"]["model_registry_name"],
        )

        # Save model locally for API deployment
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model.save_model(str(model_dir / "idews_model.json"))
        logger.info(f"Model saved to {model_dir}/idews_model.json")

        logger.info(f"MLflow run complete: {run.info.run_id}")
        logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f} | KS: {metrics['ks_statistic']:.4f} | Gini: {metrics['gini']:.4f}")

        return {
            "model":     model,
            "metrics":   metrics,
            "run_id":    run.info.run_id,
            "features":  feature_names,
            "explainer": explainer,
        }


# ----------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------
def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _check_performance_thresholds(metrics: dict, thresholds: dict) -> None:
    """Fail fast if model doesn't meet minimum quality bar."""
    if metrics["auc_roc"] < thresholds["min_auc_roc"]:
        raise ValueError(
            f"Model AUC-ROC {metrics['auc_roc']:.4f} is below minimum threshold "
            f"{thresholds['min_auc_roc']}. Investigate feature engineering or data quality."
        )
    if metrics["ks_statistic"] < thresholds["min_ks_statistic"]:
        logger.warning(
            f"KS Statistic {metrics['ks_statistic']:.4f} is below target "
            f"{thresholds['min_ks_statistic']}. Model may need tuning."
        )
    logger.info("Performance thresholds: PASSED")


# ----------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDEWS Model Training Pipeline")
    parser.add_argument("--config", default="config/model_config.yaml")
    args = parser.parse_args()
    results = train(args.config)
    logger.info(f"Training complete. Model AUC: {results['metrics']['auc_roc']:.4f}")
