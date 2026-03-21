"""
IDEWS — FastAPI Scoring Microservice
Production-grade REST API for real-time credit delinquency risk scoring.

Endpoints:
  POST /v1/predict   — Score a single account (< 50ms SLA)
  GET  /v1/explain   — Get SHAP explanation for an account
  GET  /v1/health    — Liveness check
  GET  /v1/metrics   — Operational metrics (Prometheus-ready)
  GET  /docs         — Interactive Swagger UI (auto-generated)

Run locally:
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Run via Docker:
  docker compose up api
"""

import time
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.schemas import (
    ScoringRequest, ScoringResponse, HealthResponse, MetricsResponse, RiskFactor
)
from src.features.behavioral import engineer_features, get_all_feature_names
from src.models.explainability import explain_single_account, REASON_CODE_TEMPLATES

# ----------------------------------------------------------------
# Global state (loaded once at startup)
# ----------------------------------------------------------------
MODEL_PATH   = os.getenv("MODEL_PATH", "models/idews_model.json")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

_state = {
    "model":      None,
    "explainer":  None,
    "features":   None,
    "start_time": None,
    "stats": {
        "total_requests":    0,
        "total_time_ms":     0.0,
        "high_risk_flagged": 0,
    },
}

OPERATING_THRESHOLD = float(os.getenv("OPERATING_THRESHOLD", "0.40"))

RISK_BANDS = [
    (0.20, "LOW"),
    (0.40, "MEDIUM"),
    (0.65, "HIGH"),
    (1.01, "CRITICAL"),
]


# ----------------------------------------------------------------
# Startup / Shutdown lifecycle
# ----------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, clean up at shutdown."""
    logger.info("Starting IDEWS Scoring API ...")
    _state["start_time"] = time.time()

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        logger.warning(
            f"Model file not found at {MODEL_PATH}. "
            "Run 'python -m src.models.train' first."
        )
    else:
        try:
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            _state["model"] = model
            _state["features"] = get_all_feature_names()

            # Pre-warm SHAP explainer (slow on first call — do it at startup)
            logger.info("Warming up SHAP explainer ...")
            _state["explainer"] = shap.TreeExplainer(model)
            logger.info(f"Model loaded: {MODEL_PATH} (version {MODEL_VERSION})")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    yield  # Application runs here

    logger.info("IDEWS Scoring API shutting down.")


# ----------------------------------------------------------------
# FastAPI App
# ----------------------------------------------------------------
app = FastAPI(
    title="IDEWS — Credit Delinquency Early Warning API",
    description=(
        "Production-grade real-time credit risk scoring API. "
        "Predicts probability of default 60-90 days in advance using "
        "XGBoost with SHAP explainability and SR 11-7 compliant adverse action codes."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------
@app.post(
    "/v1/predict",
    response_model=ScoringResponse,
    summary="Score an account for delinquency risk",
    tags=["Scoring"],
)
async def predict(request: ScoringRequest) -> ScoringResponse:
    """
    Score a single credit card account for delinquency risk.

    Returns a risk score (0–1), risk band, top SHAP factors, and
    regulatory-compliant adverse action reason codes.

    **SLA: < 50ms response time**
    """
    start = time.time()
    _state["stats"]["total_requests"] += 1

    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using 'python -m src.models.train'."
        )

    try:
        # Build raw feature DataFrame from request
        raw_data = {k: v for k, v in request.model_dump().items() if k != "account_id"}
        raw_df = pd.DataFrame([raw_data])

        # Engineer behavioural features
        featured_df = engineer_features(raw_df)
        features = _state["features"]
        X = featured_df[features]

        # Score with XGBoost
        risk_score = float(_state["model"].predict_proba(X)[0, 1])
        risk_flag  = risk_score >= OPERATING_THRESHOLD
        risk_band  = _get_risk_band(risk_score)
        brs        = float(featured_df["behavioural_risk_score"].iloc[0])

        # SHAP explanation
        explanation = explain_single_account(
            _state["explainer"], X, features, top_n=5
        )

        risk_factors = [
            RiskFactor(
                feature=f["feature"],
                shap_value=f["shap_value"],
                feature_value=f["feature_value"],
                direction=f["direction"],
                reason_code=f["reason_code"],
            )
            for f in explanation["top_risk_factors"]
        ]

        elapsed_ms = (time.time() - start) * 1000
        _state["stats"]["total_time_ms"] += elapsed_ms
        if risk_flag:
            _state["stats"]["high_risk_flagged"] += 1

        logger.info(
            f"Scored {request.account_id}: "
            f"score={risk_score:.3f} band={risk_band} ({elapsed_ms:.1f}ms)"
        )

        return ScoringResponse(
            account_id=request.account_id,
            risk_score=round(risk_score, 4),
            risk_flag=risk_flag,
            risk_band=risk_band,
            behavioural_risk_score=round(brs, 1),
            top_risk_factors=risk_factors,
            adverse_action_codes=explanation["adverse_action_codes"],
            model_version=MODEL_VERSION,
            response_time_ms=round(elapsed_ms, 2),
        )

    except Exception as e:
        logger.error(f"Scoring error for {request.account_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.get(
    "/v1/health",
    response_model=HealthResponse,
    summary="API health check",
    tags=["Operations"],
)
async def health() -> HealthResponse:
    """Liveness probe — used by Kubernetes/Vertex AI for health monitoring."""
    uptime = time.time() - _state["start_time"] if _state["start_time"] else 0
    return HealthResponse(
        status="healthy" if _state["model"] is not None else "degraded",
        model_loaded=_state["model"] is not None,
        model_version=MODEL_VERSION,
        uptime_seconds=round(uptime, 1),
    )


@app.get(
    "/v1/metrics",
    response_model=MetricsResponse,
    summary="Operational metrics",
    tags=["Operations"],
)
async def metrics() -> MetricsResponse:
    """Returns operational metrics for monitoring dashboards."""
    stats = _state["stats"]
    avg_ms = (
        stats["total_time_ms"] / stats["total_requests"]
        if stats["total_requests"] > 0 else 0.0
    )
    return MetricsResponse(
        total_requests=stats["total_requests"],
        avg_response_ms=round(avg_ms, 2),
        high_risk_flagged=stats["high_risk_flagged"],
        model_version=MODEL_VERSION,
    )


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "IDEWS Credit Delinquency Early Warning API",
        "version": MODEL_VERSION,
        "docs":    "/docs",
        "health":  "/v1/health",
    }


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def _get_risk_band(score: float) -> str:
    for threshold, band in RISK_BANDS:
        if score < threshold:
            return band
    return "CRITICAL"
