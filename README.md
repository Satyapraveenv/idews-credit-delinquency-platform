# IDEWS — Intelligent Delinquency Early Warning System

> **Production-grade AI/ML platform that detects credit card delinquency risk 60–90 days before the first missed payment.**

[![CI Pipeline](https://github.com/Satyapraveenv/idews-credit-delinquency-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/Satyapraveenv/idews-credit-delinquency-platform/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com)
[![GCP](https://img.shields.io/badge/Cloud-GCP%20Vertex%20AI-4285F4)](https://cloud.google.com/vertex-ai)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2)](https://mlflow.org)

---

## Why IDEWS?

Credit card delinquency costs banks over **$500 billion annually** worldwide. Most banks still react *after* a customer misses a payment — by then, recovery rates drop below 40%.

IDEWS flips this model. By detecting early behavioural signals — payment pattern shifts, utilisation spikes, minimum-payment streaks — **60 to 90 days before the first missed payment**, collections and risk teams can act preventively, not reactively.

**Who benefits:**
- **Risk Officers** — daily updated PD scores across the entire portfolio
- **Collections Teams** — prioritised intervention lists with advance warning
- **Model Risk / Audit** — full SR 11-7 compliance, SHAP explanations, audit trail
- **CROs and CFOs** — quantifiable reduction in net credit losses

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              IDEWS — 5-Layer Architecture            │
├───────────────┬─────────────────────────────────────┤
│  DATA SOURCES │ Credit Bureau · Transactions ·       │
│               │ Behavioural · Macroeconomic          │
├───────────────┼─────────────────────────────────────┤
│ FEATURE STORE │ BigQuery · Vertex AI Feature Store   │
│               │ 25 raw → 35 engineered features      │
├───────────────┼─────────────────────────────────────┤
│   ML ENGINE   │ XGBoost · SHAP Explainability        │
│               │ MLflow Registry · Model Validation   │
├───────────────┼─────────────────────────────────────┤
│  SCORING API  │ FastAPI · Vertex AI Endpoint         │
│               │ <50ms SLA · REST/JSON                │
├───────────────┼─────────────────────────────────────┤
│ APPLICATIONS  │ Risk Dashboard · Collections ·       │
│               │ Origination · Regulatory Reports     │
└───────────────┴─────────────────────────────────────┘
        Cross-cutting: MLOps & Governance
        (Evidently · GitHub Actions · SR 11-7 Docs)
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data | BigQuery, Apache Beam | Data warehouse & pipelines |
| Data Quality | Great Expectations | Validation gates |
| Feature Store | Vertex AI Feature Store | Online/offline features |
| ML Model | XGBoost + scikit-learn | Delinquency prediction |
| Explainability | SHAP | Adverse action reason codes |
| Experiment Tracking | MLflow | Run comparison & registry |
| Drift Monitoring | Evidently AI | PSI-based drift detection |
| Serving API | FastAPI + Uvicorn | Sub-50ms scoring |
| Cloud MLOps | GCP Vertex AI Pipelines | Training & deployment |
| Dashboard | Streamlit + Plotly | Risk monitoring UI |
| CI/CD | GitHub Actions | Automated quality gates |
| Containerisation | Docker + Compose | Local & cloud deployment |

---

## Quick Start (Local)

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Git

### 1. Clone and Install

```bash
git clone https://github.com/Satyapraveenv/idews-credit-delinquency-platform.git
cd idews-credit-delinquency-platform
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Downloads UCI dataset, engineers features, trains XGBoost, logs to MLflow
python -m src.models.train
```

Expected output:
```
AUC-ROC: 0.8247  |  KS: 0.4821  |  Gini: 0.6494  |  F1: 0.6103
Model saved to models/idews_model.json
```

### 3. Start the Full Stack

```bash
docker compose -f docker/docker-compose.yml up
```

| Service | URL |
|---------|-----|
| Scoring API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Risk Dashboard | http://localhost:8501 |
| MLflow Tracking | http://localhost:5000 |

### 4. Score an Account

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "account_id": "ACC_001",
    "LIMIT_BAL": 150000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_0": 1, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 130000, "BILL_AMT2": 120000, "BILL_AMT3": 110000,
    "BILL_AMT4": 95000,  "BILL_AMT5": 80000,  "BILL_AMT6": 70000,
    "PAY_AMT1": 3500, "PAY_AMT2": 3000, "PAY_AMT3": 2800,
    "PAY_AMT4": 3200, "PAY_AMT5": 2900, "PAY_AMT6": 3100
  }'
```

Response:
```json
{
  "account_id": "ACC_001",
  "risk_score": 0.6823,
  "risk_flag": true,
  "risk_band": "HIGH",
  "behavioural_risk_score": 67.4,
  "top_risk_factors": [
    {
      "feature": "util_rate_m1",
      "shap_value": 0.2341,
      "direction": "increases",
      "reason_code": "High credit utilisation in the most recent billing cycle"
    }
  ],
  "adverse_action_codes": [
    "High credit utilisation in the most recent billing cycle",
    "Steadily increasing credit utilisation trend"
  ],
  "model_version": "1.0.0",
  "response_time_ms": 12.4
}
```

---

## Project Structure

```
idews-credit-delinquency-platform/
│
├── src/
│   ├── data/
│   │   ├── ingestion.py          # Dataset download & loading
│   │   └── validation.py         # Great Expectations quality gates
│   ├── features/
│   │   └── behavioral.py         # 35 engineered behavioural features
│   ├── models/
│   │   ├── train.py              # End-to-end training pipeline
│   │   ├── evaluate.py           # AUC-ROC, KS, Gini metrics
│   │   └── explainability.py     # SHAP analysis & adverse action codes
│   ├── api/
│   │   ├── main.py               # FastAPI application
│   │   └── schemas.py            # Pydantic request/response models
│   ├── monitoring/
│   │   └── drift_detector.py     # Evidently PSI drift detection
│   └── dashboard/
│       └── app.py                # Streamlit monitoring dashboard
│
├── vertex/
│   ├── training_pipeline.py      # Vertex AI Kubeflow pipeline
│   └── deploy_endpoint.py        # GCP endpoint deployment
│
├── tests/
│   ├── test_features.py          # Feature engineering unit tests
│   └── test_api.py               # API contract tests
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   └── docker-compose.yml        # Full local dev stack
│
├── .github/workflows/
│   ├── ci.yml                    # PR quality gates
│   └── model-retrain.yml         # Scheduled retraining
│
├── config/
│   ├── model_config.yaml         # Hyperparameters & thresholds
│   ├── feature_config.yaml       # Feature definitions
│   └── monitoring_config.yaml    # Drift alert thresholds
│
└── README.md
```

---

## Model Performance (Baseline)

Trained on UCI Credit Card Default dataset (30,000 accounts, 6-month history):

| Metric | Value | Threshold |
|--------|-------|-----------|
| AUC-ROC | ~0.82 | ≥ 0.80 ✅ |
| KS Statistic | ~0.48 | ≥ 0.35 ✅ |
| Gini Coefficient | ~0.64 | ≥ 0.60 ✅ |
| F1 Score | ~0.61 | ≥ 0.55 ✅ |
| False Positive Rate | ~0.06 | ≤ 0.08 ✅ |

> *Note: Actual results vary by training run. Model meets SR 11-7 minimum performance requirements.*

---

## GCP Vertex AI Deployment

Deploy to Google Cloud with your $100 free credit:

```bash
# Set your project
export GCP_PROJECT=your-gcp-project-id
export GCP_REGION=us-central1

# Deploy training pipeline
python vertex/training_pipeline.py \
  --project $GCP_PROJECT \
  --region $GCP_REGION

# Deploy scoring endpoint
python vertex/deploy_endpoint.py \
  --project $GCP_PROJECT \
  --region $GCP_REGION
```

**Estimated cost:** $20–40 for the full prototype (within $100 free credit).
**Tip:** Shut down the Vertex AI Endpoint after each demo session — that's where most cost accumulates.

---

## Regulatory Compliance

| Framework | How IDEWS Addresses It |
|-----------|----------------------|
| **SR 11-7** (Fed Model Risk) | Full model documentation, champion/challenger, validation reports |
| **IFRS 9** | Daily PD scoring aligned to Stage 1/2/3 ECL staging |
| **ECOA / Reg B** | SHAP-powered adverse action reason codes for every high-risk flag |
| **Basel IV** | Accurate PD/LGD estimation for capital calculation |

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Feature tests only
pytest tests/test_features.py -v

# API tests only
pytest tests/test_api.py -v
```

---

## About the Author

**Satya Praveen** — Senior Consultant & AI Innovation Lead

- 20 years across BFSI, Telecom, SaaS, EdTech
- ISB Executive MBA (INSEAD Singapore immersion)
- CPMAI Certified | $25M+ career delivery portfolio
- Built 2 Quality Engineering Centres of Excellence
- Hands-on with n8n, Vapi, MCP, OpenAI API

[LinkedIn](https://linkedin.com/in/Satyapraveenv) · [GitHub](https://github.com/Satyapraveenv)

---

*IDEWS is a portfolio demonstration project using the publicly available UCI Credit Card Default dataset. It is not affiliated with any bank or financial institution.*
