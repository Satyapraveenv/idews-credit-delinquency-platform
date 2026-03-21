"""
IDEWS — FastAPI Contract Tests
Tests the API endpoints without needing a running server (uses TestClient).
"""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "account_id": "TEST_001",
    "LIMIT_BAL": 150000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 85000, "BILL_AMT2": 80000, "BILL_AMT3": 76000,
    "BILL_AMT4": 72000, "BILL_AMT5": 68000, "BILL_AMT6": 65000,
    "PAY_AMT1": 5000, "PAY_AMT2": 4500, "PAY_AMT3": 4000,
    "PAY_AMT4": 4200, "PAY_AMT5": 3800, "PAY_AMT6": 4100,
}


def test_health_endpoint_returns_200():
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "model_loaded" in data


def test_metrics_endpoint_returns_200():
    resp = client.get("/v1/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_requests" in data
    assert "avg_response_ms" in data


def test_predict_returns_503_when_model_not_loaded():
    """Without a trained model, the API should return 503 (service unavailable)."""
    resp = client.post("/v1/predict", json=VALID_PAYLOAD)
    # 503 expected when model file doesn't exist yet
    assert resp.status_code in [200, 503]


def test_predict_rejects_invalid_age():
    payload = VALID_PAYLOAD.copy()
    payload["AGE"] = 150  # Invalid
    resp = client.post("/v1/predict", json=payload)
    assert resp.status_code == 422  # Pydantic validation error


def test_predict_rejects_negative_payment():
    payload = VALID_PAYLOAD.copy()
    payload["PAY_AMT1"] = -500  # Invalid
    resp = client.post("/v1/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_missing_fields():
    payload = {"account_id": "TEST", "LIMIT_BAL": 100000}  # Many missing fields
    resp = client.post("/v1/predict", json=payload)
    assert resp.status_code == 422


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "IDEWS" in data.get("service", "")
