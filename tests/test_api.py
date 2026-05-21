"""FastAPI endpoint tests."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)

_VALID_CUSTOMER = {
    "gender": "Male",
    "senior_citizen": "No",
    "partner": "Yes",
    "dependents": "No",
    "tenure_months": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "DSL",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 56.95,
    "total_charges": 683.40,
}


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_predict_accepts_valid_payload():
    """When model artifacts are present → 200; when absent → 503. Both are valid."""
    response = client.post("/predict", json=_VALID_CUSTOMER)
    assert response.status_code in (200, 503)


def test_predict_response_schema():
    """If the model is loaded, the response must contain the expected fields."""
    response = client.post("/predict", json=_VALID_CUSTOMER)
    if response.status_code == 200:
        body = response.json()
        assert "churn_probability" in body
        assert "churn_prediction" in body
        assert "threshold" in body
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert isinstance(body["churn_prediction"], bool)


def test_predict_missing_field_returns_422():
    bad = _VALID_CUSTOMER.copy()
    del bad["tenure_months"]
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_negative_tenure_returns_422():
    bad = {**_VALID_CUSTOMER, "tenure_months": -1}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_latency_header_present():
    response = client.get("/health")
    assert "x-latency-ms" in response.headers
