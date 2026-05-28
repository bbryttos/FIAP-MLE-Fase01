import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.models.mlp import MLP

VALID_PAYLOAD = {
    "senior_citizen": 0,
    "tenure": 12,
    "monthly_charges": 65.5,
    "total_charges": 786.0,
    "gender": "Male",
    "partner": "Yes",
    "dependents": "No",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
}


@pytest.fixture
def client():
    from unittest.mock import MagicMock

    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = np.zeros((1, 30), dtype=np.float32)

    mock_model = MLP(input_dim=30, hidden_dims=[32, 16])
    mock_model.eval()

    with TestClient(app) as c:
        app.state.preprocessor = mock_preprocessor
        app.state.model = mock_model
        app.state.model_loaded = True
        yield c

    app.state.model_loaded = False


@pytest.fixture
def client_no_model():
    with TestClient(app) as c:
        app.state.model_loaded = False
        yield c


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "version" in data


def test_predict_valid_payload_returns_200(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_predict_response_schema(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()
    assert "churn_probability" in data
    assert "prediction" in data
    assert "risk_level" in data


def test_predict_probability_in_range(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    prob = response.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_binary_prediction(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    prediction = response.json()["prediction"]
    assert prediction in {0, 1}


def test_predict_risk_level_valid(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    risk = response.json()["risk_level"]
    assert risk in {"low", "medium", "high"}


def test_predict_missing_field_returns_422(client):
    incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "tenure"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_invalid_senior_citizen_returns_422(client):
    payload = {**VALID_PAYLOAD, "senior_citizen": 5}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_categorical_returns_422(client):
    payload = {**VALID_PAYLOAD, "gender": "Other"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_model_not_loaded_returns_503(client_no_model):
    response = client_no_model.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503
    assert response.json()["detail"] == "Model not available"
