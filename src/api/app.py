import logging
import os
import time
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Annotated

from src.models.mlp import ChurnMLP, predict_proba

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

PIPELINE_PATH = os.getenv("PIPELINE_PATH", "models/preprocessing_pipeline.joblib")
MODEL_PATH = os.getenv("MODEL_PATH", "models/mlp_model.pt")
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

_state: dict = {"pipeline": None, "model": None}


def _load_artifacts() -> None:
    if not os.path.exists(PIPELINE_PATH) or not os.path.exists(MODEL_PATH):
        logger.warning(
            "Model artifacts not found at %s / %s — /predict will return 503",
            PIPELINE_PATH,
            MODEL_PATH,
        )
        return

    _state["pipeline"] = joblib.load(PIPELINE_PATH)

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = ChurnMLP(
        input_dim=ckpt["input_dim"],
        hidden_dims=ckpt["hidden_dims"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _state["model"] = model
    logger.info("Artifacts loaded — input_dim=%d", ckpt["input_dim"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts customer churn probability for a telecom operator.",
    version="0.1.0",
    lifespan=lifespan,
)


class CustomerFeatures(BaseModel):
    gender: str = Field(..., examples=["Male"])
    senior_citizen: str = Field(..., examples=["No"])
    partner: str = Field(..., examples=["Yes"])
    dependents: str = Field(..., examples=["No"])
    tenure_months: int = Field(..., ge=0)
    phone_service: str = Field(..., examples=["Yes"])
    multiple_lines: str = Field(..., examples=["No"])
    internet_service: str = Field(..., examples=["DSL"])
    online_security: str = Field(..., examples=["No"])
    online_backup: str = Field(..., examples=["Yes"])
    device_protection: str = Field(..., examples=["No"])
    tech_support: str = Field(..., examples=["No"])
    streaming_tv: str = Field(..., examples=["No"])
    streaming_movies: str = Field(..., examples=["No"])
    contract: str = Field(..., examples=["Month-to-month"])
    paperless_billing: str = Field(..., examples=["Yes"])
    payment_method: str = Field(..., examples=["Electronic check"])
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    threshold: float


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "path=%s method=%s status=%d latency_ms=%.2f",
        request.url.path,
        request.method,
        response.status_code,
        latency_ms,
    )
    response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
    return response


@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.get("/ready", summary="Readiness check")
async def ready():
    if _state["pipeline"] is None or _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


def _customer_to_row(customer: CustomerFeatures) -> dict:
    return {
        "Gender": customer.gender,
        "Senior Citizen": customer.senior_citizen,
        "Partner": customer.partner,
        "Dependents": customer.dependents,
        "Tenure Months": customer.tenure_months,
        "Phone Service": customer.phone_service,
        "Multiple Lines": customer.multiple_lines,
        "Internet Service": customer.internet_service,
        "Online Security": customer.online_security,
        "Online Backup": customer.online_backup,
        "Device Protection": customer.device_protection,
        "Tech Support": customer.tech_support,
        "Streaming TV": customer.streaming_tv,
        "Streaming Movies": customer.streaming_movies,
        "Contract": customer.contract,
        "Paperless Billing": customer.paperless_billing,
        "Payment Method": customer.payment_method,
        "Monthly Charges": customer.monthly_charges,
        "Total Charges": customer.total_charges,
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict churn")
async def predict(customer: CustomerFeatures):
    if _state["pipeline"] is None or _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = _state["pipeline"].transform(pd.DataFrame([_customer_to_row(customer)]))
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        raise HTTPException(status_code=422, detail=f"Preprocessing error: {exc}") from exc

    prob = float(predict_proba(_state["model"], X)[0])
    logger.info("churn_prob=%.4f threshold=%.2f", prob, THRESHOLD)

    return PredictionResponse(
        churn_probability=prob,
        churn_prediction=prob >= THRESHOLD,
        threshold=THRESHOLD,
    )


@app.post("/predict-batch", response_model=BatchPredictionResponse, summary="Predict churn for multiple customers")
async def predict_batch(customers: Annotated[list[CustomerFeatures], Field(min_length=1, max_length=1000)]):
    if _state["pipeline"] is None or _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        rows = [_customer_to_row(c) for c in customers]
        X = _state["pipeline"].transform(pd.DataFrame(rows))
    except Exception as exc:
        logger.error("Batch preprocessing failed: %s", exc)
        raise HTTPException(status_code=422, detail=f"Preprocessing error: {exc}") from exc

    probs = predict_proba(_state["model"], X)
    logger.info("batch_size=%d", len(customers))

    predictions = [
        PredictionResponse(
            churn_probability=float(prob),
            churn_prediction=float(prob) >= THRESHOLD,
            threshold=THRESHOLD,
        )
        for prob in probs
    ]
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))
