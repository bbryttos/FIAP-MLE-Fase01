import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import Depends, FastAPI, HTTPException, Request

from src.api.schemas import (
    BatchPredictionOutput,
    ClienteInput,
    HealthOutput,
    PredictionOutput,
)
from src.models.mlp import ChurnMLP, predict_proba
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_state: dict = {"pipeline": None, "model": None, "input_dim": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = Path(settings.models_dir)
    logger.info("Loading model artifacts from {}", str(models_dir))
    try:
        # Tenta pipeline novo (build_preprocessing_pipeline) depois o legado
        pipeline_path = models_dir / "preprocessing_pipeline.joblib"
        legacy_path = models_dir / "preprocessor.joblib"
        if pipeline_path.exists():
            _state["pipeline"] = joblib.load(pipeline_path)
        elif legacy_path.exists():
            _state["pipeline"] = joblib.load(legacy_path)
        else:
            raise FileNotFoundError(f"No pipeline found in {models_dir}")

        cfg_path = models_dir / "model_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            input_dim = cfg.get("input_dim")
            hidden_dims = cfg.get("hidden_dims", [64, 32, 16])
        else:
            logger.warning("model_config.json ausente — inferindo input_dim via preprocessor")
            input_dim = None
            hidden_dims = [64, 32, 16]

        # Carrega checkpoint (novo formato .pt com state_dict + metadata)
        pt_path = models_dir / "mlp_model.pt"
        legacy_pt = models_dir / "mlp_weights.pt"
        if pt_path.exists():
            ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
            input_dim = ckpt.get("input_dim", input_dim)
            hidden_dims = ckpt.get("hidden_dims", hidden_dims)
            model = ChurnMLP(input_dim=input_dim, hidden_dims=hidden_dims)
            model.load_state_dict(ckpt["state_dict"])
        elif legacy_pt.exists():
            state_dict = torch.load(legacy_pt, map_location="cpu", weights_only=True)
            if input_dim is None:
                raise RuntimeError("input_dim desconhecido e model_config.json ausente")
            model = ChurnMLP(input_dim=input_dim, hidden_dims=hidden_dims)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"No model .pt found in {models_dir}")

        model.eval()
        _state["model"] = model
        _state["input_dim"] = input_dim
        logger.info("Model ready — input_dim={}", input_dim)

    except Exception as exc:
        logger.error("Failed to load model artifacts: {}", exc)

    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="Churn Prediction API",
    description="Previsão de churn para operadora de telecomunicações — FIAP Tech Challenge Fase 1",
    version=settings.api_version,
    lifespan=lifespan,
)


@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "method={} path={} status={} latency_ms={:.2f}",
        request.method, request.url.path, response.status_code, latency_ms,
    )
    response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
    return response


def _require_model(request: Request) -> Any:
    if _state["model"] is None or _state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not available")
    return _state


ModelState = Annotated[Any, Depends(_require_model)]


def _customer_to_df(cliente: ClienteInput) -> pd.DataFrame:
    """Converte ClienteInput para DataFrame compatível com o preprocessor."""
    return pd.DataFrame([{
        "tenure": cliente.tenure,
        "monthly_charges": cliente.monthly_charges,
        "total_charges": cliente.total_charges,
        "senior_citizen": cliente.senior_citizen,
        "gender": cliente.gender,
        "partner": cliente.partner,
        "dependents": cliente.dependents,
        "phone_service": cliente.phone_service,
        "multiple_lines": cliente.multiple_lines,
        "internet_service": cliente.internet_service,
        "online_security": cliente.online_security,
        "online_backup": cliente.online_backup,
        "device_protection": cliente.device_protection,
        "tech_support": cliente.tech_support,
        "streaming_tv": cliente.streaming_tv,
        "streaming_movies": cliente.streaming_movies,
        "contract": cliente.contract,
        "paperless_billing": cliente.paperless_billing,
        "payment_method": cliente.payment_method,
    }])


def _risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "high"
    if prob >= 0.4:
        return "medium"
    return "low"


@app.get("/health", response_model=HealthOutput, tags=["Monitoring"])
async def health():
    return HealthOutput(
        status="ok",
        model_loaded=_state["model"] is not None,
        version=settings.api_version,
    )


@app.get("/ready", tags=["Monitoring"], summary="Readiness check")
async def ready():
    if _state["pipeline"] is None or _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
async def predict(cliente: ClienteInput, bundle: ModelState):
    try:
        X = bundle["pipeline"].transform(_customer_to_df(cliente)).astype(np.float32)
    except Exception as exc:
        logger.error("Preprocessing failed: {}", exc)
        raise HTTPException(status_code=422, detail=f"Preprocessing error: {exc}") from exc

    prob = float(predict_proba(bundle["model"], X)[0])
    prediction = int(prob >= 0.5)
    risk = _risk_level(prob)
    logger.info("churn_prob={:.4f} pred={} risk={}", prob, prediction, risk)

    return PredictionOutput(
        churn_probability=round(prob, 4),
        prediction=prediction,
        risk_level=risk,
    )


@app.post("/predict-batch", response_model=BatchPredictionOutput, tags=["Inference"])
async def predict_batch(
    clientes: Annotated[list[ClienteInput], ...],
    bundle: ModelState,
):
    if len(clientes) == 0 or len(clientes) > 1000:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 1000")

    try:
        df = pd.concat([_customer_to_df(c) for c in clientes], ignore_index=True)
        X = bundle["pipeline"].transform(df).astype(np.float32)
    except Exception as exc:
        logger.error("Batch preprocessing failed: {}", exc)
        raise HTTPException(status_code=422, detail=f"Preprocessing error: {exc}") from exc

    probs = predict_proba(bundle["model"], X)
    logger.info("batch_size={}", len(clientes))

    predictions = [
        PredictionOutput(
            churn_probability=round(float(p), 4),
            prediction=int(float(p) >= 0.5),
            risk_level=_risk_level(float(p)),
        )
        for p in probs
    ]
    return BatchPredictionOutput(predictions=predictions, count=len(predictions))
