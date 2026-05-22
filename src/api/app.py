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

from src.api.schemas import ClienteInput, HealthOutput, PredictionOutput
from src.models.mlp import MLP
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = Path(settings.models_dir)
    app.state.model_loaded = False
    app.state.preprocessor = None
    app.state.model = None

    logger.info("Loading model artifacts from {dir}", dir=models_dir)
    try:
        preprocessor = joblib.load(models_dir / "preprocessor.joblib")
        app.state.preprocessor = preprocessor

        cfg_path = models_dir / "model_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                input_dim = json.load(f)["input_dim"]
        else:
            # fallback para modelos treinados antes da persistência de metadata
            logger.warning("model_config.json ausente — inferindo input_dim via preprocessor")
            dummy = {
                "tenure": [0], "MonthlyCharges": [0.0], "TotalCharges": [0.0],
                "SeniorCitizen": [0], "gender": ["Male"], "Partner": ["No"],
                "Dependents": ["No"], "PhoneService": ["No"], "MultipleLines": ["No"],
                "InternetService": ["No"], "OnlineSecurity": ["No internet service"],
                "OnlineBackup": ["No internet service"], "DeviceProtection": ["No internet service"],
                "TechSupport": ["No internet service"], "StreamingTV": ["No internet service"],
                "StreamingMovies": ["No internet service"], "Contract": ["Month-to-month"],
                "PaperlessBilling": ["No"], "PaymentMethod": ["Mailed check"],
            }
            input_dim = preprocessor.transform(pd.DataFrame(dummy)).shape[1]

        model = MLP(input_dim=input_dim, hidden_dims=[128, 64, 32])
        model.load_state_dict(
            torch.load(models_dir / "mlp_weights.pt", map_location="cpu", weights_only=True)
        )
        model.eval()
        app.state.model = model
        app.state.model_loaded = True
        logger.info("Model ready (input_dim={dim})", dim=input_dim)
    except Exception as exc:
        logger.error("Failed to load model artifacts: {error}", error=exc)

    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="Churn Prediction API",
    description="Previsao de churn para operadora de telecomunicacoes — FIAP Tech Challenge Fase 1",
    version=settings.api_version,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency_ms = round((time.time() - start) * 1000, 2)
    logger.info(
        "method={method} path={path} status={status} latency_ms={latency}",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency=latency_ms,
    )
    return response


def _require_model(request: Request) -> Any:
    if not getattr(request.app.state, "model_loaded", False):
        raise HTTPException(status_code=503, detail="Model not available")
    return request.app.state


ModelState = Annotated[Any, Depends(_require_model)]


@app.get("/health", response_model=HealthOutput, tags=["Monitoring"])
def health(request: Request):
    return HealthOutput(
        status="ok",
        model_loaded=getattr(request.app.state, "model_loaded", False),
        version=settings.api_version,
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict(cliente: ClienteInput, bundle: ModelState):
    try:
        df = pd.DataFrame([cliente.model_dump()])
        X = bundle.preprocessor.transform(df).astype(np.float32)
        X_tensor = torch.FloatTensor(X)

        with torch.no_grad():
            logit = bundle.model(X_tensor)
            prob = torch.sigmoid(logit).item()

        prediction = int(prob >= 0.5)
        risk_level = "high" if prob >= 0.7 else "medium" if prob >= 0.4 else "low"

        logger.info(
            "prediction: churn_prob={prob} pred={pred} risk={risk}",
            prob=round(prob, 4),
            pred=prediction,
            risk=risk_level,
        )

        return PredictionOutput(
            churn_probability=round(prob, 4),
            prediction=prediction,
            risk_level=risk_level,
        )

    except Exception as exc:
        logger.error("Prediction error: {error}", error=exc)
        raise HTTPException(status_code=500, detail="Internal prediction error") from exc
