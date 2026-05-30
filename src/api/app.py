"""
Churn Prediction API — FIAP Tech Challenge Fase 1

Segurança implementada:
- API Key via header X-API-Key (autenticação simples)
- JWT (JSON Web Token) via header Authorization: Bearer <token>
- Rate Limiting: máximo de requisições por IP por janela de tempo
- CORS: controle de origens permitidas
- Logging estruturado (loguru) sem print()
- Middleware de latência (X-Latency-Ms)

Observabilidade:
- trace_id por requisição (X-Trace-ID)
- Métricas Prometheus (/metrics)
- Logs JSON estruturados com trace_id
"""

import json
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

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

# ── Configurações de segurança ────────────────────────────────────────────────
JWT_SECRET_KEY = getattr(settings, "jwt_secret_key", "churn-secret-key-fiap-tech-challenge-2026")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60
API_KEY = getattr(settings, "api_key", "churn-api-key-fiap-2026")
RATE_LIMIT_REQUESTS = getattr(settings, "rate_limit_requests", 100)
RATE_LIMIT_WINDOW = getattr(settings, "rate_limit_window", 60)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
USERS_DB = {
    "admin": {"password": pwd_context.hash("admin123"), "role": "admin"},
    "user": {"password": pwd_context.hash("user123"), "role": "user"},
}

# ── Estado da aplicação ───────────────────────────────────────────────────────
_state: dict = {"pipeline": None, "model": None, "input_dim": None}
_request_history: dict = defaultdict(deque)

# ── Métricas Prometheus ───────────────────────────────────────────────────────
PREDICTIONS_TOTAL = Counter(
    "churn_predictions_total",
    "Total de predições realizadas",
    ["auth_method", "risk_level"],
)
PREDICTION_LATENCY = Histogram(
    "churn_prediction_latency_seconds",
    "Latência das predições em segundos",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
REQUEST_LATENCY = Histogram(
    "churn_request_latency_seconds",
    "Latência total das requisições em segundos",
    ["method", "endpoint", "status"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
REQUESTS_TOTAL = Counter(
    "churn_requests_total",
    "Total de requisições recebidas",
    ["method", "endpoint", "status"],
)
MODEL_LOADED = Gauge(
    "churn_model_loaded",
    "Indica se o modelo está carregado (1=sim, 0=não)",
)
LOGIN_ATTEMPTS = Counter(
    "churn_login_attempts_total",
    "Total de tentativas de login",
    ["status"],
)
RATE_LIMIT_HITS = Counter(
    "churn_rate_limit_hits_total",
    "Total de requisições bloqueadas por rate limiting",
)
CHURN_PROBABILITY = Histogram(
    "churn_prediction_probability",
    "Distribuição das probabilidades de churn preditas",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = Path(settings.models_dir)
    logger.info("Loading model artifacts from {}", str(models_dir))
    try:
        pipeline_path = models_dir / "preprocessor.joblib"
        legacy_path = models_dir / "preprocessing_pipeline.joblib"
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
            input_dim = None
            hidden_dims = [64, 32, 16]

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
            model = ChurnMLP(input_dim=input_dim, hidden_dims=hidden_dims)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"No model .pt found in {models_dir}")

        model.eval()
        _state["model"] = model
        _state["input_dim"] = input_dim
        MODEL_LOADED.set(1)
        logger.info("Model ready — input_dim={}", input_dim)

    except Exception as exc:
        MODEL_LOADED.set(0)
        logger.error("Failed to load model artifacts: {}", exc)

    yield
    logger.info("API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description="""
API de predição de churn para operadora de telecomunicações.

**FIAP Tech Challenge Fase 1**

## Autenticação
- **JWT**: `POST /auth/login` → use o token em `Authorization: Bearer <token>`
- **API Key**: header `X-API-Key: <chave>`

## Observabilidade
- Métricas Prometheus em `/metrics`
- `X-Trace-ID` em todas as respostas
- `X-Latency-Ms` em todas as respostas
""",
    version=settings.api_version,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Monitoring", "description": "Health checks e métricas"},
        {"name": "Auth", "description": "Autenticação e tokens JWT"},
        {"name": "Inference", "description": "Predições de churn"},
    ],
)
http_bearer = HTTPBearer(auto_error=False)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Middlewares ───────────────────────────────────────────────────────────────
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    now = time.time()
    history = _request_history[client_ip]

    while history and now - history[0] > RATE_LIMIT_WINDOW:
        history.popleft()

    if len(history) >= RATE_LIMIT_REQUESTS:
        retry_after = int(RATE_LIMIT_WINDOW - (now - history[0]))
        RATE_LIMIT_HITS.inc()
        logger.warning("Rate limit exceeded for IP={}", client_ip)
        return JSONResponse(
            status_code=429,
            content={
                "detail": f"Limite excedido: {RATE_LIMIT_REQUESTS} req/{RATE_LIMIT_WINDOW}s",
                "retry_after": retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )

    history.append(now)
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS)
    response.headers["X-RateLimit-Remaining"] = str(RATE_LIMIT_REQUESTS - len(history))
    return response


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Middleware de observabilidade — trace_id, latência e métricas Prometheus."""
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id
    start = time.perf_counter()

    response = await call_next(request)

    latency = time.perf_counter() - start
    latency_ms = latency * 1000
    endpoint = request.url.path
    method = request.method
    status_code = str(response.status_code)

    # Métricas Prometheus
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint, status=status_code).observe(latency)
    REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status=status_code).inc()

    # Logging estruturado com trace_id
    logger.info(
        "trace_id={} method={} path={} status={} latency_ms={:.2f}",
        trace_id, method, endpoint, status_code, latency_ms,
    )

    # Headers de observabilidade
    response.headers["X-Trace-ID"] = trace_id
    response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
    return response


# ── Funções JWT ───────────────────────────────────────────────────────────────
def create_access_token(username: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": username, "role": role, "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer),
) -> dict:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token ausente. Use: Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return {"username": payload["sub"], "role": payload["role"]}
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token inválido ou expirado: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# ── Funções API Key ───────────────────────────────────────────────────────────
def verify_api_key(x_api_key: str | None = Header(default=None)) -> str:
    if x_api_key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API Key ausente.")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API Key inválida.")
    return x_api_key


# ── Dependency: modelo disponível ─────────────────────────────────────────────
def _require_model(request: Request) -> Any:
    if _state["model"] is None or _state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not available")
    return _state


ModelState = Annotated[Any, Depends(_require_model)]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _customer_to_df(cliente: ClienteInput) -> pd.DataFrame:
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


# ── Endpoints públicos ────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthOutput, tags=["Monitoring"])
async def health():
    return HealthOutput(
        status="ok",
        model_loaded=_state["model"] is not None,
        version=settings.api_version,
    )


@app.get("/ready", tags=["Monitoring"])
async def ready():
    if _state["pipeline"] is None or _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.get("/metrics", tags=["Monitoring"], summary="Métricas Prometheus")
async def metrics():
    """Expõe métricas no formato Prometheus para scraping."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Auth ──────────────────────────────────────────────────────────────────────
@app.post("/auth/login", tags=["Auth"])
async def login(username: str, password: str):
    """Login e geração de token JWT. Usuários: admin/admin123, user/user123"""
    user = USERS_DB.get(username)
    if not user or not pwd_context.verify(password, user["password"]):
        LOGIN_ATTEMPTS.labels(status="failed").inc()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuário ou senha incorretos.")
    LOGIN_ATTEMPTS.labels(status="success").inc()
    token = create_access_token(username, user["role"])
    logger.info("Login successful for user={}", username)
    return {"access_token": token, "token_type": "bearer", "expires_in": JWT_EXPIRE_MINUTES * 60}


@app.get("/auth/me", tags=["Auth"])
async def me(current_user: dict = Depends(verify_token)):
    return {"username": current_user["username"], "role": current_user["role"], "message": "Autenticado!"}


# ── Inferência ────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
async def predict(
    cliente: ClienteInput,
    bundle: ModelState,
    current_user: dict = Depends(verify_token),
):
    """Predição de churn para um cliente. Requer JWT."""
    try:
        X = bundle["pipeline"].transform(_customer_to_df(cliente)).astype(np.float32)
    except Exception as exc:
        logger.error("Preprocessing failed: {}", exc)
        raise HTTPException(status_code=422, detail=f"Preprocessing error: {exc}") from exc

    start = time.perf_counter()
    prob = float(predict_proba(bundle["model"], X)[0])
    PREDICTION_LATENCY.observe(time.perf_counter() - start)

    prediction = int(prob >= 0.5)
    risk = _risk_level(prob)

    PREDICTIONS_TOTAL.labels(auth_method="jwt", risk_level=risk).inc()
    CHURN_PROBABILITY.observe(prob)

    logger.info("user={} churn_prob={:.4f} pred={} risk={}", current_user["username"], prob, prediction, risk)
    return PredictionOutput(churn_probability=round(prob, 4), prediction=prediction, risk_level=risk)


@app.post("/predict-apikey", response_model=PredictionOutput, tags=["Inference"])
async def predict_apikey(
    cliente: ClienteInput,
    bundle: ModelState,
    _: str = Depends(verify_api_key),
):
    """Predição de churn para um cliente. Requer API Key."""
    try:
        X = bundle["pipeline"].transform(_customer_to_df(cliente)).astype(np.float32)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Preprocessing error: {exc}") from exc

    start = time.perf_counter()
    prob = float(predict_proba(bundle["model"], X)[0])
    PREDICTION_LATENCY.observe(time.perf_counter() - start)

    prediction = int(prob >= 0.5)
    risk = _risk_level(prob)

    PREDICTIONS_TOTAL.labels(auth_method="api_key", risk_level=risk).inc()
    CHURN_PROBABILITY.observe(prob)

    return PredictionOutput(churn_probability=round(prob, 4), prediction=prediction, risk_level=risk)


@app.post("/predict-batch", response_model=BatchPredictionOutput, tags=["Inference"])
async def predict_batch(
    clientes: Annotated[list[ClienteInput], ...],
    bundle: ModelState,
    current_user: dict = Depends(verify_token),
):
    """Predição em batch para múltiplos clientes (1 a 1000). Requer JWT."""
    if len(clientes) == 0 or len(clientes) > 1000:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 1000")

    try:
        df = pd.concat([_customer_to_df(c) for c in clientes], ignore_index=True)
        X = bundle["pipeline"].transform(df).astype(np.float32)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Preprocessing error: {exc}") from exc

    probs = predict_proba(bundle["model"], X)
    logger.info("user={} batch_size={}", current_user["username"], len(clientes))

    predictions = [
        PredictionOutput(
            churn_probability=round(float(p), 4),
            prediction=int(float(p) >= 0.5),
            risk_level=_risk_level(float(p)),
        )
        for p in probs
    ]

    for p in probs:
        risk = _risk_level(float(p))
        PREDICTIONS_TOTAL.labels(auth_method="jwt_batch", risk_level=risk).inc()
        CHURN_PROBABILITY.observe(float(p))

    return BatchPredictionOutput(predictions=predictions, count=len(predictions))
