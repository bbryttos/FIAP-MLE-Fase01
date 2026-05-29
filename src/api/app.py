"""
Churn Prediction API — FIAP Tech Challenge Fase 1

Segurança implementada:
- API Key via header X-API-Key (autenticação simples)
- JWT (JSON Web Token) via header Authorization: Bearer <token>
- Rate Limiting: máximo de requisições por IP por janela de tempo
- CORS: controle de origens permitidas
- Logging estruturado (loguru) sem print()
- Middleware de latência (X-Latency-Ms)
"""

import json
import time
from collections import defaultdict
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Annotated
from typing import Any
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import Depends
from fastapi import FastAPI
from fastapi import Header
from fastapi import HTTPException
from fastapi import Request
from fastapi import status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jose import JWTError
from jose import jwt
from passlib.context import CryptContext

from src.api.schemas import BatchPredictionOutput
from src.api.schemas import ClienteInput
from src.api.schemas import HealthOutput
from src.api.schemas import PredictionOutput
from src.models.mlp import ChurnMLP
from src.models.mlp import predict_proba
from src.utils.config import settings
from src.utils.logger import get_logger

from fastapi.security import HTTPBearer
from fastapi.security import HTTPAuthorizationCredentials

logger = get_logger(__name__)

# ── Configurações de segurança ────────────────────────────────────────────────
# Em produção, gere com: openssl rand -hex 32
# e coloque no .env como JWT_SECRET_KEY
JWT_SECRET_KEY = getattr(settings, "jwt_secret_key", "churn-secret-key-fiap-tech-challenge-2026")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60

# API Key simples para comunicação entre serviços
API_KEY = getattr(settings, "api_key", "churn-api-key-fiap-2026")

# Rate Limiting
RATE_LIMIT_REQUESTS = 100   # máximo de requisições
RATE_LIMIT_WINDOW = 60      # janela em segundos

# Usuários para demonstração do JWT
# Em produção: use banco de dados com senhas hasheadas
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
USERS_DB = {
    "admin": {
        "password": pwd_context.hash("admin123"),
        "role": "admin",
    },
    "user": {
        "password": pwd_context.hash("user123"),
        "role": "user",
    },
}

# ── Estado da aplicação ───────────────────────────────────────────────────────
_state: dict = {"pipeline": None, "model": None, "input_dim": None}

# Rate limiting: armazena timestamps por IP
_request_history: dict = defaultdict(deque)


# ── Lifespan: carregamento dos artefatos ──────────────────────────────────────
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
            logger.warning("model_config.json ausente — inferindo input_dim via preprocessor")
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


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description="""
API de predição de churn para operadora de telecomunicações.

**FIAP Tech Challenge Fase 1**

## Autenticação

Dois métodos disponíveis:

### API Key (comunicação entre serviços)
Envie o header `X-API-Key: <sua-chave>` nas requisições.

### JWT (usuários autenticados)
1. `POST /auth/login` com usuário e senha
2. Use o `access_token` retornado no header `Authorization: Bearer <token>`
""",
    version=settings.api_version,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Monitoring", "description": "Health checks e status da API"},
        {"name": "Auth", "description": "Autenticação e geração de tokens JWT"},
        {"name": "Inference", "description": "Predições de churn"},
    ],
)
http_bearer = HTTPBearer(auto_error=False)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Middlewares ───────────────────────────────────────────────────────────────
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting por IP — máximo RATE_LIMIT_REQUESTS por RATE_LIMIT_WINDOW segundos."""
    client_ip = request.client.host
    now = time.time()
    history = _request_history[client_ip]

    # Remove requisições fora da janela de tempo
    while history and now - history[0] > RATE_LIMIT_WINDOW:
        history.popleft()

    if len(history) >= RATE_LIMIT_REQUESTS:
        retry_after = int(RATE_LIMIT_WINDOW - (now - history[0]))
        logger.warning("Rate limit exceeded for IP={}", client_ip)
        return JSONResponse(
            status_code=429,
            content={
                "detail": f"Limite excedido: {RATE_LIMIT_REQUESTS} requisições por {RATE_LIMIT_WINDOW}s",
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
async def latency_middleware(request: Request, call_next):
    """Middleware de latência — adiciona X-Latency-Ms no header de resposta."""
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "method={} path={} status={} latency_ms={:.2f}",
        request.method, request.url.path, response.status_code, latency_ms,
    )
    response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
    return response


# ── Funções JWT ───────────────────────────────────────────────────────────────
def create_access_token(username: str, role: str) -> str:
    """Cria um token JWT com expiração."""
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {
        "sub": username,
        "role": role,
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
) -> dict:
    """Valida o token JWT do header Authorization: Bearer <token>."""
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
def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> str:
    """Valida API Key do header X-API-Key."""
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key ausente. Envie o header X-API-Key.",
        )
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key inválida.",
        )
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
    """Verifica status da API e se o modelo está carregado."""
    return HealthOutput(
        status="ok",
        model_loaded=_state["model"] is not None,
        version=settings.api_version,
    )


@app.get("/ready", tags=["Monitoring"], summary="Readiness check")
async def ready():
    """Retorna 503 se o modelo não estiver carregado."""
    if _state["pipeline"] is None or _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


# ── Endpoints de autenticação ─────────────────────────────────────────────────
@app.post("/auth/login", tags=["Auth"], summary="Login e geração de token JWT")
async def login(username: str, password: str):
    """
    Autentica o usuário e retorna um token JWT.

    Usuários disponíveis para teste:
    - admin / admin123
    - user / user123
    """
    user = USERS_DB.get(username)
    if not user or not pwd_context.verify(password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuário ou senha incorretos.",
        )
    token = create_access_token(username, user["role"])
    logger.info("Login successful for user={}", username)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": JWT_EXPIRE_MINUTES * 60,
    }


@app.get("/auth/me", tags=["Auth"], summary="Informações do usuário autenticado")
async def me(current_user: dict = Depends(verify_token)):
    """Retorna informações do usuário autenticado via JWT."""
    return {
        "username": current_user["username"],
        "role": current_user["role"],
        "message": "Você está autenticado!",
    }


# ── Endpoints de inferência (protegidos) ──────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["Inference"],
    summary="Predição de churn — protegido por JWT ou API Key",
)
async def predict(
    cliente: ClienteInput,
    bundle: ModelState,
    current_user: dict = Depends(verify_token),
):
    """
    Realiza predição de churn para um cliente.

    **Requer autenticação JWT** — obtenha o token em `/auth/login`.
    """
    try:
        X = bundle["pipeline"].transform(_customer_to_df(cliente)).astype(np.float32)
    except Exception as exc:
        logger.error("Preprocessing failed: {}", exc)
        raise HTTPException(status_code=422, detail=f"Preprocessing error: {exc}") from exc

    prob = float(predict_proba(bundle["model"], X)[0])
    prediction = int(prob >= 0.5)
    risk = _risk_level(prob)
    logger.info(
        "user={} churn_prob={:.4f} pred={} risk={}",
        current_user["username"], prob, prediction, risk,
    )
    return PredictionOutput(
        churn_probability=round(prob, 4),
        prediction=prediction,
        risk_level=risk,
    )


@app.post(
    "/predict-apikey",
    response_model=PredictionOutput,
    tags=["Inference"],
    summary="Predição de churn — protegido por API Key",
)
async def predict_apikey(
    cliente: ClienteInput,
    bundle: ModelState,
    _: str = Depends(verify_api_key),
):
    """
    Realiza predição de churn para um cliente.

    **Requer API Key** no header `X-API-Key` — ideal para comunicação entre serviços.
    """
    try:
        X = bundle["pipeline"].transform(_customer_to_df(cliente)).astype(np.float32)
    except Exception as exc:
        logger.error("Preprocessing failed: {}", exc)
        raise HTTPException(status_code=422, detail=f"Preprocessing error: {exc}") from exc

    prob = float(predict_proba(bundle["model"], X)[0])
    prediction = int(prob >= 0.5)
    risk = _risk_level(prob)
    logger.info("apikey churn_prob={:.4f} pred={} risk={}", prob, prediction, risk)
    return PredictionOutput(
        churn_probability=round(prob, 4),
        prediction=prediction,
        risk_level=risk,
    )


@app.post(
    "/predict-batch",
    response_model=BatchPredictionOutput,
    tags=["Inference"],
    summary="Predição em batch — protegido por JWT",
)
async def predict_batch(
    clientes: Annotated[list[ClienteInput], ...],
    bundle: ModelState,
    current_user: dict = Depends(verify_token),
):
    """
    Realiza predição de churn para múltiplos clientes (1 a 1000).

    **Requer autenticação JWT**.
    """
    if len(clientes) == 0 or len(clientes) > 1000:
        raise HTTPException(status_code=422, detail="Batch size must be between 1 and 1000")

    try:
        df = pd.concat([_customer_to_df(c) for c in clientes], ignore_index=True)
        X = bundle["pipeline"].transform(df).astype(np.float32)
    except Exception as exc:
        logger.error("Batch preprocessing failed: {}", exc)
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
    return BatchPredictionOutput(predictions=predictions, count=len(predictions))