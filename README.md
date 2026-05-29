# Churn Prediction — FIAP Tech Challenge Fase 1

Modelo preditivo de churn para operadora de telecomunicações.
Pipeline end-to-end: EDA → Baselines → MLP (PyTorch) → API (FastAPI) → Monitoramento.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green)
![MLflow](https://img.shields.io/badge/MLflow-2.10%2B-blue)
![Ruff](https://img.shields.io/badge/linting-ruff-purple)

---

## Contexto

Uma operadora de telecomunicações enfrenta perda acelerada de clientes. Este projeto
constrói um sistema preditivo de churn do zero até o modelo servido via API, aplicando
boas práticas de Machine Learning Engineering:

- **Modelo principal:** Rede neural MLP treinada com PyTorch
- **Baselines:** DummyClassifier, LogisticRegression, RandomForest (com RandomizedSearchCV), GradientBoosting
- **Rastreamento:** MLflow (parâmetros, métricas, artefatos)
- **Serving:** API REST com FastAPI + Pydantic (single e batch inference)
- **Segurança:** JWT + API Key + Rate Limiting + CORS
- **Monitoramento:** Drift detection com KS test + PSI

Baixe o dataset [Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) e salve em `data/raw/Telco_customer_churn.csv`.

## Estrutura

```
src/
├── api/app.py              # FastAPI — /predict, /predict-batch, /health, /ready, /auth/*
├── data/preprocessing.py   # Transformers sklearn + load_data()
├── models/
│   ├── baseline.py         # DummyClassifier, LogReg, RF, GBT
│   └── mlp.py              # ChurnMLP (PyTorch) + EarlyStopping
└── monitoring/
    └── drift_detection.py  # KS test + PSI para monitoramento pós-deploy
train.py                    # Script de treino com MLflow + RandomizedSearchCV
tests/
├── test_schema.py          # Validação do schema do dataset (pandera)
├── test_smoke.py           # Smoke tests: pipeline e MLP
└── test_api.py             # Testes da API FastAPI
docs/
├── model_card.md           # Model Card completo
└── monitoring_plan.md      # Plano de monitoramento
notebooks/
└── 01_eda_baselines.ipynb  # EDA + baselines
```

---

## Setup

### Pré-requisitos
- Python 3.10, 3.11 ou 3.12 (3.13+ não suportado pelo torch 2.2.x)
- [uv](https://docs.astral.sh/uv/) — gerenciador de pacotes

### Instalação do uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Via pip
pip install uv
```

### Instalação do projeto

```bash
git clone git@github.com:bbryttos/FIAP-MLE-Fase01.git
cd FIAP-MLE-Fase01

# Cria o ambiente e instala todas as dependências
uv sync --extra dev

# Configura as variáveis de ambiente
cp .env.example .env
```

### Sem uv (alternativa com pip)

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

### Valida a instalação

```bash
uv run python -c "from src.utils.config import settings; print('Seed:', settings.seed)"
# Saída esperada: Seed: 42
```

---

## Comandos

| Comando | Descrição |
|---|---|
| `make install` | Instala todas as dependências |
| `make train` | Treina baselines + RF tuned + MLP, loga no MLflow, salva artefatos |
| `make run` | Sobe a API FastAPI em `http://localhost:8000` |
| `make mlflow-ui` | Abre o MLflow UI em `http://localhost:5001`* |
| `make test` | Roda todos os testes com pytest |
| `make lint` | Verifica estilo com ruff |
| `make clean` | Remove caches e artefatos temporários |

> *A porta 5000 é reservada pelo AirPlay Receiver no macOS Monterey+.
> Para usar a porta 5000 no macOS: **System Settings → AirDrop & Handoff → AirPlay Receiver → desligar**.
> Usuários Linux/Windows podem usar a porta 5000 normalmente.

---

## Arquitetura

```
Telco_customer_churn.csv
         │
    load_data()              # drop leakage cols, separa X e y
         │
  preprocessing_pipeline     # TotalChargesImputer → BinaryEncoder
         │                   # → ColumnTransformer (num: impute+clip+scale / cat: OHE)
    train/val/test split
    (estratificado, seed=42)
         │
    ┌────┴────────────────────────────┐
    │  Baselines (sklearn)             │  Dummy, LogReg, RF, GBT
    │  + RandomizedSearchCV (RF)       │  20 iter, 5-fold, scoring=AUC
    └────┬────────────────────────────┘
         │
    ┌────┴────────────────┐
    │   ChurnMLP (PyTorch) │   [input → 128 → 64 → 32 → 1]
    │   BCEWithLogitsLoss  │   BatchNorm + Dropout + EarlyStopping
    │   + pos_weight       │   Adam + ReduceLROnPlateau
    └────┬────────────────┘
         │
      MLflow tracking (params, métricas, artefatos)
         │
    FastAPI /predict         # JWT auth + Pydantic validation + latency middleware
    FastAPI /predict-batch   # Inferência vetorizada (até 1000 clientes)
```

---

## Autenticação da API

A API possui dois métodos de autenticação:

### JWT (usuários autenticados)

```bash
# 1. Faz login e obtém o token
curl -X POST "http://localhost:8000/auth/login?username=admin&password=admin123"

# Resposta:
# {
#   "access_token": "eyJ...",
#   "token_type": "bearer",
#   "expires_in": 3600
# }

# 2. Usa o token nas requisições
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer eyJ..." \
  -H "Content-Type: application/json" \
  -d '{...}'
```

**Usuários disponíveis para teste:**

| Usuário | Senha | Papel |
|---|---|---|
| `admin` | `admin123` | admin |
| `user` | `user123` | user |

> Em produção: substitua por banco de dados com senhas hasheadas e gere o `JWT_SECRET_KEY` com `openssl rand -hex 32`.

### API Key (comunicação entre serviços)

```bash
curl -X POST http://localhost:8000/predict-apikey \
  -H "X-API-Key: churn-api-key-fiap-2026" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

> Em produção: defina `API_KEY` no `.env` com um valor forte gerado por `openssl rand -hex 16`.

---

## Endpoints da API

| Método | Endpoint | Auth | Descrição |
|---|---|---|---|
| GET | `/health` | Público | Status da API e se modelo está carregado |
| GET | `/ready` | Público | Readiness check — 503 se modelo não carregado |
| POST | `/auth/login` | Público | Login e geração de token JWT |
| GET | `/auth/me` | JWT | Dados do usuário autenticado |
| POST | `/predict` | JWT | Predição para um cliente |
| POST | `/predict-apikey` | API Key | Predição para um cliente (serviços) |
| POST | `/predict-batch` | JWT | Predição para múltiplos clientes (até 1000) |

### Exemplo de requisição

```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
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
    "payment_method": "Electronic check"
  }'
```

### Exemplo de resposta

```json
{
  "churn_probability": 0.7422,
  "prediction": 1,
  "risk_level": "high"
}
```

---

## Rate Limiting

A API limita **100 requisições por 60 segundos** por IP.
Ao exceder o limite, retorna `429 Too Many Requests`:

```json
{
  "detail": "Limite excedido: 100 requisições por 60s",
  "retry_after": 45
}
```

Os headers `X-RateLimit-Limit` e `X-RateLimit-Remaining` são retornados em todas as respostas.

---

## Métricas Principais

| Modelo | AUC-ROC | F1 | Recall |
|---|---|---|---|
| DummyClassifier | 0.52 | 0.29 | 0.29 |
| LogisticRegression | 0.85 | 0.62 | 0.80 |
| RandomForest | 0.82 | 0.54 | 0.48 |
| GradientBoosting | 0.84 | 0.59 | 0.52 |
| **MLP (PyTorch)** | **0.84** | **0.62** | **0.79** |

*Execute `make train` para resultados exatos no seu ambiente.*

---

## Equipe

| Nome | RM | E-mail |
|---|---|---|
| Anna Luiza de Angelis Souza Freitas | RM375350 | annaluizafreitas17@hotmail.com |
| Bruno Brito de Souza | RM374808 | brunobrito.learning@gmail.com |
| Fellipe Resende Bastos | RM373040 | fbastos95@gmail.com |
| German Eduardo Brunner | RM375046 | brunner.brunner@gmail.com |
| Marcelo da Cruz Salvador | RM375166 | macrusal@gmail.com |

---

## Etapas do Projeto

| Etapa | Foco | Status |
|---|---|---|
| 1 | EDA + Baselines + MLflow | ✅ Concluída |
| 2 | MLP PyTorch + comparação de modelos | ✅ Concluída |
| 3 | Refatoração + FastAPI + testes + Makefile | ✅ Concluída |
| 4 | Model Card + README + vídeo STAR + deploy | ✅ Concluída |

---

MIT License