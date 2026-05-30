# 🔮 Churn Prediction — FIAP Tech Challenge Fase 1

> Modelo preditivo de churn para operadora de telecomunicações.
> Pipeline end-to-end: EDA → Baselines → MLP (PyTorch) → API (FastAPI) → Deploy (Docker + CI/CD).

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green)
![MLflow](https://img.shields.io/badge/MLflow-2.10%2B-blue)
![Ruff](https://img.shields.io/badge/linting-ruff-purple)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-black)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 📋 Contexto

Uma operadora de telecomunicações enfrenta perda acelerada de clientes. Este projeto
constrói um sistema preditivo de churn do zero até o modelo servido via API, aplicando
boas práticas de Machine Learning Engineering:

- **Modelo principal:** Rede neural MLP treinada com PyTorch
- **Baselines:** DummyClassifier, LogisticRegression, RandomForest (com RandomizedSearchCV), GradientBoosting
- **Rastreamento:** MLflow (parâmetros, métricas, artefatos)
- **Serving:** API REST com FastAPI + Pydantic (single e batch inference)
- **Segurança:** JWT + API Key + Rate Limiting + CORS
- **Monitoramento:** Drift detection com KS test + PSI
- **Deploy:** Docker multi-stage + CI/CD via GitHub Actions + AWS ECS Fargate

---

## 🗂️ Estrutura do Projeto

```
FIAP-MLE-Fase01/
├── src/
│   ├── api/
│   │   ├── app.py          # FastAPI — /predict, /predict-batch, /health, /ready, /auth/*
│   │   └── schemas.py      # Schemas Pydantic (entrada e saída)
│   ├── data/
│   │   ├── preprocessing.py # Transformers sklearn + load_data()
│   │   └── schema.py        # Validação Pandera do dataset
│   ├── features/
│   │   └── engineering.py   # Feature engineering
│   ├── models/
│   │   ├── baseline.py      # DummyClassifier, LogReg, RF, GBT
│   │   └── mlp.py           # ChurnMLP (PyTorch) + EarlyStopping
│   ├── monitoring/
│   │   └── drift_detection.py # KS test + PSI para monitoramento pós-deploy
│   ├── training/
│   │   └── train.py         # Script de treino com MLflow
│   └── utils/
│       ├── config.py        # Configuração centralizada (pydantic-settings)
│       └── logger.py        # Logging estruturado (loguru)
├── tests/
│   ├── test_api.py          # Testes da API FastAPI (com JWT)
│   ├── test_model.py        # Testes do MLP PyTorch
│   ├── test_preprocessing.py # Testes de pré-processamento
│   ├── test_schema.py       # Validação do schema do dataset (pandera)
│   └── test_smoke.py        # Smoke tests: pipeline e MLP
├── notebooks/
│   └── 01_eda_baselines.ipynb # EDA + baselines
├── data/
│   ├── raw/                 # dataset original (não versionado)
│   └── processed/           # features processadas (não versionado)
├── models/                  # artefatos treinados (não versionados)
├── docs/
│   ├── model_card.md        # Model Card: performance, limitações e vieses
│   └── monitoring_plan.md   # Plano de monitoramento
├── .github/
│   └── workflows/
│       ├── ci.yml           # CI: lint + testes em todo PR
│       └── cd.yml           # CD: build e push Docker para GHCR
├── pyproject.toml           # dependências + config de ferramentas (single source of truth)
├── Makefile                 # atalhos de comandos
├── Dockerfile               # imagem multi-stage para produção
├── .env.example             # template de variáveis de ambiente
└── README.md
```

---

## 🚀 Setup Rápido

### Pré-requisitos
- Python 3.10, 3.11 ou 3.12 (3.13+ não suportado pelo torch 2.2.x)
- [uv](https://docs.astral.sh/uv/) — gerenciador de pacotes
- Git
- Make (opcional, mas recomendado)

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

### Dataset

Baixe o dataset [Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) e coloque em:

```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## ⚙️ Comandos

| Comando | Descrição |
|---|---|
| `make install` | Instala todas as dependências |
| `make train` | Treina baselines + MLP, loga no MLflow, salva artefatos |
| `make run` | Sobe a API FastAPI em `http://localhost:8000` |
| `make mlflow-ui` | Abre o MLflow UI em `http://localhost:5001`* |
| `make test` | Roda todos os testes com pytest |
| `make lint` | Verifica estilo com ruff |
| `make clean` | Remove caches e artefatos temporários |

> *A porta 5000 é reservada pelo AirPlay Receiver no macOS Monterey+.
> Para usar a porta 5000: **System Settings → AirDrop & Handoff → AirPlay Receiver → desligar**.
> Usuários Linux/Windows podem usar a porta 5000 normalmente.

---

## 🏗️ Arquitetura

```
WA_Fn-UseC_-Telco-Customer-Churn.csv
         │
    load_data()              # renomeia colunas, separa X e y
         │
  preprocessing_pipeline     # ColumnTransformer
         │                   # num: StandardScaler / cat: OneHotEncoder
    train/val/test split
    (estratificado, seed=42)
         │
    ┌────┴────────────────────────────┐
    │  Baselines (sklearn)             │  Dummy, LogReg, RF, GBT
    │  MLflow tracking                 │  params, métricas, artefatos
    └────┬────────────────────────────┘
         │
    ┌────┴────────────────┐
    │   ChurnMLP (PyTorch) │   [input(59) → 128 → 64 → 32 → 1]
    │   BCEWithLogitsLoss  │   BatchNorm + Dropout(0.3) + EarlyStopping
    │   + pos_weight       │   Adam + ReduceLROnPlateau
    └────┬────────────────┘
         │
      MLflow tracking (params, métricas, artefatos)
         │
    FastAPI                  # JWT + API Key + Rate Limiting + CORS
    ├── /predict             # predição individual (JWT)
    ├── /predict-apikey      # predição individual (API Key)
    └── /predict-batch       # predição em lote até 1000 (JWT)
         │
    Docker (multi-stage)     # python:3.12-slim + uv + usuário não-root
         │
    GitHub Actions CI/CD     # lint + testes + build + push GHCR
```

---

## 🔐 Autenticação da API

### JWT (usuários autenticados)

```bash
# 1. Login
curl -X POST "http://localhost:8000/auth/login?username=admin&password=admin123"
# Retorna: { "access_token": "eyJ...", "token_type": "bearer" }

# 2. Predição com token
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer eyJ..." \
  -H "Content-Type: application/json" \
  -d '{...payload...}'
```

**Usuários disponíveis para teste:**

| Usuário | Senha | Papel |
|---|---|---|
| `admin` | `admin123` | admin |
| `user` | `user123` | user |

> Em produção: banco de dados com senhas hasheadas e `JWT_SECRET_KEY` gerado com `openssl rand -hex 32`.

### API Key (comunicação entre serviços)

```bash
curl -X POST http://localhost:8000/predict-apikey \
  -H "X-API-Key: churn-api-key-fiap-2026" \
  -H "Content-Type: application/json" \
  -d '{...payload...}'
```

---

## 🌐 Endpoints da API

| Método | Endpoint | Auth | Descrição |
|---|---|---|---|
| GET | `/health` | Público | Status da API e modelo |
| GET | `/ready` | Público | Readiness check (503 se modelo não carregado) |
| POST | `/auth/login` | Público | Login e geração de token JWT |
| GET | `/auth/me` | JWT | Dados do usuário autenticado |
| POST | `/predict` | JWT | Predição para um cliente |
| POST | `/predict-apikey` | API Key | Predição para um cliente (serviços) |
| POST | `/predict-batch` | JWT | Predição em lote (até 1000 clientes) |

### Exemplo de requisição e resposta

```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "senior_citizen": 0, "tenure": 12, "monthly_charges": 65.5,
    "total_charges": 786.0, "gender": "Male", "partner": "Yes",
    "dependents": "No", "phone_service": "Yes", "multiple_lines": "No",
    "internet_service": "Fiber optic", "online_security": "No",
    "online_backup": "Yes", "device_protection": "No", "tech_support": "No",
    "streaming_tv": "No", "streaming_movies": "No",
    "contract": "Month-to-month", "paperless_billing": "Yes",
    "payment_method": "Electronic check"
  }'
```

```json
{
  "churn_probability": 0.7422,
  "prediction": 1,
  "risk_level": "high"
}
```

---

## 🛡️ Rate Limiting

A API limita **100 requisições por 60 segundos** por IP.
Ao exceder, retorna `429 Too Many Requests` com `retry_after` em segundos.
Headers `X-RateLimit-Limit` e `X-RateLimit-Remaining` em todas as respostas.

---

## 🐳 Docker

### Build e execução simples

```bash
docker build -t churn-prediction:latest .
docker run -p 8000:8000 \
  -e JWT_SECRET_KEY=<sua-chave> \
  -e API_KEY=<sua-api-key> \
  churn-prediction:latest
```

### Stack completa com Prometheus e Grafana

```bash
docker-compose up -d
```

| Serviço | URL |
|---|---|
| API | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin123) |

### Performance de build

O Dockerfile usa cache do uv (`--mount=type=cache`) para otimizar builds:

| Execução | Tempo |
|---|---|
| Primeira (cache vazio) | ~19 minutos |
| Subsequentes (cache populado) | ~2 segundos |

> O cache é mantido localmente pelo Docker. No CI/CD o cache é gerenciado via GitHub Actions cache.

---

## 🔄 CI/CD

| Workflow | Trigger | O que faz |
|---|---|---|
| `ci.yml` | Todo push e PR | Lint (ruff) + 38 testes (pytest) |
| `cd.yml` | Merge para `main` | Build Docker + push para GHCR |

---

## 📊 Métricas Principais

| Modelo | AUC-ROC | F1 | Recall |
|---|---|---|---|
| DummyClassifier | 0.52 | 0.29 | 0.29 |
| LogisticRegression | 0.85 | 0.62 | 0.80 |
| RandomForest | 0.82 | 0.54 | 0.48 |
| GradientBoosting | 0.84 | 0.59 | 0.52 |
| **MLP (PyTorch)** | **0.84** | **0.62** | **0.79** |

*Execute `make train` para resultados exatos no seu ambiente.*

---

## 🧪 Testes

38 testes cobrindo: smoke, schema (pandera), API (com JWT), model e preprocessing.

```bash
make test
# ou
uv run pytest tests/ -v
```

---

## 👥 Equipe

| Nome | RM | E-mail | Papel                              |
|---|---|---|------------------------------------|
| Anna Luiza de Angelis Souza Freitas | RM375350 | annaluizafreitas17@hotmail.com | Dados / Machine Learning Engineering |
| Bruno Brito de Souza | RM374808 | brunobrito.learning@gmail.com | Dados / Machine Learning Engineering |
| Fellipe Resende Bastos | RM373040 | fbastos95@gmail.com | Dados / Machine Learning Engineering |
| German Eduardo Brunner | RM375046 | brunner.brunner@gmail.com | Dados / Machine Learning Engineering |
| Marcelo da Cruz Salvador | RM375166 | macrusal@gmail.com | Software Engineering |

---

## 📐 Etapas do Projeto

| Etapa | Foco | Status |
|---|---|---|
| 1 | EDA + Baselines + MLflow | ✅ Concluída |
| 2 | MLP PyTorch + comparação de modelos | ✅ Concluída |
| 3 | Refatoração + FastAPI + testes + Makefile | ✅ Concluída |
| + | Segurança API: JWT + API Key + Rate Limiting + CORS | ✅ Concluída |
| + | 38 testes: smoke, schema, API (JWT), model, preprocessing | ✅ Concluída |
| + | Logging estruturado (loguru) + config centralizado | ✅ Concluída |
| + | Validação de dados com Pandera | ✅ Concluída |
| 4 | Model Card + README + Docker multi-stage + CI/CD GitHub Actions | ✅ Concluída |
| + | Docker: multi-stage build com uv + usuário não-root + healthcheck | ✅ Concluída |
| + | Docker: cache de build (19min → 1.6s na segunda execução) | ✅ Concluída |
| + | CI: lint + 38 testes automáticos em todo PR (GitHub Actions) | ✅ Concluída |
| + | CD: build e push automático para GHCR no merge para main | ✅ Concluída |
| + | Observabilidade: Prometheus /metrics + trace_id + X-Trace-ID | ✅ Concluída |
| + | Docker Compose: API + Prometheus + Grafana | ✅ Concluída |
| 5 | Deploy AWS ECS Fargate | 🔄 Em andamento |

---

## 📄 Licença

MIT License