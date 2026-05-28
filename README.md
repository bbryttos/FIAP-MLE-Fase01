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
- **Monitoramento:** Drift detection com KS test + PSI

Baixe o dataset [Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) e salve em `data/raw/telco_churn.csv`.

## Estrutura

```
src/
├── api/app.py              # FastAPI — /predict, /predict-batch, /health, /ready
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

```bash
python -m venv .venv && source .venv/bin/activate
make install
```

Copie o dataset para a pasta `data/`:

```bash
cp /caminho/para/Telco_customer_churn.xlsx data/
```

---

## Comandos

| Comando | Descrição |
|---------|-----------|
| `make train` | Treina baselines + RF tuned + MLP, loga no MLflow, salva artefatos |
| `make run` | Sobe a API FastAPI em `http://localhost:8000` |
| `make mlflow` | Abre o MLflow UI em `http://localhost:5000` |
| `make test` | Roda todos os testes com pytest |
| `make lint` | Verifica estilo com ruff |
| `make clean` | Remove caches e artefatos temporários |

---

## Arquitetura

```
Telco_customer_churn.xlsx
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
    │   ChurnMLP (PyTorch) │   [input → 64 → 32 → 16 → 1]
    │   BCEWithLogitsLoss  │   BatchNorm + Dropout + EarlyStopping
    │   + pos_weight       │   Adam + ReduceLROnPlateau
    └────┬────────────────┘
         │
      MLflow tracking (params, métricas, artefatos)
         │
    FastAPI /predict         # Pydantic validation + latency middleware
    FastAPI /predict-batch   # Inferência vetorizada (até 1000 clientes)
```

---

## Endpoints da API

```bash
# Predição individual
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male", "senior_citizen": "No", "partner": "Yes",
    "dependents": "No", "tenure_months": 12, "phone_service": "Yes",
    "multiple_lines": "No", "internet_service": "DSL",
    "online_security": "No", "online_backup": "Yes",
    "device_protection": "No", "tech_support": "No",
    "streaming_tv": "No", "streaming_movies": "No",
    "contract": "Month-to-month", "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 56.95, "total_charges": 683.40
  }'
```

Resposta:
```json
{
  "churn_probability": 0.73,
  "churn_prediction": true,
  "threshold": 0.5
}
```

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Status da API e se modelo está carregado |
| GET | `/ready` | Readiness check — 503 se modelo não carregado |
| POST | `/predict` | Predição para um cliente |
| POST | `/predict-batch` | Predição vetorizada para múltiplos clientes |

---

## Métricas Principais

| Modelo | AUC-ROC | F1 | PR-AUC |
|--------|---------|----|--------|
| DummyClassifier | ~0.50 | ~0.27 | ~0.27 |
| LogisticRegression | ~0.84 | ~0.60 | ~0.68 |
| RandomForest | ~0.83 | ~0.60 | ~0.66 |
| RF Tuned (RandomizedSearchCV) | ~0.85 | ~0.62 | ~0.69 |
| GradientBoosting | ~0.85 | ~0.62 | ~0.70 |
| **MLP (PyTorch)** | **~0.86** | **~0.63** | **~0.72** |

*Execute `make train` para resultados exatos no seu ambiente.*

---

## Equipe

| Nome | RM | E-mail |
|------|----|--------|
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
