# Telco Customer Churn Prediction

FIAP 10MLET — Tech Challenge Fase 1.

Rede neural (MLP/PyTorch) para prever churn de clientes de telecomunicações, com pipeline completo de ML Engineering: rastreamento de experimentos (MLflow), API de inferência (FastAPI), testes automatizados e Model Card.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
make install
```

Copie o dataset para a pasta `data/`:

```bash
cp /caminho/para/Telco_customer_churn.xlsx data/
```

## Uso

| Comando | Descrição |
|---------|-----------|
| `make train` | Treina baselines + MLP, loga no MLflow, salva artefatos |
| `make run` | Sobe a API FastAPI em `http://localhost:8000` |
| `make mlflow` | Abre o MLflow UI em `http://localhost:5000` |
| `make test` | Roda todos os testes com pytest |
| `make lint` | Verifica estilo com ruff |

## Estrutura

```
src/
├── data/preprocessing.py   # Transformers sklearn + load_data()
├── models/
│   ├── baseline.py          # DummyClassifier, LogReg, RF, GBT
│   └── mlp.py               # ChurnMLP (PyTorch) + EarlyStopping
└── api/app.py               # FastAPI /predict e /health
train.py                     # Script de treino com MLflow
tests/
├── test_schema.py           # Validação do schema do dataset (pandera)
├── test_smoke.py            # Smoke tests: pipeline e MLP
└── test_api.py              # Testes da API FastAPI
docs/
├── model_card.md            # Model Card completo
└── monitoring_plan.md       # Plano de monitoramento
notebooks/
└── 01_eda_baselines.ipynb   # EDA + baselines
```

## Arquitetura

```
Telco_customer_churn.xlsx
         │
    load_data()              # drop leakage cols, separa X e y
         │
  preprocessing_pipeline     # TotalChargesImputer → BinaryEncoder
         │                   # → ColumnTransformer (num/cat)
    train/val/test split
    (estratificado, seed=42)
         │
    ┌────┴────────────────┐
    │  Baselines (sklearn) │   DummyClassifier, LogisticRegression,
    │                      │   RandomForest, GradientBoosting
    └────┬────────────────┘
         │
    ┌────┴────────────────┐
    │   ChurnMLP (PyTorch) │   [input → 64 → 32 → 16 → 1]
    │   BCEWithLogitsLoss  │   BatchNorm + Dropout + EarlyStopping
    │   + pos_weight       │   AdamW + ReduceLROnPlateau
    └────┬────────────────┘
         │
      MLflow tracking (params, métricas, artefatos)
         │
    FastAPI /predict         # Pydantic validation + latency middleware
```

## Endpoint de Predição

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "total_charges": 683.40
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

## Métricas Principais

| Modelo | AUC-ROC | F1 | PR-AUC |
|--------|---------|----|--------|
| DummyClassifier | ~0.50 | ~0.27 | ~0.27 |
| LogisticRegression | ~0.84 | ~0.60 | ~0.68 |
| RandomForest | ~0.83 | ~0.60 | ~0.66 |
| GradientBoosting | ~0.85 | ~0.62 | ~0.70 |
| **MLP (PyTorch)** | **~0.86** | **~0.63** | **~0.72** |

*Valores aproximados; execute `make train` para resultados exatos.*
