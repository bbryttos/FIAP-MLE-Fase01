# Churn Prediction — FIAP Tech Challenge Fase 1
Repositório para o tech challenger da FIAP - Fase 1 - Machine Learning Engineering.


## Alunos:
* Anna Luiza de Angelis Souza Freitas- RM375350- e-mail annaluizafreitas17@hotmail.com
* Bruno Brito de Souza - RM374808 - brunobrito.learning@gmail.com
* Fellipe Resende Bastos - RM373040 - fbastos95@gmail.com
* German Eduardo Brunner - RM375046 - brunner.brunner@gmail.com

## Resumo do projeto
Rede neural MLP (PyTorch) para previsão de churn em operadora de telecomunicações, comparada com baselines Scikit-Learn, rastreada com MLflow e servida via FastAPI.

## Setup

```bash
pip install -e ".[dev]"
```

## Dados

Baixe o dataset [Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) e salve em `data/raw/telco_churn.csv`.

## Uso

```bash
# Treinar todos os modelos e registrar no MLflow
make train

# Ver experimentos no MLflow UI
make mlflow-ui   # acesse http://localhost:5000

# Subir a API de inferência
make run         # acesse http://localhost:8000/docs

# Rodar testes
make test

# Lint
make lint
```

## Estrutura

```
churn-prediction/
├── data/
│   └── raw/            # dataset original (nao versionado)
├── docs/
│   └── model_card.md   # documentacao do modelo
├── models/             # artefatos salvos (nao versionados)
├── notebooks/          # exploracao e EDA
├── src/
│   ├── api/            # FastAPI (app.py, schemas.py)
│   ├── data/           # preprocessing.py
│   ├── features/       # engineering.py
│   ├── models/         # baseline.py, mlp.py
│   └── training/       # train.py
└── tests/              # pytest
```

## API

`POST /predict` — recebe features do cliente, retorna probabilidade de churn.

`GET /health` — health check.

Documentacao interativa: `http://localhost:8000/docs`

## Bibliotecas

- **PyTorch** — MLP
- **Scikit-Learn** — pipelines e baselines
- **MLflow** — tracking de experimentos
- **FastAPI** — API de inferencia
- **Pandera** — validacao de schema
- **ruff** — linting
