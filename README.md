# 🔮 Churn Prediction — FIAP Tech Challenge Fase 1

> Modelo preditivo de churn para operadora de telecomunicações.  
> Pipeline end-to-end: EDA → Baselines → MLP (PyTorch) → API (FastAPI) → Deploy (AWS).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green)
![MLflow](https://img.shields.io/badge/MLflow-2.10%2B-blue)
![Ruff](https://img.shields.io/badge/linting-ruff-purple)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 📋 Contexto

Uma operadora de telecomunicações enfrenta perda acelerada de clientes. Este projeto
constrói um sistema preditivo de churn do zero até o modelo servido via API, aplicando
boas práticas de Machine Learning Engineering:

- **Modelo principal:** Rede neural MLP treinada com PyTorch
- **Baselines:** DummyClassifier e Regressão Logística (Scikit-Learn)
- **Rastreamento:** MLflow (parâmetros, métricas, artefatos)
- **Serving:** API REST com FastAPI + Pydantic
- **Deploy:** Docker + AWS ECS Fargate (bônus)

---

## 🗂️ Estrutura do Projeto

```
FIAP-MLE-Fase01/
├── src/
│   ├── api/           # FastAPI — /predict, /health
│   ├── data/          # loaders e pré-processamento
│   ├── features/      # feature engineering e pipelines sklearn
│   ├── models/        # MLP PyTorch e baselines
│   ├── training/      # loops de treino, avaliação e MLflow tracking
│   └── utils/         # logging estruturado (loguru) e config centralizado
├── tests/             # pytest: smoke test, schema (pandera), API
├── notebooks/         # EDA exploratória e análise de baselines
├── data/
│   ├── raw/           # dataset original (não versionado)
│   └── processed/     # features processadas (não versionado)
├── models/            # artefatos treinados (não versionados)
├── docs/
│   └── model_card.md  # Model Card: performance, limitações e vieses
├── .github/
│   └── workflows/     # CI/CD GitHub Actions
├── pyproject.toml     # dependências + config de ferramentas (single source of truth)
├── Makefile           # atalhos de comandos
├── Dockerfile         # imagem multi-stage para produção
├── .env.example       # template de variáveis de ambiente
└── README.md
```

---

## 🚀 Setup Rápido

### Pré-requisitos
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) — gerenciador de pacotes e ambientes virtuais
- Git
- Make (opcional, mas recomendado)

### Instalação do uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Via pip (qualquer OS)
pip install uv
```

> O `uv` substitui o pip + venv com resolução de dependências muito mais rápida.
> O arquivo `uv.lock` garante que todos do grupo usem exatamente as mesmas versões.

### 1. Clone o repositório
```bash
git clone git@github.com:bbryttos/FIAP-MLE-Fase01.git
cd FIAP-MLE-Fase01
```

### 2. Crie o ambiente virtual e instale as dependências
```bash
uv sync --extra dev
```

> O `uv` cria automaticamente o `.venv`, resolve e instala todas as dependências
> definidas no `pyproject.toml` (produção + dev). O `uv.lock` garante reprodutibilidade total.

**PyCharm:** File → Settings → Python Interpreter → Add → Existing → selecione `.venv/bin/python`

**VSCode:** `Ctrl+Shift+P` → "Python: Select Interpreter" → selecione `.venv/bin/python`

### 3. Configure as variáveis de ambiente
```bash
cp .env.example .env
# edite o .env conforme seu ambiente
```

### 4. Valide a instalação
```bash
uv run python -c "from src.utils.config import settings; print('Seed:', settings.seed)"
# Saída esperada: Seed: 42
```

> **Sem uv?** Também é possível usar pip:
> ```bash
> python -m venv .venv && source .venv/bin/activate
> pip install --upgrade pip setuptools wheel
> pip install -e ".[dev]"
> ```

---

## ⚙️ Comandos Disponíveis

```bash
make install-dev    # instala todas as dependências
make lint           # verifica qualidade do código (ruff)
make format         # formata o código
make test           # executa os testes com cobertura
make train-mlp      # treina a rede neural MLP
make run-api        # sobe a API em modo dev (localhost:8000)
make mlflow-ui      # abre o MLflow UI (localhost:5000)
make docker-build   # builda a imagem Docker
```

> Com `uv`, prefixe os comandos com `uv run` se o ambiente não estiver ativado:
> `uv run pytest tests/ -v`

---

## 🧪 Testes

O projeto mantém cobertura mínima de 3 testes obrigatórios:

| Teste | Descrição |
|---|---|
| Smoke test | valida que o modelo carrega e faz predição |
| Schema test | valida schema do dataset com Pandera |
| API test | valida endpoints `/predict` e `/health` |

```bash
make test
# ou
uv run pytest tests/ -v
```

---

## 📊 Dataset

**Telco Customer Churn (IBM)** — dataset público de telecomunicações.

- 7.043 registros · 21 features · target binário (`Churn`: Yes/No)
- ~26% de churn — desbalanceamento tratado com SMOTE (`imbalanced-learn`)
- Download: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Após o download, coloque o arquivo em:
```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## 🔬 Experimentos e MLflow

Todos os experimentos são rastreados no MLflow:

```bash
make mlflow-ui
# ou
uv run mlflow ui --port 5000
# Acesse: http://localhost:5000
```

Métricas rastreadas: `AUC-ROC`, `PR-AUC`, `F1-Score`, `Precision`, `Recall`

---

## 🌐 API de Inferência

```bash
make run-api
# ou
uv run uvicorn src.api.main:app --reload --port 8000
# Acesse: http://localhost:8000/docs
```

### Endpoints

| Método | Endpoint | Descrição |
|---|---|---|
| GET | `/health` | verifica status da API |
| POST | `/predict` | retorna probabilidade de churn |

### Exemplo de requisição
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "MonthlyCharges": 65.5, "TotalCharges": 786.0}'
```

---

## 🐳 Docker

```bash
make docker-build
docker run -p 8000:8000 churn-prediction:latest
```

---

## ☁️ Deploy AWS (Bônus)

Deploy via AWS ECS Fargate + ECR com GitHub Actions.  
Documentação de arquitetura: [`docs/model_card.md`](docs/model_card.md)

---

## 📐 Etapas do Projeto

| Etapa | Foco | Status |
|---|---|---|
| 1 | EDA + ML Canvas + Baselines + MLflow | 🔄 Em andamento |
| 2 | MLP PyTorch + comparação de modelos | ⏳ Pendente |
| 3 | Refatoração + FastAPI + testes + Makefile | ⏳ Pendente |
| 4 | Model Card + README + vídeo STAR + deploy | ⏳ Pendente |

---

## 👥 Equipe

| Nome | RM                     |e-mail| Papel                                |
|---|------------------------|---|--------------------------------------|
| Anna Luiza de Angelis Souza Freitas |RM375350|annaluizafreitas17@hotmail.com| Dados / Machine Learning Engineering                            |
| Bruno Brito de Souza |RM374808|brunobrito.learning@gmail.com| Dados / Machine Learning Engineering                           |
| Fellipe Resende Bastos |RM373040|fbastos95@gmail.com| Dados / Machine Learning Engineering |
| German Eduardo Brunner |RM375046|brunner.brunner@gmail.com| Dados / Machine Learning Engineering                           |
| Marcelo da Cruz Salvador |RM375166|macrusal@gmail.com| Dados / Machine Learning Engineering                           |

---

## 📄 Licença

MIT License
