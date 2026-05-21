.PHONY: install lint format test train run mlflow

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/ train.py
	ruff format --check src/ tests/ train.py

format:
	ruff format src/ tests/ train.py

test:
	pytest tests/ -v

train:
	python train.py

run:
	uvicorn src.api.app:app --reload --port 8000

mlflow:
	mlflow ui --port 5000
