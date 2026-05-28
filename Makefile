.PHONY: install lint lint-fix format test train run mlflow clean

install:
	uv sync --extra dev

lint:
	ruff check src/ tests/ train.py

lint-fix:
	ruff check --fix src/ tests/ train.py

format:
	ruff format src/ tests/ train.py

test:
	pytest tests/ -v

train:
	python train.py

run:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
