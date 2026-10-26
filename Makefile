all: 
	poetry run python src/

install:
	poetry install

test:
	poetry run pytest

audit:
	poetry run pip-audit --skip-editable

MLFLOW_PORT ?= 5000

mlflow:
	poetry run mlflow server --host 127.0.0.1 --port $(MLFLOW_PORT) --backend-store-uri sqlite:///mlflow.db

.PHONY: all install test audit mlflow
