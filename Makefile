all: 
	poetry run python src/

install:
	poetry install

test:
	poetry run pytest

audit:
	poetry run pip-audit --skip-editable

mlflow:
	poetry run mlflow server --host 127.0.0.1 --port 5000

.PHONY: all install test audit mlflow
