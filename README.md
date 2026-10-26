# stores-sales-forecasting-poc

A medium article is available here : https://medium.com/@antoine.caille/what-factors-influence-grocery-sales-in-ecuador-fab4f4c4636d


## Getting started

Requires Python 3.12 or 3.13. On macOS, XGBoost also needs OpenMP: `brew install libomp`.

Install dependencies :

```sh
$ make install
```

Download the dataset. The Kaggle CLI reads its API token from `~/.kaggle/kaggle.json`, which must stay out of the repository and be readable only by you (`chmod 600 ~/.kaggle/kaggle.json`) :

```sh
$ poetry run kaggle competitions download -c store-sales-time-series-forecasting
$ mkdir -p data/raw
$ unzip store-sales-time-series-forecasting.zip -d data/raw
```

Paths, features and the train/validation ratio live in `config/data/base.yaml`.

## Tracking server

Training and evaluation log to MLflow. Start a server bound to localhost only, since it has no authentication :

```sh
$ make mlflow
```

Runs and registered models are stored in `mlflow.db` and `mlartifacts/` at the repository root, both ignored by git.

The pipeline uses `MLFLOW_TRACKING_URI` when it is set and `http://127.0.0.1:5000` otherwise. On macOS the AirPlay Receiver also listens on port 5000 and answers 403; disable it or use another port :

```sh
$ make mlflow MLFLOW_PORT=5001
$ export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
```

## Usage

```sh
$ poetry run python src/ -g -s                        # build and save the processed datasets
$ poetry run python src/ -t -m xgboost -c default -s  # train, register and save a model
$ poetry run python src/ -e                           # evaluate the models listed in config/pipeline/base.yaml
$ poetry run python src/ -o -m xgboost                # search XGBoost hyperparameters
```

Registered model names are `<model> <config> - YYYY-MM-DD HHhMM` (UTC). `config/pipeline/base.yaml` lists the registered models the evaluation pipeline compares.

## Model files

Models are stored in data-only formats, so loading one never runs code from the file: XGBoost as JSON, Keras as `.keras` (loaded in safe mode) and Prophet as JSON. The y-scaler of a scaled model is logged to its MLflow run as `scaler.json`. Local copies go to `models_save/<model>/` and are not committed, except the XGBoost model the SHAP notebooks load.

## Checks

```sh
$ make test   # unit and pipeline tests on a synthetic dataset
$ make audit  # known vulnerabilities in the installed dependencies
```
