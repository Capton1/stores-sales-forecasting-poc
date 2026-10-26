# stores-sales-forecasting-poc

Forecasts the total daily sales of the Corporación Favorita grocery stores in Ecuador, using the data of the Kaggle competition [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting). It compares XGBoost, LSTM and Prophet models, tracks them in MLflow and explains the best one with SHAP.

The analysis is written up in the Medium article [What factors influence grocery sales in Ecuador?](https://medium.com/@antoine.caille/what-factors-influence-grocery-sales-in-ecuador-fab4f4c4636d).

## How it works

The pipeline sums the sales of every store and product family per day. That gives one series of 1,688 days, from 2013-01-01 to 2017-08-15. Each day carries:
- the oil price, interpolated over missing days;
- the number of products on promotion;
- the average store cluster;
- the day type (normal, holiday, event or additional day).

The first 90% of days (to 2017-02-27) train the models. The last 169 days (2017-02-28 to 2017-08-15) are the validation set.

XGBoost adds calendar features, a 7-day moving average of sales and, in the `earthquake` configs, a flag on 2016-04-16 to 2016-04-19, the day of the Ecuador earthquake and the three days after. Its cross-validation uses `TimeSeriesSplit`, so a model never trains on days after the ones it is scored on.

Validation MSE of the best config of each model type:

| Model | Config | Validation MSE |
| --- | --- | --- |
| XGBoost | `best_params_earthquake` | 3.47e9 |
| LSTM | `simple_batch_size_128_look_back_5` | 9.08e9 |
| Prophet | `default` | 2.43e10 |

XGBoost scores are reproducible to the bit. LSTM training starts from random weights, and Prophet's optimizer differs slightly between platforms, so their scores vary by a few percent between runs.

## Getting started

Requires Python 3.12 or 3.13 and [Poetry](https://python-poetry.org/). On macOS, XGBoost also needs OpenMP: `brew install libomp`.

```sh
$ make install
```

### Dataset

The Kaggle CLI needs an API token and an account that has accepted the competition rules.

1. On kaggle.com, go to Settings → API and create a token.
2. Save the token as `~/.kaggle/access_token`. A legacy `kaggle.json` key also works, saved as `~/.kaggle/kaggle.json`.
3. Make the token readable only by you (`chmod 600`), and keep it out of the repository.
4. Accept the rules on the [competition page](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/rules).

```sh
$ mkdir -p data/raw
$ poetry run kaggle competitions download -c store-sales-time-series-forecasting -p data/raw
$ unzip data/raw/store-sales-time-series-forecasting.zip -d data/raw && rm data/raw/store-sales-time-series-forecasting.zip
```

Paths, features, the target and the train/validation ratio live in `config/data/base.yaml`.

### Tracking server

Training and evaluation log to MLflow. The server has no authentication, so `make mlflow` binds it to localhost only:

```sh
$ make mlflow
```

Runs and registered models are stored in `mlflow.db` and `mlartifacts/` at the repository root, and git ignores both.

The pipeline uses `MLFLOW_TRACKING_URI` when it is set and `http://127.0.0.1:5000` otherwise. On macOS the AirPlay Receiver also listens on port 5000 and answers 403. Either disable it or use another port:

```sh
$ make mlflow MLFLOW_PORT=5001
$ export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
```

## Usage

```sh
$ poetry run python src/ -g -s                        # build and save the processed datasets
$ poetry run python src/ -t -m xgboost -c default -s  # train, register and save one config
$ poetry run python src/ -t -m all -s                 # train, register and save every config
$ poetry run python src/ -e                           # evaluate the models listed in config/pipeline/base.yaml
$ poetry run python src/ -o -m xgboost                # search XGBoost hyperparameters
```

| Option | Meaning |
| --- | --- |
| `-m lstm\|xgboost\|prophet\|all` | Model type |
| `-c NAME` | Model config, a top-level key of `config/<model>/<model>.yaml` (default `default`) |
| `-t` / `-g` / `-e` / `-o` | Train, generate datasets, evaluate, or optimize |
| `-s` | Save the datasets, or register the model in MLflow and save a local copy |
| `-l RUN_ID` | With `-t`, start training from the model logged in this MLflow run |
| `-d NAME` / `-p NAME` | Data and pipeline config files under `config/data/` and `config/pipeline/` (default `base`) |

Registered models are named `<model> <config> - YYYY-MM-DD HHhMM`, in UTC. `config/pipeline/base.yaml` lists the registered models that `-e` compares. The evaluation writes each model's MSE to `logs/validation/` and prints the best model.

The hyperparameter search runs 48 Bayesian (TPE) trials and writes the best parameters to `models_save/optim/xgboost_best_hyperparameters.json`.

## Model files

Models are stored in data-only formats, so loading one never runs code from the file:
- XGBoost as JSON;
- Keras as `.keras`, loaded in safe mode;
- Prophet as JSON.

The scaler of a scaled model is logged to its MLflow run as `scaler.json`. The pipeline refuses MLflow models stored as pickle or skops. Local copies go to `models_save/<model>/` and are not committed, except the XGBoost model that the SHAP notebooks load.

XGBoost is pinned to 2.1. XGBoost 3 grows different trees from the same data, and its validation scores no longer match the article. scikit-learn stays on 1.5 because XGBoost 2.1 cannot load models under scikit-learn 1.6 or later.

## Project layout

| Path | Contents |
| --- | --- |
| `src/preprocessing/` | Loading, merging and cleaning the raw files into the daily series |
| `src/models/` | Model builders, training helpers and model persistence |
| `src/pipeline/` | The `Factory` that prepares data and trains, saves and loads models; the training, evaluation and optimization pipelines |
| `config/` | Hydra configs for data, pipelines and each model's configs |
| `notebooks/` | Exploratory analysis, model comparisons and SHAP explanations behind the article |
| `medium/` | The article and its figures |
| `tests/` | Unit and pipeline tests on a synthetic dataset with the competition's schema |

## Checks

```sh
$ make test   # unit and pipeline tests
$ make audit  # known vulnerabilities in the installed dependencies
```
