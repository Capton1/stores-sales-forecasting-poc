import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error

from models.prophet import prophet_train
from pipeline.factory import Factory
from pipeline.launcher import evaluate_model, generate_datasets
from hyperopt import hp

from pipeline.optim import get_space_of_hyperparameters
from preprocessing.process_raw_data import load_data

XGBOOST_CONFIG = {
    "fit_params": {"verbose": False},
    "build_params": {
        "objective": "reg:squarederror",
        "learning_rate": 0.1,
        "max_depth": 3,
        "n_estimators": 50,
        "eval_metric": "rmse",
        "early_stopping_rounds": 10,
    },
    "_name": "xgboost",
}

LSTM_CONFIGS = {
    "simple": {
        "fit_params": {"epochs": 1},
        "build_params": {"optimizer": "adam", "loss": "mse", "l1": 0.01, "l2": 0.01},
        "look_back": 5,
        "type": "simple",
        "_name": "lstm",
    },
    "multivariate": {
        "fit_params": {"epochs": 1},
        "build_params": {"optimizer": "adam", "loss": "mse"},
        "look_back": 5,
        "type": "multivariate",
        "_name": "lstm",
    },
}


def _fit(data_config, model_config, name="test"):
    f = Factory(data_config)
    f.set_save_name(f"{model_config['_name']} {name}")
    f.fit(load_data(data_config["paths"]), model_config=dict(model_config), model_config_name=name)
    return f


def _reload(data_config, model_config, name):
    f = Factory(data_config)
    f.fit(load_data(data_config["paths"]))
    f.load_model(f"{name}.h5", dict(model_config), use_mlflow=False)
    return f


def test_xgboost_local_round_trip(data_config, mlflow_tracking):
    with mlflow.start_run():
        f = _fit(data_config, XGBOOST_CONFIG).save_model()
    assert f.val_mse > 0

    reloaded = _reload(data_config, XGBOOST_CONFIG, "xgboost test")

    np.testing.assert_allclose(reloaded.predict(), f.predict(), rtol=1e-6)
    assert mean_squared_error(reloaded.get_y_val(), reloaded.predict()) > 0


@pytest.mark.parametrize("lstm_type", ["simple", "multivariate"])
def test_lstm_local_round_trip(data_config, mlflow_tracking, lstm_type):
    config = LSTM_CONFIGS[lstm_type]
    with mlflow.start_run():
        f = _fit(data_config, config, lstm_type).save_model()

    reloaded = _reload(data_config, config, f"lstm {lstm_type}")

    np.testing.assert_allclose(reloaded.predict(), f.predict(), rtol=1e-5)


def test_prophet_local_round_trip(data_config, mlflow_tracking):
    config = {
        "build_params": {"seasonality_mode": "additive", "changepoint_prior_scale": 0.05},
        "_name": "prophet",
    }
    f = Factory(data_config)
    f.fit(load_data(data_config["paths"]))
    f.model_config = dict(config)
    f._build_generator(f.train_df)
    f._build_model()
    prophet_train(f.model, f.train_generator)
    f.model_config_name = "test"
    f.val_mse = 0.0
    with mlflow.start_run() as run:
        f.set_save_name("prophet test").save_model()

    reloaded = _reload(data_config, config, "prophet test")
    from_mlflow = Factory(data_config).load_model(run.info.run_id, config)
    from_mlflow.val_df = f.val_df

    np.testing.assert_allclose(reloaded.predict(), f.predict())
    np.testing.assert_allclose(from_mlflow.predict(), f.predict())


@pytest.mark.parametrize("base_config", [XGBOOST_CONFIG, LSTM_CONFIGS["simple"]], ids=["xgboost", "lstm"])
def test_scaled_model_evaluates_from_the_registry(data_config, mlflow_tracking, base_config):
    config = {**base_config, "scaled": True}
    model_type = config["_name"]
    with mlflow.start_run():
        f = _fit(data_config, config, "default_scaled").save_model(save_localy=False)
    evaluator = Factory(data_config)
    evaluator.fit(load_data(data_config["paths"]))
    mse = evaluate_model(evaluator, model_type, f"{model_type} default_scaled", dict(config))

    # the registered model and logged scaler predict exactly as the trained ones
    expected = mean_squared_error(evaluator.get_y_val(), f.predict())
    assert mse == pytest.approx(round(expected, 2), abs=0.01)


def test_generate_datasets_saves_csvs(data_config):
    f = generate_datasets(data_config, save=True)

    for split, df in (("train", f.train_df), ("validation", f.val_df)):
        saved = pd.read_csv(data_config["paths"]["processed"][split], index_col=0)
        assert len(saved) == len(df)


def test_xgboost_hyperparameter_search(data_config):
    space = {
        **get_space_of_hyperparameters("xgboost"),
        "n_estimators": hp.choice("n_estimators", [10]),
    }

    best = Factory(data_config).bayesian_optim(
        load_data(data_config["paths"]), "xgboost", 2, parameters=space
    )

    assert set(best) == set(space)
