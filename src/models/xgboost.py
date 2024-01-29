import pathlib
import pickle
from typing import Any, Dict, Tuple

import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor


def _build_xgboost(
    n_estimators: int,
    max_depth: int,
    loss: str,
    learning_rate: float,
    eval_metric: str,
    early_stopping_rounds: int,
) -> XGBRegressor:
    """
    Build an XGBoost model.

    Returns:
        XGBRegressor: The XGBoost model.
    """
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        objective=f"reg:{loss}",
        max_depth=max_depth,
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )

    return model


def build_xgboost(
    model_config: Dict[str, Any], load_model_name: str = None, use_mlflow=True
) -> XGBRegressor:
    """
    Build an XGBoost model.

    Args:
        build_params (Dict[str, Any]): The parameters used for building the model.

    Returns:
        XGBRegressor: The XGBoost model.
    """
    if load_model_name:
        if not use_mlflow:
            return pickle.load(
                open(
                    f"{pathlib.Path(model_config['save_path']).absolute()}/{load_model_name}.h5",
                    "rb",
                )
            )
        return mlflow.pyfunc.load_model(f"models:/{load_model_name}/latest")

    return _build_xgboost(**model_config["build_params"])


def xgboost_train(
    model: XGBRegressor,
    generator: Tuple[pd.DataFrame, pd.DataFrame],
    parameters: Dict[str, Any],
) -> XGBRegressor:
    """
    Train the XGBoost model.

    Args:
        model (XGBRegressor): The XGBoost model.
        generator (Tuple[pd.DataFrame, pd.DataFrame]): The data generator containing the input data and target data.
        parameters (Dict[str, Any]): The parameters to be passed to the XGBoost model.

    Returns:
        XGBRegressor: The trained XGBoost model.
    """
    model.fit(*generator, **parameters)

    return model


def xgboost_cross_validation_train(
    model: XGBRegressor,
    generator: Tuple[pd.DataFrame, pd.DataFrame],
    parameters: Dict[str, Any],
) -> float:
    """
    Cross validate the XGBoost model.

    Args:
        model (XGBRegressor): The XGBoost model to be cross-validated.
        generator (Tuple[pd.DataFrame, pd.DataFrame]): A tuple containing the training and validation data.
        parameters (Dict[str, Any]): Additional parameters for the XGBoost model.

    Returns:
        Tuple[XGBRegressor, float]: A tuple containing the trained XGBoost model and the average mean squared error.
    """
    logs = {
        "split": 1,
        "pred_log": [],
        "target_log": [],
        "mse_log": [],
    }

    df = pd.merge(generator[0], generator[1], left_index=True, right_index=True)
    n_splits = int((1 // 0.1) - 1)  # 10% test size at each split
    time_series_split = TimeSeriesSplit(n_splits=n_splits, gap=24)

    for i_train, i_val in time_series_split.split(df):
        train = df.iloc[i_train]
        val = df.iloc[i_val]
        
        train_in = train.drop(columns=["sales"])
        train_out = train["sales"]
        val_in = val.drop(columns=["sales"])
        val_out = val["sales"]

        model.fit(
            train_in,
            train_out,
            eval_set=[(train_in, train_out), (val_in, val_out)],
            **parameters,
        )

        logs["target_log"].extend(list(val_out))

        pred = model.predict(val_in)
        logs["pred_log"].extend(list(pred))

        mse = mean_squared_error(val_out, pred)
        logs["mse_log"].append((f'split {logs["split"]}', mse))

        logs["split"] += 1

    avg_mse = mean_squared_error(logs["target_log"], logs["pred_log"])

    return avg_mse