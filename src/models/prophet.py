import pathlib
import pickle
from typing import Any, Dict

import mlflow
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation
from sklearn.metrics import mean_squared_error


def _build_prophet(parameters: Dict[str, str]) -> Prophet:
    """
    Build a Prophet model.

    Returns:
        Prophet: The Prophet model.
    """
    return Prophet(**parameters)


def build_prophet(
    model_config: Dict[str, Any], load_model_name: str = None, use_mlflow=True
) -> Prophet:
    """
    Builds a Prophet model based on the given configuration.

    Args:
        model_config (Dict[str, Any]): The configuration parameters for building the model.
        load_model_name (str, optional): The name of the pre-trained model to load. Defaults to None.

    Returns:
        Prophet: The built Prophet model.
    """
    if load_model_name:
        if not mlflow:
            return pickle.load(
                open(
                    f"{pathlib.Path(model_config['save_path']).absolute()}/{load_model_name}.h5",
                    "rb",
                )
            )
        return mlflow.prophet.load_model(f"models:/{load_model_name}/latest")

    return _build_prophet(model_config["build_params"])


def prophet_train(model: Prophet, generator: pd.DataFrame) -> Prophet:
    """
    Train the Prophet model.

    Args:
        model (Prophet): The Prophet model.
        generator (pd.DataFrame): The data generator.

    Returns:
        Prophet: The trained Prophet model.
    """
    model.fit(generator)
    return model


def prophet_cross_validation_train(model: Prophet, generator: pd.DataFrame) -> float:
    """
    Cross validate the Prophet model.
    """
    model.fit(generator)

    # means that we cut off the first 3 years of data from the training set, and always make predictions 90 days into the future.
    df_cv = cross_validation(
        model,
        initial="1096 days",
        period="90 days",
        horizon="90 days",
        disable_tqdm=True,
    )

    return mean_squared_error(df_cv.y, df_cv.yhat)
