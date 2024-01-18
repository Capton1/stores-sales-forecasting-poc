import pathlib
import pickle
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import pandas as pd
from xgboost import XGBRegressor


def _build_xgboost(
    n_estimators: int, max_depth: int, loss: str, learning_rate: float
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
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )

    return model


def build_xgboost(
    model_config: Dict[str, Any], load_model_name: str = None
) -> XGBRegressor:
    """
    Build an XGBoost model.

    Args:
        build_params (Dict[str, Any]): The parameters used for building the model.

    Returns:
        XGBRegressor: The XGBoost model.
    """
    if load_model_name:
        return pickle.load(
            open(
                f"{pathlib.Path(model_config['save_path']).absolute()}/{load_model_name}.h5",
                "rb",
            )
        )

    return _build_xgboost(**model_config["build_params"])


def xgboost(
    train: bool = False,
    load_model: str = None,
    generator: Tuple[pd.DataFrame, pd.DataFrame] = None,
    save: str = False,
    n_estimators: int = 1000,
    max_depth: int = 6,
    loss: str = "squarederror",
    learning_rate: float = 0.01,
    save_path: str = None,
) -> XGBRegressor:
    """
    Train or load an XGBoost model for stores sales forecasting.

    Args:
        train (bool, optional): Whether to train the model. Defaults to False.
        load_model (str, optional): The name of the model to load. Defaults to None.
        generator (Tuple[TimeseriesGenerator], optional): The generator used for training the model.
            Required if train=True. Defaults to None.
        save (str, optional): Whether to save the trained model. Defaults to False.
        epochs (int, optional): The number of training epochs. Defaults to 1.
        max_depth (int, optional): Max tree depth. Defaults to 6.
        n_estimators (str, optional): The number of boosting rounds. Defaults to "000.
        loss (str, optional): The loss function to use. Defaults to "squarederror".
        learning_rate (float, optional): The learning rate for training. Defaults to 0.01.
        save_path (str, optional): The path to save the trained model. Defaults to None.

    Returns:
        XGBRegressor: The trained or loaded XGBoost model.
    """
    if not train and not load_model:
        raise ValueError("You must either train or load a model")

    if train and generator is None:
        raise ValueError("You must provide a generator if you want to train a model")

    if load_model:
        model = pickle.load(
            open(f"{pathlib.Path(save_path).absolute()}/{load_model}.h5", "rb")
        )
    else:
        model = build_xgboost(n_estimators, max_depth, loss, learning_rate)

    if train:
        model.fit(*generator, eval_set=[generator], eval_metric=["rmse"], verbose=True)

    if save:
        model_name = f'xgboost {str(datetime.now(timezone.utc)).split(".")[0]}'

        pickle.dump(
            model, open(f"{pathlib.Path(save_path).absolute()}/{model_name}.h5", "wb")
        )

        print(f"Model saved as {model_name}")

    return model
