import pathlib
import pickle
from datetime import datetime, timezone
from typing import Tuple

from keras.preprocessing.sequence import TimeseriesGenerator
from xgboost import XGBRegressor


def build_xgboost(n_estimators: int, loss: str, learning_rate: float) -> XGBRegressor:
    """
    Build an XGBoost model.

    Returns:
        XGBRegressor: The XGBoost model.
    """
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        objective=f"reg:{loss}",
        random_state=42,
        n_jobs=-1,
    )

    return model


def xgboost(
    train: bool = False,
    load_model: str = None,
    generator: Tuple[TimeseriesGenerator] = None,
    save: str = False,
    epochs: int = 1,
    batch_size: int = 64,
    n_estimators: str = "adam",
    loss: str = "squarederror",
    learning_rate: float = 0.01,
    save_path: str = None,
) -> XGBRegressor:
    if not train and not load_model:
        raise ValueError("You must either train or load a model")

    if train and generator is None:
        raise ValueError("You must provide a generator if you want to train a model")

    if load_model:
        model = pickle.load(
            open(f"{pathlib.Path(save_path).absolute()}/{load_model}.h5", "rb")
        )
    else:
        model = build_xgboost(n_estimators, loss, learning_rate)

    if train:
        model.fit(
            generator,
            epochs=epochs,
            batch_size=batch_size,
        )

    if save:
        model_name = f'{type} {str(datetime.now(timezone.utc)).split(".")[0]}'

        pickle.dump(
            model, open(f"{pathlib.Path(save_path).absolute()}/{model_name}.h5", "wb")
        )

        print(f"Model saved as {model_name}")

    return model
