import pathlib
import pickle
from datetime import datetime, timezone

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import performance_metrics


def prophet(
    train: bool = False,
    load_model: str = None,
    generator: pd.DataFrame = None,
    save: str = False,
    save_path: str = None,
) -> Prophet:
    """
    Train or load a Prophet model for stores sales forecasting.

    Args:
        train (bool, optional): Whether to train the model. Defaults to False.
        load_model (str, optional): The name of the model to load. Defaults to None.
        generator (Tuple[TimeseriesGenerator], optional): The generator used for training the model.
            Required if train=True. Defaults to None.
        save (str, optional): Whether to save the trained model. Defaults to False.
        save_path (str, optional): The path to save the trained model. Defaults to None.

    Returns:
        Prophet: The trained or loaded Prophet model.
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
        model = Prophet()

    if train:
        model.fit(generator)

    if save:
        model_name = f'prophet {str(datetime.now(timezone.utc)).split(".")[0]}'

        pickle.dump(model, open(f"{save_path}/{model_name}.h5", "wb"))

        print(f"Model saved as {model_name}")

    return model
