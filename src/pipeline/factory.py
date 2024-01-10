import math
import pathlib
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
from keras.models import Model
from sklearn.metrics import mean_squared_error

from models.lstm import lstm
from models.training_helpers import (get_features_and_target,
                                     get_multiple_input_timeseries_generator,
                                     get_single_input_timeseries_generator)
from preprocessing.get_data import collect_data
from preprocessing.prepare_data import prepare_training_data


def generate_train_val_sets(data_path: Dict[str, str], save: bool = True):
    """Generates train and validation sets.

    Args:
        df (pd.DataFrame): original train set
        val_ratio (float, optional): ratio of the validation set. Defaults to 0.85.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train set and validation set
    """
    train, _ = collect_data(data_path, save=save)
    prepare_training_data(
        train, val_ratio=0.85, save=save, save_path=data_path["trusted"]
    )


def load_model_and_generator(
    train,
    model_type,
    X,
    y,
    save=False,
    model_name=None,
    config_model=None,
    save_models_path=None,
) -> Model:
    generator = None

    if model_type == "lstm":
        if config_model["type"] == "multivariate":
            X_continuous = X[["onpromotion", "dcoilwtico", "cluster"]]
            X_categorical = X.drop(["onpromotion", "dcoilwtico", "cluster"], axis=1)

            generator = get_multiple_input_timeseries_generator(
                X_continuous,
                X_categorical,
                y,
                look_back=config_model["look_back"],
                batch_size=1,
            )
        else:
            generator = get_single_input_timeseries_generator(
                X, y, look_back=config_model["look_back"], batch_size=1
            )

        model = lstm(
            train=train,
            load_model=model_name,
            generator=generator,
            save=save,
            epochs=config_model["epochs"],
            batch_size=config_model["batch_size"],
            optimizer=config_model["optimizer"],
            loss=config_model["loss"],
            learning_rate=config_model["learning_rate"],
            look_back=config_model["look_back"],
            type=config_model["type"],
            save_path=save_models_path,
        )

    elif model_type == "xgboost":
        model = None
    elif model_type == "prophet":
        model = None
    else:
        raise ValueError("Model not found")

    return model, generator


def train_model(
    config,
    model_type: str,
    save: bool = True,
    model_name: str = None,
    limit: int = 1000,
):
    """Trains the model.

    Args:
        config (Dict): configuration dictionary
        model_name (str): model to use
        save (bool, optional): whether to save the model. Defaults to True.
    """
    trusted_data_files = config["data"]["paths"]["trusted"]
    config_model = config["optimizer"]["models"][model_type]
    save_models_path = config["data"]["paths"]["models"][model_type]

    X_train, y_train = get_features_and_target(
        pd.read_csv(trusted_data_files["train"], index_col=0)[-limit:]
    )

    load_model_and_generator(
        True,
        model_type,
        X_train,
        y_train,
        save,
        model_name,
        config_model,
        save_models_path,
    )


def eval_model(config, model_type: str, model_name: str) -> float:
    """Trains the model.

    Args:
        config (Dict): configuration dictionary
        model_name (str): model to use
        save (bool, optional): whether to save the model. Defaults to True.
    """
    trusted_data_files = config["data"]["paths"]["trusted"]
    config_model = config["optimizer"]["models"][model_type]
    save_models_path = config["data"]["paths"]["models"][model_type]

    X_val, y_val = get_features_and_target(
        pd.read_csv(trusted_data_files["val"], index_col=0)
    )

    model, val_gen = load_model_and_generator(
        False,
        model_type,
        X_val,
        y_val,
        save=False,
        model_name=model_name,
        config_model=config_model,
        save_models_path=save_models_path,
    )

    pred = model.predict(val_gen)
    mse = round(mean_squared_error(y_val, pred), 2)

    return mse


def validation_pipeline(models_to_validate, config):
    """
    Runs the validation pipeline for the given models and configuration.

    Args:
        models_to_validate (dict): A dictionary containing the models to validate, grouped by model type.
        config (dict): The configuration settings for the pipeline.

    Returns:
        None
    """
    filename = f'{str(datetime.now(timezone.utc)).split(".")[0]}'
    print(f"Writing validation results in {filename} ...")

    min_mse, best_type, best_model = math.inf, None, None
    with open(
        f'{pathlib.Path(config["data"]["paths"]["logs"]).absolute()}/{filename}', "w"
    ) as file:
        for model_type, models in models_to_validate.items():
            for model_name in models:
                mse = eval_model(config, model_type, model_name)
                model_name = model_name.split()[-2:]
                file.write(f"Type: {model_type} Name: {model_name} mse: {mse}")

                if mse < min_mse:
                    min_mse, best_type, best_model = mse, model_type, model_name

    print("\nValidation pipeline is done")
    print(f"Best model is {best_type} {best_model} with mse: {min_mse}")


def launch_pipeline(
    config,
    model,
    train: bool = False,
    generate: bool = False,
    validate: bool = False,
    save: bool = False,
    load_model: str = None,
    limit: int = 1000,
):
    """Launches the pipeline."""
    if generate:
        generate_train_val_sets(config["data"]["paths"], save=save)

    if train:
        train_model(config, model, save=save, model_name=load_model, limit=limit)

    if validate:
        print(config["pipeline"]["validation"]["models"])
        validation_pipeline(config["pipeline"]["validation"]["models"], config)
