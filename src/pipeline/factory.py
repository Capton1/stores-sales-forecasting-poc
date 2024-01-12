import math
import pathlib
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
from keras.models import Model
from sklearn.metrics import mean_squared_error

from models.lstm import lstm
from models.prophet import prophet
from models.training_helpers import (
    get_features_and_target,
    get_multiple_input_timeseries_generator,
    get_prophet_df,
    get_single_input_timeseries_generator,
)
from models.xgboost import xgboost
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
        train, val_ratio=0.9, save=save, save_path=data_path["trusted"]
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
    """
    Loads a model and generator based on the specified parameters.

    Args:
        train (bool): Whether to train the model or not.
        model_type (str): The type of model to load.
        X (pandas.DataFrame): The input features.
        y (pandas.Series): The target variable.
        save (bool, optional): Whether to save the model or not. Defaults to False.
        model_name (str, optional): The name of the model to load. Defaults to None.
        config_model (dict, optional): The configuration parameters for the model. Defaults to None.
        save_models_path (str, optional): The path to save the models. Defaults to None.

    Returns:
        tuple: A tuple containing the loaded model and generator.
    """
    if model_type == "lstm":
        # get right generator
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
        elif config_model["type"] == "simple":
            generator = get_single_input_timeseries_generator(
                X, y, look_back=config_model["look_back"], batch_size=1
            )
        else:
            raise ValueError("Model type not found")

        print("LSTM model loaded")

        model = lstm(
            train=train,
            load_model=model_name,
            generator=generator,
            save=save,
            epochs=config_model["epochs"],
            batch_size=config_model["batch_size"],
            optimizer=config_model["optimizer"],
            loss=config_model["loss"],
            look_back=config_model["look_back"],
            type=config_model["type"],
            save_path=save_models_path,
        )

    elif model_type == "xgboost":
        generator = (X, y)

        print("XGBoost model loaded")

        model = xgboost(
            train=train,
            load_model=model_name,
            generator=generator,
            save=save,
            n_estimators=config_model["n_estimators"],
            max_depth=config_model["max_depth"],
            loss=config_model["loss"],
            learning_rate=config_model["learning_rate"],
            save_path=save_models_path,
        )

    elif model_type == "prophet":
        generator = get_prophet_df(y)

        print("Prophet model loaded")

        model = prophet(
            train=train,
            load_model=model_name,
            generator=generator,
            save=save,
            save_path=save_models_path,
        )
    else:
        raise ValueError("Model not found")

    return model, generator


def train_model(
    config,
    model_type: str,
    save: bool = True,
    model_name: str = None,
    limit: int = 1520,
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


def eval_model(
    config, model_type: str, model_name: str, X_val: pd.DataFrame, y_val: pd.DataFrame
) -> float:
    """Trains the model.

    Args:
        config (Dict): configuration dictionary
        model_name (str): model to use
        save (bool, optional): whether to save the model. Defaults to True.
    """
    print(f"Evaluating model {model_type} {model_name} ...")

    config_model = config["optimizer"]["models"][model_type]
    save_models_path = config["data"]["paths"]["models"][model_type]

    # handle different model type for lstm + predict only 1 value
    if model_type == "lstm":
        config_model["type"] = model_name.split()[0]
        config_model["look_back"] = 1

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

    if model_type == "lstm":
        pred = model.predict(val_gen).reshape(-1)
        mse = round(mean_squared_error(y_val[1:], pred), 2)
    elif model_type == "xgboost":
        pred = model.predict(X_val)
        mse = round(mean_squared_error(y_val, pred), 2)
    elif model_type == "prophet":
        future = model.make_future_dataframe(
            periods=len(y_val), freq="D", include_history=False
        )
        forecast = model.predict(future)
        mse = round(mean_squared_error(y_val["sales"], forecast["yhat"]), 2)
    else:
        raise ValueError("Model not found")

    return mse


def validation_pipeline(models_to_validate, config, limit: int):
    """
    Runs the validation pipeline for the given models and configuration.

    Args:
        models_to_validate (dict): A dictionary containing the models to validate, grouped by model type.
        config (dict): The configuration settings for the pipeline.

    Returns:
        None
    """
    filename = f'{str(datetime.now(timezone.utc)).split(".")[0]}'
    print(f"Writing validation results in {filename} ... \n")

    trusted_data_files = config["data"]["paths"]["trusted"]
    X_val, y_val = get_features_and_target(
        pd.read_csv(trusted_data_files["val"], index_col=0)[:limit]
    )

    min_mse, best_type, best_model = math.inf, None, None
    with open(
        f'{pathlib.Path(config["data"]["paths"]["logs"]["validation"]).absolute()}/{filename}',
        "w+",
    ) as file:
        for model_type, models in models_to_validate.items():
            for model_name in models:
                mse = eval_model(config, model_type, model_name, X_val, y_val)
                file.write(f"Type: {model_type} - Name: {model_name} - MSE: {mse} \n")

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
    limit: int = 1520,
):
    """Launches the pipeline."""
    if generate:
        generate_train_val_sets(config["data"]["paths"], save=save)

    if train:
        train_model(config, model, save=save, model_name=load_model, limit=limit)

    if validate:
        validation_pipeline(
            config["pipeline"]["validation"]["models"], config, limit=limit
        )
