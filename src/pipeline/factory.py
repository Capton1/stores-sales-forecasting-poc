import math
import pathlib
import pickle
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
from keras.models import Model
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from models.lstm import build_lstm
from models.prophet import build_prophet
from models.training_helpers import (
    build_lstm_generator,
    get_features_and_target,
    get_prophet_df,
)
from models.xgboost import build_xgboost
from preprocessing.helpers import train_val_split
from preprocessing.process_raw_data import (
    get_preprocessing_pipeline,
    load_data,
    process_data,
)


class Factory:
    def __init__(self, data_config: Dict[str, Any]):
        self.data_config = data_config
        self.model_config = None

        self.X_train = None
        self.X_val = None
        self.generator = None

        self.model = None

    def get_attributes(self):
        return self.__dict__

    def set_model_config(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        return self

    def get_model_config(self):
        return self.model_config

    def _fit_data(self, X: pd.DataFrame):
        try:
            self.df = process_data(X, data_config=self.data_config)
            self.X_train, self.X_val = train_val_split(
                self.df, val_ratio=self.data_config["validation_ratio"]
            )
        except:
            raise ValueError("Unexpected error while processing data.")

        return self

    def _build_generator(self, model_config: Dict[str, Any]):
        if self.X_train is None:
            raise ValueError("Data not found. Please fit the data first")

        self.model_config = model_config
        self.fit_params = (
            dict(model_config["fit_params"]) if "fit_params" in model_config else None
        )

        if self.model_config["_name"] == "lstm":
            self.generator, self.input_data_shape = build_lstm_generator(
                self.X_train, self.model_config, self.data_config["target"]
            )
        elif self.model_config["_name"] == "xgboost":
            self.generator = get_features_and_target(
                self.X_train, self.data_config["target"]
            )
        elif self.model_config["_name"] == "prophet":
            self.generator = get_prophet_df(self.X_train)
        else:
            raise ValueError("Model not found")

        return self

    def _build_model(self, load_model_name: str = None):
        if self.generator is None:
            raise ValueError("Generator not found. Please fit the data first")

        if self.model_config["_name"] == "lstm":
            self.model = build_lstm(
                self.input_data_shape, self.model_config, load_model_name
            )
        elif self.model_config["_name"] == "xgboost":
            self.model = build_xgboost(self.model_config, load_model_name)
        elif self.model_config["_name"] == "prophet":
            self.model = build_prophet(self.model_config, load_model_name)
        else:
            raise ValueError("Model not found")

        return self

    def _fit_model(self, model_config: Dict[str, Any], load_model_name: str = None):
        if self.X_train is None:
            raise ValueError("Data not found. Please fit the data first")

        self._build_generator(model_config)
        self._build_model(load_model_name)

        if self.model_config["_name"] == "xgboost":
            print(self.fit_params)
            self.fit_params["eval_set"] = [self.generator]

            self.model.fit(
                *self.generator,
                **self.fit_params,
            )
        else:
            if self.fit_params is not None:
                self.model.fit(
                    self.generator,
                    **self.fit_params,
                )
            else:
                self.model.fit(self.generator)

        return self

    def fit(
        self,
        df: pd.DataFrame,
        model_config: Dict[str, Any] = None,
        load_model_name: str = None,
    ):
        if not model_config and not load_model_name:
            return self._fit_data(df)

        self._fit_data(df)
        self._fit_model(model_config, load_model_name)

        return self

    def fit_transform(self, df):
        self.fit(df)
        return self.X_train

    def save_model(self, model_name: str = None):
        if not self.model:
            raise ValueError("Model not found. Please fit the data first")

        if not model_name:
            model_name = f'{self.model_config["_name"]} {str(datetime.now(timezone.utc)).split(".")[0]}'

        pickle.dump(
            self.model,
            open(
                f"{pathlib.Path(self.model_config['save_path']).absolute()}/{model_name}.h5",
                "wb",
            ),
        )

        print(f"Model saved as {model_name}")

        return self

    def save_data(self):
        if self.X_train is None or self.X_val is None:
            raise ValueError("Data not found. Please fit the data first")

        file_name = self.data_config["paths"]["processed"]["train"].split("/")[-1]
        self.X_train.to_csv(
            f"{pathlib.Path(self.data_config['paths']['processed']['train']).absolute()}"
        )
        print(f"Processed training set saved as {file_name}")

        file_name = self.data_config["paths"]["processed"]["validation"].split("/")[-1]
        self.X_val.to_csv(
            f"{pathlib.Path(self.data_config['paths']['processed']['validation']).absolute()}"
        )
        print(f"Processed validation set saved as {file_name}")

        return self


def generate_datasets(data_config: Dict[str, Any], save: bool = False):
    """Generates train and validation sets.

    Args:
        data_path (Dict[str, Any]): paths to data
        save (bool, optional): whether to save the data. Defaults to False.
    """
    df = load_data(data_config["paths"])
    f = Factory(data_config)
    f.fit(df)

    if save:
        f.save_data()


def generate_models(
    data_config: Dict[str, Any],
    model_config: Dict[str, Any],
    save: bool = False,
    load_model_name: str = None,
):
    """Generates models.

    Args:
        data_path (Dict[str, Any]): paths to data
        save (bool, optional): whether to save the data. Defaults to False.
    """
    df = load_data(data_config["paths"])
    f = Factory(data_config)
    f.fit(df, model_config=model_config, load_model_name=load_model_name)

    if save:
        f.save_model()


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
    model_config,
    train: bool = False,
    generate: bool = False,
    validate: bool = False,
    save: bool = False,
    load_model_name: str = None,
    limit: int = 1520,
):
    """Launches the pipeline."""
    print("Launching pipeline ...")

    if generate:
        generate_datasets(config["data"], save=save)

    if train:
        generate_models(
            config["data"],
            config[model][model_config],
            save=save,
            load_model_name=load_model_name,
        )

    if validate:
        validation_pipeline(
            config["pipeline"]["validation"]["models"], config, limit=limit
        )
