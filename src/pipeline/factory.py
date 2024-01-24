import math
import mlflow
import numpy as np
import pathlib
import pickle
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
from sklearn.metrics import mean_squared_error

from models.lstm import build_lstm, lstm_cross_validation_train
from models.prophet import build_prophet, prophet_cross_validation_train
from models.training_helpers import (build_lstm_generator,
                                     generate_ml_features, get_prophet_df)
from models.xgboost import build_xgboost, xgboost_cross_validation_train
from preprocessing.helpers import train_val_split
from preprocessing.process_raw_data import load_data, process_data

from pipeline.helpers import get_config_from_model_name, convert_config_to_dict


class Factory:
    def __init__(self, data_config: Dict[str, Any]):
        self.data_config = data_config
        self.model_config = None
        self.model_config_name = None

        self.train_df = None
        self.val_df = None
        self.train_generator = None  # possible type : TimeseriesGenerator, pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]

        self.model = None
        self.input_data_shape = None

    def get_attributes(self):
        return self.__dict__

    def get_df(self):
        return self.train_df, self.val_df

    def set_model_config(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        return self

    def set_model_type(self, model_type: str):
        self.model_config["_name"] = model_type
        return self

    def get_model_config(self):
        return self.model_config

    def get_model(self):
        return self.model

    def get_y_val(self):
        return self.val_df[self.data_config["target"]]

    def _fit_data(self, X: pd.DataFrame):
        if self.train_df is not None:
            return self

        print("Fitting data ...")

        try:
            self.df = process_data(X, data_config=self.data_config)
            self.train_df, self.val_df = train_val_split(
                self.df, val_ratio=self.data_config["validation_ratio"]
            )
        except Exception as e:
            raise ValueError("Unexpected error while processing data: ", e)

        return self

    def _build_generator(self, model_config: Dict[str, Any] = None):
        if self.train_df is None:
            raise ValueError("Data not found. Please fit the data first")

        if self.model_config is None:
            self.model_config = model_config
            self.fit_params = (
                model_config["fit_params"]
                if "fit_params" in model_config
                else None
            )
        if self.model_config["_name"] == "lstm":
            self.train_generator, self.input_data_shape = build_lstm_generator(
                self.train_df, self.model_config, self.data_config["target"]
            )
        elif self.model_config["_name"] == "xgboost":
            self.train_generator = generate_ml_features(
                self.train_df, self.data_config["target"]
            )
        elif self.model_config["_name"] == "prophet":
            self.train_generator = get_prophet_df(self.train_df, self.val_df)
        else:
            raise ValueError("Model not found")

        return self

    def _build_model(self, load_model_name: str = None, use_mlflow = True):
        if self.train_generator is None and load_model_name is None:
            raise ValueError(
                "Generator not found. Please fit the data first or load a model"
            )

        self.model_config["save_path"] = self.data_config["paths"]["models"][self.model_config["_name"]]

        if self.model_config["_name"] == "lstm":
            self.model = build_lstm(
                self.input_data_shape, self.model_config, load_model_name, use_mlflow
            )
        elif self.model_config["_name"] == "xgboost":
            self.model = build_xgboost(self.model_config, load_model_name, use_mlflow)
        elif self.model_config["_name"] == "prophet":
            self.model = build_prophet(self.model_config, load_model_name, use_mlflow)
        else:
            raise ValueError("Model not found")

        return self

    def _fit_model(
        self,
        model_config: Dict[str, Any],
        model_config_name: str = "default",
        load_model_name: str = None,
    ):
        if self.train_df is None:
            raise ValueError("Data not found. Please fit the data first")

        self.model_config_name = model_config_name

        self._build_generator(model_config)
        self._build_model(load_model_name)

        if self.model_config["_name"] == "lstm":
            self.val_mse = lstm_cross_validation_train(
                self.model, self.train_generator, self.fit_params
            )
        elif self.model_config["_name"] == "xgboost":
            self.val_mse = xgboost_cross_validation_train(
                self.model, self.train_generator, self.fit_params
            )
        elif self.model_config["_name"] == "prophet":
            self.val_mse = prophet_cross_validation_train(
                self.model, self.train_generator
            )
        else:
            raise ValueError("Model type not found")

        return self

    def fit(
        self,
        df: pd.DataFrame,
        model_config: Dict[str, Any] = None,
        model_config_name: str = None,
        load_model_name: str = None,
    ):
        if self.train_df is not None and load_model_name:
            self._build_model(load_model_name)

        if (model_config and not model_config_name) or (
            model_config_name and not model_config
        ):
            raise ValueError("Please provide both model_config and model_config_name")

        if not model_config and not load_model_name:
            return self._fit_data(df)

        self._fit_data(df)
        self._fit_model(model_config, model_config_name, load_model_name)

        return self

    def fit_model(
        self,
        model_config: Dict[str, Any],
        model_config_name: str = "default",
        load_model_name: str = None,
    ):
        if self.train_df is None:
            raise ValueError("Data not found. Please fit the data first")

        self._fit_model(
            model_config[model_config_name], model_config_name, load_model_name
        )

        return self

    def fit_transform(self, df):
        self.fit(df)
        return self.train_df

    def predict(self, data: pd.DataFrame = None):
        if not self.model:
            raise ValueError("Model not found. Please fit the data first")

        if self.model_config["_name"] == "lstm":
            if data is not None:
                self.input_data_shape = data.shape
            else:
                # concatenate the last `look_back` days of the training set to the validation set in order to predict the first day of the validation set
                val_lstm = pd.concat([self.train_df[-self.model_config["look_back"]:], self.val_df])
                data, shape = build_lstm_generator(
                    val_lstm, self.model_config, self.data_config["target"]
                )
                self.input_data_shape = shape
            
            res = []
            for x in data:
                if self.model_config["type"] == "multivariate":
                    res.append(self.model.predict((x[0][0], x[0][1]))[0][0])
                else:
                    res.append(self.model.predict(x[0])[0][0])
            return res
        elif self.model_config["_name"] == "xgboost":
            data = (
                generate_ml_features(self.val_df, self.data_config["target"])[0]
                if data is None
                else data
            )
            return self.model.predict(data)
        elif self.model_config["_name"] == "prophet":
            print(type(self.model))
            future = self.model.make_future_dataframe(
                periods=len(self.val_df), freq="D", include_history=False
            )
            return self.model.predict(future)["yhat"]
        else:
            raise ValueError("Model not found")

    def load_model(self, load_model_name: str, model_config: Dict[str, Any] = None, use_mlflow = True):
        if not self.model_config and not model_config:
            raise ValueError("Please provide model_config")

        if ".h5" in load_model_name:
            load_model_name = load_model_name.split(".")[0]

        self.model_config = model_config
        self._build_model(load_model_name, use_mlflow)

        return self

    def save_model(self, use_mlflow: bool = True):
        if not self.model:
            raise ValueError("Model not found. Please fit the data first")

        model_name = f'{self.model_config["_name"]} {self.model_config_name} - {str(datetime.now(timezone.utc)).split(".")[0].rsplit(":", maxsplit=1)[0]}'

        if self.model_config["_name"] == "lstm":
            mlflow.tensorflow.log_model(self.model, artifact_path="model", registered_model_name=model_name)
        elif self.model_config["_name"] == "xgboost":
            mlflow.sklearn.log_model(self.model, artifact_path="model", registered_model_name=model_name)
        else:
            mlflow.prophet.log_model(self.model, artifact_path="model", registered_model_name=model_name)
        
        mlflow.set_tags({"model_type": self.model_config["_name"]})
        mlflow.log_param("model_config", self.model_config)
        mlflow.log_metric("MSE", self.val_mse)
            
        if not use_mlflow:            
            pickle.dump(
                self.model,
                open(
                    f"{pathlib.Path(self.data_config['paths']['models'][self.model_config['_name']]).absolute()}/{model_name}.h5",
                    "wb",
                ),
            )

            print(f"Model saved as {model_name}")

        return self

    def save_data(self):
        if self.train_df is None or self.val_df is None:
            raise ValueError("Data not found. Please fit the data first")

        file_name = self.data_config["paths"]["processed"]["train"].split("/")[-1]
        self.train_df.to_csv(
            f"{pathlib.Path(self.data_config['paths']['processed']['train']).absolute()}"
        )
        print(f"Processed training set saved as {file_name}")

        file_name = self.data_config["paths"]["processed"]["validation"].split("/")[-1]
        self.val_df.to_csv(
            f"{pathlib.Path(self.data_config['paths']['processed']['validation']).absolute()}"
        )
        print(f"Processed validation set saved as {file_name}")

        return self


def generate_datasets(data_config: Dict[str, Any], save: bool = False) -> Factory:
    """Generates train and validation sets.

    Args:
        data_path (Dict[str, Any]): paths to data
        save (bool, optional): whether to save the data. Defaults to False.
    """
    print("Generating datasets ...")

    df = load_data(data_config["paths"])
    f = Factory(data_config)
    f.fit(df)

    if save:
        f.save_data()

    return f


def generate_models(
    data_config: Dict[str, Any],
    model_config: Dict[str, Any],
    model_config_name: str,
    save: bool = False,
    load_model_name: str = None,
) -> Factory:
    """Generates models.

    Args:
        data_path (Dict[str, Any]): paths to data
        save (bool, optional): whether to save the data. Defaults to False.
    """
    df = load_data(data_config["paths"])
    f = Factory(data_config)
    f.fit(
        df,
        model_config=model_config,
        model_config_name=model_config_name,
        load_model_name=load_model_name,
    )

    if save:
        f.save_model()

    return f


def evaluate_model(f: Factory, model_type: str, model_name: str, model_config: Dict[str, Any]) -> float:
    """
    Evaluates the given model.

    Args:
        f (Factory): The factory object containing the model.
        model_type (str): The type of the model.
        model_name (str): The name of the model.

    Returns:
        float: The mean squared error of the model.
    """
    print(f"Evaluating model {model_name} ...")
    
    model_config["save_path"] = f.data_config["paths"]["models"][model_type]

    f.load_model(model_name, model_config)
    y_val =  f.get_y_val()
    pred = f.predict()
    
    #if model_type == "lstm" and model_config["type"] == "multivariate":
    #    pred = pred[:, -1]

    mse = round(mean_squared_error(y_val, pred), 2)

    return mse


def validation_pipeline(
    models_to_validate: Dict[str, Any], config: Dict[str, Any]
):
    """
    Runs the validation pipeline for the given models and configuration.

    Args:
        models_to_validate (dict): A dictionary containing the models to validate, grouped by model type.
        config (dict): The configuration settings for the pipeline.

    Returns:
        None
    """
    data_config = config["data"]
    
    filename = f'Validation - {str(datetime.now(timezone.utc)).split(".")[0].rsplit(":", maxsplit=1)[0]}'
    print(f"Writing validation results in `{filename}` ... \n")

    df = load_data(data_config["paths"])
    f = Factory(data_config)
    f.fit(df)

    min_mse, best_type, best_model = math.inf, None, None
    with open(
        f'{pathlib.Path(data_config["paths"]["logs"]["validation"]).absolute()}/{filename}',
        "w+",
    ) as file:
        for model_type, models in models_to_validate.items():
            if models is None:
                continue
            for model_name in models:
                model_config = get_config_from_model_name(
                    model_name, model_type, config
                )
                mse = evaluate_model(f, model_type, model_name, model_config)
                file.write(f"Type: {model_type} - Name: {model_name} - MSE: {mse} \n")

                if mse < min_mse:
                    min_mse, best_type, best_model = mse, model_type, model_name

    print("\nValidation pipeline is done")
    print(f"Best model is {best_type} {best_model} with mse: {min_mse}")


def launch_pipeline(
    config,
    model,
    model_config_name,
    train: bool = False,
    generate: bool = False,
    validate: bool = False,
    save: bool = False,
    load_model_name: str = None,
    limit: int = 1520,
):
    """Launches the pipeline."""
    print("Launching pipeline ...")

    config = convert_config_to_dict(config)

    if generate:
        generate_datasets(config["data"], save=save)

    if train:
        generate_models(
            config["data"],
            config[model][model_config_name],
            model_config_name=model_config_name,
            save=save,
            load_model_name=load_model_name,
        )

    if validate:
        validation_pipeline(config["pipeline"]["validation"]["models"], config)
