import math
import pathlib
import pickle
from datetime import datetime, timezone
from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from models.lstm import build_lstm, lstm_cross_validation_train
from models.prophet import build_prophet, prophet_cross_validation_train
from models.training_helpers import (build_lstm_generator,
                                     generate_ml_features, get_prophet_df)
from models.xgboost import build_xgboost, xgboost_cross_validation_train
from pipeline.helpers import convert_config_to_dict, get_config_from_model_name
from preprocessing.helpers import train_val_split
from preprocessing.process_raw_data import load_data, process_data


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

        self.y_scaler = None
        self.save_name = None

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

    def set_save_name(self, save_name: str):
        self.save_name = save_name
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
            self.model_config = dict(model_config)
            self.fit_params = (
                dict(model_config["fit_params"])
                if "fit_params" in model_config
                else None
            )

        scaler = (
            MinMaxScaler(feature_range=(0, 1))
            if ("scaled" in self.model_config and self.model_config["scaled"])
            else None
        )


        if self.model_config["_name"] == "lstm":
            (
                self.train_generator,
                self.input_data_shape,
                self.y_scaler,
            ) = build_lstm_generator(
                self.train_df, self.model_config, self.data_config["target"], scaler
            )
        elif self.model_config["_name"] == "xgboost":
            self.train_generator, self.y_scaler = generate_ml_features(
                self.train_df, self.data_config["target"], scaler
            )
        elif self.model_config["_name"] == "prophet":
            self.train_generator = get_prophet_df(self.train_df, self.val_df)
        else:
            raise ValueError("Model not found")

        return self

    def _build_model(self, load_model_name: str = None, use_mlflow=True):
        if self.train_generator is None and load_model_name is None:
            raise ValueError(
                "Generator not found. Please fit the data first or load a model"
            )

        self.model_config["save_path"] = self.data_config["paths"]["models"][
            self.model_config["_name"]
        ]

        if self.model_config["_name"] == "lstm":
            self.model = build_lstm(
                self.input_data_shape,
                self.model_config,
                load_model_name,
                use_mlflow,
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

        scaler = (
            MinMaxScaler(feature_range=(0, 1))
            if "scaled" in self.model_config and self.model_config["scaled"]
            else None
        )

        if self.model_config["_name"] == "lstm":
            if data is not None:
                self.input_data_shape, self.y_scaler = data.shape, scaler
            else:
                # concatenate the last `look_back` days of the training set to the validation set in order to predict the first day of the validation set
                val_lstm = pd.concat(
                    [self.train_df[-self.model_config["look_back"] :], self.val_df]
                )
                data, shape, self.y_scaler = build_lstm_generator(
                    val_lstm, self.model_config, self.data_config["target"], scaler
                )
                self.input_data_shape = shape

            res = []
            for x in data:
                elt = (
                    [x[0][0], x[0][1]]
                    if self.model_config["type"] == "multivariate"
                    else x[0]
                )
                res.append(self.model.predict(elt)[0][0])

            return self.y_scaler.inverse_transform(np.reshape(res, (-1, 1))) if scaler else res
        elif self.model_config["_name"] == "xgboost":
            if data is not None:
                self.y_scaler = scaler
            else:
                Xy, self.y_scaler = generate_ml_features(self.val_df, self.data_config["target"], scaler)
                data = Xy[0]
                
            return (
                self.y_scaler.inverse_transform(self.model.predict(data).reshape(-1, 1))
                if scaler
                else self.model.predict(data)
            )
        elif self.model_config["_name"] == "prophet":
            future = self.model.make_future_dataframe(
                periods=len(self.val_df), freq="D", include_history=False
            )
            return self.model.predict(future)["yhat"]
        else:
            raise ValueError("Model not found")

    def load_model(
        self, load_model_name: str, model_config: Dict[str, Any] = None, use_mlflow=True
    ):
        if not self.model_config and not model_config:
            raise ValueError("Please provide model_config")

        if ".h5" in load_model_name:
            load_model_name = load_model_name.split(".")[0]

        self.model_config = dict(model_config)
        self._build_model(load_model_name, use_mlflow)

        return self

    def save_model(self, use_mlflow: bool = True):
        if not self.model:
            raise ValueError("Model not found. Please fit the data first")

        if not self.save_name:
            model_name = f'{self.model_config["_name"]} {self.model_config_name} - {str(datetime.now(timezone.utc)).split(".")[0].rsplit(":", maxsplit=1)[0]}'
        else:
            model_name = self.save_name

        if self.model_config["_name"] == "lstm":
            mlflow.tensorflow.log_model(
                self.model,
                artifact_path="model",
                registered_model_name=model_name,
            )
        elif self.model_config["_name"] == "xgboost":
            mlflow.sklearn.log_model(
                self.model,
                artifact_path="model",
                registered_model_name=model_name,
            )
        else:
            mlflow.prophet.log_model(
                self.model,
                artifact_path="model",
                registered_model_name=model_name,
            )

        mlflow.set_tags({"model_type": self.model_config["_name"]})
        mlflow.log_param("model_type", self.model_config["_name"])
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
    save_name: str = None,
) -> Factory:
    """Generates models.

    Args:
        data_path (Dict[str, Any]): paths to data
        save (bool, optional): whether to save the data. Defaults to False.
    """
    df = load_data(data_config["paths"])
    f = Factory(data_config)

    if save_name:
        f.set_save_name(save_name)

    f.fit(
        df,
        model_config=model_config,
        model_config_name=model_config_name,
        load_model_name=load_model_name,
    )

    if save:
        f.save_model()

    return f


def evaluate_model(
    f: Factory, model_type: str, model_name: str, model_config: Dict[str, Any]
) -> float:
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
    if "scaled" not in model_config:
        model_config["scaled"] = False

    f.load_model(model_name, model_config)
    y_val = f.get_y_val()
    pred = f.predict()

    mse = round(mean_squared_error(y_val, pred), 2)

    return mse


def validation_pipeline(models_to_validate: Dict[str, Any], config: Dict[str, Any]):
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
    experiment_id: int = 0,
):
    """Launches the pipeline."""
    print("Launching pipeline ...")

    config = convert_config_to_dict(config)

    if generate:
        generate_datasets(config["data"], save=save)

    if train:
        if model == "all":
            mlflow.start_run(experiment_id=experiment_id)
            print("Training all models ...")
            for i in ["lstm", "xgboost", "prophet"]:
                for j in config[i]:
                    print(f"Training {i} {j} ...")
                    save_name = f'{i} {j} - {str(datetime.now(timezone.utc)).split(".")[0].rsplit(":", maxsplit=1)[0]}'
                    with mlflow.start_run(nested=True, run_name=save_name):
                        generate_models(
                            config["data"],
                            config[i][j],
                            model_config_name=j,
                            save=save,
                            save_name=save_name,
                        )
        else:
            save_name = f'{model} {model_config_name} - {str(datetime.now(timezone.utc)).split(".")[0].rsplit(":", maxsplit=1)[0]}'
            if save:
                mlflow.start_run(experiment_id=experiment_id, run_name=save_name)

            generate_models(
                config["data"],
                config[model][model_config_name],
                model_config_name=model_config_name,
                save=save,
                load_model_name=load_model_name,
                save_name=save_name,
            )

    if validate:
        validation_pipeline(config["pipeline"]["validation"]["models"], config)
