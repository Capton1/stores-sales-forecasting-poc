import pathlib
from functools import partial
from typing import Any, Dict

import mlflow
import pandas as pd
from hyperopt import Trials, fmin, tpe
from sklearn.preprocessing import MinMaxScaler

from models.lstm import build_lstm, lstm_cross_validation_train
from models.persistence import save_model_locally, scaler_to_dict
from models.prophet import build_prophet, prophet_cross_validation_train
from models.training_helpers import (build_lstm_generator,
                                     generate_ml_features, get_basic_model,
                                     get_prophet_df)
from models.xgboost import (build_xgboost, xgboost_cross_validation_train,
                            xgboost_objective_function)
from pipeline.helpers import load_scaler_from_model, timestamped_name
from preprocessing.helpers import train_val_split
from preprocessing.process_raw_data import process_data
from xgboost import XGBRegressor


class Factory:
    def __init__(self, data_config: Dict[str, Any]):
        self.data_config = data_config
        self.model_config = None
        self.model_config_name = None
        self.registered_model_name = None

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
    
    def get_generator(self):
        return self.train_generator

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

    def _scale_target(self, df: pd.DataFrame, scaler: MinMaxScaler = None) -> pd.DataFrame:
        """Return `df` with its target on the scale the model works on."""
        if scaler is None:
            return df
        target = self.data_config["target"]
        df = df.copy()
        df[[target]] = scaler.transform(df[[target]])
        return df

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
            raise ValueError(f"Unexpected error while processing data: {e}") from e

        return self

    def _build_generator(self, df: pd.DataFrame, model_config: Dict[str, Any] = None):
        if self.model_config is None:
            self.model_config = dict(model_config)
            self.fit_params = (
                dict(model_config["fit_params"])
                if "fit_params" in model_config
                else None
            )

        self._fit_data(df)

        # the scaler sees the training split only, so no validation data leaks into it
        if self.model_config.get("scaled") and self.y_scaler is None:
            self.y_scaler = MinMaxScaler(feature_range=(0, 1)).fit(
                self.train_df[[self.data_config["target"]]]
            )
        train_df = self._scale_target(self.train_df, self.y_scaler)

        if self.model_config["_name"] == "lstm":
            (
                self.train_generator,
                self.input_data_shape,
            ) = build_lstm_generator(
                train_df,
                self.model_config,
                self.data_config["target"],
            )
        elif self.model_config["_name"] == "xgboost":
            self.train_generator = generate_ml_features(
                train_df,
                self.data_config["target"],
                self.model_config.get("earthquake", False),
            )
        elif self.model_config["_name"] == "prophet":
            self.train_generator = get_prophet_df(
                train_df, self._scale_target(self.val_df, self.y_scaler)
            )
        else:
            raise ValueError("Model not found")

        return self

    def build_generator(self, include_val_df = False):
        if self.train_df is None:
            raise ValueError("Data not found. Please fit the data first")
        
        if include_val_df and self.val_df is not None:
            self.train_df = pd.concat([self.train_df, self.val_df])
        
        self._build_generator(self.train_df, self.model_config)

        return self

    def _build_model(self, load_model_name: str = None, use_mlflow=True):
        if self.train_generator is None and load_model_name is None:
            raise ValueError(
                "Generator not found. Please fit the data first or load a model"
            )

        if load_model_name:
            self.registered_model_name = load_model_name

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
        df: pd.DataFrame,
        model_config: Dict[str, Any],
        model_config_name: str = "default",
        load_model_name: str = None,
    ):
        self.model_config_name = model_config_name

        self._build_generator(df, model_config)
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

        self._fit_model(df, model_config, model_config_name, load_model_name)

        return self

    def fit_transform(self, df):
        self.fit(df)
        return self.train_df

    def predict(self, data: pd.DataFrame = None):
        if not self.model:
            raise ValueError("Model not found. Please fit the data first")

        # a loaded model brings the scaler it was trained with
        scaler = None
        if self.model_config.get("scaled"):
            scaler = (
                load_scaler_from_model(self.registered_model_name)
                if self.registered_model_name
                else self.y_scaler
            )

        if self.model_config["_name"] == "lstm":
            if data is None:
                # concatenate the last `look_back` days of the training set to the validation set in order to predict the first day of the validation set
                val_lstm = pd.concat(
                    [self.train_df[-self.model_config["look_back"] :], self.val_df]
                )
                data, shape = build_lstm_generator(
                    self._scale_target(val_lstm, scaler),
                    self.model_config,
                    self.data_config["target"],
                )
                self.input_data_shape = shape

            # the first time step of each window's prediction
            res = self.model.predict(data, verbose=0)[:, 0].reshape(-1)

            return scaler.inverse_transform(res.reshape(-1, 1)) if scaler else res
        elif self.model_config["_name"] == "xgboost":
            if data is None:
                data = generate_ml_features(
                    self._scale_target(self.val_df, scaler),
                    self.data_config["target"],
                    self.model_config.get("earthquake", False),
                )[0]

            return (
                scaler.inverse_transform(self.model.predict(data).reshape(-1, 1))
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

        load_model_name = load_model_name.removesuffix(".h5")

        if model_config:
            self.model_config = dict(model_config)
        self._build_model(load_model_name, use_mlflow)

        return self

    def save_model(self, save_localy: bool = True):
        if not self.model:
            raise ValueError("Model not found. Please fit the data first")

        if not self.save_name:
            model_name = timestamped_name(self.model_config["_name"], self.model_config_name)
        else:
            model_name = self.save_name

        if self.model_config["_name"] == "lstm":
            mlflow.keras.log_model(
                self.model,
                name="model",
                registered_model_name=model_name,
            )
        elif self.model_config["_name"] == "xgboost":
            mlflow.xgboost.log_model(
                self.model,
                name="model",
                registered_model_name=model_name,
            )
        else:
            mlflow.prophet.log_model(
                self.model,
                name="model",
                registered_model_name=model_name,
            )

        mlflow.set_tags({"model_type": self.model_config["_name"]})
        mlflow.log_param("model_type", self.model_config["_name"])
        mlflow.log_param("model_config", self.model_config)
        mlflow.log_metric("MSE", self.val_mse)

        if self.y_scaler:
            mlflow.log_dict(scaler_to_dict(self.y_scaler), "scaler.json")

        if save_localy:
            save_model_locally(
                self.model,
                self.model_config["_name"],
                self.data_config["paths"]["models"][self.model_config["_name"]],
                model_name,
            )

            print(f"Model saved as {model_name}")

        return self

    def save_data(self):
        if self.train_df is None or self.val_df is None:
            raise ValueError("Data not found. Please fit the data first")

        for df, split in ((self.train_df, "train"), (self.val_df, "validation")):
            path = pathlib.Path(self.data_config["paths"]["processed"][split]).absolute()
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path)
            print(f"Processed {split} set saved as {path.name}")

        return self

    def bayesian_optim(
        self, df: pd.DataFrame, model_type: str, n_iter: int, parameters: Dict[str, Any]
    ):
        self.model = get_basic_model(model_type)
        self.model_config = {"_name": model_type, "earthquake": True, "scaled": False}

        self._build_generator(df, self.model_config)

        if model_type == "lstm":
            pass
        elif model_type == "xgboost":
            objective = partial(
                xgboost_objective_function, generator=self.train_generator
            )
            return fmin(
                fn=objective,
                space=parameters,
                algo=tpe.suggest,
                max_evals=n_iter,
                trials=Trials(),
            )
        elif model_type == "prophet":
            pass
        else:
            raise ValueError("Unknown model type")
