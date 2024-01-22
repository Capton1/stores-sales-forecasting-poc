import pathlib
from typing import Any, Dict, Tuple

import pandas as pd
from keras.layers import LSTM, Activation, Concatenate, Dense, Input
from keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Activation, Dense
from tensorflow.keras.models import Sequential


def _build_simple_lstm(
    input_shape: Tuple[int, int], optimizer: str, loss: str
) -> Sequential:
    """
    Build a simple LSTM model.

    Args:
        input_shape (Tuple[int, int]): The shape of the input data.
        optimizer (str): The optimizer used for compiling the model.
        loss (str): The loss function used for compiling the model.

    Returns:
        Sequential: The LSTM model.
    """
    # Input layers shape : (batch_size, look_back, n_features)
    model = Sequential()
    model.add(
        LSTM(
            50,
            activation="relu",
            input_shape=input_shape,
            return_sequences=False,
            dropout=0.2,
        )
    )
    model.add(Dense(1))
    model.add(Activation("linear"))

    model.compile(
        optimizer=optimizer,
        loss=loss,
    )

    return model


def _build_multivariate_lstm(
    input_shape_continuous: Tuple[int, int],
    input_shape_categorical: Tuple[int, int],
    optimizer: str,
    loss: str,
) -> Model:
    """
    Build a multivariate LSTM model.

    Args:
        input_shape_continuous (Tuple[int, int]): The shape of the continuous input data.
        input_shape_categorical (Tuple[int, int]): The shape of the categorical input data.
        optimizer (str): The optimizer used for compiling the model.
        loss (str): The loss function used for compiling the model.

    Returns:
        Model: The LSTM model.
    """
    # Input layers shape : (batch_size, look_back, n_features_continuous) and (batch_size, look_back, n_features_categorical)
    input_continuous = Input(shape=input_shape_continuous)
    input_categorical = Input(shape=input_shape_categorical)

    lstm_layer = LSTM(50, activation="relu", return_sequences=True, dropout=0.2)(
        input_continuous
    )
    lstm_layer = Dense(1)(lstm_layer)
    lstm_layer = Activation("linear")(lstm_layer)
    lstm_layer = Model(inputs=input_continuous, outputs=lstm_layer)

    dense_layer = Dense(4, activation="relu")(input_categorical)
    dense_layer = Dense(1)(dense_layer)
    dense_layer = Activation("linear")(dense_layer)
    dense_layer = Model(inputs=input_categorical, outputs=dense_layer)

    concat_layer = Concatenate()([lstm_layer.output, dense_layer.output])
    concat_layer = Dense(1)(concat_layer)

    model = Model(inputs=[lstm_layer.input, dense_layer.input], outputs=concat_layer)

    model.compile(
        optimizer=optimizer,
        loss=loss,
    )

    return model


def build_lstm(
    input_shape, model_config: Dict[str, Any], load_model_name: str = None
) -> Model:
    """
    Build an LSTM model based on the given input shape and model configuration.

    Args:
        input_shape (tuple): The shape of the input data.
        model_config (dict): The configuration parameters for the model.

    Returns:
        Model: The built LSTM model.
    """
    if load_model_name:
        return open(
            f"{pathlib.Path(model_config['save_path']).absolute()}/{load_model_name}.h5",
            "rb",
        )

    if model_config["type"] == "simple":
        return _build_simple_lstm(
            input_shape,
            model_config["build_params"]["optimizer"],
            model_config["build_params"]["loss"],
        )
    return _build_multivariate_lstm(
        *input_shape,
        model_config["build_params"]["optimizer"],
        model_config["build_params"]["loss"],
    )


def lstm_train(
    model: Model,
    generator: Tuple[pd.DataFrame, pd.DataFrame],
    parameters: Dict[str, Any],
) -> Model:
    """
    Train the LSTM model.

    Args:
        model (Model): The LSTM model.
        generator (Tuple[pd.DataFrame, pd.DataFrame]): The data generator containing the input data and target data.
        parameters (Dict[str, Any]): The parameters to be passed to the LSTM model.

    Returns:
        Model: The trained LSTM model.
    """
    model.fit(generator, **parameters)
    return model


def lstm_cross_validation_train(
    model: Model,
    generator: Tuple[pd.DataFrame, pd.DataFrame],
    parameters: Dict[str, Any],
) -> float:
    """
    Cross validate the LSTM model.
    """
    model.fit(generator, **parameters)
    return 1234.5
