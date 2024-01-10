import pathlib
import pickle
from datetime import datetime, timezone
from typing import Tuple

from keras.layers import LSTM, Activation, Concatenate, Dense, Input
from keras.models import Model, Sequential
from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.layers import LSTM, Activation, Dense
from tensorflow.keras.models import Sequential


def build_simple_lstm(
    input_shape: Tuple[int, int], optimizer: str, loss: str, learning_rate: float
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
        learning_rate=0.01,
    )

    return model


def build_multivariate_lstm(
    input_shape_continuous: Tuple[int, int],
    input_shape_categorical: Tuple[int, int],
    optimizer: str,
    loss: str,
    learning_rate: float,
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
    input_continuous = Input(shape=input_shape_continuous)
    input_categorical = Input(shape=input_shape_categorical)

    lstm_layer = LSTM(50, activation="relu", return_sequences=True, dropout=0.2)(
        input_continuous
    )
    lstm_layer = Dense(1)(lstm_layer)
    lstm_layer = Activation("linear")(lstm_layer)
    lstm_layer = Model(inputs=input_continuous, outputs=lstm_layer)

    dense_layer = Dense(50, activation="relu")(input_categorical)
    dense_layer = Dense(1)(dense_layer)
    dense_layer = Activation("linear")(dense_layer)
    dense_layer = Model(inputs=input_categorical, outputs=dense_layer)

    concat_layer = Concatenate()([lstm_layer.output, dense_layer.output])
    concat_layer = Dense(1)(concat_layer)

    model = Model(inputs=[lstm_layer.input, dense_layer.input], outputs=concat_layer)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        learning_rate=learning_rate,
    )

    return model


def lstm(
    train: bool = False,
    load_model: str = None,
    generator: Tuple[TimeseriesGenerator] = None,
    save: str = False,
    epochs: int = 1,
    batch_size: int = 64,
    optimizer: str = "adam",
    loss: str = "mse",
    learning_rate: float = 0.01,
    look_back: int = 30,
    type: str = "simple",
    save_path: str = None,
) -> Model:
    """
    LSTM model for training or loading a pre-trained model.

    Args:
        train (bool, optional): Flag indicating whether to train the model. Defaults to False.
        load_model (bool, optional): Flag indicating whether to load a pre-trained model. Defaults to False.
        generator (TimeseriesGenerator, optional): The generator used for training the model. Required if train=True. Defaults to None.
        save (bool, optional): Flag indicating whether to save the trained model. Defaults to False.
        epochs (int, optional): Number of epochs for training the model. Defaults to 1.
        batch_size (int, optional): Batch size for training the model. Defaults to 64.
        optimizer (str, optional): Optimizer for the model. Defaults to "adam".
        loss (str, optional): Loss function for the model. Defaults to "mse".
        save_path (str, optional): Path to save the trained model. Required if save=True. Defaults to None.

    Raises:
        ValueError: Raised when neither train nor load_model is True.
        ValueError: Raised when train is True but generator is not provided.

    Returns:
        Sequential: The trained or loaded LSTM model.
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
        if type == "simple":
            # input shape is look_back *  number of features
            model = build_simple_lstm(
                (look_back, generator[0][0].data.shape[2]),
                optimizer,
                loss,
                learning_rate,
            )
        elif type == "multivariate":
            # continuous and categorical data are separated in the generator
            model = build_multivariate_lstm(
                (look_back, generator[0][0][0].data.shape[2]),
                (look_back, generator[0][0][1].data.shape[2]),
                optimizer,
                loss,
                learning_rate,
            )
        else:
            raise ValueError("Model type not found")

    if train:
        model.fit(
            generator,
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=len(generator),
        )

    if save:
        model_name = f'{type} {str(datetime.now(timezone.utc)).split(".")[0]}'

        pickle.dump(
            model, open(f"{pathlib.Path(save_path).absolute()}/{model_name}.h5", "wb")
        )

        print(f"Model saved as {model_name}")

    return model
