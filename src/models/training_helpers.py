from typing import Tuple

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator, pad_sequences
from keras.utils import Sequence


def get_features_and_target(
    df: pd.DataFrame, target="sales"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """get features and target sets for model training and eval

    Args:
        df (pd.DataFrame): df with features and y
        target (str): target column name

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: features_set, target_set
    """
    return df.drop([target], axis=1), df[[target]]


def get_single_input_timeseries_generator(
    X: pd.DataFrame, y: pd.DataFrame, look_back: int, batch_size: int = 1
):
    """
    Create a time series generator for training data.

    Parameters:
    X_train (array-like): The input data.
    y_train (array-like): The target data.
    config_model (dict): Configuration parameters for the model.

    Returns:
    gen (TimeseriesGenerator): The time series generator for data.
    """
    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.float32)

    # batch size is the number of day we try to predict
    # generator look like this: (X, y) with X of shape (batch_size, look_back, n_features)
    gen = TimeseriesGenerator(X, y, length=look_back, batch_size=batch_size)

    return gen


def get_multiple_input_timeseries_generator(
    X1: pd.DataFrame,
    X2: pd.DataFrame,
    y: pd.DataFrame,
    look_back: int,
    batch_size: int = 1,
):
    """
    Create a time series generator for training data.

    Parameters:
    X1 (pd.DataFrame): The first input data.
    X2 (pd.DataFrame): The second input data.
    y (pd.DataFrame): The target data.
    look_back (int): The number of previous time steps to use as input for each prediction.
    batch_size (int, optional): The number of samples per batch. Defaults to 1.

    Returns:
    gen (MultipleInputTimeseriesGenerator): The time series generator for data.
    """
    X1 = np.asarray(X1).astype(np.float32)
    X2 = np.asarray(X2).astype(np.float32)
    y = np.asarray(y).astype(np.float32)

    # batch size is the number of day we try to predict
    # generator look like this: ([X1, X2], y) with X1 and X2 of shape (batch_size, look_back, n_features)
    multiple_gen = MultipleInputTimeseriesGenerator(
        X1, X2, y, length=look_back, batch_size=batch_size
    )

    return multiple_gen


def get_pad_sequences(X: pd.DataFrame, y: pd.DataFrame, look_back: int):
    """
    Create a time series generator for training data.

    Parameters:
    X (pd.DataFrame): The input data.
    y (pd.DataFrame): The target data.
    look_back (int): The number of time steps to look back.

    Returns:
    X_padded (np.ndarray): The padded input data.
    y_padded (np.ndarray): The padded target data.
    """
    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.float32)

    X_padded = pad_sequences(X, maxlen=look_back, dtype="float32")
    y_padded = pad_sequences(y, maxlen=look_back, dtype="float32")

    return X_padded, y_padded


class MultipleInputTimeseriesGenerator(TimeseriesGenerator):
    """
    Custom generator for generating multiple input timeseries data for training.

    Args:
        X1 (ndarray): The first input data.
        X2 (ndarray): The second input data.
        Y (ndarray): The target data.
        look_back (int): The number of time steps to look back for creating input sequences.
        batch_size (int): The batch size for generating data.

    Returns:
        tuple: A tuple containing the input batch and the corresponding target batch.
    """

    def __init__(self, X1, X2, Y, length, batch_size=1):
        if len(X1) != len(Y) or len(X2) != len(Y) or len(X1) != len(X2):
            raise ValueError(
                "Data and targets have to be"
                + f" of same length. X1 length is {len(X1)}, X2 length is {len(X2)}"
                + f" while target length is {len(Y)}"
            )

        self.X1 = X1
        self.X2 = X2
        self.Y = Y
        self.length = length
        self.batch_size = batch_size
        self.start_index = self.length
        self.end_index = len(self.Y) - 1

    def __len__(self):
        return (self.end_index - self.start_index + self.batch_size) // self.batch_size

    def __getitem__(self, index):
        """Getting items from the 2 generators and packing them"""
        i = self.start_index + self.batch_size * index
        rows = np.arange(i, min(i + self.batch_size, self.end_index + 1), 1)

        X1_batch = np.array([self.X1[row - self.length : row : 1] for row in rows])
        X2_batch = np.array([self.X2[row - self.length : row : 1] for row in rows])
        Y_batch = np.array([self.Y[row] for row in rows])

        X_batch = [X1_batch, X2_batch]

        return X_batch, Y_batch
