from typing import Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from keras.preprocessing.sequence import TimeseriesGenerator



def train_val_split(
    df: pd.DataFrame, val_ratio=0.85
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """split df in train and validation set

    Args:
        df (pd.DataFrame): original train set

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train set and validation set
    """

    val_size = int(len(df) * val_ratio)
    train_data = df.iloc[:val_size]
    val_data = df.iloc[val_size:]

    return train_data, val_data


def scale_df(df: pd.DataFrame) -> pd.DataFrame:
    """Use MinMaxScaler on a given df

    Args:
        df (pd.DataFrame): df to scale

    Returns:
        pd.DataFrame: scaled df
    """
    scaler = MinMaxScaler(feature_range=(0, 1))

    return pd.DataFrame(scaler.fit_transform(df), columns=list(df.columns))


def one_hot_encode(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """encode given column in df using one hot encoding

    Args:
        df (pd.DataFrame): df to encode
        column_name (str): name of the column to encode

    Returns:
        pd.DataFrame: encoded df
    """
    encoder = OneHotEncoder(sparse_output=False)
    ohe = encoder.fit_transform(df[[column_name]])

    res = pd.concat([df, pd.DataFrame(ohe, columns=df[column_name].unique())], axis=1)
    res.drop([column_name], axis=1, inplace=True)

    return res


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
    return df.drop(["date", target], axis=1), df[[target]]


def get_timeseries_generator(
    df: pd.DataFrame, target: str = 'sales', window_size: int = 30, batch_size: int = 1
) -> TimeseriesGenerator:
    """get a timeseries generator for a given df

    Args:
        df (pd.DataFrame): df to generate from
        target (str): target column name
        window_size (int, optional): size of the window. Defaults to 1.
        batch_size (int, optional): size of the batch. Defaults to 32.

    Returns:
        TimeseriesGenerator: timeseries generator
    """
    features, target = get_features_and_target(df, target)
    return TimeseriesGenerator(
        features.to_numpy(), target.to_numpy(), length=window_size, batch_size=batch_size
    )