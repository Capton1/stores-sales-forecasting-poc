from typing import Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from keras.preprocessing.sequence import TimeseriesGenerator


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


def get_timeseries_generator_from_df(
    df: pd.DataFrame, target: str = 'sales', window_size: int = 30, batch_size: int = 1, return_x: bool = False, return_y: bool = False,
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
    gen = TimeseriesGenerator(features.to_numpy(), target.to_numpy(), length=window_size, batch_size=batch_size)
    
    if return_y and return_x:
        return gen, features, target
    elif return_y:
        return gen, target
    elif return_x: 
        return gen, features
    else:    
        return gen