from typing import Tuple

import pandas as pd


def train_val_split(
    df: pd.DataFrame, val_ratio=0.9
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """split df in train and validation set. Split at specified ratio cause it's a time series.

    Args:
        df (pd.DataFrame): original train set

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train set and validation set
    """
    val_size = int(len(df) * val_ratio)
    train_data = df.iloc[:val_size]
    val_data = df.iloc[val_size:]

    return train_data, val_data
