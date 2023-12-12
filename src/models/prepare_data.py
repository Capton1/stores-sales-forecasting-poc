import pandas as pd
from .training_helpers import train_val_split, one_hot_encode, scale_df
from typing import Tuple

def prepare_training_data(df: pd.DataFrame, save = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    scaled_data = scale_df(df[['onpromotion', 'dcoilwtico', 'cluster']])
    df.drop(['onpromotion', 'dcoilwtico', 'cluster'], axis=1, inplace=True)
    res = pd.concat([df, scaled_data], axis=1)

    res = one_hot_encode(df, "typeholiday")

    train_set, val_set = train_val_split(res)

    if save:
        train_set.to_csv("data/trusted/training_set.csv")
        val_set.to_csv("data/trusted/validation_set.csv")

    return train_set, val_set