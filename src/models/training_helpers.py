import pandas as pd
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def train_val_split(df: pd.DataFrame, val_ratio = 0.75) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    return pd.DataFrame(scaler.fit_transform(df))

def one_hot_encode(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """encode given column in df using one hot encoding

    Args:
        df (pd.DataFrame): df to encode
        column_name (str): name of the column to encode

    Returns:
        pd.DataFrame: encoded df
    """
    encoder = OneHotEncoder(sparse_output = False)
    ohe = encoder.fit_transform(df[[column_name]])

    res = pd.concat([df, pd.DataFrame(ohe, columns=df[column_name].unique())], axis = 1)
    res.drop([column_name], axis=1, inplace=True)

    return res