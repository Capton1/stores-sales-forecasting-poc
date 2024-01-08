from typing import Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def train_val_split(
    df: pd.DataFrame, val_ratio=0.85
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


def scale_df(df: pd.DataFrame) -> pd.DataFrame:
    """Use MinMaxScaler on a given df

    Args:
        df (pd.DataFrame): df to scale

    Returns:
        pd.DataFrame: scaled df
    """
    scaler = MinMaxScaler(feature_range=(0, 1))

    return pd.DataFrame(scaler.fit_transform(df), columns=list(df.columns))


def one_hot_encode(df: pd.DataFrame, columns_name: List[str]) -> pd.DataFrame:
    """encode given column in df using one hot encoding

    Args:
        df (pd.DataFrame): df to encode
        column_name (str): name of the column to encode

    Returns:
        pd.DataFrame: encoded df
    """
    # encoder = OneHotEncoder(sparse_output=False)
    # ohe = encoder.fit_transform(df[columns_name])
    # print(ohe, df[columns_name])
    # res = pd.concat([df, pd.DataFrame(ohe, columns=df[columns_name].nunique())], axis=1)
    # res.drop([columns_name], axis=1, inplace=True)

    return pd.get_dummies(df, columns=columns_name)


def prepare_training_data(
    df: pd.DataFrame, val_ratio=0.85, save=False, save_path: Dict[str, str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare training data by scaling, encoding, and splitting the DataFrame.

    This function takes a DataFrame as input and performs the following steps:
    1. Scale the columns 'onpromotion', 'dcoilwtico', and 'cluster' using a scaler.
    2. Drop the columns 'onpromotion', 'dcoilwtico', and 'cluster' from the DataFrame.
    3. Concatenate the scaled columns with the original DataFrame.
    4. Perform one-hot encoding on the columns 'typeholiday', 'family', and 'typestores'.
    5. Split the DataFrame into training and validation sets.
    6. Optionally save the training and validation sets to CSV files.

    Args:
        df (pd.DataFrame): The input DataFrame.
        save (bool, optional): Whether to save the training and validation sets to CSV files. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and validation sets.
    """
    scaled_data = scale_df(df[["onpromotion", "dcoilwtico", "cluster"]])
    df.drop(["onpromotion", "dcoilwtico", "cluster"], axis=1, inplace=True)
    res = pd.concat([df, scaled_data], axis=1)

    res.rename(
        columns={"typeholiday": "typedays", "family": "typesoldproducts"}, inplace=True
    )

    to_encode = ["typedays", "typesoldproducts", "typestores", "state"]
    res = one_hot_encode(res, to_encode)

    # group by date
    # res = res.groupby("date").agg({c: "mean" if c in ["onpromotion", "dcoilwtico", "cluster", "sales"] else "first" for c in res.drop('date',axis=1).columns})
    res.drop(["date"], axis=1, inplace=True)

    train_set, val_set = train_val_split(res, val_ratio=val_ratio)

    if save:
        if not save_path:
            raise ValueError("save_path must be specified if save is True")
        train_set.to_csv(save_path["train"])
        val_set.to_csv(save_path["val"])

    return train_set, val_set
