from typing import Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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


def input_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """input missing dates in df

    Args:
        df (pd.DataFrame): df to input missing dates

    Returns:
        pd.DataFrame: df with missing dates inputed
    """
    # create a new df with all dates
    date_range = pd.date_range(start=df.index.min(), end=df.index.max())
    res = pd.DataFrame(index=date_range)

    # merge the two df
    res = pd.merge(res, df, how="left", left_index=True, right_index=True)

    # fill missing values in numeric columns
    res.interpolate(method="linear", inplace=True)

    # fill missing values in typedays
    res["typeholiday"] = res["typeholiday"].fillna("Normal")

    return res


def prepare_training_data(
    df: pd.DataFrame, val_ratio=0.9, save=False, save_path: Dict[str, str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare training data by scaling, encoding, and splitting the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be prepared.
        val_ratio (float, optional): The ratio of validation data to be split from the training data. Defaults to 0.9.
        save (bool, optional): Whether to save the prepared training and validation data to CSV files. Defaults to False.
        save_path (Dict[str, str], optional): The file paths to save the training and validation data.
            Required if `save` is True. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the prepared training and validation data DataFrames.

    Raises:
        ValueError: If `save` is True but `save_path` is not specified.

    """
    # group by date
    mean = ["dcoilwtico", "cluster"]
    sum = ["onpromotion", "sales"]
    res = (
        df.groupby(["date"])
        .agg({c: "mean" if c in mean else "sum" if c in sum else "first" for c in df})
        .drop(["date"], axis=1)
        .reset_index()
    )

    # apply scaling
    scaled_data = scale_df(res[["onpromotion", "dcoilwtico", "cluster"]])
    res.drop(["onpromotion", "dcoilwtico", "cluster"], axis=1, inplace=True)
    res = pd.concat([res, scaled_data], axis=1)

    # input missing dates
    res.set_index("date", inplace=True)
    res = input_missing_dates(res)

    # apply one hot encoding
    res.rename(columns={"typeholiday": "typedays"}, inplace=True)
    to_encode = ["typedays"]
    res = one_hot_encode(res, to_encode)

    train_set, val_set = train_val_split(res, val_ratio=val_ratio)

    if save:
        if not save_path:
            raise ValueError("save_path must be specified if save is True")

        train_set.to_csv(save_path["train"])
        val_set.to_csv(save_path["val"])

        print(
            f"Training and validation data saved to {save_path['train']} and {save_path['val']}"
        )

    return train_set, val_set
