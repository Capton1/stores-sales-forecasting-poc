from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def merge_df(
    df: pd.DataFrame, holidays: pd.DataFrame, oils: pd.DataFrame, stores: pd.DataFrame
) -> pd.DataFrame:
    """merge a df with holidays, oils and stores df

    Args:
        df (pd.DataFrame): train or test df
        holidays (pd.DataFrame): holidays df
        oils (pd.DataFrame): oils df
        stores (pd.DataFrame): stores df

    Returns:
        pd.DataFrame: df merged
    """
    res = pd.merge(df, oils, on="date")

    res = pd.merge(res, holidays, how="left", on="date")
    res = res.fillna("Normal")

    res = pd.merge(res, stores, how="left", on="store_nbr")

    res.rename(columns={"type_x": "typeholiday", "type_y": "typestores"}, inplace=True)

    res.drop("id", axis=1, inplace=True)

    return res


def fix_transfered_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """A holiday that is transferred officially falls on that calendar day,
    but was moved to another date by the government. Hence, they are normal days.
    This function fix this.

    Args:
        df (pd.DataFrame): train or test df

    Returns:
        pd.DataFrame: fixed df
    """
    df.loc[df["transferred"], "type"] = "Normal"
    df.loc[df["type"] == "Transfer", "type"] = "Holiday"

    # Since all of the transferred stuff has been dealt with, we will drop the column
    df = df.drop("transferred", axis=1)
    # Brige day can be considered as Holidays
    df.loc[df["type"] == "Bridge", "type"] = "Holiday"

    return df


def interpolate_oil_price(
    df: pd.DataFrame, oil_column: str = "dcoilwtico"
) -> pd.DataFrame:
    """handle NA in oil price df

    Args:
        df (pd.DataFrame): oil price df

    Returns:
        pd.DataFrame: oil price df with no NA
    """
    df[oil_column] = np.where(df[oil_column] == 0, np.nan, df[oil_column])
    df[oil_column] = df[oil_column].interpolate(limit_direction="both")

    return df


def get_train_data(
    raw_path: Dict[str, str],
    is_train_df=True,
    features=[
        "date",
        # "family",
        "onpromotion",
        "typeholiday",
        "dcoilwtico",
        # "typestores",
        "cluster",
    ],
) -> pd.DataFrame:
    """retrieve training data

    Returns:
        pd.DataFrame: training data
    """
    if is_train_df:
        df = pd.read_csv(raw_path["train"])
    else:
        df = pd.read_csv(raw_path["test"])

    df["date"] = pd.to_datetime(df["date"])

    oils = pd.read_csv(raw_path["oils"])
    oils = interpolate_oil_price(oils)
    oils["date"] = pd.to_datetime(oils["date"])

    holidays = pd.read_csv(raw_path["holidays"])
    holidays = fix_transfered_holidays(holidays)
    holidays["date"] = pd.to_datetime(holidays["date"])

    stores = pd.read_csv(raw_path["stores"])

    res = merge_df(df, holidays, oils, stores)
    res = select_features(res, features)

    # TODO : add column for paycheck (every 15th days)

    return res


def select_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Selects the specified features from the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features (list): The list of features to select.

    Returns:
        pd.DataFrame: The DataFrame with the selected features.
    """
    if "sales" in df.columns:
        df = df[features + ["sales"]]
    else:
        # case for test set
        df = df[features]

    return df


def export_df(df: pd.DataFrame, save_path: Dict[str, str], is_train_df=True) -> None:
    """
    Export a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to be exported.
        is_train_df (bool, optional): Indicates whether the DataFrame is a training set or not.
            Defaults to True.

    Returns:
        None
    """
    if is_train_df:
        df.to_csv(save_path["train"])
    else:
        df.to_csv(save_path["test"])


def collect_data(
    path: Dict[str, Any], save: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collect and process the train and test data. Save the processed data if specified.

    Args:
        path (Dict[str,str]): A dictionary containing the paths to the raw and processed data.
        save (bool, optional): Whether to save the processed data. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the processed train and test dataframes.
    """
    train = get_train_data(path["raw"], is_train_df=True)
    test = get_train_data(path["raw"], is_train_df=False)

    if save:
        export_df(train, path["processed"], is_train_df=True)
        export_df(test, path["processed"], is_train_df=False)

    return train, test
