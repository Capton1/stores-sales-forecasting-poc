from typing import Tuple

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


def interpolate_oil_price(df: pd.DataFrame) -> pd.DataFrame:
    """handle NA in oil price df

    Args:
        df (pd.DataFrame): oil price df

    Returns:
        pd.DataFrame: oil price df with no NA
    """
    df["dcoilwtico"] = np.where(df["dcoilwtico"] == 0, np.nan, df["dcoilwtico"])
    df["dcoilwtico"] = df["dcoilwtico"].interpolate(limit_direction="both")

    return df


def get_train_data(
    is_train_df=True,
    features=[
        "date",
        "family",
        "onpromotion",
        "typeholiday",
        "dcoilwtico",
        "typestores",
        "cluster",
    ],
) -> pd.DataFrame:
    """retrieve training data

    Returns:
        pd.DataFrame: training data
    """
    if is_train_df:
        df = pd.read_csv("data/raw/train.csv")
    else:
        df = pd.read_csv("data/raw/test.csv")

    df["date"] = pd.to_datetime(df["date"])

    oils = pd.read_csv("data/raw/oil.csv")
    oils = interpolate_oil_price(oils)
    oils["date"] = pd.to_datetime(oils["date"])

    holidays = pd.read_csv("data/raw/holidays_events.csv")
    holidays = fix_transfered_holidays(holidays)
    holidays["date"] = pd.to_datetime(holidays["date"])

    stores = pd.read_csv("data/raw/stores.csv")

    res = merge_df(df, holidays, oils, stores)
    res = select_features(res, features)

    # TODO : add column for paycheck (every 15th days)

    return res


def select_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    if "sales" in df.columns:
        df = df[features + ["sales"]]
    else:
        df = df[features]

    return df


def export_df(df: pd.DataFrame, is_train_df=True) -> None:
    if is_train_df:
        df.to_csv("data/processed/processed_train.csv")
    else:
        df.to_csv("data/trusted/test_set.csv")


def process_train_test_data(save=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """create processed df and store it in data/processed foler"""
    train = get_train_data(is_train_df=True)
    test = get_train_data(is_train_df=False)

    if save:
        export_df(train, is_train_df=True)
        export_df(test, is_train_df=False)

    return train, test
