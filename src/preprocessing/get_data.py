import numpy as np
import pandas as pd
from typing import Tuple

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
    
    res = res.fillna("None")
    res = pd.merge(res, stores, how="left", on="store_nbr")

    res.rename(columns={"type_x": "typeholiday", "type_y": "typestores"}, inplace = True)

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
    train_df=True,
    features=[
        "date",
        "family",
        "onpromotion",
        "typeholiday",
        "dcoilwtico",
        "state",
        "typestores",
        "cluster",
    ],
) -> pd.DataFrame:
    """retrieve training data

    Returns:
        pd.DataFrame: training data
    """
    if train_df:
        df = pd.read_csv("data/raw/train.csv")
    else:
        df = pd.read_csv("data/raw/test.csv")

    oils = pd.read_csv("data/raw/oil.csv")
    oils = interpolate_oil_price(oils)

    holidays = pd.read_csv("data/raw/holidays_events.csv")
    holidays = fix_transfered_holidays(holidays)

    stores = pd.read_csv("data/raw/stores.csv")

    res = merge_df(df, holidays, oils, stores)
    res = select_features(res, features)

    res['date'] = pd.to_datetime(res['date'])

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


def export_df(df: pd.DataFrame, train=True) -> None:
    if train:
        df.to_csv("data/processed/processed_train.csv")
    else:
        df.to_csv("data/processed/processed_test.csv")


def process_train_test_data(save = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """create processed df and store it in data/processed foler"""
    train = get_train_data(train_df=True)
    test = get_train_data(train_df=False)

    if save:
        export_df(train, train=True)
        export_df(test, train=False)

    return train, test