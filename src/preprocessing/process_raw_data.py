from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler


def merge_raw_df(
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
    res = res.fillna(0.0)

    res = pd.merge(res, holidays, how="left", on="date")
    res["type"] = res["type"].fillna("Normal")
    res["transferred"] = res["transferred"].fillna(False)

    res = pd.merge(res, stores, how="left", on="store_nbr")

    res.rename(columns={"type_x": "typedays", "type_y": "typestores"}, inplace=True)
    res.drop("id", axis=1, inplace=True)

    res["date"] = pd.to_datetime(res["date"])

    # res = group_by_date(res)

    return res


def load_data(data_paths: Dict[str, Any]) -> pd.DataFrame:
    """load raw data

    Args:
        data_config (Dict[str, Any]): data config

    Returns:
        pd.DataFrame: raw data
    """
    df = pd.read_csv(data_paths["raw"]["train"])
    holidays = pd.read_csv(data_paths["raw"]["holidays"])
    oils = pd.read_csv(data_paths["raw"]["oils"])
    stores = pd.read_csv(data_paths["raw"]["stores"])

    df = merge_raw_df(df, holidays, oils, stores)

    return df


def fix_transfered_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """A holiday that is transferred officially falls on that calendar day,
    but was moved to another date by the government. Hence, they are normal days.
    This function fix this.

    Args:
        df (pd.DataFrame): train or test df

    Returns:
        pd.DataFrame: fixed df
    """
    df.loc[df["transferred"], "typedays"] = "Normal"
    df.loc[df["typedays"] == "Transfer", "typedays"] = "Holiday"

    # Since all of the transferred stuff has been dealt with, we will drop the column
    df = df.drop("transferred", axis=1)
    # Brige day can be considered as Holidays
    df.loc[df["typedays"] == "Bridge", "typedays"] = "Holiday"

    return df


def interpolate_data(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """handle NA in oil price df

    Args:
        df (pd.DataFrame): oil price df

    Returns:
        pd.DataFrame: oil price df with no NA
    """
    df.loc[df[column] == 0, column] = np.nan
    df.loc[:, column].interpolate(limit_direction="both", inplace=True)

    return df


def select_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Selects the specified features from the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features (list): The list of features to select.

    Returns:
        pd.DataFrame: The DataFrame with the selected features.
    """
    return df[features]


def save_df(df: pd.DataFrame, save_path: str, save: bool = False) -> pd.DataFrame:
    """
    Export a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to be exported.
        save_path (str): The path where the CSV file will be saved.
        save (bool, optional): Indicates whether to save the DataFrame or not. Defaults to False.

    Returns:
        pd.DataFrame: The DataFrame to be exported.
    """

    if save:
        df.to_csv(save_path)
        print(f"Dataframe saved at {save_path}")

    return df


def min_max_scale_df(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Use MinMaxScaler on a given df

    Args:
        df (pd.DataFrame): df to scale

    Returns:
        pd.DataFrame: scaled df
    """
    scaler = MinMaxScaler(feature_range=(0, 1))

    df[features] = scaler.fit_transform(df[features])

    return df


def one_hot_encode_df(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """encode given column in df using one hot encoding

    Args:
        df (pd.DataFrame): df to encode
        column_name (str): name of the column to encode

    Returns:
        pd.DataFrame: encoded df
    """
    return pd.get_dummies(df, columns=features, dtype="bool")


def group_by_date(df: pd.DataFrame, mean=[], sum=[]) -> pd.DataFrame:
    """
    Group the DataFrame by date and aggregate the columns based on their types.

    Parameters:
        df (pd.DataFrame): The input DataFrame to be grouped.

    Returns:
        pd.DataFrame: The grouped DataFrame with aggregated values.
    """
    return (
        df.groupby(["date"])
        .agg({c: "mean" if c in mean else "sum" if c in sum else "first" for c in df})
        .drop(["date"], axis=1)
        .reset_index()
    )


def input_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """input missing dates in df

    Args:
        df (pd.DataFrame): df to input missing dates

    Returns:
        pd.DataFrame: df with missing dates inputed
    """
    # create a new df with all dates
    df.set_index("date", inplace=True)

    date_range = pd.date_range(start=df.index.min(), end=df.index.max())
    res = pd.DataFrame(index=date_range)

    # merge the two df
    res = pd.merge(res, df, how="left", left_index=True, right_index=True)

    # fill missing values in typedays
    res["typedays"] = res["typedays"].fillna("Normal")

    # fill missing values in numeric columns
    res.interpolate(method="linear", inplace=True)
    df = res

    return df


def get_preprocessing_pipeline(
    data_config: Dict[str, Any], is_test_df=False, save=False
) -> Pipeline:
    """Returns the preprocessing pipeline.

    Returns:
        Pipeline: The preprocessing pipeline.
    """
    type_df = "test" if is_test_df else "train"
    features_to_select = data_config["features"]
    if not is_test_df:
        features_to_select.append(data_config["target"])

    preprocessor = Pipeline(
        steps=[
            (
                "interpolate_oil_price",
                FunctionTransformer(interpolate_data, kw_args={"column": "dcoilwtico"}),
            ),
            (
                "fix_transfered_holidays",
                FunctionTransformer(fix_transfered_holidays),
            ),
            (
                "select_useful_features",
                FunctionTransformer(
                    select_features, kw_args={"features": features_to_select}
                ),
            ),
            (
                "export_preprocessed",
                FunctionTransformer(
                    save_df,
                    kw_args={
                        "save_path": data_config["paths"]["processed"][type_df],
                        "save": save,
                    },
                ),
            ),
            (
                "group_by_date",
                FunctionTransformer(
                    group_by_date,
                    kw_args={
                        "mean": ["dcoilwtico", "cluster"],
                        "sum": ["onpromotion", "sales"],
                    },
                ),
            ),
            (
                "min_max_scale_df",
                FunctionTransformer(
                    min_max_scale_df,
                    kw_args={"features": ["onpromotion", "dcoilwtico", "cluster"]},
                ),
            ),
            (
                "input_missing_dates",
                FunctionTransformer(input_missing_dates),
            ),
            (
                "one_hot_encode_df",
                FunctionTransformer(
                    one_hot_encode_df, kw_args={"features": ["typedays"]}
                ),
            ),
            (
                "export_processed",
                FunctionTransformer(
                    save_df,
                    kw_args={
                        "save_path": data_config["paths"]["trusted"][type_df],
                        "save": save,
                    },
                ),
            ),
        ]
    )

    return preprocessor


def drop_duplicates(df: pd.DataFrame, subset: List[str] = []) -> pd.DataFrame:
    """drop duplicates in df

    Args:
        df (pd.DataFrame): df to drop duplicates

    Returns:
        pd.DataFrame: df with duplicates dropped
    """
    return df.drop_duplicates(
        subset=subset,
        keep=False,
    )


def process_data(
    df: pd.DataFrame,
    data_config: Dict[str, Any],
) -> pd.DataFrame:
    """Process the data."""
    df = drop_duplicates(df, ["sales", "date", "store_nbr", "family"])

    df = interpolate_data(df, "dcoilwtico")
    df = fix_transfered_holidays(df)

    features_to_select = data_config["features"]
    if data_config["target"] not in features_to_select:
        features_to_select.append(data_config["target"])
    df = select_features(df, features_to_select)

    mean = ["dcoilwtico", "cluster"] if "cluster" in features_to_select else ["dcoilwtico"]
    df = group_by_date(df, mean=mean, sum=["onpromotion", "sales"])
    
    scale_columns = ["onpromotion", "dcoilwtico", "cluster"] if "cluster" in features_to_select else ["onpromotion", "dcoilwtico"]
    df = min_max_scale_df(df, scale_columns)
    df = input_missing_dates(df)
    df = one_hot_encode_df(df, ["typedays"])

    return df
