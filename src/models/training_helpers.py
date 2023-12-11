import pd.DataFrame
from typing import Tuple

def train_val_split(df: pd.DataFrame, val_ratio = 0.75) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """split df in train and validation set

    Args:
        df (pd.DataFrame): _description_

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: _description_
    """

    val_size = int(len(df) * val_ratio)
    train_data = df.iloc[:val_size]
    val_data = df.iloc[val_size:]

    return train_data, val_data 