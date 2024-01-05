from models.training_helpers import get_features_and_target
from preprocessing.get_data import collect_data
from preprocessing.prepare_data import prepare_training_data
from typing import Dict

import pandas as pd


def generate_train_val_sets(data_path: Dict[str,str], save: bool = True):
    """Generates train and validation sets.

    Args:
        df (pd.DataFrame): original train set
        val_ratio (float, optional): ratio of the validation set. Defaults to 0.85.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train set and validation set
    """
    train, _ = collect_data(data_path, save=save)    
    prepare_training_data(train, val_ratio=0.85, save=save, save_path=data_path["trusted"])

    
def train_model(config, model, save: bool = True):
    """Trains the model.

    Args:
        config (Dict): configuration dictionary
        model (str): model to use
        save (bool, optional): whether to save the model. Defaults to True.
    """
    trusted_data_files = config["data"]["paths"]["trusted"]
    
    X_train, y_train = get_features_and_target(pd.read_csv(trusted_data_files["train"], index_col=0))
    X_val, y_val = get_features_and_target(pd.read_csv(trusted_data_files["val"], index_col=0))
    
    if model == "lstm":
        lstm()
    elif model == "arima":
        pass
    elif model == "prophet":
        pass
    elif model == "prophet":
        pass
    elif model == "rnn":
        pass
    else:
        raise ValueError("Model not found")


def launch_pipeline(config, model, train: bool = False, generate: bool = False, save: bool = False):
    """Launches the pipeline."""
    if generate:
        generate_train_val_sets(config['data']['paths'], save=save)

    if train:
        train_model(config, model, save=save)