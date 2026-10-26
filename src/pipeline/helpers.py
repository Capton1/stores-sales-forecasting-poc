from datetime import datetime, timezone
from typing import Any, Dict

import mlflow
from omegaconf.dictconfig import DictConfig
from sklearn.preprocessing import MinMaxScaler

from models.persistence import run_model_uri, scaler_from_dict


def timestamped_name(*parts: str) -> str:
    """
    Name a model or log after its parts and the current UTC time.

    MLflow model names cannot contain ':', hence the `h` separator.

    Returns:
        str: `<parts joined by spaces> - YYYY-MM-DD HHhMM`.
    """
    return f'{" ".join(parts)} - {datetime.now(timezone.utc).strftime("%Y-%m-%d %Hh%M")}'


def get_config_from_model_name(
    name: str, model_type: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get the model config from the model name.

    Args:
        name (str): The model name.
        model_type (str): The type of the model.
        config (Dict[str, Any]): The configuration dictionary.

    Returns:
        Dict[str, Any]: The model config.
    """
    config_type = name.split(" - ")[0].split(model_type)[1].strip()
    return config[model_type][config_type]


def convert_config_to_dict(config):
    """
    Converts a configuration object to a dictionary.

    Args:
        config (DictConfig): The configuration object to convert.

    Returns:
        dict: The converted configuration dictionary.
    """
    config = dict(config)

    for e in config:
        if isinstance(config[e], DictConfig):
            config[e] = dict(config[e])
            for i in config[e]:
                if isinstance(config[e][i], DictConfig):
                    config[e][i] = dict(config[e][i])

    return config


def load_scaler_from_model(run_id: str) -> MinMaxScaler:
    """
    Load the scaler logged with a model.

    Args:
        run_id (str): The MLflow run id of the model.

    Returns:
        MinMaxScaler: The loaded scaler.
    """
    return scaler_from_dict(
        mlflow.artifacts.load_dict(run_model_uri(run_id, "scaler.json"))
    )


def get_run_id_from_model_name(model_name: str) -> str:
    """
    Get the run id of the latest version of a registered model.

    Args:
        model_name (str): The registered model name.

    Returns:
        str: The MLflow run id.
    """
    return (
        mlflow.MlflowClient().get_registered_model(model_name).latest_versions[0].run_id
    )
