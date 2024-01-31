import pickle
from typing import Any, Dict

import mlflow
from omegaconf.dictconfig import DictConfig
from sklearn.preprocessing import MinMaxScaler


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


def load_scaler_from_model(model_name: str) -> MinMaxScaler:
    """
    Load the scaler from the model.

    Args:
        model_name (str): The name of the model.

    Returns:
        MinMaxScaler: The loaded scaler.
    """
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{model_name}/scaler.h5",
        dst_path=f"models_save/scaler/{model_name}.h5",
    )
    scaler = pickle.load(
        open(
            local_path,
            "rb",
        )
    )
    return scaler


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    print(type(load_scaler_from_model("35684897dee349d39b0f37103f7738cc")))
