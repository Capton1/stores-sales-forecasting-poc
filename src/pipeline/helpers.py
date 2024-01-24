from typing import Any, Dict

from omegaconf.dictconfig import DictConfig


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
