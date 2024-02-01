import json
import pathlib
from typing import Any, Dict

from hyperopt import hp, space_eval

from pipeline.factory import Factory
from preprocessing.process_raw_data import load_data


def get_space_of_hyperparameters(model_type: str) -> Dict[str, Any]:
    """
    Returns a dictionary representing the search space of hyperparameters
    based on the given model type.

    Args:
        model_type (str): The type of the model.

    Returns:
        Dict[str, Any]: A dictionary representing the search space of hyperparameters.

    Raises:
        ValueError: If the model type is unknown.
    """

    if model_type == "prophet":
        pass
    elif model_type == "lstm":
        pass
    elif model_type == "xgboost":
        return {
            "learning_rate": hp.choice("learning_rate", [0.0001, 0.001, 0.01, 0.1, 1]),
            "max_depth": hp.choice("max_depth", range(3, 21, 3)),
            "gamma": hp.choice("gamma", [i / 10.0 for i in range(0, 5)]),
            "colsample_bytree": hp.choice(
                "colsample_bytree", [i / 10.0 for i in range(3, 10)]
            ),
            "reg_alpha": hp.choice("reg_alpha", [1e-5, 1e-2, 0.1, 1, 10, 100]),
            "reg_lambda": hp.choice("reg_lambda", [1e-5, 1e-2, 0.1, 1, 10, 100]),
        }
    else:
        raise ValueError("Unknown model type")


def optimize_model(data_config: Dict[str, Any], model_type: str):
    """
    Optimize the model hyperparameters.

    Args:
        config_data (Dict[str, Any]): The configuration parameters for building the model.
        model_name (str): The name of the model to optimize.
    """
    print(f"Optimizing hyperparameters for {model_type} model ...")

    df = load_data(data_config["paths"])

    space = get_space_of_hyperparameters(model_type)

    f = Factory(data_config)
    best = f.bayesian_optim(
        df,
        model_type,
        48,
        parameters=space,
    )

    res = space_eval(space, best)

    print(f"Saving best hyperparameters to {model_type}_best_hyperparameters.json")
    with open(
        f'{pathlib.Path(data_config["paths"]["optim"]).absolute()}/{model_type}_best_hyperparameters.json',
        "w+",
    ) as f:
        json.dump(res, f)

    print("\nOptimization finished")

    print(f"Best hyperparameters for {model_type} model:")
    print(res)
