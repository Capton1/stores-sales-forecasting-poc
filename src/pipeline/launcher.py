import math
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict

import mlflow
from sklearn.metrics import mean_squared_error

from pipeline.factory import Factory
from pipeline.helpers import convert_config_to_dict, get_config_from_model_name
from pipeline.optim import optimize_model
from preprocessing.process_raw_data import load_data


def generate_datasets(data_config: Dict[str, Any], save: bool = False) -> Factory:
    """Generates train and validation sets.

    Args:
        data_path (Dict[str, Any]): paths to data
        save (bool, optional): whether to save the data. Defaults to False.
    """
    print("Generating datasets ...")

    df = load_data(data_config["paths"])
    f = Factory(data_config)
    f.fit(df)

    if save:
        f.save_data()

    return f


def generate_models(
    data_config: Dict[str, Any],
    model_config: Dict[str, Any],
    model_config_name: str,
    save: bool = False,
    load_model_name: str = None,
    save_name: str = None,
) -> Factory:
    """Generates models.

    Args:
        data_path (Dict[str, Any]): paths to data
        save (bool, optional): whether to save the data. Defaults to False.
    """
    df = load_data(data_config["paths"])
    f = Factory(data_config)

    if save_name:
        f.set_save_name(save_name)

    f.fit(
        df,
        model_config=model_config,
        model_config_name=model_config_name,
        load_model_name=load_model_name,
    )

    if save:
        f.save_model()

    return f


def evaluate_model(
    f: Factory, model_type: str, model_name: str, model_config: Dict[str, Any]
) -> float:
    """
    Evaluates the given model.

    Args:
        f (Factory): The factory object containing the model.
        model_type (str): The type of the model.
        model_name (str): The name of the model.

    Returns:
        float: The mean squared error of the model.
    """
    print(f"Evaluating model {model_name} ...")

    model_config["save_path"] = f.data_config["paths"]["models"][model_type]
    if "scaled" not in model_config:
        model_config["scaled"] = False

    id = (
        mlflow.search_registered_models(filter_string=f"name = '{model_name}'")[0]
        .latest_versions[0]
        .run_id
    )

    f.load_model(id, model_config)
    y_val = f.get_y_val()
    pred = f.predict()

    mse = round(mean_squared_error(y_val, pred), 2)

    return mse


def validation_pipeline(models_to_validate: Dict[str, Any], config: Dict[str, Any]):
    """
    Runs the validation pipeline for the given models and configuration.

    Args:
        models_to_validate (dict): A dictionary containing the models to validate, grouped by model type.
        config (dict): The configuration settings for the pipeline.

    Returns:
        None
    """
    data_config = config["data"]

    filename = f'Validation - {str(datetime.now(timezone.utc)).split(".")[0].rsplit(":", maxsplit=1)[0]}'
    print(f"Writing validation results in `{filename}` ... \n")

    df = load_data(data_config["paths"])
    f = Factory(data_config)
    f.fit(df)

    min_mse, best_type, best_model = math.inf, None, None
    with open(
        f'{pathlib.Path(data_config["paths"]["logs"]["validation"]).absolute()}/{filename}',
        "w+",
    ) as file:
        for model_type, models in models_to_validate.items():
            if models is None:
                continue
            for model_name in models:
                model_config = get_config_from_model_name(
                    model_name, model_type, config
                )
                mse = evaluate_model(f, model_type, model_name, model_config)
                file.write(f"Type: {model_type} - Name: {model_name} - MSE: {mse} \n")

                if mse < min_mse:
                    min_mse, best_type, best_model = mse, model_type, model_name

    print("\nValidation pipeline is done")
    print(f"Best model is {best_type} {best_model} with mse: {min_mse}")


def launch_pipeline(
    config,
    model,
    model_config_name,
    train: bool = False,
    generate: bool = False,
    validate: bool = False,
    save: bool = False,
    load_model_name: str = None,
    limit: int = 1520,
    experiment_id: int = 0,
    perform_optimization: bool = False,
):
    """Launches the pipeline."""
    print("Launching pipeline ...")

    config = convert_config_to_dict(config)

    if generate:
        generate_datasets(config["data"], save=save)

    if train:
        if model == "all":
            mlflow.start_run(experiment_id=experiment_id)
            print("Training all models ...")
            for i in ["lstm", "xgboost", "prophet"]:
                for j in config[i]:
                    print(f"Training {i} {j} ...")
                    save_name = f'{i} {j} - {str(datetime.now(timezone.utc)).split(".")[0].rsplit(":", maxsplit=1)[0]}'
                    with mlflow.start_run(nested=True, run_name=save_name):
                        generate_models(
                            config["data"],
                            config[i][j],
                            model_config_name=j,
                            save=save,
                            save_name=save_name,
                        )
        else:
            save_name = f'{model} {model_config_name} - {str(datetime.now(timezone.utc)).split(".")[0].rsplit(":", maxsplit=1)[0]}'
            if save:
                mlflow.start_run(experiment_id=experiment_id, run_name=save_name)

            generate_models(
                config["data"],
                config[model][model_config_name],
                model_config_name=model_config_name,
                save=save,
                load_model_name=load_model_name,
                save_name=save_name,
            )

    if validate:
        validation_pipeline(config["pipeline"]["validation"]["models"], config)

    if perform_optimization:
        optimize_model(config["data"], model)
