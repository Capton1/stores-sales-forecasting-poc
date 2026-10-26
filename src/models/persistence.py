import os
import pathlib
import re
from functools import partial
from typing import Any, Dict

import keras
import numpy as np
from prophet.serialize import model_from_json, model_to_json
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# MLflow otherwise deserializes pickle and skops models, which can run code
# from the artifact. The flavors used here load without it.
os.environ.setdefault("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", "false")

_RUN_ID = re.compile(r"[0-9a-f]{32}")
_SCALER_ARRAYS = ("min_", "scale_", "data_min_", "data_max_", "data_range_")


def _load_xgboost(path: pathlib.Path) -> XGBRegressor:
    model = XGBRegressor()
    model.load_model(path)
    return model


# (suffix, save, load) per model type. Every format is data-only: loading a
# model never executes code from the file.
_FORMATS = {
    "lstm": (
        ".keras",
        keras.Model.save,
        partial(keras.models.load_model, safe_mode=True),
    ),
    "xgboost": (".json", XGBRegressor.save_model, _load_xgboost),
    "prophet": (
        ".json",
        lambda model, path: path.write_text(model_to_json(model)),
        lambda path: model_from_json(path.read_text()),
    ),
}


def model_path(model_type: str, save_dir: str, model_name: str) -> pathlib.Path:
    """
    Get the path of a locally saved model.

    Args:
        model_type (str): The type of the model.
        save_dir (str): The directory holding models of this type.
        model_name (str): The model name, without extension.

    Returns:
        pathlib.Path: The model file path, directly inside `save_dir`.

    Raises:
        ValueError: If the name would place the file anywhere else.
    """
    base = pathlib.Path(save_dir).resolve()
    path = (base / f"{model_name}{_FORMATS[model_type][0]}").resolve()
    if path.parent != base:
        raise ValueError(f"Model name {model_name!r} resolves outside {base}")
    return path


def save_model_locally(
    model: Any, model_type: str, save_dir: str, model_name: str
) -> pathlib.Path:
    """Save a model in its library's native format and return its path."""
    path = model_path(model_type, save_dir, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    _FORMATS[model_type][1](model, path)
    return path


def load_model_locally(model_type: str, save_dir: str, model_name: str) -> Any:
    """Load a model saved by `save_model_locally`."""
    return _FORMATS[model_type][2](model_path(model_type, save_dir, model_name))


def run_model_uri(run_id: str, artifact: str = "model") -> str:
    """
    Get the MLflow URI of an artifact logged in a run.

    Raises:
        ValueError: If `run_id` is not an MLflow run id.
    """
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError(f"{run_id!r} is not an MLflow run id")
    return f"runs:/{run_id}/{artifact}"


def scaler_to_dict(scaler: MinMaxScaler) -> Dict[str, Any]:
    """Serialize a fitted MinMaxScaler to JSON-compatible values."""
    state = {
        "feature_range": list(scaler.feature_range),
        "n_features_in_": int(scaler.n_features_in_),
        "n_samples_seen_": int(scaler.n_samples_seen_),
        **{name: getattr(scaler, name).tolist() for name in _SCALER_ARRAYS},
    }
    if hasattr(scaler, "feature_names_in_"):
        state["feature_names_in_"] = scaler.feature_names_in_.tolist()
    return state


def scaler_from_dict(state: Dict[str, Any]) -> MinMaxScaler:
    """Rebuild a fitted MinMaxScaler from `scaler_to_dict` output."""
    scaler = MinMaxScaler(feature_range=tuple(state["feature_range"]))
    for name in _SCALER_ARRAYS:
        setattr(scaler, name, np.asarray(state[name], dtype=np.float64))
    scaler.n_features_in_ = state["n_features_in_"]
    scaler.n_samples_seen_ = state["n_samples_seen_"]
    if "feature_names_in_" in state:
        scaler.feature_names_in_ = np.asarray(state["feature_names_in_"], dtype=object)
    return scaler
