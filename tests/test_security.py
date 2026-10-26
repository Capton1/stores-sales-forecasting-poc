import json
import os
import pathlib
import re

import mlflow
import numpy as np
import pytest
from mlflow.exceptions import MlflowException
from sklearn.preprocessing import MinMaxScaler

from models.persistence import (load_model_locally, model_path, run_model_uri,
                                scaler_from_dict, scaler_to_dict)

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name", ["../escape", "nested/model", "/etc/passwd", "../../../tmp/x", "a/../../b"]
)
def test_model_names_cannot_leave_the_save_dir(tmp_path, name):
    with pytest.raises(ValueError):
        model_path("xgboost", str(tmp_path), name)


def test_model_names_with_spaces_and_dates_resolve(tmp_path):
    path = model_path("xgboost", str(tmp_path), "xgboost default - 2024-02-13 08h45")

    assert path == tmp_path.resolve() / "xgboost default - 2024-02-13 08h45.json"


@pytest.mark.parametrize("run_id", ["", "../../etc", "abc", "0" * 32 + "/x", "Z" * 32])
def test_run_uris_require_a_run_id(run_id):
    with pytest.raises(ValueError):
        run_model_uri(run_id)


def test_scaler_round_trips_through_json():
    values = np.array([[10.0], [250.0], [40.0]])
    scaler = MinMaxScaler().fit(values)

    restored = scaler_from_dict(json.loads(json.dumps(scaler_to_dict(scaler))))

    np.testing.assert_allclose(restored.transform(values), scaler.transform(values))
    np.testing.assert_allclose(restored.inverse_transform([[0.5]]), [[130.0]])


def test_mlflow_refuses_pickled_models(mlflow_tracking):
    assert os.environ["MLFLOW_ALLOW_PICKLE_DESERIALIZATION"] == "false"
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            MinMaxScaler().fit([[0.0], [1.0]]), name="scaler", serialization_format="pickle"
        )

    with pytest.raises(MlflowException, match="MLFLOW_ALLOW_PICKLE_DESERIALIZATION"):
        mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/scaler")


def test_source_never_unpickles():
    offenders = [
        str(path.relative_to(REPO))
        for path in (REPO / "src").rglob("*.py")
        if re.search(r"import (cloud)?pickle|pickle\.loads?\(", path.read_text())
    ]

    assert offenders == []


def test_shipped_shap_model_loads_as_json():
    model = load_model_locally(
        "xgboost",
        str(REPO / "models_save" / "xgboost"),
        "xgboost best_params_earthquake - 2024-02-07 14:40",
    )

    assert model.get_booster().num_features() == 22
    assert model.get_booster().feature_names[-1] == "earthquake"
