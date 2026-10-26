import os
import pathlib

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

N_DAYS = 400
DATA_CONFIG = pathlib.Path(__file__).resolve().parents[1] / "config" / "data" / "base.yaml"


def _write_raw_data(raw_dir):
    """Write a small dataset with the schema of the Kaggle competition files."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-01-01", periods=N_DAYS, freq="D")

    rows = []
    for day, date in enumerate(dates):
        # the stores close on Christmas, so that date is missing from train.csv
        if date.strftime("%m-%d") == "12-25":
            continue
        for store in (1, 2):
            for family in ("GROCERY I", "BEVERAGES"):
                base = 200 + 50 * np.sin(day / 7) + (30 if family == "GROCERY I" else 0)
                rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "store_nbr": store,
                        "family": family,
                        "sales": max(base + rng.normal(0, 10), 0),
                        "onpromotion": int(rng.integers(0, 5)),
                    }
                )
    train = pd.DataFrame(rows)
    train.insert(0, "id", range(len(train)))
    train.to_csv(raw_dir / "train.csv", index=False)

    # oil prices exist on weekdays only, with some gaps
    oil = pd.DataFrame({"date": dates[dates.dayofweek < 5].strftime("%Y-%m-%d")})
    oil["dcoilwtico"] = np.linspace(90, 50, len(oil))
    oil.loc[oil.index % 17 == 0, "dcoilwtico"] = np.nan
    oil.to_csv(raw_dir / "oil.csv", index=False)

    pd.DataFrame(
        {
            "date": ["2015-01-01", "2015-02-16", "2015-04-03", "2015-05-01", "2015-05-11", "2015-08-10", "2015-12-24"],
            "type": ["Holiday", "Holiday", "Holiday", "Holiday", "Event", "Transfer", "Additional"],
            "locale": ["National"] * 7,
            "locale_name": ["Ecuador"] * 7,
            "description": ["a", "b", "c", "d", "e", "f", "g"],
            "transferred": [False, False, False, True, False, False, False],
        }
    ).to_csv(raw_dir / "holidays_events.csv", index=False)

    pd.DataFrame(
        {
            "store_nbr": [1, 2],
            "city": ["Quito", "Guayaquil"],
            "state": ["Pichincha", "Guayas"],
            "type": ["D", "A"],
            "cluster": [13, 4],
        }
    ).to_csv(raw_dir / "stores.csv", index=False)


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    _write_raw_data(raw)
    return raw


@pytest.fixture
def data_config(raw_dir, tmp_path):
    """The repository's data config, reading the synthetic raw files and writing under `tmp_path`."""
    config = OmegaConf.to_container(OmegaConf.load(DATA_CONFIG))
    paths = config["paths"]
    for key, path in paths["raw"].items():
        paths["raw"][key] = str(raw_dir / pathlib.PurePath(path).name)
    for group in ("processed", "trusted", "models", "logs"):
        for key, path in paths[group].items():
            paths[group][key] = str(tmp_path / path)
    paths["optim"] = str(tmp_path / paths["optim"])
    return config


@pytest.fixture
def mlflow_tracking(tmp_path):
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment("tests", artifact_location=(tmp_path / "mlartifacts").as_uri())
    experiment = mlflow.set_experiment("tests")
    yield experiment
    mlflow.end_run()
