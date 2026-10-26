import numpy as np
import pandas as pd

from models.training_helpers import (TimeseriesGenerator, build_lstm_generator,
                                     generate_ml_features, get_prophet_df)
from preprocessing.helpers import train_val_split
from preprocessing.process_raw_data import load_data, process_data
from tests.conftest import N_DAYS


def test_process_data_gives_one_complete_row_per_day(data_config):
    df = process_data(load_data(data_config["paths"]), data_config)

    assert len(df) == N_DAYS
    assert df.index.is_monotonic_increasing
    assert (df.index.to_series().diff().dropna() == pd.Timedelta("1D")).all()
    assert list(df.columns) == [
        "onpromotion",
        "dcoilwtico",
        "cluster",
        "sales",
        "typedays_Additional",
        "typedays_Event",
        "typedays_Holiday",
        "typedays_Normal",
    ]
    assert not df.isna().any().any()
    for column in ("onpromotion", "dcoilwtico", "cluster"):
        assert df[column].between(0, 1).all()


def test_process_data_maps_holiday_types(data_config):
    df = process_data(load_data(data_config["paths"]), data_config)

    # a transferred holiday is a normal day, a Transfer day is the holiday
    assert df.loc["2015-05-01", "typedays_Normal"]
    assert df.loc["2015-08-10", "typedays_Holiday"]
    # Christmas has no sales rows and is filled in as a normal day
    assert df.loc["2015-12-25", "typedays_Normal"]


def test_process_data_leaves_config_untouched(data_config):
    features = list(data_config["features"])

    process_data(load_data(data_config["paths"]), data_config)

    assert data_config["features"] == features


def test_ml_features_and_prophet_df(data_config):
    df = process_data(load_data(data_config["paths"]), data_config)
    train, val = train_val_split(df, val_ratio=0.9)

    X, y = generate_ml_features(train, earthquake=True)
    assert len(X) == len(y) == len(train)
    assert {"day_of_week", "season", "sales_mov_avg", "earthquake"} <= set(X.columns)
    assert "sales" not in X.columns

    prophet_df = get_prophet_df(train, val)
    assert list(prophet_df.columns) == ["ds", "y"]
    assert len(prophet_df) == len(df)
    assert "date" not in train.columns


def test_timeseries_generator_windows():
    data = [[i, 10 * i] for i in range(10)]
    targets = [[100 * i] for i in range(10)]
    gen = TimeseriesGenerator(np.array(data), np.array(targets), length=3, batch_size=2)

    assert len(gen) == 4
    X, y = gen[0]
    assert X.shape == (2, 3, 2)
    assert X[0].tolist() == [[0, 0], [1, 10], [2, 20]]
    assert y.tolist() == [[300], [400]]
    # the last batch is partial
    X, y = gen[3]
    assert y.tolist() == [[900]]
    assert len(list(gen)) == 4


def test_lstm_generators_shapes(data_config):
    df = process_data(load_data(data_config["paths"]), data_config)

    gen, shape = build_lstm_generator(df, {"type": "simple", "look_back": 5})
    assert shape == (5, 7)
    assert gen[0][0].shape == (1, 5, 7)

    gen, shapes = build_lstm_generator(df, {"type": "multivariate", "look_back": 5})
    assert shapes == ((5, 3), (5, 4))
    (continuous, categorical), y = gen[0]
    assert continuous.shape == (1, 5, 3)
    assert categorical.shape == (1, 5, 4)
    assert y.shape == (1, 1)
