import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys  # noqa
from argparse import ArgumentParser, Namespace

from pipeline.factory import launch_pipeline
from preprocessing.get_data import process_train_test_data
from preprocessing.prepare_data import prepare_training_data

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # noqa

from config.get_config import get_config


def main(config_files: Namespace):
    config = get_config(optimizer_config=config_files.optimizer_config)

    print(config_files)

    launch_pipeline(config, config_files.model, config_files.train)
    # train, _ = process_train_test_data(save=True)

    # train_set, val_set = prepare_training_data(train, save=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--optimizer_config", default="base", help="optimizer config file"
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        choices=["lstm", "arima", "prophet", "rnn"],
        help="model to use",
    )
    parser.add_argument("-t", "--train", help="train model", action="store_true")

    args = parser.parse_args()

    main(args)
