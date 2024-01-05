import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys  # noqa
from argparse import ArgumentParser, Namespace

from pipeline.factory import launch_pipeline

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # noqa

from config.get_config import get_config


def main(config_files: Namespace):
    config = get_config(optimizer_config=config_files.optimizer_config, data_config=config_files.data_config)

    launch_pipeline(config, config_files.model, config_files.train, config_files.generate, save=config_files.save)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--optimizer_config", default="base", help="optimizer config file")
    parser.add_argument("-d", "--data_config", default="base", help="data config file")
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        choices=["lstm", "arima", "prophet", "rnn"],
        help="model to use",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--train", help="train model", action="store_true")
    group.add_argument("-g", "--generate", help="generate data", action="store_true")
    parser.add_argument("-s", "--save", help="save data or model", action="store_true")

    args = parser.parse_args()

    if args.train and args.generate:
        raise ValueError("Cannot use both -t/--train and -g/--generate options at the same time.")

    main(args)
