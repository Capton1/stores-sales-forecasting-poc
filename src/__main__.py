import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys  # noqa
from argparse import ArgumentParser, Namespace

from pipeline.factory import launch_pipeline

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # noqa

from config.get_config import get_config


def main(config_files: Namespace):
    config = get_config(
        optimizer_config=config_files.optimizer_config,
        data_config=config_files.data_config,
        pipeline_config=config_files.pipeline_config,
    )

    launch_pipeline(
        config,
        config_files.model,
        config_files.train,
        config_files.generate,
        save=config_files.save,
        load_model=config_files.load,
        validate=config_files.eval,
        limit=config_files.limit,
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "-c", "--optimizer_config", default="base", help="optimizer config file"
    )
    parser.add_argument("-d", "--data_config", default="base", help="data config file")
    parser.add_argument(
        "-p", "--pipeline_config", default="base", help="pipeline config file"
    )

    parser.add_argument(
        "-m",
        "--model",
        choices=["lstm", "xgboost", "prophet"],
        help="model to use",
    )
    parser.add_argument(
        "--limit",
        default=1000,
        help="limit data to use",
        type=int,
        choices=range(1, 1849447),
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--train", help="train model", action="store_true")
    group.add_argument("-g", "--generate", help="generate data", action="store_true")
    group.add_argument(
        "-e", "--eval", help="launch evaluation pipeline", action="store_true"
    )

    parser.add_argument("-s", "--save", help="save data or model", action="store_true")
    parser.add_argument("-l", "--load", default=None, help="path to model to load")

    args = parser.parse_args()

    if args.train and not args.model:
        raise ValueError("The --train option requires specifying a --model.")

    if sum([args.train, args.generate, args.eval]) > 1:
        raise ValueError(
            "Cannot use both -t/--train, -g/--generate, and -e/--eval options at the same time."
        )

    main(args)
