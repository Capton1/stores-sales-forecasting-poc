from hydra import compose, initialize


def get_config(
    data_config: str = "base",
    pipeline_config: str = "base",
):
    """Get config from Hydra"""
    with initialize(version_base=None, config_path="./"):
        return compose(
            config_name="config",
            overrides=[
                f"data={data_config}",
                f"pipeline={pipeline_config}",
            ],
        )
