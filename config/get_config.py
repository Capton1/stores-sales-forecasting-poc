from hydra import compose, initialize


def get_config(
    optimizer_config: str = "base",
    data_config: str = "base",
    pipeline_config: str = "base",
):
    """Get config from Hydra"""
    with initialize(version_base=None, config_path="./"):
        return compose(
            config_name="config",
            overrides=[
                f"optimizer={optimizer_config}",
                f"data={data_config}",
                f"pipeline={pipeline_config}",
            ],
        )
