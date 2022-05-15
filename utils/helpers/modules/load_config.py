import importlib.util


def load_config(config_type: str, config_name: str):
    path = f"configs.{config_type}.{config_name}"
    loader = importlib.util.find_spec(path)

    if loader is None:
        raise Exception(
            f"No config file found for the selected {config_type} '{config_name}'"
        )

    config_module = importlib.import_module(path)
    # this requires that the variables in the config file follow this naming pattern
    config = getattr(config_module, f"{config_type}_config")

    return config

    # this version made problems when pickling the loaded configs
    # adapted from https://stackoverflow.com/a/67692/8162243
    # spec = importlib.util.spec_from_file_location(
    #     name=config_name,
    #     location=utils.paths.config(
    #         config_type=config_type,
    #         config_name=config_name,
    #         mkdir=False,
    #     ),
    # )
    # config_module = importlib.util.module_from_spec(spec)
    # try:
    #     spec.loader.exec_module(config_module)
    # except FileNotFoundError:
    #     raise Exception(
    #         "No config file found for the selected "
    #         f"config_type: {config_type}, "
    #         f"config_name: {config_name}"
    #     )
    #
    # # this requires that the variables in the config file follow this naming pattern
    # config = getattr(config_module, f"{config_type}_config")
    #
    # return config
