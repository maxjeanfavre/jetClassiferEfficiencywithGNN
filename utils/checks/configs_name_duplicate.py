from loguru import logger

import utils
from utils.helpers.modules.load_config import load_config
from utils.helpers.modules.package_names import get_package_names


def main():
    configs_dir_path = utils.paths.config(
        config_type="foo", config_name="foo", mkdir=False
    ).parent.parent
    config_types = [p.name for p in configs_dir_path.iterdir() if p.is_dir()]

    for config_type in config_types:
        dir_path = utils.paths.config(
            config_type=config_type, config_name="foo", mkdir=False
        ).parent

        config_file_names = get_package_names(dir_path=dir_path)
        # check that there are no duplicate filenames (thats a given tbf)
        assert len(set(config_file_names)) == len(config_file_names)

        config_names = []
        for config_file_name in config_file_names:
            config = load_config(config_type=config_type, config_name=config_file_name)
            config_names.append(config.name)
            # check that the filename (without the extension)
            # is equal to the Config.name
            assert config_file_name == config.name

        # check that there are no duplicates in the Config.name
        # attribute within a config_type
        assert len(set(config_names)) == len(config_names)

        logger.info(
            f"Configs name duplicate check: Checked configs of type {config_type}"
        )
