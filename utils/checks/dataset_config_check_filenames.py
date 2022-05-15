import pathlib
import re

import natsort
from loguru import logger

import utils
from utils.configs.dataset import DatasetConfig
from utils.helpers.modules.load_config import load_config
from utils.helpers.modules.package_names import get_package_names


def check_dataset_config_filenames(dataset: str):
    dataset_config: DatasetConfig = load_config(
        config_type="dataset", config_name=dataset
    )

    path = dataset_config.path

    eos_prefix = "root://eoscms.cern.ch/"
    if path.startswith(eos_prefix):
        path = path.replace(eos_prefix, "")

    path = pathlib.Path(path)

    try:
        if not path.exists():
            logger.error(f"path does not exist: {path}")
            return 0
    except PermissionError:
        logger.error(f"PermissionError when accessing {path}")
        if str(path).startswith("/eos"):
            logger.error(
                "eos is not available. "
                "Use 'kinit cern_username@CERN.CH' and "
                "'eosfusebind -g krb5 $HOME/krb5cc_$UID'"
            )
        return 0

    if not path.is_dir():
        logger.error(f"path is not a directory: {path}")
    else:
        filenames = [
            i.name
            for i in path.iterdir()
            if i.is_file() and re.search(dataset_config.filename_pattern, i.name)
        ]

        sorted_filenames = natsort.natsorted(filenames)

        logger.debug(
            f"Found {len(sorted_filenames)} files for dataset {dataset_config.name} "
            f"at {dataset_config.path}"
        )

        if sorted_filenames != list(dataset_config.filenames):
            logger.error("Existing filename list is different than the one just found")
            print(sorted_filenames)
        else:
            logger.info(
                "dataset config filenames check: "
                f"Checked dataset {dataset_config.name}. "
                f"Filename list already up to date"
            )


def main():
    dir_path = utils.paths.config(
        config_type="dataset", config_name="foo", mkdir=False
    ).parent

    dataset_config_names = get_package_names(dir_path=dir_path)

    for dataset_config_name in dataset_config_names:
        check_dataset_config_filenames(dataset=dataset_config_name)
