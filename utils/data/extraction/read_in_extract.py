from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

import utils
from utils.configs.dataset import DatasetConfig
from utils.data.dataframe_format import get_idx_from_event_n_jets
from utils.helpers.run_id_handler import RunIdHandler


def read_in_extract(
    dataset_config: DatasetConfig,
    branches: Optional[Union[str, Tuple[str, ...]]] = None,
    run_id: Optional[str] = None,
    return_all: Optional[bool] = False,
):
    logger.info("Starting read_in_extract")

    if run_id is None:
        run_ids = RunIdHandler.get_run_ids(
            dir_path=utils.paths.extracted_dataset_dir(
                dataset_name=dataset_config.name, run_id="foo", mkdir=False
            ).parent,
            only_latest=True,
        )
        if len(run_ids) == 0:
            raise ValueError(
                f"No extraction for dataset_config {dataset_config.name} found"
            )
        else:
            assert len(run_ids) == 1
            run_id = run_ids[0]
        logger.debug(f"Using latest extraction with 'run_id': {run_id}")
    else:
        logger.debug(f"Using given 'run_id': {run_id}")

    extraction_dir_path = utils.paths.extracted_dataset_dir(
        dataset_name=dataset_config.name,
        run_id=run_id,
        mkdir=False,
    )

    if branches is not None:
        if isinstance(branches, str):
            branches = [branches]
        branches = list(set(branches))

    logger.trace("Starting reading of feather file")
    df = pd.read_feather(
        path=extraction_dir_path / utils.filenames.dataset_extraction_filename,
        columns=branches,
        use_threads=True,
    )

    logger.trace("Starting reading of event_n_jets")
    event_n_jets = np.load(
        file=extraction_dir_path
        / utils.filenames.dataset_extraction_event_n_jets_filename
    )

    logger.trace("Generating index")
    idx = get_idx_from_event_n_jets(event_n_jets=event_n_jets)

    logger.trace("Setting generated index")
    df.index = idx

    logger.trace("Done with read_in_extract")

    if return_all:
        return df, event_n_jets
    else:
        return df
