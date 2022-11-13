import time
from typing import List, Optional

import numpy as np
from loguru import logger

import utils
from utils.configs.dataset import DatasetConfig
from utils.data.dataframe_format import (
    reconstruct_event_n_jets_from_groupby_zeroth_level_values,
)
from utils.data.extraction.read_in_extract import read_in_extract
from utils.data.read_in_root_files import read_in_root_files_in_chunks
from utils.helpers.run_id_handler import RunIdHandler
from utils.logging import set_up_logging_sinks


def extract_dataset(
    dataset_config: DatasetConfig,
    branches: Optional[List[str]] = None,
    num_workers: int = 10,
    chunk_size: int = 10,
    library: str = "np",
):
    run_id_handler = RunIdHandler.new_run_id(prefix="in_progress", bootstrap=False)

    extraction_dir_path = utils.paths.extracted_dataset_dir(
        dataset_name=dataset_config.name,
        run_id=run_id_handler.get_str(),
        mkdir=True,
    )

    set_up_logging_sinks(
        dir_path=extraction_dir_path,
        base_filename=utils.filenames.extraction_log,
    )

    logger.info("Starting extraction")

    if branches is None:
        branches = [
            "run",
            "event",
            "nJet",
            "Jet_pt",
            "Jet_Pt",
            "Jet_eta",
            "Jet_phi",
            "Jet_btagDeepB",
            "Jet_hadronFlavour",
            "Jet_nConstituents",
            "Jet_mass",
            "Jet_area",
        ]

    branches = list(set(branches))

    if dataset_config.branches_to_simulate is not None:
        branches_to_not_read_in = set()
        for branch_to_simulate in dataset_config.branches_to_simulate:
            col = branch_to_simulate["name"]
            branches_to_not_read_in.add(col)
        logger.debug(
            f"Will simulate and therefore not read in the branches: "
            f"{branches_to_not_read_in}"
        )
        branches = list(set(branches) - branches_to_not_read_in)
    #print("branches_to_not_read_in ",branches_to_not_read_in)    

    logger.trace("Starting reading in root files")
    df = read_in_root_files_in_chunks(
        path=dataset_config.path,
        filenames=dataset_config.filenames,
        file_limit=dataset_config.file_limit,
        key=dataset_config.key,
        expressions=branches,
        num_workers=num_workers,
        chunk_size=chunk_size,
        library=library,
    )

    if dataset_config.branches_to_simulate is not None:
        for branch_to_simulate in dataset_config.branches_to_simulate:
            col = branch_to_simulate["name"]
            expr = branch_to_simulate["expression"]
            logger.debug(
                f"Simulating the branch {repr(col)} using the expression {repr(expr)}"
            )
            df[col] = df.eval(expr=expr)

    memory_usage = df.memory_usage(deep=True).sum()
    logger.trace(f"Memory usage of df from root files: {memory_usage / (1024 ** 3)} GB")

    logger.trace("Getting event_n_jets")
    event_n_jets = reconstruct_event_n_jets_from_groupby_zeroth_level_values(df=df)
    print("event_n_jets :",event_n_jets)

    logger.debug("Saving event_n_jets")
    np.save(
        file=extraction_dir_path / utils.filenames.dataset_extraction_event_n_jets,
        arr=event_n_jets,
    )

    logger.debug("Saving feather file")
    df.reset_index(drop=True).to_feather(
        path=extraction_dir_path / utils.filenames.dataset_extraction,
        version=2,
    )

    time.sleep(10)

    df_read_in, event_n_jets_read_in = read_in_extract(
        dataset_config=dataset_config,
        branches=None,
        run_id=run_id_handler.get_str(),
        return_all=True,
    )

    event_n_jets_equal = np.array_equal(a1=event_n_jets, a2=event_n_jets_read_in)
    logger.info(f"Equality event_n_jets: {event_n_jets_equal}")

    memory_usage_read_in = df_read_in.memory_usage(deep=True).sum()
    logger.debug(
        "Memory usage of df read in from feather file: "
        f"{memory_usage_read_in / (1024 ** 3)} GB"
    )

    memory_usage_equal = memory_usage == memory_usage_read_in
    logger.debug(f"Equality memory usage: {memory_usage_equal}")

    df_equal = df.equals(df_read_in)
    logger.debug(f"Equality dfs: {df_equal}")

    if not event_n_jets_equal or not memory_usage_equal or not df_equal:
        logger.error(
            "Result of reading in the extraction does "
            "not fulfill equality requirements"
        )

        run_id_handler.prefix = "failed"
        extraction_dir_path.rename(
            target=utils.paths.extracted_dataset_dir(
                dataset_name=dataset_config.name,
                run_id=run_id_handler.get_str(),
                mkdir=False,
            )
        )
        logger.debug("Renamed result directory")
        logger.error("Failed extraction")

        raise ValueError(
            "Failed extraction. Result of reading in the extraction "
            "does not fulfill equality requirements"
        )
    else:
        run_id_handler.prefix = None
        extraction_dir_path.rename(
            target=utils.paths.extracted_dataset_dir(
                dataset_name=dataset_config.name,
                run_id=run_id_handler.get_str(),
                mkdir=False,
            )
        )
        logger.debug("Renamed result directory")
        logger.info("Successful extraction")
