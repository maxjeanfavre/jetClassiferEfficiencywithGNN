import json
import pickle
from typing import Optional

from loguru import logger

import utils
from utils.configs.dataset import DatasetConfig
from utils.configs.dataset_handling import DatasetHandlingConfig
from utils.configs.model import ModelConfig
from utils.configs.working_points_set import WorkingPointsSetConfig
from utils.data.jet_events_dataset import JetEventsDataset
from utils.helpers.columns_data_manipulators import (
    added_columns_multiple_data_manipulators,
)
from utils.helpers.log_environment import log_environment
from utils.helpers.run_id_handler import RunIdHandler
from utils.logging import set_up_logging_sinks
from utils.models.model import Model


def train_handler(
    dataset_config: DatasetConfig,
    dataset_handling_config: DatasetHandlingConfig,
    working_points_set_config: WorkingPointsSetConfig,
    model_config: ModelConfig,
    bootstrap: bool,
    run_id: Optional[str] = None,
    trained_model_path: Optional[str]=None,
) -> Model:
    if run_id is not None:
        if bootstrap:
            if not RunIdHandler.is_bootstrap_run_id(run_id=run_id):
                run_id = RunIdHandler.convert_to_bootstrap_run_id(run_id=run_id)
        run_id_handler = RunIdHandler(run_id=run_id, prefix="in_progress")
    else:
        run_id_handler = RunIdHandler.new_run_id(
            prefix="in_progress", bootstrap=bootstrap
        )

    model_dir_path = utils.paths.model_files_dir(
        dataset_name=dataset_config.name,
        dataset_handling_name=dataset_handling_config.name,
        working_points_set_name=working_points_set_config.name,
        model_name=model_config.name,
        run_id=run_id_handler.get_str(),
        mkdir=True,
    )
    print("model_dir_path :",model_dir_path) #/work/krgedia/CMSSW_10_1_0/src/Xbb/python/gnn_b_tagging_efficiency/data/QCD_Pt_300_470_MuEnrichedPt5/pt_30_1000_eta_25_cut_dataset_handling/standard_working_points_set/models/direct_tagging/in_progress_20220902_101752_261253__df8f9054-18f1-452d-8b9c-bb7a2e2430f8
    set_up_logging_sinks(
        dir_path=model_dir_path,
        base_filename=utils.filenames.train_log,
    )

     
    logger.info("Starting 'train_handler'")

    log_environment()

    if run_id is not None:
        logger.debug(f"Using supplied run_id: {run_id}")
    logger.debug(f"run_id: {run_id_handler.run_id}")

    model = model_config.model_cls(
        working_points_set_config=working_points_set_config,
        **model_config.model_init_kwargs,
    )
    print("**model_config.model_init_kwargs ", model_config.model_init_kwargs)
    branches_to_read_in = tuple(
        (
            {"run", "event", "nJet"}  # always include these
            .union(set(dataset_handling_config.get_required_columns()))
            .union(set(working_points_set_config.get_required_columns()))
            .union(set(model_config.get_required_columns()))
            .union(set(model.get_required_columns()))
        )
        - set(
            added_columns_multiple_data_manipulators(
                data_manipulators=(
                    *dataset_handling_config.data_manipulators,
                    *model_config.data_manipulators,
                )
            )
        )
    )

    print(set(dataset_handling_config.get_required_columns()))
    print(set(working_points_set_config.get_required_columns()))
    print(set(model_config.get_required_columns()))
    print(set(model.get_required_columns()))
    print(set(                                                                                                                                                                                                     added_columns_multiple_data_manipulators(                                                                                                                                                                  data_manipulators=(                                                                                                                                                                                        *dataset_handling_config.data_manipulators,                                                                                                                                                            *model_config.data_manipulators,                                                                                                                                                                   )                                                                                                                                                                                                  )))

    jds = JetEventsDataset.read_in(
        dataset_config=dataset_config,
        branches=branches_to_read_in,
    )

    jds_train = jds.split_data(
        train_size=dataset_handling_config.train_split,
        test_size=dataset_handling_config.test_split,
        return_train=True,
        return_test=False,
        random_state=dataset_handling_config.train_test_split_random_state,
    )

    del jds

    if bootstrap is True:
        n_events_before = jds_train.n_events
        jds_train = jds_train.get_bootstrap_sample(sample_size=jds_train.n_events)
        n_events_after = jds_train.n_events
        assert n_events_before == n_events_after

    data_manipulators_mode = "train"

    logger.trace(
        f"Data Manipulators from dataset_handling_config in mode: "
        f"{data_manipulators_mode}"
    )
    jds_train.manipulate(
        data_manipulators=dataset_handling_config.data_manipulators,
        mode=data_manipulators_mode,
    )

    logger.trace(
        f"Data Manipulators from model_config in mode: " f"{data_manipulators_mode}"
    )
    jds_train.manipulate(
        data_manipulators=model_config.data_manipulators,
        mode=data_manipulators_mode,
    )

    model.train(
        jds=jds_train,
        path_to_save=model_dir_path,
        **model_config.model_train_kwargs,
        trained_model_path = trained_model_path
    )

    del jds_train

    model.save(path=model_dir_path)

    with open(model_dir_path / utils.filenames.dataset_config_pickle, "wb") as f:
        pickle.dump(obj=dataset_config, file=f)

    with open(model_dir_path / utils.filenames.dataset_config_json, "w") as f:
        json.dump(obj=dataset_config.__dict__, fp=f)

    with open(model_dir_path / utils.filenames.model_config_pickle, "wb") as f:
        pickle.dump(obj=model_config, file=f)

    # with open(model_dir_path / utils.model_config_json_filename, "w") as f:
    #     json.dump(obj=model_config.__dict__, fp=f)

    logger.info("Training done")

    run_id_handler.prefix = None
    model_dir_path.rename(
        target=utils.paths.model_files_dir(
            dataset_name=dataset_config.name,
            dataset_handling_name=dataset_handling_config.name,
            working_points_set_name=working_points_set_config.name,
            model_name=model_config.name,
            run_id=run_id_handler.get_str(),
            mkdir=False,
        )
    )
    logger.info("Renamed result directory")

    return model
