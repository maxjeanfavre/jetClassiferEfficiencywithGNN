from loguru import logger

import utils
from utils.configs.dataset import DatasetConfig
from utils.configs.dataset_handling import DatasetHandlingConfig
from utils.configs.model import ModelConfig
from utils.configs.prediction_dataset_handling import PredictionDatasetHandlingConfig
from utils.configs.working_points_set import WorkingPointsSetConfig
from utils.data.jet_events_dataset import JetEventsDataset
from utils.helpers.columns_data_manipulators import (
    added_columns_multiple_data_manipulators,
)
from utils.helpers.model_predictions import ModelPredictions
from utils.logging import set_up_logging_sinks


def save_predictions_handler(
    dataset_config: DatasetConfig,
    dataset_handling_config: DatasetHandlingConfig,
    working_points_set_config: WorkingPointsSetConfig,
    model_config: ModelConfig,
    run_id: str,
    prediction_dataset_handling_config: PredictionDatasetHandlingConfig,
):
    model_dir_path = utils.paths.model_files_dir(
        dataset_name=dataset_config.name,
        dataset_handling_name=dataset_handling_config.name,
        working_points_set_name=working_points_set_config.name,
        model_name=model_config.name,
        run_id=run_id,
        mkdir=False,
    )

    set_up_logging_sinks(
        dir_path=model_dir_path,
        base_filename=utils.filenames.save_prediction_log_filename(
            prediction_dataset_handling_config_name=prediction_dataset_handling_config.name
        ),
    )

    logger.info("Starting 'save_predictions_handler'")

    model = model_config.model_cls.load(path=model_dir_path)

    branches_to_read_in = tuple(
        (
            {"run", "event", "nJet"}  # always include these
            .union(set(dataset_handling_config.get_required_columns()))
            .union(set(prediction_dataset_handling_config.get_required_columns()))
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

    jds = JetEventsDataset.read_in(
        dataset_config=dataset_config,
        branches=branches_to_read_in,
    )

    jds_test = jds.split_data(
        train_size=dataset_handling_config.train_split,
        test_size=dataset_handling_config.test_split,
        return_train=False,
        return_test=True,
        random_state=dataset_handling_config.train_test_split_random_state,
    )

    del jds

    data_manipulators_mode = "predict"

    logger.trace(
        f"Data Manipulators from dataset_handling_config in mode: "
        f"{data_manipulators_mode}"
    )
    jds_test.manipulate(
        data_manipulators=dataset_handling_config.data_manipulators,
        mode=data_manipulators_mode,
    )

    logger.trace(
        f"Data Manipulators from prediction_dataset_handling_config in mode: "
        f"{data_manipulators_mode}"
    )
    jds_test.manipulate(
        data_manipulators=prediction_dataset_handling_config.data_manipulators,
        mode=data_manipulators_mode,
    )

    logger.trace(
        f"Data Manipulators from model_config in mode: " f"{data_manipulators_mode}"
    )
    jds_test.manipulate(
        data_manipulators=model_config.data_manipulators,
        mode=data_manipulators_mode,
    )

    predictions = model.predict(
        jds=jds_test,
        working_point_configs=list(working_points_set_config.working_points),
        **model_config.model_predict_kwargs,
    )
    assert len(predictions) == len(working_points_set_config.working_points)
    logger.debug("Got predictions for all working points.")

    for working_point_config, working_point_predictions in zip(
        working_points_set_config.working_points, predictions
    ):
        mp = ModelPredictions(
            res=working_point_predictions[0], err=working_point_predictions[1]
        )

        mp.save(
            dir_path=model_dir_path,
            filename=utils.filenames.model_prediction_filename(
                working_point_name=working_point_config.name,
                prediction_dataset_handling_name=prediction_dataset_handling_config.name,
            ),
            event_n_jets=jds_test.event_n_jets,
        )

        logger.debug(
            f"Saved predictions of working point: {working_point_config.name}."
        )

    logger.info("Done with saving predictions.")
