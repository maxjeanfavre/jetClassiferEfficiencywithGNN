import copy
import json
import pickle
import shutil
import sys
from typing import List

import numpy as np
import pandas as pd
from loguru import logger

import utils
from utils.configs.dataset import DatasetConfig
from utils.configs.dataset_handling import DatasetHandlingConfig
from utils.configs.evaluation_data_manipulation import EvaluationDataManipulationConfig
from utils.configs.evaluation_model_selection import EvaluationModelSelectionConfig
from utils.configs.prediction_dataset_handling import PredictionDatasetHandlingConfig
from utils.configs.working_points_set import WorkingPointsSetConfig
from utils.data.jet_events_dataset import JetEventsDataset
from utils.evaluation.jet_variable_histogram import create_jet_variable_histogram
from utils.evaluation.leading_subleading_histograms import (
    create_leading_subleading_histograms,
)
from utils.evaluation.predictions_histogram import create_predictions_histogram
from utils.evaluation.predictions_loss import compute_predictions_loss
from utils.helpers.columns_not_present_yet import check_columns_not_present_yet
from utils.helpers.hash_dataframe import hash_df
from utils.helpers.model_predictions import ModelPredictions
from utils.helpers.run_id_handler import RunIdHandler


def evaluation_handler(
    dataset_config: DatasetConfig,
    dataset_handling_config: DatasetHandlingConfig,
    working_points_set_config: WorkingPointsSetConfig,
    prediction_dataset_handling_config: PredictionDatasetHandlingConfig,
    evaluation_model_selection_config: EvaluationModelSelectionConfig,
    evaluation_data_manipulation_configs: List[EvaluationDataManipulationConfig],
):
    # TODO(medium): set up logging
    # set_up_logging_sinks(
    #     dir_path=model_dir_path,
    #     base_filename=utils.filenames.train_log_filename,
    # )
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS!UTC}</green> | "
        "{elapsed} | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.remove(None)  # removes all existing handlers
    logger.add(
        sink=sys.stdout,
        level="TRACE",
        format=log_format,
        colorize=True,
    )

    logger.info("Evaluation starts")

    run_id_handler = RunIdHandler.new_run_id(prefix="in_progress", bootstrap=False)

    logger.debug(f"run_id: {run_id_handler.run_id}")

    jds = JetEventsDataset.read_in(
        dataset_config=dataset_config,
        branches=None,
    )

    jds_test = jds.split_data(
        train_size=dataset_handling_config.train_split,
        test_size=dataset_handling_config.test_split,
        return_train=False,
        return_test=True,
        random_state=dataset_handling_config.train_test_split_random_state,
    )

    del jds

    jds_test.manipulate(
        data_manipulators=dataset_handling_config.data_manipulators,
        mode="predict",
    )

    jds_test.manipulate(
        data_manipulators=prediction_dataset_handling_config.data_manipulators,
        mode="predict",
    )

    jds_test_df = jds_test.df
    jds_test_df_hash = hash_df(jds_test.df)
    jds_test_n_events = copy.deepcopy(jds_test.n_events)
    jds_test_n_jets = copy.deepcopy(jds_test.n_jets)
    jds_test_event_n_jets = copy.deepcopy(jds_test.event_n_jets)
    jds_test_df_columns = copy.deepcopy(jds_test.df.columns)

    del jds_test

    for working_point_config in working_points_set_config.working_points:
        jds_test = JetEventsDataset(df=jds_test_df.copy(deep=True))

        assert hash_df(jds_test.df) == jds_test_df_hash
        assert jds_test.n_events == jds_test_n_events
        assert jds_test.n_jets == jds_test_n_jets
        assert np.array_equal(jds_test.event_n_jets, jds_test_event_n_jets)
        assert np.array_equal(jds_test.df.columns, jds_test_df_columns)

        eff_pred_cols = []
        eff_err_cols = []

        comparison_pred_col = None

        for (
            evaluation_model_config
        ) in evaluation_model_selection_config.evaluation_model_configs:
            if evaluation_model_config.run_selection == "only_latest":
                only_latest = True
            elif evaluation_model_config.run_selection == "all":
                only_latest = False
            else:
                raise ValueError(
                    "Unsupported value for 'run_selection': "
                    f"{evaluation_model_config.run_selection}"
                )

            model_run_ids = RunIdHandler.get_run_ids(
                dir_path=utils.paths.model_files_dir(
                    dataset_name=dataset_config.name,
                    dataset_handling_name=dataset_handling_config.name,
                    working_points_set_name=working_points_set_config.name,
                    model_name=evaluation_model_config.model_config.name,
                    run_id="foo",
                    mkdir=False,
                ).parent,
                only_latest=only_latest,
                only_bootstrap=evaluation_model_config.only_bootstrap_runs,
            )
            if len(model_run_ids) == 0:
                logger.warning(
                    "No model runs found for model_config "
                    f"{evaluation_model_config.model_config.name}"
                )
                continue

            if evaluation_model_config.run_aggregation == "individual":
                for i, model_run_id in enumerate(model_run_ids):
                    model_dir_path = utils.paths.model_files_dir(
                        dataset_name=dataset_config.name,
                        dataset_handling_name=dataset_handling_config.name,
                        working_points_set_name=working_points_set_config.name,
                        model_name=evaluation_model_config.model_config.name,
                        run_id=model_run_id,
                        mkdir=True,
                    )

                    mp = ModelPredictions.load(
                        dir_path=model_dir_path,
                        filename=utils.filenames.model_prediction(
                            working_point_name=working_point_config.name,
                            prediction_dataset_handling_name=prediction_dataset_handling_config.name,
                        ),
                    )

                    assert mp.res.index.equals(jds_test.df.index)
                    assert mp.err.index.equals(jds_test.df.index)

                    if evaluation_model_config.display_name is not None:
                        base_name = f"{evaluation_model_config.display_name}"
                        if len(model_run_ids) > 1:
                            base_name += f" ({i + 1})"
                    else:
                        base_name = f"{evaluation_model_config.model_config.name}_{model_run_id}"
                    pred_col = f"{base_name}"
                    err_col = f"err_{base_name}"

                    check_columns_not_present_yet(
                        df=jds_test.df, cols=(pred_col, err_col)
                    )

                    jds_test.df[pred_col] = mp.res
                    jds_test.df[err_col] = mp.err

                    eff_pred_cols.append(pred_col)
                    eff_err_cols.append(err_col)
            elif evaluation_model_config.run_aggregation in ("mean", "median"):
                run_preds_df = pd.DataFrame(index=jds_test.df.index)

                for model_run_id in model_run_ids:
                    model_dir_path = utils.paths.model_files_dir(
                        dataset_name=dataset_config.name,
                        dataset_handling_name=dataset_handling_config.name,
                        working_points_set_name=working_points_set_config.name,
                        model_name=evaluation_model_config.model_config.name,
                        run_id=model_run_id,
                        mkdir=True,
                    )

                    mp = ModelPredictions.load(
                        dir_path=model_dir_path,
                        filename=utils.filenames.model_prediction(
                            working_point_name=working_point_config.name,
                            prediction_dataset_handling_name=prediction_dataset_handling_config.name,
                        ),
                    )

                    assert mp.res.index.equals(jds_test.df.index)
                    assert mp.err.index.equals(jds_test.df.index)

                    run_preds_df[model_run_id] = mp.res

                # get mean and standard deviation of the predictions

                # a = (
                #     run_preds_df.sub(run_preds_df.mean(axis=1), axis=0)
                #     .div(run_preds_df.mean(axis=1), axis=0)
                #     .describe()
                # )

                if evaluation_model_config.display_name is not None:
                    base_name = f"{evaluation_model_config.display_name}"
                else:
                    base_name = f"{evaluation_model_config.model_config.name}"

                if evaluation_model_config.run_aggregation == "mean":
                    pred_col = (
                        f"{base_name} "
                        f"(mean of {len(model_run_ids)} "
                        f"{'bootstrap ' if evaluation_model_config.only_bootstrap_runs else ''}"
                        f"runs)"
                    )
                    err_col = f"err_{pred_col}"

                    check_columns_not_present_yet(
                        df=jds_test.df, cols=(pred_col, err_col)
                    )

                    jds_test.df[pred_col] = run_preds_df.mean(axis=1)
                    jds_test.df[err_col] = run_preds_df.std(axis=1)
                elif evaluation_model_config.run_aggregation == "median":
                    pred_col = (
                        f"{base_name} "
                        f"(median of {len(model_run_ids)} "
                        f"{'bootstrap ' if evaluation_model_config.only_bootstrap_runs else ''}"
                        f"runs)"
                    )
                    err_col = f"err_{pred_col}"

                    check_columns_not_present_yet(
                        df=jds_test.df, cols=(pred_col, err_col)
                    )

                    jds_test.df[pred_col] = run_preds_df.median(axis=1)
                    jds_test.df[err_col] = run_preds_df.std(axis=1)
                else:
                    raise ValueError("You shouldn't reach this point")

                eff_pred_cols.append(pred_col)
                eff_err_cols.append(err_col)

                del run_preds_df
            else:
                raise ValueError(
                    "Unsupported value for 'run_aggregation': "
                    f"{evaluation_model_config.run_aggregation}"
                )

            if evaluation_model_config.is_comparison_base is True:
                if comparison_pred_col is not None:
                    raise ValueError(
                        "More than one model setting was marked as comparison model"
                    )
                comparison_pred_col = pred_col

        jds_test_with_predictions_df = jds_test.df
        jds_test_with_predictions_df_hash = hash_df(jds_test.df)
        jds_test_with_predictions_n_events = copy.deepcopy(jds_test.n_events)
        jds_test_with_predictions_n_jets = copy.deepcopy(jds_test.n_jets)
        jds_test_with_predictions_event_n_jets = copy.deepcopy(jds_test.event_n_jets)
        jds_test_with_predictions_df_columns = copy.deepcopy(jds_test.df.columns)

        del jds_test

        for evaluation_data_manipulation_config in evaluation_data_manipulation_configs:
            evaluation_name = (
                f"{prediction_dataset_handling_config.name}_"
                f"{evaluation_model_selection_config.name}_"
                f"{evaluation_data_manipulation_config.name}"
            )

            run_id_handler.prefix = "in_progress"
            evaluation_dir_path = utils.paths.evaluation_files_dir(
                dataset_name=dataset_config.name,
                dataset_handling_name=dataset_handling_config.name,
                working_points_set_name=working_points_set_config.name,
                evaluation_name=evaluation_name,
                run_id=run_id_handler.get_str(),
                working_point_name=working_point_config.name,
                mkdir=True,
            )

            jds_test_with_predictions = JetEventsDataset(
                df=jds_test_with_predictions_df.copy(deep=True)
            )

            assert (
                hash_df(jds_test_with_predictions.df)
                == jds_test_with_predictions_df_hash
            )
            assert (
                jds_test_with_predictions.n_events == jds_test_with_predictions_n_events
            )
            assert jds_test_with_predictions.n_jets == jds_test_with_predictions_n_jets
            assert np.array_equal(
                jds_test_with_predictions.event_n_jets,
                jds_test_with_predictions_event_n_jets,
            )
            assert np.array_equal(
                jds_test_with_predictions.df.columns,
                jds_test_with_predictions_df_columns,
            )

            if evaluation_data_manipulation_config.data_manipulators is not None:
                jds_test_with_predictions.manipulate(
                    data_manipulators=evaluation_data_manipulation_config.data_manipulators,
                    mode="foo",
                )

            # evaluation part

            evaluation_data_collection = {}

            if comparison_pred_col is not None:
                evaluation_data = compute_predictions_loss(
                    jds=jds_test_with_predictions,
                    eff_pred_cols=eff_pred_cols,
                    comparison_col=comparison_pred_col,
                )
                assert "predictions_loss" not in evaluation_data_collection
                evaluation_data_collection["predictions_loss"] = evaluation_data

            # for model_config in model_configs:
            #     model_prediction_variance(jds=jds_test, model_config=model_config)

            create_predictions_histogram(
                jds=jds_test_with_predictions,
                eff_pred_cols=eff_pred_cols,
                evaluation_dir_path=evaluation_dir_path,
            )

            evaluation_data_collection["jet_variable_histograms"] = {}
            for var_col, var_name_nice, unit in [
                # not using \text{} as this caused errors
                ["Jet_Pt", r"$p_\mathrm{T}$", "[GeV]"],
                ["Jet_eta", r"$\eta$", None],
                ["Jet_phi", r"$\phi$", None],
                ["Jet_mass", r"Jet mass", "[GeV]"],
                ["Jet_area", r"Jet area", None],
            ]:
                evaluation_data = create_jet_variable_histogram(
                    jds=jds_test_with_predictions,
                    eff_pred_cols=eff_pred_cols,
                    var_col=var_col,
                    var_name_nice=var_name_nice,
                    unit=unit,
                    evaluation_dir_path=evaluation_dir_path,
                    bins=20,
                    comparison_col=comparison_pred_col,
                )
                assert set(evaluation_data.keys()).isdisjoint(
                    set(evaluation_data_collection["jet_variable_histograms"].keys())
                )
                evaluation_data_collection["jet_variable_histograms"].update(
                    evaluation_data
                )

            evaluation_data = create_leading_subleading_histograms(
                jds=jds_test_with_predictions,
                eff_pred_cols=eff_pred_cols,
                evaluation_dir_path=evaluation_dir_path,
                comparison_col=comparison_pred_col,
            )
            assert set(evaluation_data.keys()).isdisjoint(
                set(evaluation_data_collection.keys())
            )
            evaluation_data_collection[
                "leading_subleading_histograms"
            ] = evaluation_data

            print(json.dumps(evaluation_data_collection, indent=4))

            with open(evaluation_dir_path / "evaluation_data.json", "w") as f:
                json.dump(obj=evaluation_data_collection, fp=f, indent=4)

            with open(evaluation_dir_path / "evaluation_data.pkl", "wb") as f:
                pickle.dump(obj=evaluation_data_collection, file=f)

            del jds_test_with_predictions

            run_id_handler.prefix = None
            evaluation_dir_path.rename(
                target=utils.paths.evaluation_files_dir(
                    dataset_name=dataset_config.name,
                    dataset_handling_name=dataset_handling_config.name,
                    working_points_set_name=working_points_set_config.name,
                    evaluation_name=evaluation_name,
                    run_id=run_id_handler.get_str(),
                    working_point_name=working_point_config.name,
                    mkdir=True,
                )
            )
            shutil.rmtree(path=evaluation_dir_path.parent)
