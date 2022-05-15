import copy
import json
import pickle
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

    logger.info("Evaluation starts.")

    run_id = RunIdHandler.generate_run_id(bootstrap=False)

    # branches_to_read_in = set()
    # for model_config, _, _, _ in model_settings:
    #     branches_to_read_in.update(model_config.get_required_columns())
    # branches_to_read_in = tuple(branches_to_read_in)
    branches_to_read_in = None
    # TODO(low): fix

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

    # eval_dfs = []

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

        # for (
        #     model_config,
        #     runs,
        #     run_aggregation,
        #     is_comparison_base,
        # ) in evaluation_model_selection_config.model_settings:
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
            )
            if len(model_run_ids) == 0:
                logger.warning(
                    "No model runs found for model_config "
                    f"{evaluation_model_config.model_config.name}"
                )
                continue

            if evaluation_model_config.run_aggregation == "individual":
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
                        filename=utils.filenames.model_prediction_filename(
                            working_point_name=working_point_config.name,
                            prediction_dataset_handling_name=prediction_dataset_handling_config.name,
                        ),
                    )

                    assert mp.res.index.equals(jds_test.df.index)
                    assert mp.err.index.equals(jds_test.df.index)

                    pred_col = f"res_{evaluation_model_config.model_config.name}_{model_run_id}"
                    err_col = f"err_{evaluation_model_config.model_config.name}_{model_run_id}"

                    jds_test.df[pred_col] = mp.res
                    jds_test.df[err_col] = mp.err

                    eff_pred_cols.append(pred_col)
                    eff_err_cols.append(err_col)
            elif evaluation_model_config.run_aggregation == "mean":
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
                        filename=utils.filenames.model_prediction_filename(
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

                pred_col = (
                    f"res_{evaluation_model_config.model_config.name}_mean_ensemble"
                )
                err_col = (
                    f"err_{evaluation_model_config.model_config.name}_mean_ensemble"
                )

                jds_test.df[pred_col] = run_preds_df.mean(axis=1)
                jds_test.df[err_col] = run_preds_df.std(axis=1)

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
            evaluation_dir_path = utils.paths.evaluation_files_dir(
                dataset_name=dataset_config.name,
                dataset_handling_name=dataset_handling_config.name,
                working_points_set_name=working_points_set_config.name,
                evaluation_name=(
                    f"{prediction_dataset_handling_config.name}_"
                    f"{evaluation_model_selection_config.name}_"
                    f"{evaluation_data_manipulation_config.name}"
                ),
                run_id=run_id,
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

            evaluation_data_collection = dict()

            # for model_config in model_configs:
            #     model_prediction_variance(jds=jds_test, model_config=model_config)

            create_predictions_histogram(
                jds=jds_test_with_predictions,
                eff_pred_cols=eff_pred_cols,
                evaluation_dir_path=evaluation_dir_path,
            )

            for var_col, bins in [
                [
                    "Jet_Pt",
                    np.array(
                        [-np.inf, 0, 30, 50, 70, 100, 150, 200, 300, 600, 1000, np.inf]
                    ),
                ],
                [
                    "Jet_eta",
                    np.array(
                        [
                            -np.inf,
                            -2.5,
                            -2,
                            -1.5,
                            -1,
                            -0.5,
                            0,
                            0.5,
                            1,
                            1.5,
                            2,
                            2.5,
                            np.inf,
                        ]
                    ),
                ],
            ]:
                evaluation_data = create_jet_variable_histogram(
                    jds=jds_test_with_predictions,
                    eff_pred_cols=eff_pred_cols,
                    var_col=var_col,
                    evaluation_dir_path=evaluation_dir_path,
                    # bins=bins,
                    bins=20,
                    comparison_col=comparison_pred_col,
                )
                evaluation_data_collection.update(evaluation_data)

            evaluation_data = create_leading_subleading_histograms(
                jds=jds_test_with_predictions,
                eff_pred_cols=eff_pred_cols,
                evaluation_dir_path=evaluation_dir_path,
                comparison_col=comparison_pred_col,
            )
            evaluation_data_collection.update(evaluation_data)

            print(json.dumps(evaluation_data_collection, indent=4))

            # eval_df = pd.concat(
            #     {
            #         k: pd.DataFrame.from_dict(v, "index")
            #         for k, v in evaluation_data_collection.items()
            #     },
            #     axis=0,
            # )

            # eval_dfs.append(eval_df)

            with open(evaluation_dir_path / "evaluation_data.json", "w") as f:
                json.dump(obj=evaluation_data_collection, fp=f, indent=4)

            with open(evaluation_dir_path / "evaluation_data.pkl", "wb") as f:
                pickle.dump(obj=evaluation_data_collection, file=f)

            del jds_test_with_predictions
