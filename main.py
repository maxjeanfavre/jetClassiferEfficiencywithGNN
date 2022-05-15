import argparse

import utils
from utils.configs.dataset import DatasetConfig
from utils.configs.dataset_handling import DatasetHandlingConfig
from utils.configs.evaluation_data_manipulation import EvaluationDataManipulationConfig
from utils.configs.evaluation_model_selection import EvaluationModelSelectionConfig
from utils.configs.model import ModelConfig
from utils.configs.prediction_dataset_handling import PredictionDatasetHandlingConfig
from utils.configs.working_points_set import WorkingPointsSetConfig
from utils.data.extraction.extract_dataset import extract_dataset
from utils.evaluation.evaluation_handler import evaluation_handler
from utils.helpers.modules.load_config import load_config
from utils.helpers.modules.package_names import get_package_names
from utils.models.handlers.save_prediction import save_predictions_handler
from utils.models.handlers.train import train_handler


def get_parsers():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="task", help="task to do", required=True)

    parser_train = subparsers.add_parser(
        name="train",
        help="train a model",
    )
    parser_save_predictions = subparsers.add_parser(
        name="save_predictions",
        help="save predictions of a model",
    )
    parser_evaluate = subparsers.add_parser(
        name="evaluate",
        help="run an evaluation",
    )
    parser_extract = subparsers.add_parser(
        name="extract",
        help="extract a dataset",
    )

    for p in [parser_train, parser_save_predictions, parser_evaluate, parser_extract]:
        p.add_argument(
            "--dataset",
            choices=get_package_names(
                dir_path=utils.paths.config(
                    config_type="dataset", config_name="foo", mkdir=False
                ).parent
            ),
            help="name of the dataset config",
            required=True,
        )

    for p in [parser_train, parser_save_predictions, parser_evaluate]:
        p.add_argument(
            "--dataset_handling",
            choices=get_package_names(
                dir_path=utils.paths.config(
                    config_type="dataset_handling", config_name="foo", mkdir=False
                ).parent
            ),
            help="name of the dataset handling config",
            required=True,
        )

    for p in [parser_train, parser_save_predictions, parser_evaluate]:
        p.add_argument(
            "--working_points_set",
            choices=get_package_names(
                dir_path=utils.paths.config(
                    config_type="working_points_set", config_name="foo", mkdir=False
                ).parent
            ),
            help="name of the working points set config",
            required=True,
        )

    for p in [parser_train, parser_save_predictions]:
        p.add_argument(
            "--model",
            choices=get_package_names(
                dir_path=utils.paths.config(
                    config_type="model", config_name="foo", mkdir=False
                ).parent
            ),
            help="name of the model config",
            required=True,
        )

    parser_train.add_argument(
        "--bootstrap",
        action="store_true",
        help="whether its a bootstrap run",
    )

    parser_save_predictions.add_argument(
        "--run_id",
        help="run_id of the model",
        required=True,
    )

    for p in [parser_save_predictions, parser_evaluate]:
        p.add_argument(
            "--prediction_dataset_handling",
            choices=get_package_names(
                dir_path=utils.paths.config(
                    config_type="prediction_dataset_handling",
                    config_name="foo",
                    mkdir=False,
                ).parent
            ),
            help="name of the prediction dataset handling configs",
            required=True,
        )

    parser_evaluate.add_argument(
        "--evaluation_model_selection",
        choices=get_package_names(
            dir_path=utils.paths.config(
                config_type="evaluation_model_selection",
                config_name="foo",
                mkdir=False,
            ).parent
        ),
        help="name of the evaluation model selection config",
        required=True,
    )

    parser_evaluate.add_argument(
        "--evaluation_data_manipulation",
        nargs="+",
        choices=get_package_names(
            dir_path=utils.paths.config(
                config_type="evaluation_data_manipulation",
                config_name="foo",
                mkdir=False,
            ).parent
        ),
        help="names of the evaluation data manipulation configs",
        required=True,
    )

    return (
        parser,
        parser_train,
        parser_save_predictions,
        parser_evaluate,
        parser_extract,
    )


def main():
    (
        parser,
        parser_train,
        parser_save_predictions,
        parser_evaluate,
        parser_extract,
    ) = get_parsers()

    parser_train.add_argument(
        "--run_id",
        help="manually set a run_id",
    )

    args = parser.parse_args()

    task = args.task

    dataset = args.dataset

    dataset_config: DatasetConfig = load_config(
        config_type="dataset", config_name=dataset
    )

    if task in ("train", "save_predictions", "evaluate"):
        dataset_handling = args.dataset_handling
        working_points_set = args.working_points_set

        dataset_handling_config: DatasetHandlingConfig = load_config(
            config_type="dataset_handling", config_name=dataset_handling
        )

        working_points_set_config: WorkingPointsSetConfig = load_config(
            config_type="working_points_set", config_name=working_points_set
        )

        if task in ("train", "save_predictions"):
            model = args.model

            model_config: ModelConfig = load_config(
                config_type="model", config_name=model
            )

            if task == "train":
                try:
                    bootstrap = args.bootstrap
                except AttributeError:
                    bootstrap = None

                try:
                    run_id = args.run_id
                except AttributeError:
                    run_id = None

                train_handler(
                    dataset_config=dataset_config,
                    dataset_handling_config=dataset_handling_config,
                    working_points_set_config=working_points_set_config,
                    model_config=model_config,
                    bootstrap=bootstrap,
                    run_id=run_id,
                )
            elif task == "save_predictions":
                run_id = args.run_id
                prediction_dataset_handling = args.prediction_dataset_handling

                prediction_dataset_handling_config: PredictionDatasetHandlingConfig = (
                    load_config(
                        config_type="prediction_dataset_handling",
                        config_name=prediction_dataset_handling,
                    )
                )

                save_predictions_handler(
                    dataset_config=dataset_config,
                    dataset_handling_config=dataset_handling_config,
                    working_points_set_config=working_points_set_config,
                    model_config=model_config,
                    run_id=run_id,
                    prediction_dataset_handling_config=prediction_dataset_handling_config,
                )
            else:
                raise ValueError("You shouldn't reach this point")
        elif task == "evaluate":
            prediction_dataset_handling = args.prediction_dataset_handling
            evaluation_model_selection = args.evaluation_model_selection
            evaluation_data_manipulation_list = args.evaluation_data_manipulation
            assert type(evaluation_data_manipulation_list) == list

            prediction_dataset_handling_config: PredictionDatasetHandlingConfig = (
                load_config(
                    config_type="prediction_dataset_handling",
                    config_name=prediction_dataset_handling,
                )
            )

            evaluation_model_selection_config: EvaluationModelSelectionConfig = (
                load_config(
                    config_type="evaluation_model_selection",
                    config_name=evaluation_model_selection,
                )
            )

            evaluation_data_manipulation_configs = []
            for evaluation_data_manipulation in evaluation_data_manipulation_list:
                evaluation_data_manipulation_config: EvaluationDataManipulationConfig = load_config(
                    config_type="evaluation_data_manipulation",
                    config_name=evaluation_data_manipulation,
                )
                evaluation_data_manipulation_configs.append(
                    evaluation_data_manipulation_config
                )

            evaluation_handler(
                dataset_config=dataset_config,
                dataset_handling_config=dataset_handling_config,
                working_points_set_config=working_points_set_config,
                prediction_dataset_handling_config=prediction_dataset_handling_config,
                evaluation_model_selection_config=evaluation_model_selection_config,
                evaluation_data_manipulation_configs=evaluation_data_manipulation_configs,
            )
        else:
            raise ValueError("You shouldn't reach this point")

    elif task == "extract":
        extract_dataset(dataset_config=dataset_config)

    else:
        raise ValueError("You shouldn't reach this point")


if __name__ == "__main__":
    main()
