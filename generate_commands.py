import sys

import utils
from utils.helpers.modules.package_names import get_package_names


def print_commands(file):
    configs_dir_path = utils.paths.config(
        config_type="foo", config_name="foo", mkdir=False
    ).parent.parent
    config_types = [p.name for p in configs_dir_path.iterdir() if p.is_dir()]

    config_names = {}
    for config_type in config_types:
        dir_path = utils.paths.config(
            config_type=config_type, config_name="foo", mkdir=False
        ).parent

        config_file_names = get_package_names(dir_path=dir_path)

        config_names[config_type] = config_file_names

    for dataset in config_names["dataset"]:
        print(dataset, file=file)
        print("", file=file)

        print("Extraction", file=file)
        print(f"python submit_t3.py extract --dataset={dataset}", file=file)
        print("", file=file)

        print("Training", file=file)
        for dataset_handling in config_names["dataset_handling"]:
            for working_points_set in config_names["working_points_set"]:
                for model in config_names["model"]:
                    print(
                        f"python submit_t3.py train"
                        f" --dataset={dataset}"
                        f" --dataset_handling={dataset_handling}"
                        f" --working_points_set={working_points_set}"
                        f" --model={model}"
                        f"{' --bootstrap'if model.startswith('gnn') else ''}"
                        f" --save_predictions_after_training_prediction_dataset_handling"
                        f" {' '.join(config_names['prediction_dataset_handling'])}"
                        f"{' --array=5' if model.startswith('gnn') else ''}",
                        file=file,
                    )
        print("", file=file)

        print("Evaluation", file=file)
        for dataset_handling in config_names["dataset_handling"]:
            for working_points_set in config_names["working_points_set"]:
                for prediction_dataset_handling in config_names[
                    "prediction_dataset_handling"
                ]:
                    for evaluation_model_selection in config_names[
                        "evaluation_model_selection"
                    ]:
                        print(
                            f"python submit_t3.py evaluate"
                            f" --dataset={dataset}"
                            f" --dataset_handling={dataset_handling}"
                            f" --working_points_set={working_points_set}"
                            f" --prediction_dataset_handling={prediction_dataset_handling}"
                            f" --evaluation_model_selection={evaluation_model_selection}"
                            f" --evaluation_data_manipulation"
                            f" {' '.join(config_names['evaluation_data_manipulation'])}",
                            # f" no_changes",
                            file=file,
                        )
        print("", file=file)
        print("", file=file)


def main():
    with open("commands.txt", "w") as f:
        print_commands(file=f)
    print_commands(file=sys.stdout)


if __name__ == "__main__":
    main()

