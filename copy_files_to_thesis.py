import pathlib
import shutil

import utils
from configs.dataset.QCD_Pt_300_470_MuEnrichedPt5 import (
    dataset_config as QCD_Pt_300_470_MuEnrichedPt5_dataset_config,
)
from configs.dataset.TTTo2L2Nu import dataset_config as TTTo2L2Nu_dataset_config
from configs.dataset.TTToSemiLeptonic import dataset_config as TTToSemiLeptonic_dataset_config
from configs.dataset.TTToHadronic import dataset_config as TTToHadronic_dataset_config
from configs.dataset_handling.standard_dataset_handling import (
    dataset_handling_config as standard_dataset_handling_config,
)
from configs.evaluation_data_manipulation.no_changes import (
    evaluation_data_manipulation_config as no_changes_evaluation_data_manipulation_config,
)
from configs.evaluation_model_selection.individual_gnn_and_median import (
    evaluation_model_selection_config as individual_gnn_and_median_evaluation_model_selection_config,
)
from configs.evaluation_model_selection.selection_1_median import (
    evaluation_model_selection_config as selection_1_median_evaluation_model_selection_config,
)
from configs.prediction_dataset_handling.no_changes import (
    prediction_dataset_handling_config,
)
from configs.working_points_set.standard_working_points_set import (
    working_points_set_config,
)
from utils.helpers.run_id_handler import RunIdHandler


def main():
    base_path = pathlib.Path("../doc/thesis/src/data")

    assert base_path.exists()
    assert base_path.is_dir()

    dataset_configs = [
        TTTo2L2Nu_dataset_config,
        QCD_Pt_300_470_MuEnrichedPt5_dataset_config,
    ]

    dataset_handling_config = standard_dataset_handling_config

    evaluation_model_selection_dir_names_and_configs = [
        ["evaluation", selection_1_median_evaluation_model_selection_config],
        [
            "bootstrap_evaluation",
            individual_gnn_and_median_evaluation_model_selection_config,
        ],
    ]
    evaluation_data_manipulation_config = no_changes_evaluation_data_manipulation_config

    for dataset_config in dataset_configs:
        dataset_path = base_path / dataset_config.name

        try:
            shutil.rmtree(path=dataset_path)
        except FileNotFoundError:
            pass

        # copy dataset outputs
        dataset_output_dir_path = utils.paths.dataset_outputs_dir(
            dataset_name=dataset_config.name,
            mkdir=True,
        )

        dataset_outputs_destination_path = dataset_path / "outputs"

        shutil.copytree(
            src=dataset_output_dir_path, dst=dataset_outputs_destination_path
        )

        # copy evaluation outputs
        for (
            evaluation_model_selection_dir_name,
            evaluation_model_selection_config,
        ) in evaluation_model_selection_dir_names_and_configs:
            evaluation_name = (
                f"{prediction_dataset_handling_config.name}_"
                f"{evaluation_model_selection_config.name}_"
                f"{evaluation_data_manipulation_config.name}"
            )
            run_ids = RunIdHandler.get_run_ids(
                dir_path=utils.paths.evaluation_files_dir(
                    dataset_name=dataset_config.name,
                    dataset_handling_name=dataset_handling_config.name,
                    working_points_set_name=working_points_set_config.name,
                    evaluation_name=evaluation_name,
                    run_id="foo",
                    working_point_name="foo",
                    mkdir=False,
                ).parent.parent,
                only_latest=True,
            )
            assert len(run_ids) == 1
            run_id = run_ids[0]
            evaluation_dir_path = utils.paths.evaluation_files_dir(
                dataset_name=dataset_config.name,
                dataset_handling_name=dataset_handling_config.name,
                working_points_set_name=working_points_set_config.name,
                evaluation_name=evaluation_name,
                run_id=run_id,
                working_point_name="foo",
                mkdir=False,
            ).parent

            evaluation_destination_path = (
                dataset_path / evaluation_model_selection_dir_name
            )

            shutil.copytree(src=evaluation_dir_path, dst=evaluation_destination_path)


if __name__ == "__main__":
    main()
