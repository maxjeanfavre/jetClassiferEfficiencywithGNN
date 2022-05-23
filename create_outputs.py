import utils
from utils.data.jet_events_dataset import JetEventsDataset
from utils.outputs.dataset_plots import create_dataset_plots
from utils.outputs.efficiency_map_plots import create_efficiency_map_histogram_plots
from utils.outputs.mistag_rates import save_light_jet_mistag_rates


def main():
    from configs.dataset.QCD_Pt_300_470_MuEnrichedPt5 import (
        dataset_config as QCD_Pt_300_470_MuEnrichedPt5_dataset_config,
    )
    from configs.dataset.QCD_Pt_300_470_MuEnrichedPt5_test import (
        dataset_config as QCD_Pt_300_470_MuEnrichedPt5_test_dataset_config,
    )
    from configs.dataset.TTTo2L2Nu import dataset_config as TTTo2L2Nu_dataset_config
    from configs.dataset.TTTo2L2Nu_test import (
        dataset_config as TTTo2L2Nu_test_dataset_config,
    )
    from configs.dataset_handling.standard_dataset_handling import (
        dataset_handling_config as standard_dataset_handling_config,
    )
    from configs.model.eff_map_pt_eta import model_config as eff_map_pt_eta_model_config
    from configs.working_points_set.standard_working_points_set import (
        working_points_set_config as standard_working_points_set_config,
    )

    for dataset_config in [
        TTTo2L2Nu_dataset_config,
        TTTo2L2Nu_test_dataset_config,
        QCD_Pt_300_470_MuEnrichedPt5_dataset_config,
        QCD_Pt_300_470_MuEnrichedPt5_test_dataset_config,
    ]:
        dataset_output_dir_path = utils.paths.dataset_output_dir(
            dataset_name=dataset_config.name,
            mkdir=True,
        )

        jds = JetEventsDataset.read_in(
            dataset_config=dataset_config,
            branches=None,
        )

        create_dataset_plots(
            dataset_config=dataset_config,
            output_dir_path=dataset_output_dir_path,
            jds=jds,
        )

        for working_points_set_config in [standard_working_points_set_config]:
            save_light_jet_mistag_rates(
                dataset_config=dataset_config,
                working_points_set_config=working_points_set_config,
                output_dir_path=dataset_output_dir_path,
                jds=jds,
            )

        for dataset_handling_config in [standard_dataset_handling_config]:
            for model_config in [eff_map_pt_eta_model_config]:
                create_efficiency_map_histogram_plots(
                    dataset_config=dataset_config,
                    dataset_handling_config=dataset_handling_config,
                    working_points_set_config=working_points_set_config,
                    model_config=model_config,
                )


if __name__ == "__main__":
    main()
