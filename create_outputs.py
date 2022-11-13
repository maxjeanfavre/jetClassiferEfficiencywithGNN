from utils.outputs.create_outputs import create_outputs


def main():
    from configs.dataset.QCD_Pt_300_470_MuEnrichedPt5 import (
        dataset_config as QCD_Pt_300_470_MuEnrichedPt5_dataset_config,
    )
    from configs.dataset.TTTo2L2Nu import dataset_config as TTTo2L2Nu_dataset_config
    from configs.dataset_handling.standard_dataset_handling import (
        dataset_handling_config as standard_dataset_handling_config,
    )
    from configs.model.eff_map_pt_eta import model_config as eff_map_pt_eta_model_config
    from configs.working_points_set.standard_working_points_set import (
        working_points_set_config as standard_working_points_set_config,
    )

    for dataset_config in [
        TTTo2L2Nu_dataset_config,
        #QCD_Pt_300_470_MuEnrichedPt5_dataset_config,
       
    ]:
        create_outputs(
            dataset_config=dataset_config,
            working_points_set_configs=[standard_working_points_set_config],
            dataset_handling_configs=[standard_dataset_handling_config],
            efficiency_map_model_configs=[eff_map_pt_eta_model_config],
        )


if __name__ == "__main__":
    main()
