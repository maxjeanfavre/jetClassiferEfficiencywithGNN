class Filenames:
    # logs
    train_log = "train"
    extraction_log = "extraction"
    dataset_outputs_log = "dataset_outputs"

    # dataset extraction
    dataset_extraction = "extraction.feather"
    dataset_extraction_event_n_jets = "extraction_event_n_jets.npy"

    @staticmethod
    def save_prediction_log(
        prediction_dataset_handling_config_name: str,
    ) -> str:
        return f"save_prediction_{prediction_dataset_handling_config_name}"

    # predictions
    @staticmethod
    def model_prediction(
        dataset_name: str, working_point_name: str, prediction_dataset_handling_name: str
    ) -> str:
        return f"prediction_{dataset_name}_{working_point_name}_{prediction_dataset_handling_name}.npz"

    # configs
    dataset_config_pickle = "dataset_config.pkl"
    dataset_config_json = "dataset_config.json"
    model_config_pickle = "model_config.pkl"

    # plots
    @staticmethod
    def efficiency_map_plot_surfaces(model_name: str, working_point_name: str):
        return f"{model_name}_{working_point_name}_surfaces"

    @staticmethod
    def efficiency_map_plot_bars(model_name: str, working_point_name: str):
        return f"{model_name}_{working_point_name}_bars"

    @staticmethod
    def jet_variable_normalised_histogram(
        variable: str, lower_quantile: float, upper_quantile: float
    ):
        lower_quantile_str = str(lower_quantile).replace(".", "")
        upper_quantile_str = str(upper_quantile).replace(".", "")

        return (
            f"jet_variable_normalised"
            f"_{variable}"
            f"_{lower_quantile_str}"
            f"_{upper_quantile_str}"
        )

    jet_multiplicity_normalised_histogram_inclusive = (
        "jet_multiplicity_normalised_inclusive"
    )
    jet_multiplicity_normalised_histogram_by_flavour = (
        "jet_multiplicity_normalised_by_flavour"
    )
    deepcsv_discriminator_histogram = "jet_btagDeepB_histogram"

    def __init__(self) -> None:
        pass
