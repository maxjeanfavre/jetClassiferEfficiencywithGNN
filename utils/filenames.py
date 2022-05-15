class Filenames:
    # logs
    train_log_filename = "train"
    extraction_log_filename = "extraction"
    dataset_extraction_filename = "extraction.feather"
    dataset_extraction_event_n_jets_filename = "extraction_event_n_jets.npy"

    @staticmethod
    def save_prediction_log_filename(
        prediction_dataset_handling_config_name: str,
    ) -> str:
        return f"save_prediction_{prediction_dataset_handling_config_name}"

    # predictions
    @staticmethod
    def model_prediction_filename(
        working_point_name: str, prediction_dataset_handling_name: str
    ) -> str:
        return f"prediction_{working_point_name}_{prediction_dataset_handling_name}.npz"

    # plots
    efficiency_prediction_histogram_plot = "eff_pred_hist.png"

    @staticmethod
    def jet_variable_histogram_plot(title: str) -> str:
        return f"jet_variable_histogram_{title}.png"

    @staticmethod
    def leading_subleading_histogram_plot(title: str) -> str:
        return f"leading_subleading_{title}.png"

    # configs
    dataset_config_pickle_filename = "dataset_config.pkl"
    dataset_config_json_filename = "dataset_config.json"

    model_config_pickle_filename = "model_config.pkl"

    def __init__(self) -> None:
        pass
