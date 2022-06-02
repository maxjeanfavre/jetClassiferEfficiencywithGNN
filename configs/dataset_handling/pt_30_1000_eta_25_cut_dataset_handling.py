from utils.configs.dataset_handling import DatasetHandlingConfig
from utils.data.manipulation.data_filters.eval_expression import EvalExpressionFilter
from utils.data.manipulation.data_filters.jet_multiplicity import JetMultiplicityFilter
from utils.data.manipulation.data_filters.remove_nans import RemoveNaNs

dataset_handling_config = DatasetHandlingConfig(
    name="pt_30_1000_eta_25_cut_dataset_handling",
    train_split=0.75,
    test_split=0.25,
    train_test_split_random_state=42,
    data_manipulators=(
        RemoveNaNs(
            active_modes=("train", "predict"),
            filter_full_event=True,
        ),
        JetMultiplicityFilter(
            active_modes=("train", "predict"),
            n=1,
            mode="remove",
        ),  # as they are reconstruction errors
        EvalExpressionFilter(
            description=(
                "Some basic cuts (30 <= Jet_Pt <= 1000 & abs(Jet_eta) <= 2.5)"
            ),
            active_modes=("train", "predict"),
            expression="30 <= Jet_Pt <= 1000 & abs(Jet_eta) <= 2.5",
            filter_full_event=False,
            required_columns=("Jet_Pt", "Jet_eta"),
        ),
    ),
)
