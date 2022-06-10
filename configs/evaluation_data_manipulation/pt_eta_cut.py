from utils.configs.evaluation_data_manipulation import EvaluationDataManipulationConfig
from utils.data.manipulation.data_filters.eval_expression import EvalExpressionFilter

evaluation_data_manipulation_config = EvaluationDataManipulationConfig(
    name="pt_eta_cut",
    data_manipulators=(
        EvalExpressionFilter(
            description=(
                "Some basic cuts (30 <= Jet_Pt <= 1000 & abs(Jet_eta) <= 2.5)"
            ),
            active_modes=("foo",),
            expression="30 <= Jet_Pt <= 1000 & abs(Jet_eta) <= 2.5",
            filter_full_event=False,
            required_columns=("Jet_Pt", "Jet_eta"),
        ),
    ),
)
