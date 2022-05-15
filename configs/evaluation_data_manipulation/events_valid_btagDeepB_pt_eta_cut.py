from utils.configs.evaluation_data_manipulation import EvaluationDataManipulationConfig
from utils.data.manipulation.data_filters.eval_expression import EvalExpressionFilter

evaluation_data_manipulation_config = EvaluationDataManipulationConfig(
    name="events_valid_btagDeepB_pt_eta_cut",
    data_manipulators=(
        EvalExpressionFilter(
            description=(
                "Keeps events with valid btagDeepB value (0 <= Jet_btagDeepB <= 1)"
            ),
            active_modes=("foo",),
            expression="0 <= Jet_btagDeepB <= 1",
            filter_full_event=True,
            required_columns=("Jet_btagDeepB",),
        ),
        EvalExpressionFilter(
            description=(
                "Some basic cuts (20 <= Jet_Pt <= 1000 & abs(Jet_eta) <= 2.5)"
            ),
            active_modes=("foo",),
            expression="20 <= Jet_Pt <= 1000 & abs(Jet_eta) <= 2.5",
            filter_full_event=False,
            required_columns=("Jet_Pt", "Jet_eta"),
        ),
    ),
)
