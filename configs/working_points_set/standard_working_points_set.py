from utils.configs.working_point import WorkingPointConfig
from utils.configs.working_points_set import WorkingPointsSetConfig

working_points_set_config = WorkingPointsSetConfig(
    name="standard_working_points_set",
    working_points=(
        WorkingPointConfig(
            name="btagWP_Loose_DeepCSV",
            expression="Jet_btagDeepB > 0.1241",
            required_columns=("Jet_btagDeepB",),
        ),
        WorkingPointConfig(
            name="btagWP_Medium_DeepCSV",
            expression="Jet_btagDeepB > 0.4184",
            required_columns=("Jet_btagDeepB",),
        ),
        WorkingPointConfig(
            name="btagWP_Tight_DeepCSV",
            expression="Jet_btagDeepB > 0.7527",
            required_columns=("Jet_btagDeepB",),
        ),
    ),
)
