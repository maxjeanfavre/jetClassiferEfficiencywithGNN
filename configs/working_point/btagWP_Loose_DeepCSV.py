from utils.configs.working_point import WorkingPointConfig

working_point_config = WorkingPointConfig(
    name="btagWP_Loose_DeepCSV",
    expression="Jet_btagDeepB > 0.1241",
    required_columns=("Jet_btagDeepB",),
)
