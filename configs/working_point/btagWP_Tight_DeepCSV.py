from utils.configs.working_point import WorkingPointConfig

working_point_config = WorkingPointConfig(
    name="btagWP_Tight_DeepCSV",
    expression="Jet_btagDeepB > 0.7527",
    required_columns=("Jet_btagDeepB",),
)
