from utils.configs.working_point import WorkingPointConfig

working_point_config = WorkingPointConfig(
    name="btagWP_Medium_DeepCSV",
    expression="Jet_btagDeepB > 0.4184",
    required_columns=("Jet_btagDeepB",),
)
