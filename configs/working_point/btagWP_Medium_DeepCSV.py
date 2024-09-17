from utils.configs.working_point import WorkingPointConfig

working_point_config = WorkingPointConfig(
    name="btagWP_Medium_DeepCSV",
    expression="Jet_btagDeepFlavB > 0.3040",
    required_columns=("Jet_btagDeepFlavB",),
)
