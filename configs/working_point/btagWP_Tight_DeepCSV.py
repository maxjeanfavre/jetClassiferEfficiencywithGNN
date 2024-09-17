from utils.configs.working_point import WorkingPointConfig

working_point_config = WorkingPointConfig(
    name="btagWP_Tight_DeepCSV",
    expression="Jet_btagDeepFlavB > 0.7476",
    required_columns=("Jet_btagDeepFlavB",),
)
