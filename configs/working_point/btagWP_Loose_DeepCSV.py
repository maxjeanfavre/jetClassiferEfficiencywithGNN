from utils.configs.working_point import WorkingPointConfig

working_point_config = WorkingPointConfig(
    name="btagWP_Loose_DeepCSV",
    expression="Jet_btagDeepFlavB > 0.0532",
    required_columns=("Jet_btagDeepFlavB",),
)
