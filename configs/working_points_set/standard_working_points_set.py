from configs.working_point.btagWP_Loose_DeepCSV import (
    working_point_config as loose_working_point_config,
)
from configs.working_point.btagWP_Medium_DeepCSV import (
    working_point_config as medium_working_point_config,
)
from configs.working_point.btagWP_Tight_DeepCSV import (
    working_point_config as tight_working_point_config,
)
from utils.configs.working_points_set import WorkingPointsSetConfig

working_points_set_config = WorkingPointsSetConfig(
    name="standard_working_points_set",
    working_points=(
#        loose_working_point_config,
#        medium_working_point_config,
        tight_working_point_config,
    ),
)
