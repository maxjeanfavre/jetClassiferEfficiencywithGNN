from configs.model.direct_tagging import model_config as direct_tagging_model_config
from configs.model.eff_map_pt_eta import (
    model_config as eff_map_pt_eta_model_config,
)
from configs.model.gnn import model_config as gnn_model_config
from configs.model.gnn_dropout_50 import model_config as gnn_dropout_50_model_config
from configs.model.gnn_variables_1 import (
    model_config as gnn_variables_1_model_config,
)
from utils.configs.evaluation_model import EvaluationModelConfig
from utils.configs.evaluation_model_selection import EvaluationModelSelectionConfig


evaluation_model_selection_config = EvaluationModelSelectionConfig(
    name="mean_gnn_collection_2",
    evaluation_model_configs=[
        EvaluationModelConfig(
            model_config=direct_tagging_model_config,
            run_selection="only_latest",
            run_aggregation="individual",
            is_comparison_base=True,
        ),
        EvaluationModelConfig(
            model_config=eff_map_pt_eta_model_config,
            run_selection="only_latest",
            run_aggregation="individual",
            is_comparison_base=False,
        ),
        EvaluationModelConfig(
            model_config=gnn_model_config,
            run_selection="all",
            run_aggregation="mean",
            is_comparison_base=False,
        ),
        EvaluationModelConfig(
            model_config=gnn_dropout_50_model_config,
            run_selection="all",
            run_aggregation="mean",
            is_comparison_base=False,
        ),
        EvaluationModelConfig(
            model_config=gnn_variables_1_model_config,
            run_selection="all",
            run_aggregation="mean",
            is_comparison_base=False,
        ),
    ],
)
