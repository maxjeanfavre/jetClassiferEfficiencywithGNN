from configs.model.gnn_variables_1 import model_config as gnn_variables_1_model_config
from utils.configs.evaluation_model import EvaluationModelConfig
from utils.configs.evaluation_model_selection import EvaluationModelSelectionConfig


evaluation_model_selection_config = EvaluationModelSelectionConfig(
    name=f"individual_{gnn_variables_1_model_config.name}",
    evaluation_model_configs=[
        EvaluationModelConfig(
            model_config=gnn_variables_1_model_config,
            run_selection="all",
            run_aggregation="median",
            is_comparison_base=True,
            display_name="GNN2",
        ),
        EvaluationModelConfig(
            model_config=gnn_variables_1_model_config,
            run_selection="all",
            run_aggregation="individual",
            is_comparison_base=False,
            display_name="GNN 2",
        ),
    ],
)
