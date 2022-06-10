from configs.model.gnn import model_config as gnn_model_config
from utils.configs.evaluation_model import EvaluationModelConfig
from utils.configs.evaluation_model_selection import EvaluationModelSelectionConfig


evaluation_model_selection_config = EvaluationModelSelectionConfig(
    name=f"individual_{gnn_model_config.name}_and_median",
    evaluation_model_configs=[
        EvaluationModelConfig(
            model_config=gnn_model_config,
            run_selection="all",
            run_aggregation="median",
            is_comparison_base=True,
            display_name="GNN1",
            only_bootstrap_runs=True,
        ),
        EvaluationModelConfig(
            model_config=gnn_model_config,
            run_selection="all",
            run_aggregation="individual",
            is_comparison_base=False,
            display_name="GNN 1",
            only_bootstrap_runs=True,
        ),
    ],
)
