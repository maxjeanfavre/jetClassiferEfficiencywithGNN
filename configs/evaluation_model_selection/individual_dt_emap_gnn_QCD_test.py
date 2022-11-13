from configs.model.direct_tagging_test import model_config as direct_tagging_model_config

from configs.model.gnn_test import model_config as gnn_model_config
from utils.configs.evaluation_model import EvaluationModelConfig
from utils.configs.evaluation_model_selection import EvaluationModelSelectionConfig


evaluation_model_selection_config = EvaluationModelSelectionConfig(
    name=f"individual_{gnn_model_config.name}_and_median",
    evaluation_model_configs=[
        EvaluationModelConfig(
            model_config=direct_tagging_model_config,
            run_selection="only_latest",
            run_aggregation="individual",
            is_comparison_base=True,
            display_name="Direct tagging",
        ),
        EvaluationModelConfig(
            model_config=gnn_model_config,
            run_selection="only_latest",
            run_aggregation="median",
            is_comparison_base=False,
            display_name="GNN",
            only_bootstrap_runs=True,
        ),
    ],
)
