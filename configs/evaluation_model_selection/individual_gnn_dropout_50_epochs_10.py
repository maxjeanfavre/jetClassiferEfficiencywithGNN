from configs.model.gnn_dropout_50_epochs_10 import (
    model_config as gnn_dropout_50_epochs_10_model_config,
)
from utils.configs.evaluation_model import EvaluationModelConfig
from utils.configs.evaluation_model_selection import EvaluationModelSelectionConfig


evaluation_model_selection_config = EvaluationModelSelectionConfig(
    name=f"individual_{gnn_dropout_50_epochs_10_model_config.name}",
    evaluation_model_configs=[
        EvaluationModelConfig(
            model_config=gnn_dropout_50_epochs_10_model_config,
            run_selection="all",
            run_aggregation="individual",
            is_comparison_base=False,
        ),
    ],
)
