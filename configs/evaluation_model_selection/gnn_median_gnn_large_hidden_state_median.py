from configs.model.direct_tagging import model_config as direct_tagging_model_config
from configs.model.eff_map_pt_eta import (
    model_config as eff_map_pt_eta_model_config,
)
from configs.model.gnn import model_config as gnn_model_config
from configs.model.gnn_large_hidden_state import (
    model_config as gnn_large_hidden_state_model_config,
)
from utils.configs.evaluation_model import EvaluationModelConfig
from utils.configs.evaluation_model_selection import EvaluationModelSelectionConfig

evaluation_model_selection_config = EvaluationModelSelectionConfig(
    name="gnn_median_gnn_large_hidden_state_median",
    evaluation_model_configs=[
        EvaluationModelConfig(
            model_config=direct_tagging_model_config,
            run_selection="only_latest",
            run_aggregation="individual",
            is_comparison_base=True,
            display_name="Direct tagging",
        ),
        EvaluationModelConfig(
            model_config=eff_map_pt_eta_model_config,
            run_selection="only_latest",
            run_aggregation="individual",
            is_comparison_base=False,
            display_name="Efficiency map",
        ),
        EvaluationModelConfig(
            model_config=gnn_model_config,
            run_selection="all",
            run_aggregation="median",
            is_comparison_base=False,
            display_name="GNN 1",
            only_bootstrap_runs=True,
        ),
        EvaluationModelConfig(
            model_config=gnn_large_hidden_state_model_config,
            run_selection="all",
            run_aggregation="median",
            is_comparison_base=False,
            display_name="GNN 1 large hidden state",
            only_bootstrap_runs=True,
        ),
    ],
)
