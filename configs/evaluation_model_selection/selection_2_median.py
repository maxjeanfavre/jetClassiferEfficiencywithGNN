from configs.model.direct_tagging import model_config as direct_tagging_model_config
from configs.model.eff_map_pt_eta import (
    model_config as eff_map_pt_eta_model_config,
)
from configs.model.gnn import model_config as gnn_model_config
from configs.model.gnn_large_hidden_state import (
    model_config as gnn_large_hidden_state_model_config,
)
from configs.model.gnn_x_large_hidden_state import (
    model_config as gnn_x_large_hidden_state_model_config,
)
from configs.model.gnn_one_gn_block import (
    model_config as gnn_one_gn_block_model_config,
)
from configs.model.gnn_small_hidden_state import (
    model_config as gnn_small_hidden_state_model_config,
)
from configs.model.gnn_two_gn_blocks import (
    model_config as gnn_two_gn_blocks_model_config,
)
from configs.model.gnn_variables_1 import (
    model_config as gnn_variables_1_model_config,
)
from configs.model.gnn_one_gn_block_large_hidden_state import (
    model_config as gnn_one_gn_block_large_hidden_state_model_config,
)
from utils.configs.evaluation_model import EvaluationModelConfig
from utils.configs.evaluation_model_selection import EvaluationModelSelectionConfig

evaluation_model_selection_config = EvaluationModelSelectionConfig(
    name="selection_2_median",
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
            display_name="GNN",
            only_bootstrap_runs=True,
        ),
        EvaluationModelConfig(
            model_config=gnn_variables_1_model_config,
            run_selection="all",
            run_aggregation="median",
            is_comparison_base=False,
            display_name="GNN variables 1",
            only_bootstrap_runs=True,
        ),
        EvaluationModelConfig(
            model_config=gnn_small_hidden_state_model_config,
            run_selection="all",
            run_aggregation="median",
            is_comparison_base=False,
            display_name="GNN small hidden state",
            only_bootstrap_runs=True,
        ),
        EvaluationModelConfig(
            model_config=gnn_large_hidden_state_model_config,
            run_selection="all",
            run_aggregation="median",
            is_comparison_base=False,
            display_name="GNN large hidden state",
            only_bootstrap_runs=True,
        ),
        EvaluationModelConfig(
            model_config=gnn_x_large_hidden_state_model_config,
            run_selection="all",
            run_aggregation="median",
            is_comparison_base=False,
            display_name="GNN x large hidden state",
            only_bootstrap_runs=True,
        ),
        EvaluationModelConfig(
            model_config=gnn_one_gn_block_model_config,
            run_selection="all",
            run_aggregation="median",
            is_comparison_base=False,
            display_name="GNN one gn block",
            only_bootstrap_runs=True,
        ),
        EvaluationModelConfig(
            model_config=gnn_two_gn_blocks_model_config,
            run_selection="all",
            run_aggregation="median",
            is_comparison_base=False,
            display_name="GNN two gn blocks",
            only_bootstrap_runs=True,
        ),
        EvaluationModelConfig(
            model_config=gnn_one_gn_block_large_hidden_state_model_config,
            run_selection="all",
            run_aggregation="median",
            is_comparison_base=False,
            display_name="GNN one gn block large hidden state",
            only_bootstrap_runs=True,
        ),
    ],
)
