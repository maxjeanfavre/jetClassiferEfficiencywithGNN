from configs.model.direct_tagging import model_config as direct_tagging_model_config
from configs.model.eff_map_pt_eta import (
    model_config as eff_map_pt_eta_model_config,
)
from configs.model.gnn import model_config as gnn_model_config
from configs.model.gnn_large_hidden_state import (
    model_config as gnn_large_hidden_state_model_config,
)
from configs.model.gnn_node_mass import (
    model_config as gnn_node_mass_model_config,
)
from configs.model.gnn_edge_mass import (
    model_config as gnn_edge_mass_model_config,
)
from utils.configs.evaluation_model import EvaluationModelConfig
from utils.configs.evaluation_model_selection import EvaluationModelSelectionConfig

evaluation_model_selection_config = EvaluationModelSelectionConfig(
    name="gnn_single_edge_mass",
    evaluation_model_configs=[
        EvaluationModelConfig(
            model_config=direct_tagging_model_config,
            run_selection="only_latest",
            run_aggregation="individual",
            is_comparison_base=True,
            display_name="Direct tagging",
        ),
        #EvaluationModelConfig(
        #    model_config=eff_map_pt_eta_model_config,
        #    run_selection="only_latest",
        #    run_aggregation="individual",
        #    is_comparison_base=False,
        #    display_name="Efficiency map",
        #),
        EvaluationModelConfig(
            model_config=gnn_node_mass_model_config,
            run_selection="only_latest",
            run_aggregation="median",
            is_comparison_base=False,
            display_name="GNN node Mass",
            only_bootstrap_runs=True,
        ),
        EvaluationModelConfig(
            model_config=gnn_edge_mass_model_config,
            run_selection="only_latest",
            run_aggregation="median",
            is_comparison_base=False,
            display_name="GNN edge Mass",
            only_bootstrap_runs=True,
        ),
    ],
)
