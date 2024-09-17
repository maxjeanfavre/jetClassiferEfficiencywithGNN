from configs.model.direct_tagging import model_config as direct_tagging_model_config
from configs.model.eff_map_pt_eta import (
    model_config as eff_map_pt_eta_model_config,
)
from configs.model.gnn_node_njet_edge_mass_addition import (
    model_config as gnn_node_njet_edge_mass_addition_model_config,
)
from configs.model.gnn_node_njet_edge_mass_soustraction import (
    model_config as gnn_node_njet_edge_mass_soustraction_model_config,
)
from configs.model.gnn_node_njet_edge_mass_mass_center import (
    model_config as gnn_node_njet_edge_mass_mass_center_model_config,
)
from configs.model.gnn_node_njet_edge_mass_euclidian_norm import (
    model_config as gnn_node_njet_edge_mass_euclidian_norm_model_config,
)
from configs.model.gnn import (
    model_config as gnn_model_config,
)
from utils.configs.evaluation_model import EvaluationModelConfig
from utils.configs.evaluation_model_selection import EvaluationModelSelectionConfig

evaluation_model_selection_config = EvaluationModelSelectionConfig(
    name="gnn_node_njet_edge_mass_dijet_metric",
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
            model_config=gnn_node_njet_edge_mass_euclidian_norm_model_config,
            run_selection="all",
            run_aggregation="median",
            is_comparison_base=False,
            display_name="Euclidean norm",
            only_bootstrap_runs=True,
        ),
        EvaluationModelConfig(
            model_config=gnn_node_njet_edge_mass_mass_center_model_config,
            run_selection="all",
            run_aggregation="median",
            is_comparison_base=False,
            display_name="Center of mass",
            only_bootstrap_runs=True,
        ),
        EvaluationModelConfig(
            model_config=gnn_node_njet_edge_mass_soustraction_model_config,
            run_selection="all",
            run_aggregation="median",
            is_comparison_base=False,
            display_name="Subtraction",
            only_bootstrap_runs=True,
        ),
        EvaluationModelConfig(
            model_config=gnn_node_njet_edge_mass_addition_model_config,
            run_selection="all",
            run_aggregation="median",
            is_comparison_base=False,
            display_name="Addition",
            only_bootstrap_runs=True,
        ),
    ],
)
