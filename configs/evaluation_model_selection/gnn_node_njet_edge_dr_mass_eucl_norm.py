from configs.model.direct_tagging import model_config as direct_tagging_model_config
from configs.model.gnn_node_njet import (
    model_config as gnn_node_njet_model_config,
)
#from configs.model.gnn_edge_mass import (
#    model_config as gnn_edge_mass_model_config,
#)
from configs.model.gnn_node_njet_edge_mass_euclidian_norm import (
    model_config as gnn_node_njet_edge_mass_euclidian_norm_model_config,
) 
from configs.model.gnn_node_njet_edge_eucl_norm_mass_dr import (
    model_config as gnn_node_njet_edge_eucl_norm_mass_dr_model_config,
)
from utils.configs.evaluation_model import EvaluationModelConfig
from utils.configs.evaluation_model_selection import EvaluationModelSelectionConfig

evaluation_model_selection_config = EvaluationModelSelectionConfig(
    name="gnn_node_njet_edge_dr_mass_eucl_norm",
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
            run_selection="only_latest",
            run_aggregation="median",
            is_comparison_base=False,
            display_name="GNN edge Mass dijet",
            only_bootstrap_runs=True,
        ),
        EvaluationModelConfig(
            model_config=gnn_node_njet_model_config,
            run_selection="only_latest",
            run_aggregation="median",
            is_comparison_base=False,
            display_name="GNN edge $\Delta R$",
            only_bootstrap_runs=True,
        ),
        EvaluationModelConfig(
            model_config=gnn_node_njet_edge_eucl_norm_mass_dr_model_config,
            run_selection="only_latest",
            run_aggregation="median",
            is_comparison_base=False,
            display_name="GNN edge Mass dijet + $\Delta R$",
            only_bootstrap_runs=True,
        ),
    ],
)
