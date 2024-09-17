from configs.model.direct_tagging import model_config as direct_tagging_model_config
from configs.model.eff_map_pt_eta import (
    model_config as eff_map_pt_eta_model_config,
)
from configs.model.gnn_node_njet_mass import (
    model_config as gnn_node_njet_model_config,
)
from configs.model.gnn import (
    model_config as gnn_model_config,
)
from configs.model.gnn_node_njet_edge_mass_euclidian_norm import (
    model_config as gnn_node_njet_edge_mass_euclidian_norm_model_config,
) 
from utils.configs.evaluation_model import EvaluationModelConfig
from utils.configs.evaluation_model_selection import EvaluationModelSelectionConfig

evaluation_model_selection_config = EvaluationModelSelectionConfig(
    name="gnn_node_njet_eucl_mass",
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
           run_selection="only_latest",
           run_aggregation="median",
           is_comparison_base=False,
           display_name="GNN",
           only_bootstrap_runs=False,
        ),
        EvaluationModelConfig(
           model_config=gnn_node_njet_model_config,
           run_selection="only_latest",
           run_aggregation="median",
           is_comparison_base=False,
           display_name="GNN nJet",
           only_bootstrap_runs=True,
        ),
        EvaluationModelConfig(
           model_config=gnn_node_njet_edge_mass_euclidian_norm_model_config,
           run_selection="only_latest",
           run_aggregation="median",
           is_comparison_base=False,
           display_name="GNN nJet + mass_dijet_eucl_norm",
           only_bootstrap_runs=True,
        ),
    ],
)