import copy

from configs.model.gnn_edge_mass_dijet_addition import model_config as gnn_edge_mass_dijet_model_config
from utils.preprocessing.identity import Identity
from utils.preprocessing.normalizer import Normalizer

model_config = copy.deepcopy(gnn_edge_mass_dijet_model_config)

model_config.name = "gnn_node_edge_mass_addition"

model_config.model_init_kwargs["node_features_cols"].extend(
    [
        "Jet_mass",
    ]
)

model_config.model_init_kwargs["preprocessing_pipeline"].column_preprocessors.update(
    {
        "Jet_mass": Normalizer(),
    }
)
