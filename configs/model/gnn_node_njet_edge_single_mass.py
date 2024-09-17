import copy

from configs.model.gnn_edge_mass import model_config as gnn_model_config
from utils.preprocessing.identity import Identity
from utils.preprocessing.normalizer import Normalizer

model_config = copy.deepcopy(gnn_model_config)

model_config.name = "gnn_node_njet_edge_single_mass"

model_config.model_init_kwargs["node_features_cols"].extend(
    [
        "nJet",
    ]
)

model_config.model_init_kwargs["preprocessing_pipeline"].column_preprocessors.update(
    {
        "nJet": Normalizer(),
    }
)