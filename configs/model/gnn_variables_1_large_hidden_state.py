import copy

from configs.model.gnn import model_config as gnn_model_config
from utils.preprocessing.identity import Identity
from utils.preprocessing.normalizer import Normalizer

model_config = copy.deepcopy(gnn_model_config)

model_config.name = "gnn_variables_1_large_hidden_state"

model_config.model_init_kwargs["node_features_cols"].extend(
    [
        "Jet_nConstituents",
        "nJet",
        "Jet_mass",
        "Jet_area",
    ]
)

model_config.model_init_kwargs["preprocessing_pipeline"].column_preprocessors.update(
    {
        "Jet_nConstituents": Identity(),
        "nJet": Normalizer(),
        "Jet_mass": Normalizer(),
        "Jet_area": Normalizer(),
    }
)

model_config.model_init_kwargs["edge_hidden_state_sizes"] = [
    1024,
    1024,
    1024,
    1024,
    1024,
]
model_config.model_init_kwargs["node_hidden_state_sizes"] = [
    2048,
    2048,
    2048,
    2048,
    2048,
]
model_config.model_init_kwargs["jet_efficiency_net_hidden_layers"] = [
    2048,
    512,
    128,
    64,
]
