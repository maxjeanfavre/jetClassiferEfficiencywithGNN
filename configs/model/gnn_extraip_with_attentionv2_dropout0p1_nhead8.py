0;10;1cimport copy

from configs.model.gnn_extraip_with_attentionv2_dropout0p1_nhead8 import model_config as gnn_model_config
from utils.preprocessing.identity import Identity
from utils.preprocessing.normalizer import Normalizer

model_config = copy.deepcopy(gnn_model_config)

model_config.name = "gnn_extraip_with_attentionv2_dropout0p1_nhead8"

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
