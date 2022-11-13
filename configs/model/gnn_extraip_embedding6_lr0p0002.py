import copy

from configs.model.gnn import model_config as gnn_model_config
from utils.preprocessing.identity import Identity
from utils.preprocessing.normalizer import Normalizer

model_config = copy.deepcopy(gnn_model_config)

model_config.name = "gnn_extraip_embedding6_lr0p0002"

model_config.model_init_kwargs["flavour_embedding_dim"] = 6 

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

model_config.model_train_kwargs["optimizer_cls_init_kwargs"] = {"lr": 0.0002}
model_config.model_predict_kwargs["batch_size"] = 256 
