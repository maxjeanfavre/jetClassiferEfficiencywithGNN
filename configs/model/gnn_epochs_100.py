import copy

from configs.model.gnn import model_config as gnn_model_config

model_config = copy.deepcopy(gnn_model_config)

model_config.name = "gnn_epochs_100"

model_config.model_train_kwargs["epochs"] = 100
