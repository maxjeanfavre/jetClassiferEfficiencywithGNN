import copy

from configs.model.gnn import model_config as gnn_model_config

model_config = copy.deepcopy(gnn_model_config)

model_config.name = "gnn_x_large_hidden_state"

model_config.model_init_kwargs["edge_hidden_state_sizes"] = [
    2048,
    2048,
    2048,
    2048,
    2048,
]
model_config.model_init_kwargs["node_hidden_state_sizes"] = [
    4096,
    4096,
    4096,
    4096,
    4096,
]
model_config.model_init_kwargs["jet_efficiency_net_hidden_layers"] = [
    4096,
    1024,
    256,
    64,
]

model_config.model_train_kwargs["epochs"] = 5
model_config.model_train_kwargs["batch_size"] = 256
model_config.model_predict_kwargs["batch_size"] = 256
