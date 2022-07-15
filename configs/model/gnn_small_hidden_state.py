import copy

from configs.model.gnn import model_config as gnn_model_config

model_config = copy.deepcopy(gnn_model_config)

model_config.name = "gnn_small_hidden_state"

model_config.model_init_kwargs["edge_hidden_state_sizes"] = [32, 32, 32, 32, 32]
model_config.model_init_kwargs["node_hidden_state_sizes"] = [64, 64, 64, 64, 64]
model_config.model_init_kwargs["jet_efficiency_net_hidden_layers"] = [64, 32, 16, 8]
