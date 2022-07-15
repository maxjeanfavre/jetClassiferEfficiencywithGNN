import copy

from configs.model.gnn import model_config as gnn_model_config

model_config = copy.deepcopy(gnn_model_config)

model_config.name = "gnn_one_gn_block"

model_config.model_init_kwargs["edge_hidden_state_sizes"] = [256]
model_config.model_init_kwargs["node_hidden_state_sizes"] = [512]
model_config.model_init_kwargs["jet_efficiency_net_hidden_layers"] = [512, 256, 128, 50]
