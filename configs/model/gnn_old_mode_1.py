import copy

from configs.model.gnn import model_config as gnn_model_config

model_config = copy.deepcopy(gnn_model_config)
model_config.name = "gnn_old_mode_1"
model_config.model_init_kwargs["old_mode"] = True
model_config.model_init_kwargs["old_mode_wp_idx"] = 1
