import copy

from configs.model.gnn import model_config as gnn_model_config
from utils.models.gnn_single_wp_mode import GNNSingleWPMode

model_config = copy.deepcopy(gnn_model_config)

model_config.name = "gnn_single_wp_mode"

model_config.model_cls = GNNSingleWPMode
