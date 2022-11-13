import copy

from configs.dataset.QCD_Pt_300_470_MuEnrichedPt5 import (
    dataset_config as original_dataset_config,
)

dataset_config = copy.deepcopy(original_dataset_config)

dataset_config.name += "_testv2"
dataset_config.file_limit = 1
