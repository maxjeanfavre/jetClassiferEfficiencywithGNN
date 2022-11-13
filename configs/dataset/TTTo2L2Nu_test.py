
import copy

from configs.dataset.TTTo2L2Nu import dataset_config as original_dataset_config

dataset_config = copy.deepcopy(original_dataset_config)

dataset_config.name += "_test"
dataset_config.file_limit = 1
