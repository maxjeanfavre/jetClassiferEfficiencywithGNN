from utils.configs.model import ModelConfig
from utils.models.direct_tagging import DirectTagging

model_config = ModelConfig(
    name="direct_tagging",
    data_manipulators=(),
    model_cls=DirectTagging,
    model_init_kwargs={},
    model_train_kwargs={},
    model_predict_kwargs={},
)
