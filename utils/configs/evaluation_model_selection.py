from typing import List

from utils.configs.config import Config
from utils.configs.evaluation_model import EvaluationModelConfig


class EvaluationModelSelectionConfig(Config):
    def __init__(
        self, name: str, evaluation_model_configs: List[EvaluationModelConfig]
    ) -> None:
        if (
            sum(
                [
                    evaluation_model_config.is_comparison_base
                    for evaluation_model_config in evaluation_model_configs
                ]
            )
            > 1
        ):
            raise ValueError(
                "More than one model setting was marked as comparison_base"
            )

        super().__init__(name=name)

        self.evaluation_model_configs = evaluation_model_configs
