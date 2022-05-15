from utils.configs.config import Config
from utils.configs.model import ModelConfig


class EvaluationModelConfig(Config):
    def __init__(
        self,
        model_config: ModelConfig,
        run_selection: str,
        run_aggregation: str,
        is_comparison_base: bool,
    ) -> None:
        super().__init__(
            name=(
                f"{model_config.name}"
                f"_{run_selection}"
                f"_{run_aggregation}"
                f"_{is_comparison_base}"
            )
        )

        self.model_config = model_config
        self.run_selection = run_selection
        self.run_aggregation = run_aggregation
        self.is_comparison_base = is_comparison_base
