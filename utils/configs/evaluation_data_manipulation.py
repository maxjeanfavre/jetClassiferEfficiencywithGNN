from typing import Optional, Tuple

from utils.configs.config import Config
from utils.data.manipulation.data_manipulator import DataManipulator


class EvaluationDataManipulationConfig(Config):
    def __init__(
        self, name: str, data_manipulators: Optional[Tuple[DataManipulator, ...]]
    ) -> None:
        super().__init__(name=name)

        self.data_manipulators = data_manipulators
