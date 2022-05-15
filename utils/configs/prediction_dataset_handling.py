from typing import Tuple

from utils.configs.config import Config
from utils.data.manipulation.data_manipulator import DataManipulator
from utils.helpers.columns_data_manipulators import (
    determine_required_columns_multiple_data_manipulators,
)


class PredictionDatasetHandlingConfig(Config):
    """Container for information about the handling of a dataset for predictions.

    Attributes:
        name: The name of the configuration.
        data_manipulators: Data Manipulators that should be applied at all times.
    """

    def __init__(
        self, name: str, data_manipulators: Tuple[DataManipulator, ...]
    ) -> None:
        for data_manipulator in data_manipulators:
            assert data_manipulator.active_modes == ("predict",)

        super().__init__(name=name)

        self.data_manipulators = data_manipulators

    def get_required_columns(self) -> Tuple[str, ...]:
        required_columns = determine_required_columns_multiple_data_manipulators(
            data_manipulators=self.data_manipulators
        )

        return required_columns
