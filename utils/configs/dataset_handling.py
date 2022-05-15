"""Defines the DatasetHandlingConfig class."""
from typing import Tuple

from loguru import logger

from utils.configs.config import Config
from utils.data.manipulation.data_manipulator import DataManipulator
from utils.helpers.columns_data_manipulators import (
    determine_required_columns_multiple_data_manipulators,
)


class DatasetHandlingConfig(Config):
    """Container for information about the handling of a dataset.

    Attributes:
        name: The name of the configuration.
        train_split: Float in [0, 1] to determine size of the training split.
        test_split: Float in [0, 1] to determine size of the test split.
        train_test_split_random_state: Integer to make sure that
            the data splitting is consistent across runs.
        data_manipulators: Data Manipulators that should be applied.
    """

    def __init__(
        self,
        name: str,
        train_split: float,
        test_split: float,
        train_test_split_random_state: int,
        data_manipulators: Tuple[DataManipulator, ...],
    ):
        if train_split + test_split != 1:
            logger.warning(
                f"DatasetHandlingConfig '{name}' "
                "train and test split do not sum up to 1. "
                "They have values: "
                f"train_split: {train_split}, "
                f"test_split: {test_split}"
            )

        super().__init__(name=name)

        self.train_split = train_split
        self.test_split = test_split
        self.train_test_split_random_state = train_test_split_random_state
        self.data_manipulators = data_manipulators

    def get_required_columns(self) -> Tuple[str, ...]:
        required_columns = determine_required_columns_multiple_data_manipulators(
            data_manipulators=self.data_manipulators
        )

        return required_columns
