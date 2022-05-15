"""Defines the WorkingPointsSetConfig class."""
from typing import Tuple

from utils.configs.config import Config
from utils.configs.working_point import WorkingPointConfig


class WorkingPointsSetConfig(Config):
    """Container for information about a set of working points.

    Attributes:
        name: The name of the configuration.
        working_points: Tuple of WorkingPointConfigs.
    """

    def __init__(
        self,
        name: str,
        working_points: Tuple[WorkingPointConfig, ...],
    ):
        working_point_names = [
            working_point_config.name for working_point_config in working_points
        ]
        if len(working_point_names) != len(set(working_point_names)):
            raise ValueError(
                f"Duplicate in working_point_config names: {working_point_names}"
            )

        super().__init__(name=name)

        self.working_points = working_points

    def get_working_point_set_idx_and_config_by_name(
        self, working_point_name: str
    ) -> Tuple[int, WorkingPointConfig]:
        working_point_names = [
            working_point_config.name for working_point_config in self.working_points
        ]
        try:
            working_point_idx = working_point_names.index(working_point_name)
        except ValueError:
            raise ValueError(
                f"Specified working_point_name '{working_point_name}' was not found. "
                f"Possible values are: {working_point_names}"
            )

        working_point_config = self.working_points[working_point_idx]

        assert working_point_config.name == working_point_name

        return working_point_idx, working_point_config

    def __len__(self):
        return len(self.working_points)

    def get_required_columns(self) -> Tuple[str, ...]:
        required_columns = tuple(
            set(
                i
                for working_point_config in self.working_points
                for i in working_point_config.get_required_columns()
            )
        )

        return required_columns
