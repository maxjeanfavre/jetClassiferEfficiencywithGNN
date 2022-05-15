"""Defines the WorkingPointConfig class."""
from typing import Tuple

from utils.configs.config import Config


class WorkingPointConfig(Config):
    """Container for information about a working point.

    Attributes:
        name: The name of the configuration.
        expression:
    """

    def __init__(
        self,
        name: str,
        expression: str,
        required_columns: Tuple[str, ...],
    ):
        super().__init__(name=name)

        self.expression = expression
        self._required_columns = required_columns

    def __eq__(self, other):
        """Overrides the default implementation for equality."""
        if isinstance(other, WorkingPointConfig):
            return self.name == other.name and self.expression == other.expression
        return NotImplemented

    def get_required_columns(self) -> Tuple[str, ...]:
        required_columns = tuple(i for i in self._required_columns)
        return required_columns
