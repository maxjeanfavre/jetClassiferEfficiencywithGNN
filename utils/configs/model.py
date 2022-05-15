from typing import Any, Dict, Tuple, Type

from utils.configs.config import Config
from utils.data.manipulation.data_manipulator import DataManipulator
from utils.helpers.columns_data_manipulators import (
    determine_required_columns_multiple_data_manipulators,
)
from utils.models.model import Model


class ModelConfig(Config):
    def __init__(
        self,
        name: str,
        data_manipulators: Tuple[DataManipulator, ...],
        model_cls: Type[Model],
        model_init_kwargs: Dict[str, Any],
        model_train_kwargs: Dict[str, Any],
        model_predict_kwargs: Dict[str, Any],
    ):
        super().__init__(name=name)

        self.data_manipulators = data_manipulators
        self.model_cls = model_cls
        self.model_init_kwargs = model_init_kwargs
        self.model_train_kwargs = model_train_kwargs
        self.model_predict_kwargs = model_predict_kwargs

    def get_required_columns(self) -> Tuple[str, ...]:
        required_columns = set()

        required_columns.update(
            determine_required_columns_multiple_data_manipulators(
                data_manipulators=self.data_manipulators
            )
        )

        columns_to_be_read_in = tuple(required_columns)

        return columns_to_be_read_in
