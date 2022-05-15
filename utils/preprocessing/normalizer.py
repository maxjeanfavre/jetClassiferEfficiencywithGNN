from typing import List, Optional, Union

import numpy as np

from utils.exceptions import NotFittedError
from utils.preprocessing.preprocessor import Preprocessor


class Normalizer(Preprocessor):
    def __init__(self, mean=None, std_dev=None) -> None:
        super().__init__()

        self._std_dev = None

        self.mean = mean
        self.std_dev = std_dev

    @property
    def std_dev(self):
        return self._std_dev

    @std_dev.setter
    def std_dev(self, value: Optional[Union[np.floating, int, float]]):
        if value is not None and not isinstance(value, (np.floating, int, float)):
            raise ValueError(
                f"Argument must be np.floating, int, or float. "
                f"Was of type {type(value)}"
            )
        self._std_dev = value

    def check_is_fitted(self) -> None:
        if self.mean is None and self.std_dev is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments first"
            )

    def fit(self, data):
        self.check_data(data=data)

        self.mean = np.mean(data)
        self.std_dev = np.std(data)

    def transform(self, data):
        self.check_is_fitted()
        self.check_data(data=data)

        transformed_data = data - self.mean
        if self.std_dev != 0:
            transformed_data = transformed_data / self.std_dev
        else:
            pass
        # transformed_data = (data - self.mean) / (
        #     self.std_dev if self.std_dev != 0 else 1
        # )

        return transformed_data

    def inverse_transform(self, data):
        self.check_is_fitted()
        self.check_data(data=data)

        if self.std_dev != 0:
            inverse_transformed_data = data * self.std_dev
        else:
            inverse_transformed_data = data
        inverse_transformed_data += self.mean

        return inverse_transformed_data

    def get_new_col_name(self, col_name: str) -> List[str]:
        if not isinstance(col_name, str):
            raise ValueError(
                "col_name has to be an instance of 'str'. "
                f"Got type {type(col_name)} instead"
            )
        self.check_is_fitted()
        return [f"{col_name}_normalized"]

    def __repr__(self):
        """Overrides the default implementation for representation."""
        return f"Normalizer(mean={self.mean}, std_dev={self.std_dev})"

    def __eq__(self, other):
        """Overrides the default implementation for equality."""
        if isinstance(other, Normalizer):
            return self.mean == other.mean and self.std_dev == other.std_dev
        return NotImplemented
