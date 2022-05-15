from typing import List

import numpy as np


class Preprocessor:
    def __init__(self) -> None:
        pass

    @staticmethod
    def check_data(data):
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Expected np.ndarray, got '{type(data)}' instead")
        else:
            if data.ndim != 2:
                raise ValueError(
                    f"Expected 2D array, got {data.ndim}D array instead:"
                    "\n"
                    f"array={data}."
                    "\n"
                    "Possible solution: Reshape using array.reshape(-1, 1)"
                )
            else:
                if data.shape[1] != 1:
                    raise ValueError(
                        f"Data has to be of shape (n, 1). Got shape {data.shape}"
                    )

    def check_is_fitted(self) -> None:
        # check appropriate attributes to determine whether the preprocessor is fitted
        raise NotImplementedError

    def fit(self, data) -> None:
        # self.check_data(data=data)
        # ...
        raise NotImplementedError

    def transform(self, data):
        # self.check_is_fitted()
        # self.check_data(data=data)
        # transformed_data = ...
        # return transformed_data
        raise NotImplementedError

    def inverse_transform(self, data):
        # self.check_is_fitted()
        # self.check_data(data=data)
        # inverse_transformed_data = ...
        # return inverse_transformed_data
        raise NotImplementedError

    def get_new_col_name(self, col_name: str) -> List[str]:
        # self.check_is_fitted()
        raise NotImplementedError

    # def __repr__(self):
    #     """Overrides the default implementation for representation."""
    #     raise NotImplementedError

    def __eq__(self, other):
        """Overrides the default implementation for equality."""
        raise NotImplementedError
