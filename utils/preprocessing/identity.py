from typing import List

from utils.exceptions import NotFittedError
from utils.preprocessing.preprocessor import Preprocessor


class Identity(Preprocessor):
    def __init__(self, fitted: bool = False) -> None:
        super().__init__()

        self.fitted = fitted

    def check_is_fitted(self) -> None:
        if not self.fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments first"
            )

    def fit(self, data):
        self.check_data(data=data)  # To make sure data is appropriate
        self.fitted = True

    def transform(self, data):
        self.check_is_fitted()
        self.check_data(data=data)

        transformed_data = data

        return transformed_data

    def inverse_transform(self, data):
        self.check_is_fitted()
        self.check_data(data=data)

        inverse_transformed_data = data

        return inverse_transformed_data

    def get_new_col_name(self, col_name: str) -> List[str]:
        if not isinstance(col_name, str):
            raise ValueError(
                "col_name has to be an instance of 'str'. "
                f"Had type {type(col_name)} instead"
            )
        self.check_is_fitted()
        return [col_name]

    def __repr__(self):
        """Overrides the default implementation for representation."""
        return f"Identity(fitted={self.fitted})"

    def __eq__(self, other):
        """Overrides the default implementation for equality."""
        if isinstance(other, Identity):
            return self.fitted == other.fitted
        return NotImplemented
