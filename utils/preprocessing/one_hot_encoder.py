from typing import List

import sklearn.exceptions
import sklearn.preprocessing
import sklearn.utils.validation

from utils.exceptions import NotFittedError
from utils.preprocessing.preprocessor import Preprocessor


class OneHotEncoder(Preprocessor):
    def __init__(self) -> None:
        super().__init__()

        self.enc = sklearn.preprocessing.OneHotEncoder(sparse=False)

    def check_is_fitted(self) -> None:
        # taken from sklearn.utils.validation.check_is_fitted()
        # fitted = [
        #     v for v in vars(self.enc) if v.endswith("_") and not v.startswith("__")
        # ]
        # if not fitted:  # 'not fitted' is True when fitted is empty
        try:
            sklearn.utils.validation.check_is_fitted(estimator=self.enc)
        except sklearn.exceptions.NotFittedError as sklearn_not_fitted_error:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments first"
            ) from sklearn_not_fitted_error

    def fit(self, data):
        self.check_data(data=data)

        self.enc.fit(data)

    def transform(self, data):
        self.check_is_fitted()
        self.check_data(data=data)

        transformed_data = self.enc.transform(X=data)

        return transformed_data

    def inverse_transform(self, data):
        self.check_is_fitted()
        self.check_data(data=data)

        inverse_transformed_data = self.enc.inverse_transform(X=data)

        return inverse_transformed_data

    def get_new_col_name(self, col_name: str) -> List[str]:
        self.check_is_fitted()
        return [f"{col_name}_{cat}" for cat in self.enc.categories_[0]]

    # def __repr__(self):
    #     """Overrides the default implementation for representation."""
    #     raise NotImplementedError

    def __eq__(self, other):
        """Overrides the default implementation for equality."""
        if isinstance(other, OneHotEncoder):
            return self.enc == other.enc
            # this does not work, didn't find equality in sklearn
        return NotImplemented
