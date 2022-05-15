import pytest

from utils.exceptions import NotFittedError
from utils.preprocessing.preprocessor import Preprocessor


# this test class is for methods that are implemented in the Preprocessor Class
class TestPreprocessor:
    def test_check_data(self, data, expected_exception):
        preprocessor = Preprocessor()

        with expected_exception:
            preprocessor.check_data(data=data)


# this test class is for methods that are implemented in subclasses of Preprocessor
# name chosen, so it is not discovered by pytest because it
# is not supposed to run directly but only when subclassed
class PreprocessorTestsParentClass:
    @staticmethod
    def get_unfitted_preprocessor():
        raise NotImplementedError

    @staticmethod
    def get_fitted_preprocessor():
        raise NotImplementedError

    def test_unfitted_check_is_fitted(self):
        preprocessor = self.get_unfitted_preprocessor()

        with pytest.raises(NotFittedError):
            preprocessor.check_is_fitted()

    def test_fitted_check_is_fitted(self):
        preprocessor = self.get_fitted_preprocessor()

        preprocessor.check_is_fitted()

    def test_fit_calls_check_data(self, data, expected_exception):
        preprocessor = self.get_unfitted_preprocessor()

        with expected_exception:
            preprocessor.fit(data=data)

    def test_transform_check_is_fitted(self):
        preprocessor = self.get_unfitted_preprocessor()

        with pytest.raises(NotFittedError):
            preprocessor.transform(data=[])

    def test_transform_check_data(self, data, expected_exception):
        preprocessor = self.get_fitted_preprocessor()

        with expected_exception:
            preprocessor.transform(data=data)

    def test_inverse_transform_check_is_fitted(self):
        preprocessor = self.get_unfitted_preprocessor()

        with pytest.raises(NotFittedError):
            preprocessor.inverse_transform(data=[])

    def test_inverse_transform_check_data(self, data, expected_exception):
        preprocessor = self.get_fitted_preprocessor()

        with expected_exception:
            preprocessor.inverse_transform(data=data)

    def test_get_new_col_name_check_is_fitted(self):
        preprocessor = self.get_unfitted_preprocessor()

        with pytest.raises(NotFittedError):
            preprocessor.get_new_col_name(col_name="42")

    # def test_check_is_fitted(self):
    #     assert False
    #
    # def test_fit(self):
    #     assert False
    #
    # def test_transform(self):
    #     assert False
    #
    # def test_inverse_transform(self):
    #     assert False
    #
    # def test_get_new_col_name(self):
    #     assert False
    #
    # def test___repr__(self):
    #     assert False
    #
    # def test___eq__(self):
    #     assert False


# TODO(test): in all preprocessors, make sure the result of transform and
#  inverse_transform has base None (not a view)?
# TODO(test): in all preprocessors, have to test that wrong data is detected to test
#  that I actually call self.check_is_fitted in the relevant methods
