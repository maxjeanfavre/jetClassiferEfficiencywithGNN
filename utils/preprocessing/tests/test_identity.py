from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from utils.preprocessing.identity import Identity
from utils.preprocessing.tests.test_preprocessor import PreprocessorTestsParentClass


class TestIdentity(PreprocessorTestsParentClass):
    @staticmethod
    def get_unfitted_preprocessor():
        return Identity(fitted=False)

    @staticmethod
    def get_fitted_preprocessor():
        return Identity(fitted=True)

    def test_fit(self, n):
        data = np.random.random(size=(n, 1))

        identity = Identity()

        identity.fit(data=data)

    def test_transform(self, n):
        data = np.random.random(size=(n, 1))

        identity = Identity()
        identity.fit(data=data)

        transformed_data = identity.transform(data=data)

        np.testing.assert_array_equal(
            x=transformed_data,
            y=data,
        )

    def test_inverse_transform(self, n):
        data = np.random.random(size=(n, 1))

        identity = Identity()
        identity.fit(data=data)

        transformed_data = identity.transform(data=data)

        inverse_transformed_data = identity.inverse_transform(data=transformed_data)

        np.testing.assert_array_equal(
            x=inverse_transformed_data,
            y=data,
        )

    @pytest.mark.parametrize(
        "col_name,expected_new_col_name,expected_exception",
        [
            ("42", ["42"], does_not_raise()),
            (["foo"], None, pytest.raises(ValueError)),
        ],
    )
    def test_get_new_col_name(
        self, col_name, expected_new_col_name, expected_exception
    ):
        identity = Identity(fitted=True)

        with expected_exception:
            new_col_name = identity.get_new_col_name(col_name=col_name)

            assert new_col_name == expected_new_col_name

    @pytest.mark.parametrize(
        "identity", [Identity(fitted=False), Identity(fitted=True)]
    )
    def test___repr__(self, identity):
        identity_from_repr = eval(identity.__repr__())

        assert identity == identity_from_repr

    @pytest.mark.parametrize(
        "identity_1,identity_2,equal",
        [
            (Identity(fitted=False), Identity(fitted=False), True),
            (Identity(fitted=False), Identity(fitted=True), False),
            (Identity(fitted=True), Identity(fitted=False), False),
            (Identity(fitted=True), Identity(fitted=True), True),
        ],
    )
    def test___eq__(self, identity_1, identity_2, equal):
        assert (identity_1 == identity_2) == equal
        assert (identity_2 == identity_1) == equal
