from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from utils.preprocessing.normalizer import Normalizer
from utils.preprocessing.tests.test_preprocessor import PreprocessorTestsParentClass


class TestNormalizer(PreprocessorTestsParentClass):
    @staticmethod
    def get_unfitted_preprocessor():
        return Normalizer()

    @staticmethod
    def get_fitted_preprocessor():
        return Normalizer(mean=1.1, std_dev=0.128)

    def test_fit(self, n):
        data = np.random.random(size=(n, 1))

        normalizer = Normalizer()

        normalizer.fit(data=data)

        assert normalizer.mean == np.mean(data)
        assert normalizer.std_dev == np.std(data)

    def test_transform(self, n):
        data = np.random.random(size=(n, 1))

        normalizer = Normalizer()
        normalizer.fit(data=data)

        transformed_data = normalizer.transform(data=data)

        np.testing.assert_allclose(
            actual=np.mean(transformed_data),
            desired=0,
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            actual=np.std(transformed_data),
            desired=1 if n != 1 else 0,
        )

    def test_inverse_transform(self, n):
        data = np.random.random(size=(n, 1))

        normalizer = Normalizer()
        normalizer.fit(data=data)

        transformed_data = normalizer.transform(data=data)

        inverse_transformed_data = normalizer.inverse_transform(data=transformed_data)

        np.testing.assert_allclose(
            actual=inverse_transformed_data,
            desired=data,
        )

    @pytest.mark.parametrize(
        "col_name,expected_new_col_name,expected_exception",
        [
            ("42", ["42_normalized"], does_not_raise()),
            (["foo"], None, pytest.raises(ValueError)),
        ],
    )
    def test_get_new_col_name(
        self, col_name, expected_new_col_name, expected_exception
    ):
        normalizer = self.get_fitted_preprocessor()
        with expected_exception:
            new_col_name = normalizer.get_new_col_name(col_name=col_name)

            assert new_col_name == expected_new_col_name

    @pytest.mark.parametrize(
        "normalizer",
        [
            Normalizer(mean=None, std_dev=None),
            Normalizer(mean=None, std_dev=0.128),
            Normalizer(mean=1.1, std_dev=None),
            Normalizer(mean=1.1, std_dev=0.128),
        ],
    )
    def test___repr__(self, normalizer):
        normalizer_from_repr = eval(normalizer.__repr__())

        assert normalizer == normalizer_from_repr

    @pytest.mark.parametrize("std_dev_2", [None, 0.128, 0.2])
    @pytest.mark.parametrize("mean_2", [None, 1.1, 2.1])
    @pytest.mark.parametrize("std_dev_1", [None, 0.128, 0.2])
    @pytest.mark.parametrize("mean_1", [None, 1.1, 2.1])
    def test___eq__(self, mean_1, std_dev_1, mean_2, std_dev_2):
        equal = mean_1 == mean_2 and std_dev_1 == std_dev_2
        normalizer_1 = Normalizer(mean=mean_1, std_dev=std_dev_1)
        normalizer_2 = Normalizer(mean=mean_2, std_dev=std_dev_2)
        assert (normalizer_1 == normalizer_2) == equal
        assert (normalizer_2 == normalizer_1) == equal

    @pytest.mark.parametrize(
        "std_dev,expected_exception",
        [
            (1, does_not_raise()),
            (1.0, does_not_raise()),
            (0, does_not_raise()),
            (0.0, does_not_raise()),
            ([1], pytest.raises(ValueError)),
        ],
    )
    def test_std_dev_setter(self, std_dev, expected_exception):
        with expected_exception:
            Normalizer(mean=0, std_dev=std_dev)
