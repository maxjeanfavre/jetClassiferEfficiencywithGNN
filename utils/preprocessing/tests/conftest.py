from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest


@pytest.fixture(params=[1, 2, 10, 10 ** 3, 10 ** 5])
def n(request):
    return request.param


@pytest.fixture(
    params=[
        (
            [],
            pytest.raises(ValueError, match="Expected np.ndarray, got '.*' instead"),
        ),
        (
            [1, 2],
            pytest.raises(ValueError, match="Expected np.ndarray, got '.*' instead"),
        ),
        (
            np.array(1),
            pytest.raises(
                ValueError, match="Expected 2D array, got 0D array instead:.*"
            ),
        ),
        (
            np.array([1]),
            pytest.raises(
                ValueError, match="Expected 2D array, got 1D array instead:.*"
            ),
        ),
        (
            np.array([[[1]]]),
            pytest.raises(
                ValueError, match="Expected 2D array, got 3D array instead:.*"
            ),
        ),
        (
            np.array([[1, 2]]),
            pytest.raises(
                ValueError, match=r"Data has to be of shape \(n, 1\). Got shape .*"
            ),
        ),
        (
            np.array([[1]]),
            does_not_raise(),
        ),
    ]
)
def data_expected_exception(request):
    return request.param


@pytest.fixture()
def data(request, data_expected_exception):
    return data_expected_exception[0]


@pytest.fixture()
def expected_exception(request, data_expected_exception):
    return data_expected_exception[1]
