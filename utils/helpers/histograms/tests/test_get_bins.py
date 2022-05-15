from typing import Union

import numpy as np
import pytest

from utils.helpers.histograms.get_bins import get_bin_edges_equidistant


class TestGetBinEdgesEquidistant:
    @pytest.mark.parametrize("overflow", [False, True])
    @pytest.mark.parametrize("underflow", [False, True])
    @pytest.mark.parametrize("n_bins", [1, 5, 10, 10 ** 3])
    @pytest.mark.parametrize(
        "upper",
        list(np.linspace(start=-(10 ** 5), stop=10 ** 5, num=10 ** 1, endpoint=True)),
    )
    @pytest.mark.parametrize(
        "lower",
        list(
            np.linspace(
                start=-(10 ** 5) - 1, stop=10 ** 5 - 1, num=10 ** 1, endpoint=True
            )
        ),
    )
    def test_get_bin_edges_equidistant(
        self,
        lower: Union[int, float],
        upper: Union[int, float],
        n_bins: int,
        underflow: bool,
        overflow: bool,
    ):
        if lower > upper:
            lower, upper = upper, lower

        bin_edges = get_bin_edges_equidistant(
            lower=lower,
            upper=upper,
            n_bins=n_bins,
            underflow=underflow,
            overflow=overflow,
        )

        assert isinstance(bin_edges, np.ndarray)
        assert bin_edges.ndim == 1
        assert len(bin_edges) >= 2
        assert len(bin_edges) == n_bins + 1 + underflow + overflow
        assert np.sum(np.isnan(bin_edges)) == 0

        if underflow:
            assert bin_edges[0] == -np.inf
            assert bin_edges[1] == lower
        else:
            assert bin_edges[0] == lower

        if overflow:
            assert bin_edges[-1] == np.inf
            assert bin_edges[-2] == upper
        else:
            assert bin_edges[-1] == upper

        assert np.sum(np.isinf(bin_edges)) == underflow + overflow

        bin_width = (upper - lower) / n_bins
        bin_edges_manual = []
        if underflow:
            bin_edges_manual.append(-np.inf)
        bin_edges_manual.extend([lower + i * bin_width for i in range(n_bins + 1)])
        if overflow:
            bin_edges_manual.append(np.inf)
        bin_edges_manual = np.array(bin_edges_manual)

        np.testing.assert_allclose(
            actual=bin_edges,
            desired=bin_edges_manual,
        )

    def test_errors(self):
        with pytest.raises(ValueError):
            get_bin_edges_equidistant(
                lower=0,
                upper=0,
                n_bins=1,
                underflow=False,
                overflow=False,
            )
        with pytest.raises(ValueError):
            get_bin_edges_equidistant(
                lower=0,
                upper=-1,
                n_bins=1,
                underflow=False,
                overflow=False,
            )
        with pytest.raises(ValueError):
            get_bin_edges_equidistant(
                lower=0,
                upper=1,
                n_bins=0,
                underflow=False,
                overflow=False,
            )
