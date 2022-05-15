import numpy as np
import pytest

from utils.helpers.histograms.bin_counts import compute_weighted_bin_counts_from_data
from utils.helpers.histograms.get_bins import get_bin_edges_equidistant


class TestComputeWeightedBinCountsFromData:
    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("overflow", [False, True])
    @pytest.mark.parametrize("underflow", [False, True])
    @pytest.mark.parametrize("n_bins", [1, 2, 10, 10 ** 2])
    @pytest.mark.parametrize("n", [0, 1, 2, 10, 10 ** 3, 10 ** 5])
    def test_compute_weighted_bin_counts_from_data(
        self,
        n: int,
        n_bins: int,
        underflow: bool,
        overflow: bool,
        weighted: bool,
    ):
        if n_bins == 0 and not underflow and not overflow:
            return

        x = np.random.random(n)

        bin_edges = get_bin_edges_equidistant(
            lower=0,
            upper=1,
            n_bins=n_bins,
            underflow=underflow,
            overflow=overflow,
        )

        if weighted is True:
            weights = np.random.random(n)
        else:
            weights = None

        (
            bin_counts,
            bin_counts_statistical_errors,
        ) = compute_weighted_bin_counts_from_data(
            x=x,
            bin_edges=bin_edges,
            weights=weights,
        )

        bin_counts_np, _ = np.histogram(
            a=x,
            bins=bin_edges,
            weights=weights,
        )

        np.testing.assert_allclose(
            actual=bin_counts,
            desired=bin_counts_np,
        )

    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("n_bins", [1, 2, 10, 10 ** 2])
    @pytest.mark.parametrize("n", [0, 1, 2, 10, 10 ** 3, 10 ** 5])
    def test_compute_weighted_bin_counts_with_edges_from_numpy(
        self, n, n_bins, weighted
    ):
        x = np.random.random(n)

        if weighted is True:
            weights = np.random.random(n)
        else:
            weights = None

        bin_counts_np, bin_edges_np = np.histogram(
            a=x,
            bins=n_bins,
            weights=weights,
        )

        # extend right edge slightly because in np.histogram, the outer right edge
        # is included, while in my implementation it is excluded
        bin_edges = np.copy(bin_edges_np)
        bin_edges[-1] += np.finfo(bin_edges.dtype).eps

        (
            bin_counts,
            bin_counts_statistical_errors,
        ) = compute_weighted_bin_counts_from_data(
            x=x,
            bin_edges=bin_edges,
            weights=weights,
        )

        np.testing.assert_allclose(
            actual=bin_counts,
            desired=bin_counts_np,
        )
