import itertools

import numpy as np
import pandas as pd
import pytest

from utils.efficiency_map.histogram import Histogram
from utils.helpers.basics.powerset import powerset


def create_df_and_bins(n_dims, n_bins, n_rows, under_over_flow):
    if isinstance(n_bins, int):
        bins_per_dim = tuple(n_bins for _ in range(n_dims))
    elif isinstance(n_bins, tuple):
        if len(n_bins) == 2:
            bins_per_dim = tuple(
                np.random.randint(low=n_bins[0], high=n_bins[1], size=n_dims).tolist()
            )
        else:
            raise ValueError(
                f"If n_bins is a tuple it must be of length 2. Got: {n_bins}"
            )
    else:
        raise ValueError(f"Unsupported value for argument 'n_bins': {n_bins}")

    if under_over_flow == "both":
        underflow = tuple(True for _ in range(n_dims))
        overflow = tuple(True for _ in range(n_dims))
    elif under_over_flow == "none":
        underflow = tuple(False for _ in range(n_dims))
        overflow = tuple(False for _ in range(n_dims))
    elif under_over_flow == "overflow":
        underflow = tuple(False for _ in range(n_dims))
        overflow = tuple(True for _ in range(n_dims))
    elif under_over_flow == "underflow":
        underflow = tuple(True for _ in range(n_dims))
        overflow = tuple(False for _ in range(n_dims))
    elif under_over_flow == "mixed":
        underflow = tuple((np.random.random(size=n_dims) > 0.5).tolist())
        overflow = tuple((np.random.random(size=n_dims) > 0.5).tolist())
    else:
        raise ValueError(
            f"Unsupported value for argument 'under_over_flow': {under_over_flow}"
        )

    df = pd.DataFrame(
        data=np.random.random(size=(n_rows, n_dims)),
        columns=[str(i) for i in range(n_dims)],
    )
    bins = {
        str(i): np.concatenate(
            (
                np.array([-np.inf]) if underflow[i] else np.array([]),
                np.linspace(0, 1, bins_per_dim[i] + 1),
                np.array([np.inf]) if overflow[i] else np.array([]),
            )
        )
        for i in range(n_dims)
    }

    return df, bins


def get_histogram(n_dims, n_bins, n_rows, under_over_flow):
    df, bins = create_df_and_bins(
        n_dims=n_dims,
        n_bins=n_bins,
        n_rows=n_rows,
        under_over_flow=under_over_flow,
    )

    hist = Histogram.from_df_and_bins(df=df, bins=bins)

    return hist


@pytest.mark.parametrize("n_rows", [10 ** 3])
@pytest.mark.parametrize("n_bins", [1, 2, 5, 10, 25, (1, 10), (1, 20), (10, 50)])
@pytest.mark.parametrize("n_dims", [1, 2, 3, 4])
class TestHistogram:
    @pytest.mark.parametrize(
        "under_over_flow", ["both", "overflow", "underflow", "mixed", "none"]
    )
    def test_project(self, n_dims, n_bins, n_rows, under_over_flow):
        df, bins = create_df_and_bins(
            n_dims=n_dims,
            n_bins=n_bins,
            n_rows=n_rows,
            under_over_flow=under_over_flow,
        )
        hist = Histogram.from_df_and_bins(df=df, bins=bins)
        for projection_variables in powerset(bins.keys()):
            if len(projection_variables) == 0:
                continue
            for projection_variables_permutation in itertools.permutations(
                projection_variables
            ):
                hist_projected = hist.project(
                    projection_variables=projection_variables_permutation
                )

                hist_directly = Histogram.from_df_and_bins(
                    df=df[list(projection_variables_permutation)],
                    bins={var: bins[var] for var in projection_variables_permutation},
                )
                assert hist_projected == hist_directly

    @pytest.mark.parametrize(
        "under_over_flow", ["both", "overflow", "underflow", "mixed", "none"]
    )
    def test_project_on_all_existing_variables(
        self, n_dims, n_bins, n_rows, under_over_flow
    ):
        hist = get_histogram(
            n_dims=n_dims, n_bins=n_bins, n_rows=n_rows, under_over_flow=under_over_flow
        )

        hist_projected = hist.project(projection_variables=hist.variables)

        assert hist == hist_projected

    @pytest.mark.parametrize(
        "under_over_flow", ["both", "overflow", "underflow", "mixed", "none"]
    )
    def test_without_under_over_flow(self, n_dims, n_bins, n_rows, under_over_flow):
        df, bins = create_df_and_bins(
            n_dims=n_dims,
            n_bins=n_bins,
            n_rows=n_rows,
            under_over_flow=under_over_flow,
        )
        hist = Histogram.from_df_and_bins(df=df, bins=bins)
        hist_without_under_over_flow = hist.without_under_over_flow()

        bins_without_under_over_flow = {k: v[np.isfinite(v)] for k, v in bins.items()}
        hist_without_under_over_flow_directly = Histogram.from_df_and_bins(
            df=df, bins=bins_without_under_over_flow
        )
        assert hist_without_under_over_flow == hist_without_under_over_flow_directly

    def test_without_under_over_flow_identity(self, n_dims, n_bins, n_rows):
        hist = get_histogram(
            n_dims=n_dims, n_bins=n_bins, n_rows=n_rows, under_over_flow="none"
        )

        hist_without_under_over_flow = hist.without_under_over_flow()

        assert hist_without_under_over_flow == hist

    @pytest.mark.parametrize(
        "under_over_flow", ["both", "overflow", "underflow", "mixed", "none"]
    )
    def test_without_under_over_flow_no_inf_in_result_edges(
        self, n_dims, n_bins, n_rows, under_over_flow
    ):
        hist = get_histogram(
            n_dims=n_dims, n_bins=n_bins, n_rows=n_rows, under_over_flow=under_over_flow
        )

        hist_without_under_over_flow = hist.without_under_over_flow()

        assert all(
            np.all(np.isfinite(edges)) for edges in hist_without_under_over_flow.edges
        )

    @pytest.mark.parametrize(
        "under_over_flow", ["both", "overflow", "underflow", "mixed", "none"]
    )
    def test_order_project_without_under_over_flow_irrelevant(
        self, n_dims, n_bins, n_rows, under_over_flow
    ):
        df, bins = create_df_and_bins(
            n_dims=n_dims,
            n_bins=n_bins,
            n_rows=n_rows,
            under_over_flow=under_over_flow,
        )
        hist = Histogram.from_df_and_bins(df=df, bins=bins)
        for projection_variables in powerset(bins.keys()):
            if len(projection_variables) == 0:
                continue
            for projection_variables_permutation in itertools.permutations(
                projection_variables
            ):

                hist_projected_without_under_over_flow = hist.project(
                    projection_variables=projection_variables_permutation
                ).without_under_over_flow()

                hist_without_under_over_flow_projected = (
                    hist.without_under_over_flow().project(
                        projection_variables=projection_variables_permutation
                    )
                )

                assert (
                    hist_projected_without_under_over_flow
                    == hist_without_under_over_flow_projected
                )

    @pytest.mark.parametrize(
        "under_over_flow", ["both", "overflow", "underflow", "mixed", "none"]
    )
    def test_get_bin_entry(self, n_dims, n_bins, n_rows, under_over_flow):
        df, bins = create_df_and_bins(
            n_dims=n_dims,
            n_bins=n_bins,
            n_rows=n_rows,
            under_over_flow=under_over_flow,
        )
        hist = Histogram.from_df_and_bins(df=df, bins=bins)

        res = hist.get_bin_entry(df=df)

        res_manual = []
        for _, r in df.iterrows():
            bin_indices = []
            for i, v in r.items():
                bin_idx = None
                for j in range(len(bins[i]) - 1):
                    if bins[i][j] <= v < bins[i][j + 1]:
                        bin_idx = j
                if bin_idx is None:
                    raise ValueError("Didn't find the bin")
                else:
                    bin_indices.append(bin_idx)
            hist_entry = hist.h.item(tuple(bin_indices))
            res_manual.append(hist_entry)

        res_manual = np.array(res_manual)

        np.testing.assert_array_equal(
            x=res,
            y=res_manual,
        )
