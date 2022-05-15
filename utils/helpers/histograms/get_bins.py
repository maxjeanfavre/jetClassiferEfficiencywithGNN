import numpy as np


# TODO(test): test cases where bin_edges is -np.inf, np.inf or -np.inf, 0, np.inf or -np.inf, 0 or 0, np.inf


def get_bin_edges_equidistant(
    lower: float,
    upper: float,
    n_bins: int,
    underflow: bool,
    overflow: bool,
) -> np.ndarray:
    if lower >= upper:
        raise ValueError(f"Need lower < upper. Got: lower: {lower}, upper: {upper}")
    if n_bins < 1:
        raise ValueError(f"n_bins has to be at least 1. Got: {n_bins}")

    bin_edges = np.linspace(
        start=lower,
        stop=upper,
        num=n_bins + 1,
        endpoint=True,
    )

    if underflow:
        bin_edges = np.concatenate((np.array([-np.inf]), bin_edges))
    if overflow:
        bin_edges = np.concatenate((bin_edges, np.array([np.inf])))

    assert len(bin_edges) >= 2
    assert len(bin_edges) == n_bins + 1 + underflow + overflow

    return bin_edges


def get_bin_widths_from_bin_edges(bin_edges: np.ndarray):
    # TODO(test): test it
    bin_widths = np.diff(bin_edges)

    if np.isinf(bin_widths[0]):
        bin_widths[0] = bin_widths[1]
    if np.isinf(bin_widths[-1]):
        bin_widths[-1] = bin_widths[-2]

    assert not np.any(np.isinf(bin_widths)), "inf values in 'bin_widths'"
    assert (
        len(bin_widths) == len(bin_edges) - 1
    ), "shape of 'bin_widths' is not compatible with 'bin_edges'"

    return bin_widths


def get_bin_midpoints_from_bin_edges(bin_edges: np.ndarray):
    # TODO(test): test it
    bin_midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2

    bin_widths = get_bin_widths_from_bin_edges(bin_edges=bin_edges)

    if np.isinf(bin_midpoints[0]):
        bin_midpoints[0] = bin_edges[1] - (bin_widths[0] / 2)
    if np.isinf(bin_midpoints[-1]):
        bin_midpoints[-1] = bin_edges[-2] + (bin_widths[-1] / 2)

    assert not np.any(np.isinf(bin_midpoints)), "inf values in 'bin_midpoints'"
    assert (
        len(bin_midpoints) == len(bin_edges) - 1
    ), "shape of 'bin_midpoints' is not compatible with 'bin_edges'"

    return bin_midpoints
