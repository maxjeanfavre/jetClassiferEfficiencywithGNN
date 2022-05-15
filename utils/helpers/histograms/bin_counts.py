from typing import Optional, Tuple

import numpy as np


def compute_weighted_bin_counts_from_data(
    x: np.ndarray,
    bin_edges: np.ndarray,
    weights: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    bin_indices = compute_bin_indices(
        x=x,
        bin_edges=bin_edges,
    )

    (
        bin_counts,
        bin_counts_statistical_errors,
    ) = compute_weighted_bin_counts_from_bin_indices(
        bin_indices=bin_indices,
        n_bins=len(bin_edges) - 1,
        weights=weights,
    )

    return bin_counts, bin_counts_statistical_errors


def compute_bin_indices(
    x: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    bin_indices = np.digitize(
        x=x, bins=bin_edges, right=False
    )  # these are the indices of the right edge of the bin the values fall into
    bin_indices -= 1  # now they are the bin indices

    return bin_indices


def compute_weighted_bin_counts_from_bin_indices(
    bin_indices: np.ndarray,
    n_bins: int,
    weights: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(bin_indices, np.ndarray):
        raise ValueError(
            f"bin_indices has to be np.ndarray. Got type: {type(bin_indices)}"
        )
    if bin_indices.ndim != 1:
        raise ValueError(
            f"bin_indices has to be 1D np.ndarray. Had {bin_indices.ndim} dimensions"
        )
    if np.any((bin_indices % 1) != 0):
        raise ValueError(
            "bin_indices can only contain whole numbers. "
            f"Got these unqique values: {set(bin_indices)}"
        )
    if not isinstance(n_bins, int):
        raise ValueError(f"n_bins has to be int. Got type: {type(n_bins)}")
    if weights is not None:
        if not isinstance(weights, np.ndarray):
            raise ValueError(
                f"weights has to be None or np.ndarray. Got type: {type(weights)}"
            )
        if weights.ndim != 1:
            raise ValueError(
                "If weights is not None, it has to be a 1D np.ndarray. "
                f"Had {weights.ndim} dimensions"
            )
        if weights.shape != bin_indices.shape:
            raise ValueError(
                "weights has to be in the same shape as bin_indices."
                f"Got shape: bin_indices: {bin_indices.shape}, weights: {weights.shape}"
            )

    bin_counts = []
    bin_counts_statistical_errors = []

    for bin_idx in range(n_bins):
        mask_bin = np.where(bin_indices == bin_idx)
        assert len(mask_bin) == 1
        if weights is not None:
            weights_bin = weights[mask_bin]
            bin_count = np.nansum(weights_bin)
            bin_count_statistical_error = np.sqrt(np.nansum(np.square(weights_bin)))
        else:
            bin_count = len(mask_bin[0])
            bin_count_statistical_error = np.sqrt(bin_count)

        bin_counts.append(bin_count)
        bin_counts_statistical_errors.append(bin_count_statistical_error)

    bin_counts = np.array(bin_counts)
    bin_counts_statistical_errors = np.array(bin_counts_statistical_errors)

    assert len(bin_counts) == n_bins
    assert len(bin_counts_statistical_errors) == n_bins

    return bin_counts, bin_counts_statistical_errors
