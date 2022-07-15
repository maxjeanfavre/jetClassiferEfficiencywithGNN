from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from utils.exceptions import UnexpectedError
from utils.helpers.histograms.bin_counts import (
    compute_bin_indices,
    compute_weighted_bin_counts_from_bin_indices,
)
from utils.helpers.histograms.get_bins import (
    get_bin_edges_equidistant,
    get_bin_midpoints_from_bin_edges,
    get_bin_widths_from_bin_edges,
)


def plot_evaluation_histogram(
    data: np.ndarray,
    xlabel: str,
    ylabel: str,
    weights: Dict[str, Optional[np.ndarray]],
    title: str,
    statistical_errors: bool,
    bins: Union[int, np.ndarray] = 20,
    comparison_name: Optional[str] = None,
    plot_info_string: Optional[str] = None,
) -> Tuple[plt.Figure, Dict]:
    if not isinstance(data, np.ndarray):
        raise ValueError(f"'data' has to be np.ndarray. Got type: {type(data)}")
    if data.ndim != 1:
        raise ValueError(f"'data' has to be 1d np.ndarray. Got ndim: {data.ndim}")

    if not isinstance(weights, dict):
        raise ValueError(f"'weights' has to be dict. Got type: {type(weights)}")
    for name, weight in weights.items():
        if weight is not None:
            if not isinstance(weight, np.ndarray):
                raise ValueError(
                    f"'weight' of '{name}' has to be np.ndarray. "
                    f"Got type: {type(weight)}"
                )
            if weight.ndim != 1:
                raise ValueError(
                    f"'weight' of '{name}' has to be 1d np.ndarray. "
                    f"Got ndim: {weight.ndim}"
                )

    if not isinstance(bins, (int, np.ndarray)):
        raise ValueError(f"'bins' has to be int or np.ndarray. Got type: {type(bins)}")
    if isinstance(bins, np.ndarray):
        if bins.ndim != 1:
            raise ValueError(f"'bins' has to be 1d np.ndarray. Got ndim: {bins.ndim}")

    if comparison_name is not None:
        if comparison_name not in weights:
            raise ValueError("'comparison_name' must be a key in 'weights'")

    if isinstance(bins, int):
        bin_edges = get_bin_edges_equidistant(
            lower=np.quantile(a=data, q=0.01),
            upper=np.quantile(a=data, q=0.99),
            n_bins=bins,
            # underflow=True,
            underflow=False,
            # overflow=True,
            overflow=False,
        )
    elif isinstance(bins, np.ndarray):
        bin_edges = bins
    else:
        raise UnexpectedError("You shouldn't reach this point")

    bin_widths = get_bin_widths_from_bin_edges(bin_edges=bin_edges)

    bin_midpoints = get_bin_midpoints_from_bin_edges(bin_edges=bin_edges)

    bin_indices = compute_bin_indices(
        x=data,
        bin_edges=bin_edges,
    )

    figsize = (9.5, 9.5)

    if comparison_name is not None:
        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            sharex="all",
            gridspec_kw={
                "width_ratios": [1],
                "height_ratios": [3, 1, 1],
            },
            constrained_layout=True,
            figsize=figsize,
        )
    else:
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            constrained_layout=True,
            figsize=figsize,
        )
        axes = [ax]

    fig.suptitle(title)

    hist_data = {}

    threshold_n_lines = 10  # can't be more than 10 because matplotlib default color cycle is of length 10

    if len(weights) <= threshold_n_lines:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    else:
        n_colors = len(weights)
        colors = plt.get_cmap("gist_rainbow")(np.linspace(0, 1, n_colors))

    assert len(colors) >= len(weights)
    color_cycler = iter(colors)

    used_colors = {}

    for name, weight in weights.items():
        (
            bin_counts,
            bin_counts_statistical_errors,
        ) = compute_weighted_bin_counts_from_bin_indices(
            bin_indices=bin_indices,
            n_bins=len(bin_edges) - 1,
            weights=weight,
        )

        hist_data[name] = {
            "y": bin_counts,
            "y_err": bin_counts_statistical_errors,
        }

        if (
            len(weights) > threshold_n_lines
            and comparison_name is not None
            and name == comparison_name
        ):
            color = "k"
        else:
            color = next(color_cycler)
        used_colors[name] = color

        if len(weights) > threshold_n_lines:
            if comparison_name is not None and name == comparison_name:
                label = name
            else:
                label = None
        else:
            label = name

        axes[0].errorbar(
            x=bin_midpoints,
            y=bin_counts,
            yerr=bin_counts_statistical_errors if statistical_errors else None,
            xerr=bin_widths / 2,
            capsize=5,
            ls="None",
            label=label,
            ecolor=color,
        )

    axes[0].legend()
    axes[0].set_ylabel(ylabel=ylabel)
    # axes[0].set_xscale("log")
    # axes[0].set_xticks(bin_edges[np.isfinite(bin_edges)])

    if comparison_name is None:
        axes[0].set_xlabel(xlabel=xlabel)

    if plot_info_string is not None:
        if plot_info_string.count("\n") > 9:  # more than 10 lines
            logger.warning(
                "Can't plot plot_info_string because it has too many lines: "
                f"{plot_info_string = }"
            )
        elif any(len(line) > 150 for line in plot_info_string.split("\n")):
            # at least one line is too long
            logger.warning(
                "Can't plot plot_info_string because at least one line is too long: "
                f"{plot_info_string = }"
            )
        else:
            axes[0].text(
                1,
                1,
                plot_info_string,
                horizontalalignment="right",
                verticalalignment="bottom",
                transform=axes[0].transAxes,
                fontsize="xx-small",
            )

    if comparison_name is not None:
        # plotted_something = False
        for name in hist_data.keys():
            if name != comparison_name:
                ratio_bin_counts = np.divide(
                    hist_data[name]["y"],
                    hist_data[comparison_name]["y"],
                    out=np.full(shape=len(hist_data[name]["y"]), fill_value=np.nan),
                    where=hist_data[comparison_name]["y"] != 0,
                )

                if np.any(np.isfinite(ratio_bin_counts)):
                    axes[1].errorbar(
                        x=bin_midpoints,
                        y=ratio_bin_counts,
                        yerr=None,
                        xerr=bin_widths / 2,
                        capsize=5,
                        ls="None",
                        label=name,
                        ecolor=used_colors[name],
                    )
                    # plotted_something = True

        axes[1].plot(
            bin_midpoints,
            [1 for _ in range(len(bin_midpoints))],
            "k--",
        )

        # if (
        #     plotted_something
        # ):  # only draw legend if something was plotted to avoid warning
        #     axes[1].legend()

        axes[1].set_ylabel("Ratio of central values")
        axes[1].set_ylim(
            [max(axes[1].get_ylim()[0], 0.8), min(axes[1].get_ylim()[1], 1.2)]
        )

        # plotted_something = False
        for name in hist_data.keys():
            if name != comparison_name:
                ratio_bin_counts_errors = np.divide(
                    hist_data[name]["y_err"],
                    hist_data[comparison_name]["y_err"],
                    out=np.full(shape=len(hist_data[name]["y_err"]), fill_value=np.nan),
                    where=hist_data[comparison_name]["y_err"] != 0,
                )

                if np.any(np.isfinite(ratio_bin_counts_errors)):
                    axes[2].errorbar(
                        x=bin_midpoints,
                        y=ratio_bin_counts_errors,
                        yerr=None,
                        xerr=bin_widths / 2,
                        capsize=5,
                        ls="None",
                        label=name,
                        ecolor=used_colors[name],
                    )
                    # plotted_something = True

        # if (
        #     plotted_something
        # ):  # only draw legend if something was plotted to avoid warning
        #     axes[2].legend()

        axes[2].set_xlabel(xlabel=xlabel)
        axes[2].set_ylabel("Ratio of error bars")

    return fig, hist_data
