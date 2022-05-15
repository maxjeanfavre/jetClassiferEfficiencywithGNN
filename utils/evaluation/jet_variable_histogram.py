import pathlib
from typing import Dict, List, Optional, Union

import numpy as np
from matplotlib import pyplot as plt

import utils
from utils.data.jet_events_dataset import JetEventsDataset
from utils.helpers.histograms.chi_squared import chi_squared_bin_wise
from utils.helpers.rmse import compute_rmse
from utils.plots.histogram import plot_histogram


def create_jet_variable_histogram(
    jds: JetEventsDataset,
    eff_pred_cols: List[str],
    var_col: str,
    evaluation_dir_path: pathlib.Path,
    bins: Union[int, np.ndarray] = 20,
    comparison_col: Optional[str] = None,
) -> Dict:
    evaluation_data = dict()

    for flavour in [
        "inclusive",
        # 0,
        # 4,
        5,
    ]:
        if flavour == "inclusive":
            df = jds.df
        else:
            df = jds.df[jds.df["Jet_hadronFlavour"] == flavour]

        title_snake_case = f"{var_col}_{flavour}"
        title_nice = f"{var_col} {flavour}"

        fig, hist_data = plot_histogram(
            data=df[var_col].to_numpy(),
            data_name=var_col,
            weights={
                eff_pred_col: df[eff_pred_col].to_numpy()
                for eff_pred_col in eff_pred_cols
            },
            title=title_nice,
            statistical_errors=True,
            bins=bins,
            comparison_name=comparison_col,
        )

        fig.savefig(
            fname=evaluation_dir_path
            / utils.filenames.jet_variable_histogram_plot(title=title_snake_case),
            dpi=utils.settings.plots_dpi,
        )

        plt.close(fig)

        evaluation_data[title_snake_case] = dict()

        if comparison_col is not None:
            for name in hist_data.keys():
                y_obs = hist_data[name]["y"]
                y_exp = hist_data[comparison_col]["y"]

                chi_squared = chi_squared_bin_wise(y_obs=y_obs, y_exp=y_exp)
                rmse = compute_rmse(y_true=y_exp, y_pred=y_obs)

                evaluation_data[title_snake_case][name] = {
                    "chi_squared": chi_squared,
                    "rmse": rmse,
                }

    return evaluation_data
