import pathlib
from typing import Dict, List, Optional, Union

import numpy as np
from matplotlib import pyplot as plt

import utils
from utils.data.jet_events_dataset import JetEventsDataset
from utils.helpers.histograms.distances.bhattacharyya import (
    compute_bhattacharyya_distance,
)
from utils.helpers.histograms.distances.chi_squared import chi_squared_bin_wise
from utils.helpers.histograms.distances.rmse import compute_rmse_distance
from utils.plots.evaluation_histogram import plot_evaluation_histogram
from utils.plots.save_figure import save_figure


def create_jet_variable_histogram(
    jds: JetEventsDataset,
    eff_pred_cols: List[str],
    var_col: str,
    var_name_nice: str,
    unit: Optional[str],
    evaluation_dir_path: pathlib.Path,
    bins: Union[int, np.ndarray] = 20,
    comparison_col: Optional[str] = None,
) -> Dict:
    evaluation_data = {}

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

        filename = f"jet_variable_histogram_{var_col}_{flavour}"

        if flavour == "inclusive":
            title = f"{var_name_nice} (all flavours)"
        else:
            title = f"{var_name_nice} (only {utils.flavours_niceify[flavour]} jets)"

        if unit is not None:
            xlabel = f"{var_name_nice} {unit}"
        else:
            xlabel = f"{var_name_nice}"

        fig, hist_data = plot_evaluation_histogram(
            data=df[var_col].to_numpy(),
            xlabel=xlabel,
            ylabel="Jets",
            weights={
                eff_pred_col: df[eff_pred_col].to_numpy()
                for eff_pred_col in eff_pred_cols
            },
            title=title,
            statistical_errors=True,
            bins=bins,
            comparison_name=comparison_col,
        )

        save_figure(
            fig=fig,
            path=evaluation_dir_path,
            filename=filename,
        )

        plt.close(fig=fig)

        evaluation_data[title] = {}

        if comparison_col is not None:
            for model_name in hist_data.keys():
                if model_name != comparison_col:
                    y_obs = hist_data[model_name]["y"]
                    y_exp = hist_data[comparison_col]["y"]

                    chi_squared = chi_squared_bin_wise(y_obs=y_obs, y_exp=y_exp)
                    rmse = compute_rmse_distance(y_1=y_exp, y_2=y_obs)
                    bhattacharyya = compute_bhattacharyya_distance(y_1=y_obs, y_2=y_exp)

                    evaluation_data[title][model_name] = {
                        "chi_squared": chi_squared,
                        "rmse": rmse,
                        "bhattacharyya": bhattacharyya,
                    }

    return evaluation_data
