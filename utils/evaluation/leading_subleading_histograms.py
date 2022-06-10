import pathlib
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

import utils
from utils.data.dataframe_format import (
    get_idx_from_event_n_jets,
    reconstruct_event_n_jets_from_groupby_zeroth_level_values,
)
from utils.data.jet_events_dataset import JetEventsDataset
from utils.data.manipulation.data_filters.first_n_jets import FirstNJetsFilter
from utils.data.manipulation.data_filters.jet_multiplicity import JetMultiplicityFilter
from utils.helpers.histograms.distances.bhattacharyya import (
    compute_bhattacharyya_distance,
)
from utils.helpers.histograms.distances.chi_squared import chi_squared_bin_wise
from utils.helpers.histograms.distances.rmse import compute_rmse_distance
from utils.helpers.kinematics.delta_r_two_jet_events import (
    compute_delta_r_two_jet_events,
)
from utils.helpers.kinematics.invariant_mass import compute_invariant_mass
from utils.plots.evaluation_histogram import plot_evaluation_histogram
from utils.plots.save_figure import save_figure


def create_leading_subleading_histograms(
    jds: JetEventsDataset,
    eff_pred_cols: List[str],
    evaluation_dir_path: pathlib.Path,
    comparison_col: Optional[str] = None,
):
    if comparison_col is not None:
        assert comparison_col in eff_pred_cols

    warning_threshold = 0.01
    if np.mean(jds.event_n_jets < 2) > warning_threshold:
        logger.warning(
            f"More than {warning_threshold:.4%} of "
            f"the events had less than two jets left"
        )

    df_sorted = jds.df.sort_values(
        by=["entry", "Jet_Pt"], axis=0, ascending=[True, False]
    )

    df_sorted.index = jds.df.index

    del jds

    jds_sorted = JetEventsDataset(df=df_sorted)

    del df_sorted

    jds_sorted.manipulate(
        data_manipulators=(
            JetMultiplicityFilter(
                active_modes=("foo",),
                n=1,
                mode="remove",
            ),
            FirstNJetsFilter(active_modes=("foo",), n=2),
        ),
        mode="foo",
    )

    evaluation_data = {}

    for flavour_selection in [
        "inclusive",
        [5, 5],
        # [4, 4],
        # [0, 0],
        # [5, 4],
        # [5, 0],
        # [4, 0],
    ]:
        if flavour_selection == "inclusive":
            jds_selection = jds_sorted
        else:
            assert isinstance(flavour_selection, list)
            assert len(flavour_selection) == 2
            mask = np.logical_or(
                np.logical_and(
                    (
                        jds_sorted.df.loc[(slice(None), 0), "Jet_hadronFlavour"]
                        == flavour_selection[0]
                    ).to_numpy(),
                    (
                        jds_sorted.df.loc[(slice(None), 1), "Jet_hadronFlavour"]
                        == flavour_selection[1]
                    ).to_numpy(),
                ),
                np.logical_and(
                    (
                        jds_sorted.df.loc[(slice(None), 0), "Jet_hadronFlavour"]
                        == flavour_selection[1]
                    ).to_numpy(),
                    (
                        jds_sorted.df.loc[(slice(None), 1), "Jet_hadronFlavour"]
                        == flavour_selection[0]
                    ).to_numpy(),
                ),
            )
            df_selection = jds_sorted.df[np.repeat(mask, 2)]
            event_n_jets_after_selection = (
                reconstruct_event_n_jets_from_groupby_zeroth_level_values(
                    df=df_selection
                )
            )

            new_idx = get_idx_from_event_n_jets(
                event_n_jets=event_n_jets_after_selection
            )

            df_selection.index = new_idx

            jds_selection = JetEventsDataset(df=df_selection)

            del df_selection

        df = pd.DataFrame()
        df.index.name = "entry"

        for col in eff_pred_cols:
            df[col] = (
                jds_selection.df.loc[(slice(None), 0), col].to_numpy()  # leading jet
                * jds_selection.df.loc[
                    (slice(None), 1), col
                ].to_numpy()  # sub-leading jet
            )

        # sanity check
        df_old = jds_selection.df.groupby(level=0, sort=False)[eff_pred_cols].prod(
            min_count=2
        )
        same_cols = df_old.columns.intersection(df.columns)
        assert df_old[same_cols].equals(df[same_cols])
        del df_old
        # end sanity check

        df["invariant_mass"] = compute_invariant_mass(
            pt_1=jds_selection.df["Jet_Pt"].to_numpy()[0::2],
            pt_2=jds_selection.df["Jet_Pt"].to_numpy()[1::2],
            eta_1=jds_selection.df["Jet_eta"].to_numpy()[0::2],
            eta_2=jds_selection.df["Jet_eta"].to_numpy()[1::2],
            phi_1=jds_selection.df["Jet_phi"].to_numpy()[0::2],
            phi_2=jds_selection.df["Jet_phi"].to_numpy()[1::2],
        )

        df["delta_r"] = compute_delta_r_two_jet_events(
            event_n_jets=jds_selection.event_n_jets,
            eta=jds_selection.df["Jet_eta"].to_numpy(),
            phi=jds_selection.df["Jet_phi"].to_numpy(),
        )

        for var_col, var_name_nice, unit in [
            ["invariant_mass", r"$M_{1,2}$", "[GeV]"],
            ["delta_r", r"$\Delta R$", None],
        ]:
            if flavour_selection == "inclusive":
                filename = f"leading_subleading_{var_col}_{flavour_selection}"
                title_full = (
                    f"{var_name_nice} leading and sub-leading jet pair (all flavours)"
                )
                title_short = f"{var_name_nice} (all jet flavours)"
            else:
                filename = f"leading_subleading_{var_col}_{flavour_selection[0]}_{flavour_selection[1]}"
                title_full = (
                    f"{var_name_nice} leading and sub-leading jet pair "
                    "(flavours: "
                    f"{utils.flavours_niceify[flavour_selection[0]]} "
                    f"{utils.flavours_niceify[flavour_selection[1]]}"
                    ")"
                )
                title_short = (
                    f"{var_name_nice} "
                    "("
                    f"{utils.flavours_niceify[flavour_selection[0]]} "
                    f"{utils.flavours_niceify[flavour_selection[1]]}"
                    ")"
                )

            if unit is not None:
                xlabel = f"{var_name_nice} {unit}"
            else:
                xlabel = f"{var_name_nice}"

            fig, hist_data = plot_evaluation_histogram(
                data=df[var_col].to_numpy(),
                xlabel=xlabel,
                ylabel="Events",
                weights={
                    eff_pred_col: df[eff_pred_col].to_numpy()
                    for eff_pred_col in eff_pred_cols
                },
                title=title_full,
                statistical_errors=True,
                bins=20,
                comparison_name=comparison_col,
            )

            save_figure(
                fig=fig,
                path=evaluation_dir_path,
                filename=filename,
            )

            plt.close(fig=fig)

            evaluation_data[title_short] = {}

            if comparison_col is not None:
                for model_name in hist_data.keys():
                    if model_name != comparison_col:
                        y_obs = hist_data[model_name]["y"]
                        y_exp = hist_data[comparison_col]["y"]

                        chi_squared = chi_squared_bin_wise(y_obs=y_obs, y_exp=y_exp)
                        rmse = compute_rmse_distance(y_1=y_exp, y_2=y_obs)
                        bhattacharyya = compute_bhattacharyya_distance(
                            y_1=y_obs, y_2=y_exp
                        )

                        evaluation_data[title_short][model_name] = {
                            "chi_squared": chi_squared,
                            "rmse": rmse,
                            "bhattacharyya": bhattacharyya,
                        }

    return evaluation_data
