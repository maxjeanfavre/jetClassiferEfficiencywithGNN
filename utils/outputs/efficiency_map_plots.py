import pathlib
from typing import List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import utils
from utils.configs.dataset import DatasetConfig
from utils.configs.dataset_handling import DatasetHandlingConfig
from utils.configs.model import ModelConfig
from utils.configs.working_points_set import WorkingPointsSetConfig
from utils.efficiency_map.histogram import Histogram
from utils.helpers.run_id_handler import RunIdHandler
from utils.models.binned_efficiency_map_model import BinnedEfficiencyMapModel


def plot_2d_histograms_as_3d_bars(
    hists: List[Histogram],
    titles: List[str],
    fig_title: str,
):
    # TODO(low): https://stackoverflow.com/questions/18602660/matplotlib-bar3d-clipping-problems
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(fig_title)

    for i, (hist, title) in enumerate(zip(hists, titles)):
        x_name = hist.variables[0]
        y_name = hist.variables[1]

        x_edges = hist.edges[0]
        y_edges = hist.edges[1]

        x_grid, y_grid = np.meshgrid(x_edges[:-1], y_edges[:-1])

        x = x_grid.ravel()
        y = y_grid.ravel()
        z = np.zeros_like(y)
        dx = np.tile(np.diff(x_edges), x_grid.shape[0])
        dy = np.repeat(np.diff(y_edges), y_grid.shape[1])
        dz = np.nan_to_num(hist.h).T.ravel()
        # TODO(test): test with histogram where the plot is known (e.g. one peak in
        #  a certain corner) to make sure using .T is correct

        color_map = cm.get_cmap("jet")
        max_height = np.max(dz)  # get range so we can normalize
        min_height = np.min(dz)
        # scale each dz to [0,1], and get their rgb values
        rgba = [color_map((k - min_height) / max_height) for k in dz]

        ax = fig.add_subplot(1, len(hists), 1 + i, projection="3d")

        ax.bar3d(
            x=x,
            y=y,
            z=z,
            dx=dx,
            dy=dy,
            dz=dz,
            shade=True,
            color=rgba,
        )
        ax.set_title(title)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)

    return fig


def plot_2d_histograms_as_3d_surfaces(
    hists: List[Histogram],
    titles: List[str],
    fig_title: str,
):
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(fig_title)

    for i, (hist, title) in enumerate(zip(hists, titles)):
        x_name = hist.variables[0]
        y_name = hist.variables[1]

        x_edges = hist.edges[0]
        y_edges = hist.edges[1]

        x_edges_midpoints = (x_edges[1:] + x_edges[:-1]) / 2
        y_edges_midpoints = (y_edges[1:] + y_edges[:-1]) / 2

        x_grid, y_grid = np.meshgrid(
            x_edges_midpoints,
            y_edges_midpoints,
        )

        ax = fig.add_subplot(1, len(hists), 1 + i, projection="3d")
        ax.plot_surface(
            x_grid,
            y_grid,
            hist.h.T,
            cmap="jet",
            alpha=0.5,
        )
        ax.set_title(title)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)

    return fig


def create_efficiency_map_histogram_plots(
    dataset_config: DatasetConfig,
    dataset_handling_config: DatasetHandlingConfig,
    working_points_set_config: WorkingPointsSetConfig,
    model_config: ModelConfig,
    output_dir_path: pathlib.Path,
):
    variables_to_plot = ("Jet_Pt", "Jet_eta")

    assert model_config.model_cls == BinnedEfficiencyMapModel
    assert model_config.model_init_kwargs["separation_cols"] == ("Jet_hadronFlavour",)

    model_run_ids = RunIdHandler.get_run_ids(
        dir_path=utils.paths.model_files_dir(
            dataset_name=dataset_config.name,
            dataset_handling_name=dataset_handling_config.name,
            working_points_set_name=working_points_set_config.name,
            model_name=model_config.name,
            run_id="foo",
            mkdir=False,
        ).parent,
        only_latest=False,
    )

    for model_run_id in model_run_ids:
        model_dir_path = utils.paths.model_files_dir(
            dataset_name=dataset_config.name,
            dataset_handling_name=dataset_handling_config.name,
            working_points_set_name=working_points_set_config.name,
            model_name=model_config.name,
            run_id=model_run_id,
            mkdir=True,
        )

        model = BinnedEfficiencyMapModel.load(path=model_dir_path)

        assert model.bem.separation_cols == ("Jet_hadronFlavour",)

        bem = model.bem.project(
            projection_variables=variables_to_plot,
        ).without_under_over_flow()

        for working_point, working_point_histograms in bem.histograms.items():
            hists = []
            titles = []
            flavours = sorted(working_point_histograms.keys())
            for flavour in flavours:  # to have consistent order of flavours
                flavour_eff_hist = working_point_histograms[flavour]["eff_mode"]
                hists.append(flavour_eff_hist)
                titles.append(f"{bem.separation_cols[0]}: {flavour}")

            fig_surfaces = plot_2d_histograms_as_3d_surfaces(
                hists=hists,
                titles=titles,
                fig_title=working_point,
            )

            fig_surfaces.savefig(
                fname=model_dir_path / f"efficiency_map_{working_point}_surfaces.png",
                dpi=utils.settings.plots_dpi,
                bbox_inches="tight",
            )
            fig_surfaces.savefig(
                fname=output_dir_path / f"{model_config.name}_{working_point}_surfaces.png",
                dpi=utils.settings.plots_dpi,
                bbox_inches="tight",
            )

            plt.close(fig=fig_surfaces)

            fig_bars = plot_2d_histograms_as_3d_bars(
                hists=hists,
                titles=titles,
                fig_title=working_point,
            )

            fig_bars.savefig(
                fname=model_dir_path / f"efficiency_map_{working_point}_bars.png",
                dpi=utils.settings.plots_dpi,
                bbox_inches="tight",
            )
            fig_bars.savefig(
                fname=output_dir_path / f"{model_config.name}_{working_point}_bars.png",
                dpi=utils.settings.plots_dpi,
                bbox_inches="tight",
            )

            plt.close(fig=fig_bars)
