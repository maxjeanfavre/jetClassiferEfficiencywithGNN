import pathlib

import matplotlib.pyplot as plt

import utils
from utils.configs.dataset import DatasetConfig
from utils.configs.dataset_handling import DatasetHandlingConfig
from utils.configs.model import ModelConfig
from utils.configs.working_points_set import WorkingPointsSetConfig
from utils.helpers.run_id_handler import RunIdHandler
from utils.models.binned_efficiency_map_model import BinnedEfficiencyMapModel
from utils.plots.histograms_as_3d import (
    plot_2d_histograms_as_3d_bars,
    plot_2d_histograms_as_3d_surfaces,
)
from utils.plots.save_figure import save_figure


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

        x_label = variables_to_plot[0]
        y_label = variables_to_plot[1]
        x_label = utils.branches_niceify.get(x_label, x_label)
        y_label = utils.branches_niceify.get(y_label, y_label)
        z_label = r"Efficiency [%]"

        for working_point, working_point_histograms in bem.histograms.items():
            hists = []
            titles = []
            flavours = sorted(working_point_histograms.keys())
            for flavour in flavours:  # to have consistent order of flavours
                flavour_eff_hist = working_point_histograms[flavour]["eff_mode"]
                assert flavour_eff_hist.variables == variables_to_plot
                hists.append(flavour_eff_hist)
                titles.append(
                    f"{utils.branches_niceify.get(bem.separation_cols[0], bem.separation_cols[0])}: "
                    f"{utils.flavours_niceify.get(flavour, flavour)}"
                )

            fig_surfaces = plot_2d_histograms_as_3d_surfaces(
                hists=hists,
                titles=titles,
                fig_title=(
                    f"{utils.working_points_niceify.get(working_point, working_point)}"
                    " Working Point"
                ),
                z_scaler=100,  # to make it percent
                x_label=x_label,
                y_label=y_label,
                z_label=z_label,
            )

            for path in [model_dir_path, output_dir_path]:
                save_figure(
                    fig=fig_surfaces,
                    path=path,
                    filename=utils.filenames.efficiency_map_plot_surfaces(
                        model_name=model_config.name,
                        working_point_name=working_point,
                    ),
                    bbox_inches="tight",
                )

            plt.close(fig=fig_surfaces)

            fig_bars = plot_2d_histograms_as_3d_bars(
                hists=hists,
                titles=titles,
                fig_title=(
                    f"{utils.working_points_niceify.get(working_point, working_point)}"
                    " Working Point"
                ),
                z_scaler=100,  # to make it percent
                x_label=x_label,
                y_label=y_label,
                z_label=z_label,
            )

            for path in [model_dir_path, output_dir_path]:
                save_figure(
                    fig=fig_bars,
                    path=path,
                    filename=utils.filenames.efficiency_map_plot_bars(
                        model_name=model_config.name,
                        working_point_name=working_point,
                    ),
                    bbox_inches="tight",
                )

            plt.close(fig=fig_bars)
