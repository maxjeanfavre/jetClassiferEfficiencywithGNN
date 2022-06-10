import pathlib
from typing import Optional

import matplotlib.pyplot as plt

import utils
from utils.configs.dataset import DatasetConfig
from utils.data.jet_events_dataset import JetEventsDataset
from utils.plots.save_figure import save_figure


# TODO: move plot to utils/plots
def create_jet_variable_histograms(
    dataset_config: DatasetConfig,
    output_dir_path: pathlib.Path,
    jds: Optional[JetEventsDataset] = None,
):
    vars_and_quantiles = [
        ["Jet_Pt", [0, 1]],
        ["Jet_Pt", [0, 0.999]],
        ["Jet_eta", [0, 1]],
        ["Jet_phi", [0, 1]],
        ["Jet_nConstituents", [0, 1]],
        ["Jet_nConstituents", [0, 0.999]],
        ["Jet_nConstituents", [0, 0.99]],
        ["Jet_mass", [0, 1]],
        ["Jet_mass", [0, 0.999]],
        ["Jet_mass", [0, 0.99]],
        ["Jet_area", [0, 1]],
        ["Jet_area", [0.001, 0.999]],
        ["Jet_area", [0.01, 0.99]],
    ]

    if jds is None:
        jds = JetEventsDataset.read_in(
            dataset_config=dataset_config,
            branches=("Jet_hadronFlavour", *set(var for var, _ in vars_and_quantiles)),
        )

    for var, quantiles in vars_and_quantiles:
        assert len(quantiles) == 2
        assert quantiles[0] < quantiles[1]
        assert all(0 <= q <= 1 for q in quantiles)

        flavours = sorted(set(jds.df["Jet_hadronFlavour"].to_numpy()))
        var_data = []
        label_data = []
        colour_data = []
        for flavour in flavours:  # not groupby to have consistent order of flavours
            var_data.append(
                jds.df.loc[jds.df["Jet_hadronFlavour"] == flavour, var].to_numpy()
            )
            label_data.append(utils.flavours_niceify[flavour])
            colour_data.append(utils.settings.quark_flavour_colours[flavour])
        fig, ax = plt.subplots()
        ax.hist(
            x=var_data,
            bins=30,
            range=tuple(jds.df[var].quantile(q=quantiles)),
            density=True,
            label=label_data,
            histtype="barstacked",
            color=colour_data,
        )

        ax.set_xlabel(utils.branches_niceify.get(var, var))

        ax.legend()

        title = (
            f"Normalised histogram of {utils.branches_niceify.get(var, var)} in the "
            f"{utils.datasets_niceify.get(dataset_config.name, dataset_config.name)}"
            " dataset."
        )
        if quantiles[0] != 0 or quantiles[1] != 1:
            if quantiles[0] != 0:
                below_text = f"below the {quantiles[0]:.2%} quantile"
            else:
                below_text = None
            if quantiles[1] != 1:
                above_text = f"above the {quantiles[1]:.2%} quantile"
            else:
                above_text = None
            outlier_text = " Outliers "
            if below_text is not None:
                outlier_text += below_text
                if above_text is not None:
                    outlier_text += " and "
            if above_text:
                outlier_text += above_text
            outlier_text += " were removed."
            title += outlier_text
        title += "\n"
        fig.suptitle(title, wrap=True)
        fig.tight_layout()

        save_figure(
            fig=fig,
            path=output_dir_path,
            filename=utils.filenames.jet_variable_normalised_histogram(
                variable=var,
                lower_quantile=quantiles[0],
                upper_quantile=quantiles[1],
            ),
        )

        plt.close(fig=fig)
