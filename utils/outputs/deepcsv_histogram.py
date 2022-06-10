import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

import utils
from utils.configs.dataset import DatasetConfig
from utils.data.jet_events_dataset import JetEventsDataset
from utils.plots.save_figure import save_figure


# TODO: move plot to utils/plots
def create_deepcsv_discriminator_histogram(
    dataset_config: DatasetConfig,
    output_dir_path: pathlib.Path,
    jds: Optional[JetEventsDataset] = None,
):
    """Plot distribution of DeepCSV P(b) + P(bb) discriminator
    for jets with valid values separated by flavour"""
    if jds is None:
        jds = JetEventsDataset.read_in(
            dataset_config=dataset_config,
            branches=None,
        )

    valid_btagDeepB_selection = jds.df.eval("0 <= Jet_btagDeepB <= 1")

    flavours = sorted(set(jds.df["Jet_hadronFlavour"].to_numpy()))
    var_data = []
    label_data = []
    colour_data = []
    for flavour in flavours:
        var_data.append(
            jds.df.loc[
                (valid_btagDeepB_selection) & (jds.df["Jet_hadronFlavour"] == flavour),
                "Jet_btagDeepB",
            ].to_numpy()
        )
        label_data.append(utils.flavours_niceify[flavour])
        colour_data.append(utils.settings.quark_flavour_colours[flavour])

    fig, ax = plt.subplots()
    fig.suptitle(
        (
            "Histogram of the DeepCSV discriminator in the "
            f"{utils.datasets_niceify.get(dataset_config.name, dataset_config.name)}"
            " dataset."
        ),
        wrap=True,
    )

    ax.hist(
        x=var_data,
        bins=np.linspace(0, 1, 30),
        label=label_data,
        histtype="barstacked",
        edgecolor="black",
        # linewidth=0.75,
        color=colour_data,
    )

    ax.set_yscale("log")

    ax.set_xlabel("DeepCSV discriminator")
    ax.set_ylabel("Jets")

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig=fig,
        path=output_dir_path,
        filename=utils.filenames.deepcsv_discriminator_histogram,
    )

    plt.close(fig=fig)
