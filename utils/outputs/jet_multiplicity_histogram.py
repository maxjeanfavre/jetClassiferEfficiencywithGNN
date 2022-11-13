import pathlib
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import utils
from utils.configs.dataset import DatasetConfig
from utils.data.jet_events_dataset import JetEventsDataset
from utils.plots.save_figure import save_figure


# TODO: move plot to utils/plots
def create_jet_multiplicity_histogram_inclusive(
    dataset_config: DatasetConfig,
    output_dir_path: pathlib.Path,
    jds: Optional[JetEventsDataset] = None,
):
    if jds is None:
        jds = JetEventsDataset.read_in(
            dataset_config=dataset_config,
            branches="nJet",
        )

    # sanity check
    a = jds.df.loc[(slice(None), 0), "nJet"].to_numpy()
    assert np.array_equal(a, jds.event_n_jets)
    # end sanity check

    max_n_jets = np.max(jds.event_n_jets)
    x = np.arange(max_n_jets + 1)
    height = np.bincount(jds.event_n_jets)
    assert len(height) == max_n_jets + 1

    height_normalized = height / np.sum(height)

    matplotlib.rcParams['font.sans-serif'] = 'Arial'    
    fig, ax = plt.subplots()
    ax.text(0.00, 1.01, 'CMS Simulation Preliminary',transform=ax.transAxes)
    caption = (
        f"Normalised histogram of the jet multiplicity in the "
        f"{utils.datasets_niceify.get(dataset_config.name, dataset_config.name)}"
        " dataset."
    )
    txt = ax.text(0.00,-0.12,caption,va="top",transform=ax.transAxes,wrap=True)
    fig_xsize, fig_ysize = fig.get_size_inches()*fig.dpi
    #below works if tight layout is switched off
    txt._get_wrap_line_width = lambda : fig_xsize
    #fig.suptitle(
    #    (
    #        "Normalised histogram of the jet multiplicity in the "
    #        f"{utils.datasets_niceify.get(dataset_config.name, dataset_config.name)}"
    #        " dataset."
    #    ),
    #    wrap=True,
    #)
    ax.bar(
        x=x,
        height=height_normalized,
    )
    ax.set_xlim([0, 20])  # TODO(critical): change to do it properly
    fig.tight_layout()
    save_figure(
        fig=fig,
        path=output_dir_path,
        filename=utils.filenames.jet_multiplicity_normalised_histogram_inclusive,
    )

    plt.close(fig=fig)


# TODO: move plot to utils/plots
def create_jet_multiplicity_histogram_by_flavour(
    dataset_config: DatasetConfig,
    output_dir_path: pathlib.Path,
    jds: Optional[JetEventsDataset] = None,
):
    if jds is None:
        jds = JetEventsDataset.read_in(
            dataset_config=dataset_config,
            branches=("nJet", "Jet_hadronFlavour"),
        )

    # sanity check
    a = jds.df.loc[(slice(None), 0), "nJet"].to_numpy()
    assert np.array_equal(a, jds.event_n_jets)
    # end sanity check

    flavours = sorted(set(jds.df["Jet_hadronFlavour"].to_numpy()))

    matplotlib.rcParams['font.sans-serif'] = 'Arial'    
    fig, axes = plt.subplots(len(flavours), 1, sharex="all")
    #fig.suptitle(
    #    (
    #        "Jet multiplicity by flavour in the "
    #        f"{utils.datasets_niceify.get(dataset_config.name, dataset_config.name)}"
    #        " dataset."
    #    ),
    #    wrap=True,
    #)

    for i, flavour in enumerate(
        flavours
    ):  # not groupby to have consistent order of flavours
        flavour_jets_per_event = (
            jds.df[jds.df["Jet_hadronFlavour"] == flavour]
            .groupby(level=0, sort=False)
            .size()
            .to_numpy()
        )
        height = np.bincount(flavour_jets_per_event)
        axes[i].set_title(utils.flavours_niceify[flavour])
        axes[i].bar(
            x=np.arange(len(height)),
            height=height,
            color=utils.settings.quark_flavour_colours[flavour],
        )
        axes[i].set_xlim([0, 20])  # TODO(critical): change to do it properly
        axes[i].text(0.00, 1.01, 'CMS Simulation Preliminary',transform=axes[i].transAxes)
    caption = (
        f"Jet multiplicity by flavour in the "
        f"{utils.datasets_niceify.get(dataset_config.name, dataset_config.name)}"
        " dataset."
    )
    txt = axes[-1].text(0.00,-0.25,caption,va="top",transform=axes[-1].transAxes,wrap=True)
    fig_xsize, fig_ysize = fig.get_size_inches()*fig.dpi
    #below works if tight layout is switched off
    txt._get_wrap_line_width = lambda : fig_xsize

    fig.tight_layout()

    save_figure(
        fig=fig,
        path=output_dir_path,
        filename=utils.filenames.jet_multiplicity_normalised_histogram_by_flavour,
    )

    plt.close(fig=fig)
