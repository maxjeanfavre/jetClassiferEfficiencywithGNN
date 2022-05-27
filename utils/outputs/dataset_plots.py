import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

import utils
from utils.configs.dataset import DatasetConfig
from utils.data.jet_events_dataset import JetEventsDataset
from utils.data.manipulation.data_filters.eval_expression import EvalExpressionFilter


def create_dataset_plots(
    dataset_config: DatasetConfig,
    output_dir_path: pathlib.Path,
    jds: Optional[JetEventsDataset] = None,
):
    vars_and_quantiles = [
        ["Jet_Pt", [0, 1]],
        ["Jet_Pt", [0, 0.999]],
        ["Jet_eta", [0, 1]],
    ]

    if jds is None:
        jds = JetEventsDataset.read_in(
            dataset_config=dataset_config,
            branches=("nJet", *set(var for var, _ in vars_and_quantiles)),
        )

    ### Plot jet multiplicity distribution

    # sanity check
    a = jds.df.loc[(slice(None), 0), "nJet"].to_numpy()
    assert np.array_equal(a, jds.event_n_jets)
    # end sanity check

    max_n_jets = np.max(jds.event_n_jets)
    x = np.arange(max_n_jets + 1)
    height = np.bincount(jds.event_n_jets)
    assert len(height) == max_n_jets + 1

    height_normalized = height / np.sum(height)

    fig, ax = plt.subplots()
    fig.suptitle(
        f"Normalized histogram of the jet multiplicity in {dataset_config.name}",
        wrap=True,
    )
    ax.bar(
        x=x,
        height=height_normalized,
    )
    fig.tight_layout()
    fig.savefig(
        fname=output_dir_path / "jet_multiplicity_normalized.png",
        dpi=utils.settings.plots_dpi,
    )

    plt.close(fig=fig)

    ### Plot variable distributions
    for var, quantiles in vars_and_quantiles:
        flavours = sorted(set(jds.df["Jet_hadronFlavour"].to_numpy()))
        var_data = []
        label_data = []
        for flavour in flavours:  # not groupby to have consistent order of flavours
            var_data.append(
                jds.df.loc[jds.df["Jet_hadronFlavour"] == flavour, var].to_numpy()
            )
            label_data.append(flavour)
        fig, ax = plt.subplots()
        ax.hist(
            x=var_data,
            bins=30,
            range=tuple(jds.df[var].quantile(q=quantiles)),
            density=True,
            label=label_data,
            histtype="barstacked",
        )
        ax.legend()

        title = f"Normalized histogram of {var} in {dataset_config.name}."
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

        fig.savefig(
            fname=output_dir_path
            / f"jet_variable_normalized_{var}_{quantiles[0]}_{quantiles[1]}.png",
            dpi=utils.settings.plots_dpi,
        )

        plt.close(fig=fig)

    ### Plot distribution of DeepCSV P(b) + P(bb) discriminator
    ### for jets with valid values separated by flavour
    # TODO(low): use given jds but have to make sure to not change it
    jds = JetEventsDataset.read_in(
        dataset_config=dataset_config,
        branches=("Jet_btagDeepB", "Jet_hadronFlavour"),
    )

    jds.manipulate(
        data_manipulators=(
            EvalExpressionFilter(
                description=(
                    "Keeps jets with valid btagDeepB value (0 <= Jet_btagDeepB <= 1)"
                ),
                active_modes=("foo",),
                expression="0 <= Jet_btagDeepB <= 1",
                filter_full_event=False,
                required_columns=("Jet_btagDeepB",),
            ),
        ),
        mode="foo",
    )

    flavours = sorted(set(jds.df["Jet_hadronFlavour"].to_numpy()))
    flavours.reverse()
    var_data = []
    label_data = []
    for flavour in flavours:
        var_data.append(
            jds.df.loc[
                jds.df["Jet_hadronFlavour"] == flavour, "Jet_btagDeepB"
            ].to_numpy()
        )
        label_data.append(flavour)

    fig, ax = plt.subplots()
    fig.suptitle(
        f"Histogram of the DeepCSV discriminator in {dataset_config.name}", wrap=True
    )

    ax.hist(
        x=var_data,
        bins=np.linspace(0, 1, 30),
        label=label_data,
        histtype="barstacked",
        edgecolor="black",
        linewidth=0.75,
    )

    ax.set_yscale("log")

    ax.set_xlabel("DeepCSV discriminator")
    ax.set_ylabel("Jets")

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        fname=output_dir_path / "jet_btagDeepB_histogram.png",
        dpi=utils.settings.plots_dpi,
    )

    plt.close(fig=fig)
