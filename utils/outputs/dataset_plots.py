import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from utils.configs.dataset import DatasetConfig
from utils.data.jet_events_dataset import JetEventsDataset


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
    fig.savefig(fname=output_dir_path / "jet_multiplicity_normalized.png", dpi=300)

    ### Plot variable distributions
    for var, quantiles in vars_and_quantiles:
        fig, ax = plt.subplots()
        ax.hist(
            x=jds.df[var].to_numpy(),
            bins=30,
            range=tuple(jds.df[var].quantile(q=quantiles)),
            density=True,
        )

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
            outlier_text = "Outliers "
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
            dpi=300,
        )
