import pathlib
from typing import List

from matplotlib import pyplot as plt

import utils
from utils.data.jet_events_dataset import JetEventsDataset
from utils.plots.predictions_histogram import (
    plot_predictions_histogram,
)


def create_predictions_histogram(
    jds: JetEventsDataset,
    eff_pred_cols: List[str],
    evaluation_dir_path: pathlib.Path,
):
    fig = plot_predictions_histogram(
        df=jds.df,
        eff_pred_cols=eff_pred_cols,
    )

    fig.savefig(
        fname=evaluation_dir_path
        / utils.filenames.efficiency_prediction_histogram_plot,
        dpi=utils.settings.plots_dpi,
    )

    plt.close(fig=fig)
