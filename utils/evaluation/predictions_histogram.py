import pathlib
from typing import List

from matplotlib import pyplot as plt

from utils.data.jet_events_dataset import JetEventsDataset
from utils.plots.predictions_histogram import (
    plot_predictions_histogram,
)
from utils.plots.save_figure import save_figure


def create_predictions_histogram(
    jds: JetEventsDataset,
    eff_pred_cols: List[str],
    evaluation_dir_path: pathlib.Path,
):
    fig = plot_predictions_histogram(
        df=jds.df,
        eff_pred_cols=eff_pred_cols,
    )

    filename = "eff_pred_hist"
    save_figure(
        fig=fig,
        path=evaluation_dir_path,
        filename=filename,
    )

    plt.close(fig=fig)
