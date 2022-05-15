from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_predictions_histogram(
    df: pd.DataFrame, eff_pred_cols: List[str], n_bins: int = 20
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 15))

    fig.suptitle("Histogram of efficiency predictions of different models")

    for col in eff_pred_cols:
        ax.hist(
            x=df[col],
            bins=np.linspace(0, 1, n_bins + 1),
            label=f"{col} Sum: {df[col].sum():.1f}. NaN: {df[col].isna().sum()}.",
            histtype="step",
        )

    ax.legend()

    return fig
