import pathlib

import matplotlib.pyplot as plt


def save_figure(fig: plt.Figure, path: pathlib.Path, filename: str, **kwargs):
    fig.savefig(fname=path / f"{filename}.pdf", **kwargs)

    fig.savefig(fname=path / f"{filename}.pgf", **kwargs)
