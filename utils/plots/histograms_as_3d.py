from typing import List, Optional

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from utils.efficiency_map.histogram import Histogram


def plot_2d_histograms_as_3d_bars(
    hists: List[Histogram],
    titles: List[str],
    fig_title: str,
    z_scaler: Optional[float] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    z_label: Optional[str] = None,
):
    for hist in hists:
        assert len(hist.edges) == 2

    # TODO(low): https://stackoverflow.com/questions/18602660/matplotlib-bar3d-clipping-problems
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(fig_title)

    for i, (hist, title) in enumerate(zip(hists, titles)):
        x_edges = hist.edges[0]
        y_edges = hist.edges[1]

        x_grid, y_grid = np.meshgrid(x_edges[:-1], y_edges[:-1])

        x = x_grid.ravel()
        #print("x ",x)
        y = y_grid.ravel()
        #print("y ",y)
        z = np.zeros_like(y)
        #print("z ",z)
        dx = np.tile(np.diff(x_edges), x_grid.shape[0])
        #print("dx ",dx)
        dy = np.repeat(np.diff(y_edges), y_grid.shape[1])
        #print("dy ",dy)
        dz = np.nan_to_num(hist.h).T.ravel()
        #print("dz ",dz)

        if z_scaler is not None:
            dz = z_scaler * dz
        # TODO(test): test with histogram where the plot is known (e.g. one peak in
        #  a certain corner) to make sure using .T is correct

        color_map = cm.get_cmap("jet")
        max_height = np.max(dz)  # get range so we can normalize
        min_height = np.min(dz)
        # scale each dz to [0,1], and get their rgb values
        rgba = [color_map((k - min_height) / max_height) for k in dz]

        matplotlib.rcParams['font.sans-serif'] = 'Arial'    
        ax = fig.add_subplot(1, len(hists), 1 + i, projection="3d")

        ax.bar3d(
            x=x,
            y=y,
            z=z,
            dx=dx,
            dy=dy,
            dz=dz,
            shade=True,
            color=rgba,
        )
        ax.set_title(title)
        #ax.text(0.00, 0.00, 1.01, 'CMS Simulation Preliminary',transform=ax.transAxes)
        if x_label is None:
            x_label = hist.variables[0]
        if y_label is None:
            y_label = hist.variables[1]
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if z_label is not None:
            ax.set_zlabel(z_label)

    return fig


def plot_2d_histograms_as_3d_surfaces(
    hists: List[Histogram],
    titles: List[str],
    fig_title: str,
    z_scaler: Optional[float] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    z_label: Optional[str] = None,
):
    for hist in hists:
        assert len(hist.edges) == 2

    matplotlib.rcParams['font.sans-serif'] = 'Arial'    
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(fig_title)

    for i, (hist, title) in enumerate(zip(hists, titles)):
        x_edges = hist.edges[0]
        y_edges = hist.edges[1]

        x_edges_midpoints = (x_edges[1:] + x_edges[:-1]) / 2
        y_edges_midpoints = (y_edges[1:] + y_edges[:-1]) / 2

        x_grid, y_grid = np.meshgrid(
            x_edges_midpoints,
            y_edges_midpoints,
        )

        X = x_grid
        Y = y_grid
        Z = hist.h.T

        if z_scaler is not None:
            Z = z_scaler * Z

        ax = fig.add_subplot(1, len(hists), 1 + i, projection="3d")
        ax.plot_surface(
            X=X,
            Y=Y,
            Z=Z,
            cmap="jet",
            alpha=0.5,
        )
        ax.set_title(title)
        ax.text(0.00, 0.00, 1.01, 'CMS Simulation Preliminary',transform=ax.transAxes)
        if x_label is None:
            x_label = hist.variables[0]
        if y_label is None:
            y_label = hist.variables[1]
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if z_label is not None:
            ax.set_zlabel(z_label)

    return fig
