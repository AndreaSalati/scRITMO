import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scritmo.linear_regression import (
    harmonic_function_exp,
    harmonic_function,
    polar_genes_pandas,
    fit_periodic_spline,
)
from scritmo.basics import w, rh, ind2
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from adjustText import adjust_text
from scritmo.beta import Beta

import seaborn as sns
from scipy.stats import vonmises
from .utils import polar_plot


def hist(x, bins=30):
    _, _, _ = plt.hist(x, bins=bins)
    plt.show()
    return


def phist(
    x,
    bins=30,
    title="",
    inner_ring_size=0.0,
    color=None,
    show_rlabels=True,
    show_grid=True,
):
    """Quick polar histogram"""
    polar_plot(
        title=title,
        inner_ring_size=inner_ring_size,
        show_rlabels=show_rlabels,
        show_grid=show_grid,
    )
    _, _, _ = plt.hist(x, bins=bins, color=color)
    plt.show()
    return


def plot_stacked_polar(
    phases_dict,
    bins=30,
    figsize=(6, 6),
    normalize=False,
    inner_ring_size=0.0,
    title="",
):
    """
    Plot stacked polar histograms from a dictionary of phase populations.

    Parameters
    ----------
    phases_dict : dict
        Keys = population labels, values = lists/arrays of angles (radians).
    bins : int
        Number of angular bins (default 30).
    figsize : tuple
        Figure size.
    normalize : bool
        If True, normalize each bin so its total height = 1 (proportions).
    """

    ax = polar_plot(title=title, inner_ring_size=inner_ring_size)

    # Histogram bins
    bins = np.linspace(0, 2 * np.pi, bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = np.diff(bins)

    # Initialize bottom
    bottom = np.zeros(len(bin_centers))

    # Colors
    colors = plt.cm.tab10.colors

    # Compute histograms
    histograms = {}
    for label, phases in phases_dict.items():
        phases = np.mod(np.asarray(phases), 2 * np.pi)  # wrap into [0, 2π]
        counts, _ = np.histogram(phases, bins=bins)
        histograms[label] = counts

    # Normalize if requested
    if normalize:
        total_per_bin = np.sum(list(histograms.values()), axis=0)
        total_per_bin[total_per_bin == 0] = 1  # avoid div by zero
        for label in histograms:
            histograms[label] = histograms[label] / total_per_bin

    # Plot
    for i, (label, counts) in enumerate(histograms.items()):
        ax.bar(
            bin_centers,
            counts,
            width=width,
            bottom=bottom,
            color=colors[i % len(colors)],
            edgecolor="k",
            alpha=0.8,
            label=label,
        )
        bottom += counts

    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    return ax
