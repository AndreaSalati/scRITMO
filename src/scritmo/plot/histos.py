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

import seaborn as sns
from scipy.stats import vonmises
from .utils import polar_plot
import numpy as np


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
    phases,
    time,
    bins=30,
    normalize=False,
    inner_ring_size=0.0,
    title="",
):
    """
    Plot stacked polar histograms for each unique time value.

    Parameters
    ----------
    phases : array-like
        Array of angles (radians), shape (n,).
    time : array-like
        Array of time values, shape (n,). Must be same length as phases.
    bins : int
        Number of angular bins (default 30).
    figsize : tuple
        Figure size.
    normalize : bool
        If True, normalize each bin so its total height = 1 (proportions).
    """

    ax = polar_plot(title=title, inner_ring_size=inner_ring_size)

    # Histogram bins
    bins_arr = np.linspace(0, 2 * np.pi, bins + 1)
    bin_centers = (bins_arr[:-1] + bins_arr[1:]) / 2
    width = np.diff(bins_arr)

    # Initialize bottom
    bottom = np.zeros(len(bin_centers))

    # Colors
    colors = plt.cm.tab10.colors

    # Unique time points
    unique_times = np.unique(time)
    histograms = {}

    for t in unique_times:
        mask = time == t
        phases_t = np.mod(np.asarray(phases)[mask], 2 * np.pi)
        counts, _ = np.histogram(phases_t, bins=bins_arr)
        histograms[t] = counts

    # Normalize if requested
    if normalize:
        total_per_bin = np.sum(list(histograms.values()), axis=0)
        total_per_bin[total_per_bin == 0] = 1  # avoid div by zero
        for t in histograms:
            histograms[t] = histograms[t] / total_per_bin

    # Plot
    for i, t in enumerate(unique_times):
        counts = histograms[t]
        ax.bar(
            bin_centers,
            counts,
            width=width,
            bottom=bottom,
            color=colors[i % len(colors)],
            edgecolor="k",
            alpha=0.8,
            label=f"{t}",
        )
        bottom += counts

    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    return ax


def plot_phase_polar_population(
    phases,
    time,
    amplitude=None,
    plot_type="histogram",
    cmap_name="twilight",
    bins=30,
    ylim=None,
    color_order="linear",
    inner_ring_size=0.0,
    title="",
    show_rlabels=True,
    show_grid=True,
    hist_density=True,
):
    """
    Plot phase population on a polar plot using polar_plot for consistent styling.

    Parameters
    ----------
    phases : array-like
        Phase angles (radians).
    time : array-like
        Time points (same length as phases).
    amplitude : array-like, optional
        Amplitude values for scatter plot.
    plot_type : str
        "histogram", "scatter", or "density".
    cmap_name : str
        Colormap name.
    bins : int
        Number of bins for histogram.
    ylim : float, optional
        Y-axis limit for scatter plot.
    color_order : str
        "linear" or "high_contrast".
    inner_ring_size : float
        Passed to polar_plot.
    title : str
        Plot title.
    show_rlabels : bool
        Show radial labels.
    show_grid : bool
        Show grid.
    """
    import matplotlib.pyplot as plt

    # Prepare data
    phases = np.asarray(phases)
    time = np.asarray(time)
    if amplitude is not None:
        amplitude = np.asarray(amplitude)

    unique_times = np.unique(time)
    cmap = plt.get_cmap(cmap_name)

    # Set up polar axes using polar_plot
    ax = polar_plot(
        title=title,
        inner_ring_size=inner_ring_size,
        show_rlabels=show_rlabels,
        show_grid=show_grid,
    )
    fig = ax.figure

    # Color selection
    if color_order == "linear":
        base_colors = cmap(np.linspace(0, 1, len(unique_times), endpoint=False))
    else:
        # fallback: just use linear if no get_high_contrast_colors
        base_colors = cmap(np.linspace(0, 1, len(unique_times), endpoint=False))

    theta_vals = np.linspace(0, 2 * np.pi, 200, endpoint=False)

    for i, t in enumerate(unique_times):
        color = base_colors[i]
        theta_ref = 2 * np.pi * (t % 24) / 24

        mask = time == t
        ph = phases[mask]
        if amplitude is not None:
            amps = amplitude[mask]

        if plot_type == "density":
            # If you have a sample grouping, add here
            if "samples" in locals():
                samples = np.asarray(samples)
                for j, sam in enumerate(np.unique(samples[mask])):
                    ph_s = ph[samples[mask] == sam]
                    if len(ph_s) > 1:
                        kappa, loc, scale = vonmises.fit(ph_s, method="analytical")
                        dens = vonmises.pdf(theta_vals, kappa, loc=loc, scale=scale)
                        ax.fill_between(theta_vals, 0, dens, color=color, alpha=0.3)
                        ax.plot(theta_vals, dens, color=color, alpha=0.7)
                    else:
                        ax.plot(ph_s, [0.1], "o", color=color, alpha=0.9, markersize=4)
            else:
                if len(ph) > 1:
                    kappa, loc, scale = vonmises.fit(ph, method="analytical")
                    dens = vonmises.pdf(theta_vals, kappa, loc=loc, scale=scale)
                    ax.fill_between(theta_vals, 0, dens, color=color, alpha=0.3)
                    ax.plot(theta_vals, dens, color=color, alpha=0.7)
                else:
                    ax.plot(ph, [0.1], "o", color=color, alpha=0.9, markersize=4)

        elif plot_type == "scatter" and amplitude is not None:
            ax.scatter(
                ph,
                amps,
                s=10,
                color=color,
                alpha=0.5,
                edgecolors="none",
            )
            if ylim is not None:
                ax.set_ylim(0, ylim)
        elif plot_type == "histogram":
            ax.hist(
                ph,
                bins=bins,
                density=hist_density,
                alpha=0.5,
                color=color,
                label=f"{t:.0f}h",
                range=(0, 2 * np.pi),
            )

    rmax = ax.get_rmax()
    for i, t in enumerate(unique_times):
        color = base_colors[i]
        theta_ref = 2 * np.pi * (t % 24) / 24
        ax.plot(
            [theta_ref, theta_ref],
            [0, rmax],
            linestyle="--",
            color=color,
            linewidth=1.5,
        )

    ax.set_title(
        f"{title or 'Phase Population'} – {plot_type.capitalize()} Plot", va="bottom"
    )
    if plot_type == "histogram":
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    fig.tight_layout()
    return fig, ax
