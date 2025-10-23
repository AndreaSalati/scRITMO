import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scritmo.basics import w, rh, ind2
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from adjustText import adjust_text

import seaborn as sns
from scipy.stats import vonmises


def xy(color="red", linestyle="--", legend_label=""):
    plt.axline(
        (0, 0),
        slope=1,
        color=color,
        linestyle=linestyle,
        label=legend_label,
    )


def polar_plot(
    title="",
    inner_ring_size=0,
    show_rlabels=True,
    show_grid=True,
    theta_min=0,
    theta_max=2 * np.pi,
    aperture=None,
    n_phase_ticks=6,
    xtick_fontsize=20,
):
    """
    This function returns the ax object that can be used to plot the polar plot
    or histogram.

    Parameters
    ----------
    title : str
        The title of the plot.
    inner_ring_size : float
        The size of the inner ring, negative number is suggested.
    show_rlabels : bool, optional
        If False, removes radial axis labels (numbers). Default is True.
    show_grid : bool, optional
        If False, removes polar grid lines. Default is True.
    theta_min : float, optional
        Minimum angle (in radians) to display. Default is 0.
    theta_max : float, optional
        Maximum angle (in radians) to display. Default is 2π.
    aperture : float, optional
        If provided, sets theta_min = -aperture/2 and theta_max = aperture/2.
    n_phase_ticks : int, optional
        Number of angular tick labels. Default is 6.
    xtick_fontsize : int or float, optional
        Font size for the angular tick labels (e.g., "0h", "6h"). Default is 10.
    """
    if aperture is not None:
        theta_min = -aperture / 2
        theta_max = aperture / 2

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    # Set tick positions
    ax.set_xticks(np.linspace(0, 2 * np.pi, n_phase_ticks, endpoint=False))
    # Create labels
    x_ticks_labels = [f"{int(i * 24 / n_phase_ticks)}h" for i in range(n_phase_ticks)]
    # Set labels with specified fontsize
    ax.set_xticklabels(x_ticks_labels, fontsize=xtick_fontsize)

    ax.set_title(title)
    ax.set_rorigin(inner_ring_size)
    if not show_rlabels:
        ax.set_yticklabels([])
    if not show_grid:
        ax.grid(False)
    # Set the angular span
    ax.set_thetamin(np.degrees(theta_min))
    ax.set_thetamax(np.degrees(theta_max))
    return ax


def scatter_with_labels(
    x,
    y,
    labels,
    fontsize=10,
    arrowstyle="-",
    color_arr="black",
    adjust=True,
):
    """
    Plots a scatter and annotates it with non-overlapping labels.

    Args:
        x (array-like): x-coordinates
        y (array-like): y-coordinates
        labels (array-like): list of strings for annotation
        fontsize (int): font size of the labels
        s (int): scatter marker size
    """
    texts = []
    for xi, yi, label in zip(x, y, labels):
        texts.append(plt.text(xi, yi, label, fontsize=fontsize))

    if adjust:
        adjust_text(
            texts, arrowprops=dict(arrowstyle=arrowstyle, color=color_arr, lw=0.5)
        )
    else:

        # annotate
        for i, label in enumerate(labels):
            plt.text(x[i], y[i], label, fontsize=fontsize)


def integrated_histo(v, tr_vec, normalize=True):
    """
    This function takes as input samples belonging to (ePDF)
    and returns the integrated histogram (eCDF)
    eCDF = int (from -inf to x) ePDF(x) dx
    """

    # integrate amplitude histogram
    int_v = np.zeros(len(tr_vec))
    for i, tr in enumerate(tr_vec):
        int_v[i] = np.sum(v < tr)

    if normalize:
        int_v = int_v / v.shape[0]

    return int_v


def reversed_integrated_histo(v, tr_vec=None, normalize=False):
    """
    This function takes as input samples belonging to (ePDF)
    and returns the reversed integrated histogram (eCDF)
    """

    # integrate amplitude histogram
    int_v = np.zeros(len(tr_vec))
    for i, tr in enumerate(tr_vec):
        int_v[i] = np.sum(v > tr)

    if normalize:
        int_v = int_v / v.shape[0]

    return int_v


def paper_style():
    plt.rcParams.update(
        {
            "figure.figsize": (7, 5),  # Default figure size (width, height) in inches
            "axes.titlesize": 18,  # Title font size
            "axes.labelsize": 16,  # Axis label font size
            "xtick.labelsize": 14,  # X-tick label font size
            "ytick.labelsize": 14,  # Y-tick label font size
            "legend.fontsize": 14,  # Legend font size
            "legend.title_fontsize": 16,  # Legend title font size
            # 'font.family': 'serif',              # Font family (you can change this to 'sans-serif' or 'monospace')
            "font.serif": ["Arial"],  # Font choice, adjust to your needs
            "axes.linewidth": 1.5,  # Width of the axis lines
            "lines.linewidth": 2.0,  # Line width for plots
            "axes.spines.top": False,  # Disable top spine
            "axes.spines.right": False,  # Disable right spine
            "legend.frameon": False,  # Disable legend box
            "savefig.dpi": 300,  # Set DPI for saving figures, important for publication-quality figures
            "savefig.format": "pdf",  # Default file format when saving figures
        }
    )


def hexbin_with_marginals(
    x,
    y,
    density="linear",  # 'linear'  or 'log'
    gridsize=50,
    hist_bins=50,
    figsize=(8, 8),
    cmap="viridis",
):
    """
    Hex-bin joint plot with perfectly aligned marginal histograms.

    Parameters
    ----------
    x, y       : 1-D array-likes
    density    : 'linear' or 'log'   (colour scale for hexbin)
    gridsize   : hexbin grid size
    hist_bins  : #bins in marginal histograms
    figsize    : figure size
    cmap       : matplotlib colormap

    Returns
    -------
    ax_joint   : main Axes (for titles / labels)
    """
    # Prepare ranges once so both hexbin & hist share them
    pad = 1e-9
    x_min, x_max = np.min(x) - pad, np.max(x) + pad
    y_min, y_max = np.min(y) - pad, np.max(y) + pad
    extent = (x_min, x_max, y_min, y_max)

    bins_arg = "log" if density == "log" else None

    # ── 1. Main axis ────────────────────────────────────────────────────
    fig, ax_joint = plt.subplots(figsize=figsize)
    divider = make_axes_locatable(ax_joint)

    # ── 2. Attach marginal axes (share limits automatically) ───────────
    ax_xhist = divider.append_axes("top", size=1.2, pad=0.1, sharex=ax_joint)
    ax_yhist = divider.append_axes("right", size=1.2, pad=0.1, sharey=ax_joint)

    # ── 3. Joint hex-bin ───────────────────────────────────────────────
    hb = ax_joint.hexbin(
        x,
        y,
        gridsize=gridsize,
        bins=bins_arg,
        cmap=cmap,
        extent=extent,
    )
    ax_joint.set_xlim(x_min, x_max)
    ax_joint.set_ylim(y_min, y_max)

    # ── 4. Marginal histograms (exact same ranges) ─────────────────────
    ax_xhist.hist(x, bins=hist_bins, range=(x_min, x_max), color="gray")
    ax_yhist.hist(
        y, bins=hist_bins, range=(y_min, y_max), orientation="horizontal", color="gray"
    )

    # tidy marginal axes
    ax_xhist.tick_params(bottom=False, labelbottom=False)
    ax_yhist.tick_params(left=False, labelleft=False)

    # ── 5. Colour-bar ──────────────────────────────────────────────────
    cb = fig.colorbar(hb, ax=ax_joint, pad=0.02)
    cb.set_label("log10(N)" if density == "log" else "count")

    return ax_joint
