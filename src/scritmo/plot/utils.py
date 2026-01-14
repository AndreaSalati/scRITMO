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
    ax=None,
):
    """
    This function configures and returns a polar ax object.
    If 'ax' is provided, it configures that axis.
    If 'ax' is None, it creates a new figure and polar axis.

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
    ax : matplotlib.axes.Axes, optional
        An existing polar axes object to configure. Must have
        projection='polar'. If None, a new figure and axes are created.
    """
    if aperture is not None:
        theta_min = -aperture / 2
        theta_max = aperture / 2

    # --- This is the key change ---
    # If no axis is provided, create a new one
    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, projection="polar")
    # ------------------------------

    # The rest of the function operates on 'ax' as before
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
    ax=None,
    fontsize=10,
    arrowstyle="-",
    color_arr="black",
    adjust=True,
    scatter=False,
):
    """
    Plots a scatter and annotates it with non-overlapping labels on a specific ax.
    """
    if ax is None:
        ax = plt.gca()

    # Create the scatter plot
    if scatter:
        ax.scatter(x, y)

    texts = []
    for xi, yi, label in zip(x, y, labels):
        texts.append(ax.text(xi, yi, label, fontsize=fontsize))

    if adjust:
        # adjust_text needs the ax to calculate overlaps correctly
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle=arrowstyle, color=color_arr, lw=0.5),
        )

    return ax


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




def adjust_polar_text(df, theta_col, r_col, ax, label_col=None, fontsize=9, **kwargs):
    """
    Robustly adjusts text on a polar plot by projecting to a Cartesian grid
    that matches the EXACT physical dimensions of the target subplot.
    """
    # 1. Extract Data
    r = df[r_col].values
    theta = df[theta_col].values
    labels = df.index if label_col is None else df[label_col].values

    # 2. Project to Cartesian (North=0, Clockwise)
    x_cart = r * np.sin(theta)
    y_cart = r * np.cos(theta)

    # --- THE CRITICAL FIX ---
    # Calculate the actual size of subplot 'ax' in inches
    # ax.get_position() returns relative bbox (0-1). We multiply by figure size.
    bbox = ax.get_position()
    fig_width, fig_height = ax.figure.get_size_inches()

    # Dimensions in inches (with a small safety floor to prevent 0-size errors)
    width_inch = max(bbox.width * fig_width, 1.0)
    height_inch = max(bbox.height * fig_height, 1.0)

    # 3. Create Dummy Figure with the CORRECT scaled size
    # Now adjust_text will struggle with space just like the real plot,
    # forcing it to find better positions.
    fig_temp, ax_temp = plt.subplots(figsize=(width_inch, height_inch))

    # Match limits to preserve aspect ratio / density
    limit = max(r) * 1.5
    ax_temp.set_xlim(-limit, limit)
    ax_temp.set_ylim(-limit, limit)

    # Plot invisible obstacles
    scat_size = kwargs.pop("s", 50)
    scat = ax_temp.scatter(x_cart, y_cart, s=scat_size, color="red", alpha=0)

    texts_temp = []
    for x, y, label in zip(x_cart, y_cart, labels):
        # Pre-Alignment Logic
        ha = "left" if x >= 0 else "right"
        va = "bottom" if y >= 0 else "top"

        # Pass fontsize so algorithm knows real text collision size
        texts_temp.append(ax_temp.text(x, y, label, ha=ha, va=va, fontsize=fontsize))

    # 4. Run adjust_text
    adjust_text(
        texts_temp,
        ax=ax_temp,
        add_objects=[scat],
        **kwargs,
    )

    # 5. Map Back to Polar and Plot
    adjusted_texts = []
    for i, t in enumerate(texts_temp):
        x_adj, y_adj = t.get_position()

        # Convert Cartesian back to Polar
        r_adj = np.sqrt(x_adj**2 + y_adj**2)
        theta_adj = np.arctan2(x_adj, y_adj)

        # Add text to real axis
        new_text = ax.text(
            theta_adj,
            r_adj,
            t.get_text(),
            ha=t.get_horizontalalignment(),
            va=t.get_verticalalignment(),
            fontsize=fontsize,
            zorder=100,
        )
        adjusted_texts.append(new_text)

        if "arrowprops" in kwargs:
            ax.annotate(
                "",
                xy=(theta[i], r[i]),
                xytext=(theta_adj, r_adj),
                arrowprops=kwargs["arrowprops"],
                zorder=99,
            )

    plt.close(fig_temp)
    return adjusted_texts
