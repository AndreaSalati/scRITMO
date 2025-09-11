import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .linear_regression import harmonic_function_exp, harmonic_function
from scritmo.linear_regression import polar_genes_pandas
from .basics import w, rh, ind2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from adjustText import adjust_text
from .beta import Beta

import seaborn as sns
from .linear_regression import fit_periodic_spline
from scipy.stats import vonmises


def xy(color="red", linestyle="--"):
    plt.axline(
        (0, 0),
        slope=1,
        color=color,
        linestyle=linestyle,
    )


def polar_plot(title="", inner_ring_size=0):
    """
    This function returns the ax object that can be used to plot the polar plot
    or histogram
    Input:
    title: the title of the plot
    inner_ring_size: the size of the inner ring, negative number
        is suggested
    """
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # ax.set_rlim(0, 1)
    # ax.set_rticks([0, 0.5, 1, 1.5, 2])
    ax.set_rlabel_position(0)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    ax.set_xticklabels(np.arange(24))
    ax.set_title(title)

    ax.set_rorigin(inner_ring_size)
    return ax


def polar_plot_shifts(title: str = "", inner_ring_size: float = 0, angle: float = None):
    """
    Returns a polar-Axes in which the 24 hour‐ticks are labeled by their
    signed offsets from “midnight” (θ=0), and—if 'angle' is given—we only
    display the wedge θ ∈ [−angle/2, +angle/2].  The radial‐grid labels
    are placed on the LEFT boundary of that wedge.

    Parameters
    ----------
    title : str, optional
        The title string to put on top of the plot.
    inner_ring_size : float, optional
        If negative, the “zero radius” circle is pushed inward by that amount.
        (In other words, r=0 is actually at radius == inner_ring_size < 0.)
    angle : float or None, optional
        The TOTAL angular span (in radians) you want to see, *centered* on θ=0.
        - If None, we show the full circle (no theta‐clipping).
        - If angle = π/2, we show only θ ∈ [−π/4, +π/4], i.e. ±3 h.
        - If angle = π, we show only θ ∈ [−π/2, +π/2], i.e. ±6 h.

    Returns
    -------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        A polar‐projection Axes on which you can now call .plot(…) or .bar(…), etc.
    """
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection="polar")

    # 1) Put θ=0 at North, and make positive θ go *clockwise*:
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # 2) Compute the 24 “signed” angles in [−π, +π]:
    #
    #    For each hour h = 0..23, the usual angle is
    #       θ_h = (h / 24) * 2π   ∈ [0, 2π).
    #    We shift into [−π, +π] by:
    #       signed_angle_h = (θ_h + π) mod 2π  − π.
    #    That way, h=1 sits at +2π/24, h=23 sits at −2π/24, etc.
    angles_abs = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    signed_angles = (angles_abs + np.pi) % (2 * np.pi) - np.pi

    # 3) Compute the signed‐hour label for each angle,
    #    rounding to the nearest integer instead of truncating:
    raw_values = signed_angles * (24.0 / (2 * np.pi))
    signed_hours = np.round(raw_values).astype(int)
    # (Optional) If you’d prefer “+12” instead of “−12” at exactly θ=±π, you can do:
    signed_hours[signed_hours == -12] = 12

    # 4) Sort by signed_angles so ticks go in increasing order from −π to +π:
    idx = np.argsort(signed_angles)
    sorted_signed_angles = signed_angles[idx]
    sorted_labels = signed_hours[idx].astype(str)

    # 5) Place those 24 ticks at the “signed” angles, labeling them by signed_hours:
    ax.set_xticks(sorted_signed_angles)
    ax.set_xticklabels(sorted_labels)

    # 6) Add title & “inner ring”:
    ax.set_title(title)
    ax.set_rorigin(inner_ring_size)

    # 7) If the user requested a finite wedge (angle != None):
    #       → Clip to θ ∈ [−angle/2, +angle/2].
    #       → Put r‐labels on the LEFT boundary of that wedge.
    if angle is not None:
        half = angle / 2.0
        ax.set_thetalim(-half, half)

        # The left‐boundary is at θ = −half (in radians).
        # Convert to degrees, then place r‐labels there:
        rlabel_deg = -(half * 180.0 / np.pi)
        ax.set_rlabel_position(rlabel_deg)
    else:
        # Full circle & r‐labels straight up at θ=0°:
        ax.set_rlabel_position(0)

    ax.grid(True)
    return ax


def scatter_with_labels(
    x, y, labels, fontsize=10, s=200, arrowstyle="-", color_arr="black"
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
    adjust_text(texts, arrowprops=dict(arrowstyle=arrowstyle, color=color_arr, lw=0.5))


def biplot(
    loadings,
    vectors,
    pcs,
    labels=None,
    arrowscale=1.0,
    dotsize=1,
    fontsize=8,
    plot_vec=True,
):
    """
    Used for cool PCA plots

    loadings: the loadings of the PCA
    vectors: the vectors of the PCA
    pcs: the pcs to plot
    labels: the labels of the genes
    arrowscale: the scale of the arrows
    dotsize: the size of the dots
    fontsize: the fontsize of the labels
    plot_vec: if True, plot the vectors
    """
    pc1, pc2 = pcs
    xs = loadings[:, pc1]
    ys = loadings[:, pc2]
    n = vectors.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, s=dotsize, color="orange")

    if plot_vec:
        vectors = vectors * arrowscale
        for i in range(n):
            plt.arrow(0, 0, vectors[i, pc1], vectors[i, pc2], color="purple", alpha=0.1)
            if not labels is None:
                plt.text(
                    vectors[i, pc1] * 1.15,
                    vectors[i, pc2] * 1.15,
                    labels[i],
                    color="darkblue",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                )

    plt.xlabel("PC{}".format(pc1))
    plt.ylabel("PC{}".format(pc2))


def plot_count_histo(x, normalize=False, **kwargs):

    bin_x, bin_y = np.unique(x, return_counts=True)
    if normalize:
        bin_y = bin_y / bin_y.sum()
    plt.bar(bin_x, bin_y, width=0.9, align="center", **kwargs)


def polar_plot_params_g(
    df, genes_to_plot=None, title="", amp_lim=0.5, cartesian=False, s=20, fontisize=12
):
    """
    Takes as input a pandas dataframe with columns "amp", "phase", "mu"
    and plots the genes in a polar plot
    gene names need to be on df.index

    """
    ax = polar_plot(title=title)

    if cartesian:
        df = polar_genes_pandas(df)

    if genes_to_plot is None:
        genes_to_plot = df.index

    for j, gene in enumerate(df.index):
        if gene not in genes_to_plot:
            continue
        amp, phase, mu = df.iloc[j, 0:3]
        ax.scatter(phase, amp, s=s)
        # annotate
        if amp < amp_lim:
            continue
        ax.annotate(gene, (phase, amp), fontsize=fontisize)


def polar_plot_params_g2(
    df,
    genes_to_plot=None,
    title="",
    amp_lim=[0.0, 5.0],
    s=20,
    fontisize=12,
    col_names=["amp", "phase"],
):
    """
    Takes as input a pandas dataframe with columns "amp", "phase"
    doesn't metter the order of the columns
    """
    ax = polar_plot(title=title)

    if genes_to_plot is None:
        genes_to_plot = df.index

    for j, gene in enumerate(df.index):

        amp, phase = df[col_names].iloc[j]

        if gene not in genes_to_plot:
            continue

        if amp < amp_lim[0] or amp > amp_lim[1]:
            continue

        ax.scatter(phase, amp, s=s)
        # annotate

        ax.annotate(gene, (phase, amp), fontsize=fontisize)


# def integrated_std_histo(v, tr_vec=None):
#     if tr_vec is None:
#         tr_vec = np.linspace(0, 6, 100)
#     # integrate amplitude histogram
#     int_v = np.zeros(len(tr_vec))
#     for i, tr in enumerate(tr_vec):
#         int_v[i] = np.sum(v < tr)
#     int_v = int_v / int_v[-1]

#     return tr_vec, int_v


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
            # 'font.serif': ['Times New Roman'],   # Font choice, adjust to your needs
            "axes.linewidth": 1.5,  # Width of the axis lines
            "lines.linewidth": 2.0,  # Line width for plots
            "axes.spines.top": False,  # Disable top spine
            "axes.spines.right": False,  # Disable right spine
            "legend.frameon": False,  # Disable legend box
            "savefig.dpi": 300,  # Set DPI for saving figures, important for publication-quality figures
            "savefig.format": "pdf",  # Default file format when saving figures
        }
    )


def bin_data(
    adata,
    covariate,
    genes,
    layer,
    n_bins,
):

    # Get expression values
    expression = adata[:, genes].layers[layer].toarray().squeeze()
    n_genes = expression.shape[1]

    # Bin the data
    bin_edges = np.linspace(covariate.min(), covariate.max(), n_bins + 1)
    bin_indices = np.digitize(covariate, bin_edges) - 1
    bin_indices[bin_indices == n_bins] = 0  # Wrap around for circular data

    # Calculate the center of each bin
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate mean expression for each bin
    binned_expr = np.zeros((n_bins, n_genes))
    for i in range(n_bins):
        # Check if there are data points in this bin
        if np.sum(bin_indices == i) > 0:
            binned_expr[i] = np.mean(expression[bin_indices == i], axis=0)
        else:
            binned_expr[i] = np.nan  # No data in this bin

    # Plot binned data
    valid_bins = ~np.isnan(binned_expr)

    return bin_centers, binned_expr


def plot_circadian_data(
    adata,
    phis,
    g,
    ax=None,
    layer="spliced",
    n_bins=None,
    alpha=0.7,
    s=1,
    log_bin_y=False,
    jitter=0.0,
):
    """
    Plot expression values of a gene over circadian phase.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    phis : array-like
        Array of circadian phases (in radians).
    g : str
        Gene name to plot.
    ax : matplotlib.axes.Axes or None
        Axis to plot on. If None, a new figure and axis are created.
    layer : str
        Layer in `adata.layers` to extract expression from.
    n_bins : int or None
        Number of bins for aggregating expression. If None, raw points are plotted.
    alpha : float
        Transparency of scatter points.
    s : float
        Size of scatter points.
    log_bin_y : bool
        If True, log-transform the binned expression values.
    jitter : float
        Standard deviation of Gaussian noise added to phase values (in hours) for visual separation.
    """
    if ax is None:
        fig, ax = plt.subplots()

    w = 2 * np.pi / 24
    rh = w**-1
    phi_x = np.linspace(0, 2 * np.pi, 100)

    # Get expression values
    if layer is None:
        expression = adata[:, g].X.toarray().squeeze()
    else:
        expression = adata[:, g].layers[layer].toarray().squeeze()

    if n_bins is not None:
        # Bin the data
        bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
        bin_indices = np.digitize(phis, bin_edges) - 1
        # Wrap around for circular data
        bin_indices[bin_indices == n_bins] = 0

        # Calculate the center of each bin
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate mean expression for each bin
        binned_expr = np.zeros(n_bins)
        for i in range(n_bins):
            # Check if there are data points in this bin
            if np.sum(bin_indices == i) > 0:
                binned_expr[i] = np.mean(expression[bin_indices == i])
            else:
                binned_expr[i] = np.nan  # No data in this bin

        # Plot binned data
        valid_bins = ~np.isnan(binned_expr)
        if log_bin_y:
            binned_expr[valid_bins] = np.log(binned_expr[valid_bins])
        ax.scatter(
            bin_centers[valid_bins] * rh,
            binned_expr[valid_bins],
            s=s,
            alpha=alpha,
            label="binned data",
            marker="o",
        )
    else:
        x = (phis * rh) + np.random.normal(0, jitter, size=phis.shape)
        # Plot original data points
        ax.scatter(
            x,
            expression,
            s=s,
            label="data",
            alpha=alpha,
        )

    # Plot details
    plt.title(f"{g}")
    ax.set_xlabel("Circadian phase")
    ax.set_ylabel("Normalized expression")
    ax.legend()

    return ax


def plot_circadian_data_and_fit(
    adata,
    phis,
    g,
    params_g,
    ax=None,
    layer="spliced",
    n_bins=None,
    alpha=0.7,
    line_color="red",
    log_bin_y=False,
    exp=True,
    columns_names=["amp", "phase", "a_0"],
    s=10,
    jitter=0.0,
):
    """
    Creates a plot of circadian expression data and GLM fit, returning an axis object.
    If n_bins is provided, data will be binned by phase before plotting.

    Parameters:
        adata (AnnData): AnnData object containing gene data.
        phis (array): Phase values for scatter plot.
        g (str): Gene name or index.
        params_g (DataFrame): Parameter dataframe for GLM fit.
        ax (matplotlib.axes._axes.Axes, optional): Axis to plot on. If None, creates a new one.
        layer (str): Layer of AnnData object to use. Default is "spliced".
        n_bins (int, optional): Number of bins to use for data. If None, no binning is performed.
        alpha (float): Transparency level for scatter points.
        line_color (str): Color for the GLM fit line.
        log_bin_y (bool): If True, apply log transformation to binned y values.
        exp (bool): If True, apply exponential transformation to GLM fit.
        columns_names (list): Column names in params_g for amp, phase, and mu.
        jitter : float
            Standard deviation of Gaussian noise added to phase values (in hours) for visual separation.

    Returns:
        matplotlib.axes._axes.Axes: Axis object with the plot.
    """
    # First plot the data points using plot_circadian_data
    ax = plot_circadian_data(
        adata,
        phis,
        g,
        ax=ax,
        layer=layer,
        n_bins=n_bins,
        alpha=alpha,
        log_bin_y=log_bin_y,
        s=s,
        jitter=jitter,
    )

    # Now add the GLM fit
    w = 2 * np.pi / 24
    rh = w**-1
    phi_x = np.linspace(0, 2 * np.pi, 100)

    params_g = Beta(params_g)
    y = params_g.predict(phi_x, exp_base=exp)

    ind_g = ind2(params_g.index, g)[0]
    y = y[:, ind_g]

    # amp, phase, mu = params_g[columns_names].loc[g]
    # y = amp * np.cos(phi_x - phase) + mu
    # if exp:
    #     y = np.exp(y)

    ax.plot(
        phi_x * rh,
        y,
        c=line_color,
        label="GLM fit",
    )

    # Update legend since we've added a new line
    ax.legend()

    return ax


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


def bin_data_simple(data, covariate, n_bins):
    """
    Bin an arbitrary data array according to a numeric covariate.

    Parameters
    ----------
    data : np.ndarray
        Either a 1-D array of length n_samples, or a 2-D array of shape (n_samples, n_features).
    covariate : np.ndarray
        A 1-D array of length n_samples, giving the continuous covariate for each sample.
    n_bins : int
        The number of equal-width bins to create along the range of `covariate`.

    Returns
    -------
    bin_centers : np.ndarray, shape (n_bins,)
        The center (midpoint) of each of the n_bins.
    binned_means : np.ndarray
        If data was 2-D of shape (n_samples, n_features), returns shape (n_bins, n_features),
        where row k is the mean of data[samples_in_bin_k, :]. Bins with no samples are NaN.
        If data was 1-D (length n_samples), returns shape (n_bins,) of means per bin.
    """
    # Convert data to 2D: (n_samples, n_features). If already 2D, this is a no-op.
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)  # now shape = (n_samples, 1)

    n_samples, n_features = arr.shape

    # Step 1: compute equally spaced bin edges over the covariate
    cov = np.asarray(covariate)
    if cov.ndim != 1 or cov.shape[0] != n_samples:
        raise ValueError("covariate must be a 1D array of length n_samples")

    cov_min, cov_max = cov.min(), cov.max()
    bin_edges = np.linspace(cov_min, cov_max, n_bins + 1)

    # Step 2: assign each sample to a bin
    # np.digitize returns values in 1..(n_bins+1), so subtract 1 for 0..n_bins
    bin_idx = np.digitize(cov, bin_edges) - 1
    # If any covariate == cov_max, np.digitize will give index = n_bins, so wrap to n_bins - 1
    bin_idx[bin_idx == n_bins] = n_bins - 1

    # Step 3: compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Step 4: allocate output array and fill by averaging
    binned_means = np.full((n_bins, n_features), np.nan, dtype=float)
    for k in range(n_bins):
        mask = bin_idx == k
        if np.any(mask):
            # compute mean over axis=0 (i.e. over all samples in bin k)
            binned_means[k, :] = arr[mask, :].mean(axis=0)
        # else leave as NaN

    # If original data was 1-D, return a 1-D array of length n_bins
    if data.ndim == 1:
        binned_means = binned_means.squeeze()  # shape (n_bins,)

    return bin_centers, binned_means


def plot_phase_polar(
    cell_type,
    time,
    phase,
    sample_name=None,
    amplitude=None,
    plot_type="density",
    cmap_name="twilight",
    bins=30,
    ylim=None,
):
    """
    Polar plots of phase data for each cell type.

    Parameters
    ----------
    cell_type : array-like (n_cells,)
        Label of each cell (e.g. string or int), loops over unique values.
    time : array-like (n_cells,)
        Zeitgeber time (ZT) in hours for each cell.
    phase : array-like (n_cells,)
        Inferred phase in radians for each cell.
    sample_name : array-like (n_cells,), optional
        Sample identifier for each cell; required for 'density'.
    amplitude : array-like (n_cells,), optional
        Amplitude or count per cell; required for 'scatter'.
    plot_type : {'density', 'scatter', 'histogram'}
    cmap_name : str
        Cyclic colormap name (default 'twilight').
    bins : int
        Number of bins for histogram (only if plot_type='histogram').
    """
    # validate inputs
    if plot_type == "density" and sample_name is None:
        raise ValueError("`sample_name` is required for density plots")
    if plot_type == "scatter" and amplitude is None:
        raise ValueError("`amplitude` is required for scatter plots")

    cell_types = np.unique(cell_type)
    cmap = plt.get_cmap(cmap_name)

    for ct in cell_types:
        mask_ct = cell_type == ct
        times_ct = time[mask_ct]
        phases_ct = phase[mask_ct]
        samples_ct = sample_name[mask_ct] if sample_name is not None else None
        amps_ct = amplitude[mask_ct] if amplitude is not None else None
        zts = np.unique(times_ct)

        # set up polar axes
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.set_rorigin(-0.3)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.grid(False)

        if plot_type == "scatter":
            max_amp = np.max(amps_ct)
            ticks = np.linspace(0, max_amp, num=5)
            ax.set_yticks(ticks)
            ax.set_ylim(0, max_amp * 1.1)
        else:
            ax.set_yticklabels([])

        # angular ticks every 6h
        hour_ticks = np.arange(0, 24, 6)
        angles = 2 * np.pi * hour_ticks / 24
        ax.set_xticks(angles)
        ax.set_xticklabels([f"ZT{h:02d}" for h in hour_ticks], fontsize=10)

        base_colors = cmap(np.linspace(0, 1, len(zts), endpoint=False))
        theta_vals = np.linspace(0, 2 * np.pi, 200, endpoint=False)

        for i, zt in enumerate(zts):
            color = base_colors[i]
            theta_ref = 2 * np.pi * (zt % 24) / 24
            # ax.plot(
            #     [theta_ref, theta_ref],
            #     [0, ax.get_rmax()],
            #     linestyle="--",
            #     color=color,
            #     linewidth=1.5,
            # )

            mask_zt = times_ct == zt
            ph = phases_ct[mask_zt]

            if plot_type == "density":
                for j, sam in enumerate(np.unique(samples_ct[mask_zt])):
                    ph_s = ph[samples_ct[mask_zt] == sam]
                    if len(ph_s) > 1:
                        # fit returns (kappa, loc, scale)
                        kappa, loc, scale = vonmises.fit(ph_s, method="analytical")
                        dens = vonmises.pdf(theta_vals, kappa, loc=loc, scale=scale)
                        alpha_fill = np.linspace(
                            0.2, 0.6, len(np.unique(samples_ct[mask_zt]))
                        )[j]
                        alpha_line = np.linspace(
                            0.5, 1.0, len(np.unique(samples_ct[mask_zt]))
                        )[j]
                        lw = np.linspace(0.8, 1.8, len(np.unique(samples_ct[mask_zt])))[
                            j
                        ]
                        ax.fill_between(
                            theta_vals, 0, dens, color=color, alpha=alpha_fill
                        )
                        ax.plot(
                            theta_vals,
                            dens,
                            color=color,
                            alpha=alpha_line,
                            linewidth=lw,
                        )
                    else:
                        ax.plot(ph_s, 0.1, "o", color=color, alpha=0.9, markersize=4)

            elif plot_type == "scatter":
                ax.scatter(
                    ph,
                    amps_ct[mask_zt],
                    s=10,
                    color=color,
                    alpha=0.2,
                    edgecolors="none",
                )
                # change ylim
                if ylim is not None:
                    ax.set_ylim(0, ylim)

            elif plot_type == "histogram":
                ax.hist(
                    ph,
                    bins=bins,
                    density=True,
                    alpha=0.5,
                    color=color,
                    label=f"{zt:.0f}h",
                )

        rmax = ax.get_rmax()
        # print(f"rmax: {rmax}")
        for i, zt in enumerate(zts):
            theta_ref = 2 * np.pi * (zt % 24) / 24
            ax.plot(
                [theta_ref, theta_ref],
                [0, rmax],
                linestyle="--",
                color=color,
                linewidth=1.5,
            )
        ax.set_title(f"{ct} – {plot_type.capitalize()} Plot", va="bottom")
        if plot_type == "histogram":
            ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
        plt.tight_layout()
        plt.show()


def plot_phase_polar_single_ct(
    ct,
    cell_type_list,
    time,
    phase,
    sample_name=None,
    amplitude=None,
    plot_type="density",
    cmap_name="twilight",
    bins=30,
    ylim=None,
):
    """
    same thing as before, jsut returns the ax and fig objects
    and does one gene at the time
    """
    # validate inputs
    if plot_type == "density" and sample_name is None:
        raise ValueError("`sample_name` is required for density plots")
    if plot_type == "scatter" and amplitude is None:
        raise ValueError("`amplitude` is required for scatter plots")

    cmap = plt.get_cmap(cmap_name)

    mask_ct = cell_type_list == ct
    times_ct = time[mask_ct]
    phases_ct = phase[mask_ct]
    samples_ct = sample_name[mask_ct] if sample_name is not None else None
    amps_ct = amplitude[mask_ct] if amplitude is not None else None
    zts = np.unique(times_ct)

    # set up polar axes
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.set_rorigin(-0.3)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.grid(False)

    if plot_type == "scatter":
        max_amp = np.max(amps_ct)
        ticks = np.linspace(0, max_amp, num=5)
        ax.set_yticks(ticks)
        ax.set_ylim(0, max_amp * 1.1)
    else:
        ax.set_yticklabels([])

    # angular ticks every 6h
    hour_ticks = np.arange(0, 24, 6)
    angles = 2 * np.pi * hour_ticks / 24
    ax.set_xticks(angles)
    ax.set_xticklabels([f"ZT{h:02d}" for h in hour_ticks], fontsize=10)

    base_colors = cmap(np.linspace(0, 1, len(zts), endpoint=False))
    theta_vals = np.linspace(0, 2 * np.pi, 200, endpoint=False)

    for i, zt in enumerate(zts):
        color = base_colors[i]
        theta_ref = 2 * np.pi * (zt % 24) / 24
        ax.plot(
            [theta_ref, theta_ref],
            [0, ax.get_rmax()],
            linestyle="--",
            color=color,
            linewidth=1.5,
        )

        mask_zt = times_ct == zt
        ph = phases_ct[mask_zt]

        if plot_type == "density":
            for j, sam in enumerate(np.unique(samples_ct[mask_zt])):
                ph_s = ph[samples_ct[mask_zt] == sam]
                if len(ph_s) > 1:
                    # fit returns (kappa, loc, scale)
                    kappa, loc, scale = vonmises.fit(ph_s, method="analytical")
                    dens = vonmises.pdf(theta_vals, kappa, loc=loc, scale=scale)
                    alpha_fill = np.linspace(
                        0.2, 0.6, len(np.unique(samples_ct[mask_zt]))
                    )[j]
                    alpha_line = np.linspace(
                        0.5, 1.0, len(np.unique(samples_ct[mask_zt]))
                    )[j]
                    lw = np.linspace(0.8, 1.8, len(np.unique(samples_ct[mask_zt])))[j]
                    ax.fill_between(theta_vals, 0, dens, color=color, alpha=alpha_fill)
                    ax.plot(
                        theta_vals,
                        dens,
                        color=color,
                        alpha=alpha_line,
                        linewidth=lw,
                    )
                else:
                    ax.plot(ph_s, 0.1, "o", color=color, alpha=0.9, markersize=4)

        elif plot_type == "scatter":
            ax.scatter(
                ph,
                amps_ct[mask_zt],
                s=10,
                color=color,
                alpha=0.2,
                edgecolors="none",
            )
            # change ylim
            if ylim is not None:
                ax.set_ylim(0, ylim)

        elif plot_type == "histogram":
            ax.hist(
                ph,
                bins=bins,
                density=True,
                alpha=0.5,
                color=color,
                label=f"{zt:.0f}h",
            )
    rmax = ax.get_rmax()
    for i, zt in enumerate(zts):
        color = base_colors[i]
        theta_ref = 2 * np.pi * (zt % 24) / 24
        ax.plot(
            [theta_ref, theta_ref],
            [0, rmax],
            linestyle="--",
            color=color,
            linewidth=1.5,
        )

    ax.set_title(f"{ct} – {plot_type.capitalize()} Plot", va="bottom")
    if plot_type == "histogram":
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    return fig, ax


def plot_cae_per_sample(sample_name, time_mod, cad, plot_dir):
    """
    Generates and saves a boxplot with jitter points showing the Circular
    Absolute Error (CAE) for each sample, grouped by external phase.

    This function annotates each box with the Median Absolute Error (MAE).

    Parameters:
    ----------
    sample_name:

    time_mod : np.ndarray
        A NumPy array indicating the external phase (e.g., ZT) for each cell.
        Must be the same length as the number of cells in `adata`.

    cad : np.ndarray
        A NumPy array containing the Circular Absolute Distance/Error values
        for each cell. Must be the same length as `time_mod`.

    plot_dir : str
        The directory path where the output plot 'jitter_plot.pdf' will be saved.
        The path should end with a '/'.
    """
    # --- Data Preparation ---
    sample_name_ct = sample_name
    unique_phases = np.unique(time_mod)

    # Lists to hold data for plotting
    cad_grouped = []
    labels = []
    colors = []
    mae_annotations = []
    x_scatter = []
    y_scatter = []
    c_scatter = []

    # Get a color map for the different phases
    cmap = plt.colormaps["twilight"]
    phase_colors = cmap(np.linspace(0, 1, len(unique_phases), endpoint=False))

    # --- Group Data by Phase and Sample ---
    for i, phase in enumerate(unique_phases):
        samples_in_phase = np.unique(sample_name_ct[time_mod == phase])
        n_samples = len(samples_in_phase)
        base_color = phase_colors[i]
        # Create different alpha levels for samples within the same phase
        alphas = np.linspace(0.4, 0.8, n_samples)

        for j, sample in enumerate(samples_in_phase):
            # Create a mask to select cells for the current phase and sample
            mask = (time_mod == phase) & (sample_name_ct == sample)
            cad_per_sample = cad[mask] * rh
            cad_grouped.append(cad_per_sample)

            # Create a label for the x-axis tick
            labels.append(f"ZT{int(phase)}\n{sample}")

            # Assign a unique color with a specific alpha
            current_color = (*base_color[:3], alphas[j])
            colors.append(current_color)

            # Compute Median Absolute Error for this sample's annotation
            mae = np.median(cad_per_sample)
            box_index = len(cad_grouped)  # 1-based index for plotting
            mae_annotations.append((box_index, f"MAE={mae:.2f}h"))

            # Generate jittered x-coordinates for the scatter plot
            xs = box_index + np.random.uniform(-0.15, 0.15, size=len(cad_per_sample))
            x_scatter.extend(xs)
            y_scatter.extend(cad_per_sample)
            c_scatter.extend([current_color] * len(cad_per_sample))

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the boxplot
    bp = ax.boxplot(cad_grouped, patch_artist=True, showfliers=False)

    # Set the face color for each box
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    # Overlay the jittered scatter plot
    ax.scatter(x_scatter, y_scatter, color=c_scatter, alpha=0.6, s=12, edgecolor="none")

    # Add MAE annotations above each box
    for x_pos, text in mae_annotations:
        # Position the text relative to the max value of the data in the box
        ymax = np.max(cad_grouped[x_pos - 1])
        ax.text(x_pos, ymax * 0.9, text, ha="center", va="top", fontsize=8)

    # --- Final Touches ---
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Circular Absolute Error (h)")
    ax.set_title("CAE per Sample within Each External Phase—with per-sample MAE")
    plt.tight_layout()
    return fig, ax


def amp_inflation_plot(
    params_g,
    par_refit,
    mask_inf,
    full_name_titles="",
    annotate_names=False,
    show_marginals=True,
    max_val=4.0,
):
    """
    Creates a jointplot with options to annotate genes and hide marginal plots.

    Args:
        params_g (pd.DataFrame): DataFrame with initial data.
        par_refit (pd.DataFrame): DataFrame with refitted data.
        mask_inf (np.array): Boolean mask for inference genes.
        full_name_titles (str): Title suffix for the plot.
        annotate_names (bool): If True, annotates the inference genes.
        show_marginals (bool): If False, hides the top and right marginal plots.

    Returns:
        tuple: A tuple containing the matplotlib Figure and Axes objects.
    """
    # Prepare data for plotting
    x = params_g["log2fc"][~mask_inf]
    y = par_refit["log2fc"][~mask_inf]
    x_inf = params_g["log2fc"][mask_inf]
    y_inf = par_refit["log2fc"][mask_inf]
    inf_genes = params_g.index[mask_inf]

    percent_inflation = (y > x).sum() / len(x) * 100

    # Create the initial jointplot
    g = sns.jointplot(x=x, y=y, alpha=0.5)

    # Plot the second scatter plot for the inference genes
    g.ax_joint.scatter(x_inf, y_inf, color="orange", s=20, label="Inference genes")

    # Add annotations for inference genes if requested
    if annotate_names:
        for gene, x_coord, y_coord in zip(inf_genes, x_inf, y_inf):
            g.ax_joint.text(x_coord + 0.05, y_coord, gene, fontsize=8)

    # Add the KDE plot and a diagonal line
    g.plot_joint(sns.kdeplot, color="r", zorder=1, levels=6, alpha=1)
    g.ax_joint.axline((0, 0), slope=1, ls="--", c=".3", label="y=x")

    # --- Hide Marginal Plots ---
    # Hide the marginal plots if requested
    if not show_marginals:
        g.ax_marg_x.set_visible(False)
        g.ax_marg_y.set_visible(False)
    # ---------------------------

    # Set titles and labels
    g.figure.suptitle(
        f"Amp inflation {full_name_titles} ({percent_inflation:.1f}%)", fontsize=16
    )
    g.ax_joint.set_xlabel("Initial Amp (log2FC)")
    g.ax_joint.set_ylabel("Inferred Amp (log2FC)")

    # Manually create the legend
    handles, labels = g.ax_joint.get_legend_handles_labels()
    handles.insert(0, g.ax_joint.collections[0])
    labels.insert(0, "Rhythmic genes")
    g.ax_joint.legend(handles, labels)
    g.ax_joint.set_xlim((None, max_val))
    g.ax_joint.set_ylim((None, max_val))

    plt.tight_layout()

    return g.fig, g.ax_joint


def desynchrony_time_plot(
    context_label, df_desync, agg_df_data, agg_df_0, n_df="adaptive"
):
    """
    Generates a plot for a specific context_label, showing cSTD vs. External Time.

    Args:
        context_label (str): The context_label (e.g., 'celltype') to filter data for.
        df_desync (pd.DataFrame): DataFrame with BIO data.
        agg_df_data (pd.DataFrame): DataFrame with aggregated DATA.
        agg_df_0 (pd.DataFrame): DataFrame with 'perfect sync' data.
        n_df (int or str): Number of degrees of freedom for spline fitting. If 'adaptive', it will be set based on the number of data points.

    Returns:
        tuple: A tuple containing the matplotlib figure and axis objects (fig, ax).
    """
    period = 24
    rh = 24 / (2 * np.pi)  # Conversion factor from radians to hours
    # Define a consistent color palette and plot settings
    color_map = {"BIO": "#1f77b4", "DATA": "#ff7f0e", "perfect sync": "#2ca02c"}

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Data sources to plot ---
    # A list of tuples: (label, dataframe, x_col, y_col, color, transform_func)
    sources = [
        (
            "BIO",
            df_desync[df_desync.context == context_label],
            "ext_time_hours",
            "simulated_phases_cStd",
            color_map["BIO"],
            lambda x, y: (x, y),
        ),
        (
            "DATA",
            agg_df_data[agg_df_data.context == context_label],
            "ext_time",
            "circSTD",
            color_map["DATA"],
            lambda x, y: (x * rh, y * rh),
        ),
        (
            "perfect sync",
            agg_df_0[agg_df_0.context == context_label],
            "ext_time",
            "circSTD",
            color_map["perfect sync"],
            lambda x, y: (x * rh, y * rh),
        ),
    ]

    # --- Loop through each data source for the current context ---
    for label, df_source, x_col, y_col, color, transform in sources:
        if df_source.empty:
            print(f"  - No data for '{label}' in {context_label}. Skipping.")
            continue

        x_raw = df_source[x_col].values
        y_raw = df_source[y_col].values
        x_data, y_data = transform(x_raw, y_raw)

        ax.scatter(x_data, y_data, label=label, color=color, alpha=0.6, s=30, zorder=2)

        if n_df == "adaptive":
            n_df = 3 if len(x_data) < 8 else 4

        predict_spline = fit_periodic_spline(
            x_data,
            y_data,
            df=n_df,
            period=24,
        )

        x_smooth = np.linspace(0, period, 400)
        y_smooth_pred = predict_spline(x_smooth)

        ax.plot(x_smooth, y_smooth_pred, color=color, linewidth=2.5, zorder=3)

    # --- Finalize and customize the plot ---
    ax.set_title(
        f"Group cSTD vs. External Time (Context: {context_label})", fontsize=16
    )
    ax.set_xlabel("External Time [h]", fontsize=12)
    ax.set_ylabel("Group cSTD [h]", fontsize=12)
    ax.set_xlim(-1, 24)
    ax.set_ylim(0, None)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(title="Data Type")
    fig.tight_layout()

    return fig, ax


def plot_dual_loss(loss_epochs, mad_epochs, title="Training Metrics Over Epochs"):
    """
    Creates a plot with two y-axes for loss and MAD.

    Args:
        loss_epochs (list or np.ndarray): List of loss values per epoch.
        mad_epochs (list or np.ndarray): List of MAD values per epoch.
        title (str): Title for the plot.
    """
    # Create the figure and the first axes (for loss)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the loss on the left y-axis
    color_loss = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color_loss)
    ax1.plot(loss_epochs, color=color_loss, label="Loss")
    ax1.tick_params(axis="y", labelcolor=color_loss)

    # Create a secondary y-axis for MAD
    ax2 = ax1.twinx()  # instantiates a second axes that shares the same x-axis

    # Plot the MAD on the right y-axis
    color_mad = "tab:blue"
    ax2.set_ylabel("MAD", color=color_mad)  # we already handled the x-label with ax1
    ax2.plot(mad_epochs, color=color_mad, label="MAD")
    ax2.tick_params(axis="y", labelcolor=color_mad)

    # Add a title and legend to the plot
    fig.suptitle(title)
    fig.tight_layout()  # to prevent labels from overlapping

    # Manually create a legend that includes both lines
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="best")

    # return axis
    return fig, ax1, ax2
