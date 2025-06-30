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

from scipy.stats import vonmises

def xy():
    plt.axline(
        (0, 0),
        slope=1,
        color="red",
        linestyle="--",
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
    plot_type='density',
    cmap_name='twilight',
    bins=30,
    ylim=None
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
    if plot_type == 'density' and sample_name is None:
        raise ValueError("`sample_name` is required for density plots")
    if plot_type == 'scatter' and amplitude is None:
        raise ValueError("`amplitude` is required for scatter plots")

    cell_types = np.unique(cell_type)
    cmap = plt.get_cmap(cmap_name)

    for ct in cell_types:
        mask_ct = (cell_type == ct)
        times_ct = time[mask_ct]
        phases_ct = phase[mask_ct]
        samples_ct = sample_name[mask_ct] if sample_name is not None else None
        amps_ct = amplitude[mask_ct] if amplitude is not None else None
        zts = np.unique(times_ct)

        # set up polar axes
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_rorigin(-0.3)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.grid(False)

        if plot_type == 'scatter':
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
            ax.plot([theta_ref, theta_ref], [0, ax.get_rmax()],
                    linestyle='--', color=color, linewidth=1.5)

            mask_zt = (times_ct == zt)
            ph = phases_ct[mask_zt]

            if plot_type == 'density':
                for j, sam in enumerate(np.unique(samples_ct[mask_zt])):
                    ph_s = ph[samples_ct[mask_zt] == sam]
                    if len(ph_s) > 1:
                        # fit returns (kappa, loc, scale)
                        kappa, loc, scale = vonmises.fit(ph_s, method='analytical')
                        dens = vonmises.pdf(theta_vals, kappa, loc=loc, scale=scale)
                        alpha_fill = np.linspace(0.2, 0.6, len(np.unique(samples_ct[mask_zt])))[j]
                        alpha_line = np.linspace(0.5, 1.0, len(np.unique(samples_ct[mask_zt])))[j]
                        lw = np.linspace(0.8, 1.8, len(np.unique(samples_ct[mask_zt])))[j]
                        ax.fill_between(theta_vals, 0, dens, color=color, alpha=alpha_fill)
                        ax.plot(theta_vals, dens, color=color, alpha=alpha_line, linewidth=lw)
                    else:
                        ax.plot(ph_s, 0.1, 'o', color=color, alpha=0.9, markersize=4)

            elif plot_type == 'scatter':
                ax.scatter(ph, amps_ct[mask_zt], s=10, color=color,
                           alpha=0.2, edgecolors='none')
                #change ylim
                if ylim is not None:
                    ax.set_ylim(0, ylim)

            elif plot_type == 'histogram':
                ax.hist(ph, bins=bins, density=True, alpha=0.5,
                        color=color, label=f"{zt}h")

        ax.set_title(f"{ct} – {plot_type.capitalize()} Plot", va='bottom')
        if plot_type == 'histogram':
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        plt.tight_layout()
        plt.show()
