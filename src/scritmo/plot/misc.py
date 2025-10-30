import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scritmo.basics import w, rh, ind2
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from adjustText import adjust_text
from scritmo.beta import Beta

import seaborn as sns
from scipy.stats import vonmises


def polar_plot_shifts(
    title: str = "",
    inner_ring_size: float = 0,
    angle: float = None,
    show_radial_grid: bool = True,  # Replaces show_grid
    show_angular_grid: bool = True,  # Replaces show_grid
    show_rlabels: bool = True,
    xtick_fontsize: float = 10,
    ax=None,
):
    """
    Returns a polar-Axes in which the 24 hour‐ticks are labeled by their
    signed offsets from “midnight” (θ=0), and—if 'angle' is given—we only
    display the wedge θ ∈ [−angle/2, +angle/2].  The radial‐grid labels
    are placed on the LEFT boundary of that wedge.

    This version is based on the original function, with added parameters
    for grid/label visibility and font size.

    Parameters
    ----------
    title : str, optional
        The title string to put on top of the plot.
    inner_ring_size : float, optional
        If negative, the “zero radius” circle is pushed inward by that amount.
    angle : float or None, optional
        The TOTAL angular span (in radians) you want to see, *centered* on θ=0.
    show_radial_grid : bool, optional
        If False, removes radial grid lines. Default is True.
    show_angular_grid : bool, optional
        If False, removes angular grid lines. Default is True.
    show_rlabels : bool, optional
        If False, removes radial axis labels (numbers). Default is True.
    xtick_fontsize : int or float, optional
        Font size for the angular tick labels (e.g., "-6", "0", "+6"). Default is 10.
    ax : matplotlib.axes.Axes, optional
        An existing polar axes object to configure. Must have
        projection='polar'. If None, a new figure and axes are created.

    Returns
    -------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        A polar‐projection Axes on which you can now call .plot(…) or .bar(…), etc.
    """
    # If no axis is provided, create a new one
    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, projection="polar")

    # 1) Put θ=0 at North, and make positive θ go *clockwise*:
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # 2) Compute the 24 “signed” angles in [−π, +π]:
    angles_abs = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    signed_angles = (angles_abs + np.pi) % (2 * np.pi) - np.pi

    # 3) Compute the signed‐hour label for each angle:
    raw_values = signed_angles * (24.0 / (2 * np.pi))
    signed_hours = np.round(raw_values).astype(int)
    signed_hours[signed_hours == -12] = 12

    # 4) Sort by signed_angles so ticks go in increasing order from −π to +π:
    idx = np.argsort(signed_angles)
    sorted_signed_angles = signed_angles[idx]
    sorted_labels = signed_hours[idx].astype(str)

    # 5) Place those 24 ticks at the “signed” angles, labeling them by signed_hours:
    ax.set_xticks(sorted_signed_angles)
    ax.set_xticklabels(sorted_labels, fontsize=xtick_fontsize)  # Added fontsize

    # 6) Add title & “inner ring”:
    #    (Scaled title fontsize based on xtick_fontsize)
    ax.set_title(title, fontsize=xtick_fontsize * 1.2)
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

    # 8) Apply visibility settings
    if not show_rlabels:
        ax.set_yticklabels([])

    # Apply grid visibility separately
    ax.yaxis.grid(show_radial_grid)  # Radial lines
    ax.xaxis.grid(show_angular_grid)  # Angular lines

    return ax


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


def plot_count_histo(x, normalize=False, **kwargs):
    """
    Plots a histogram of count data using unique values as bins.
    Parameters
    ----------
    x : array-like
        Input data to be histogrammed. Should be a 1D array of count values.
    normalize : bool, optional
        If True, the histogram is normalized to form a probability distribution (default is False).
    **kwargs
        Additional keyword arguments passed to `matplotlib.pyplot.bar`.
    Returns
    -------
    None
        This function creates a bar plot and does not return any value.
    Examples
    --------
    >>> plot_count_histo([1, 2, 2, 3, 3, 3])
    >>> plot_count_histo([1, 1, 2, 2, 2, 3], normalize=True, color='red')
    """

    bin_x, bin_y = np.unique(x, return_counts=True)
    if normalize:
        bin_y = bin_y / bin_y.sum()
    plt.bar(bin_x, bin_y, width=0.9, align="center", **kwargs)


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
