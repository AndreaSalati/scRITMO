import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .linear_regression import harmonic_function_exp, harmonic_function
from scritmo.linear_regression import polar_genes_pandas
from .basics import w, rh


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


def polar_plot_params_g(df, genes_to_plot=None, title="", amp_lim=0.5, cartesian=False):
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
        ax.scatter(phase, amp)
        # annotate
        if amp < amp_lim:
            continue
        ax.annotate(gene, (phase, amp))


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
):
    if ax is None:
        fig, ax = plt.subplots()

    w = 2 * np.pi / 24
    rh = w**-1
    phi_x = np.linspace(0, 2 * np.pi, 100)

    # Get expression values
    expression = adata[:, g].layers[layer].toarray().squeeze()

    if n_bins is not None:
        # Bin the data
        bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
        bin_indices = np.digitize(phis, bin_edges) - 1
        bin_indices[bin_indices == n_bins] = 0  # Wrap around for circular data

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
        ax.scatter(
            bin_centers[valid_bins] * rh,
            binned_expr[valid_bins],
            s=20,
            alpha=alpha,
            label="binned data",
            marker="o",
        )
    else:
        # Plot original data points
        ax.scatter(
            phis * rh,
            expression,
            s=5,
            label="data",
            c=adata.obs.ZTmod,
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
):
    """
    Creates a plot of circadian expression data and GLM fit, returning an axis object.
    If n_bins is provided, data will be binned by phase before plotting.

    Parameters:
        adata (AnnData): AnnData object containing gene data.
        phis (array): Phase values for scatter plot.
        g (str): Gene name or index.
        counts (array): Normalization counts for spliced layer data.
        params_g (DataFrame): Parameter dataframe for GLM fit.
        ax (matplotlib.axes._axes.Axes, optional): Axis to plot on. If None, creates a new one.
        layer (str): Layer of AnnData object to use. Default is "spliced".
        n_bins (int, optional): Number of bins to use for data. If None, no binning is performed.

    Returns:
        matplotlib.axes._axes.Axes: Axis object with the plot.
    """
    # First plot the data points using plot_circadian_data
    ax = plot_circadian_data(
        adata, phis, g, ax=ax, layer=layer, n_bins=n_bins, alpha=alpha
    )

    # Now add the GLM fit
    w = 2 * np.pi / 24
    rh = w**-1
    phi_x = np.linspace(0, 2 * np.pi, 100)

    ax.plot(
        phi_x * rh,
        harmonic_function_exp(phi_x * rh, params_g.loc[g][:3], omega=w),
        c="red",
        label="GLM fit",
    )

    # Update legend since we've added a new line
    ax.legend()

    return ax
