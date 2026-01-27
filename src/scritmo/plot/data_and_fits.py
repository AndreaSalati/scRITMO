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
from .utils import polar_plot


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


# def plot_circadian_data(
#     adata,
#     phis,
#     g,
#     ax=None,
#     layer="spliced",
#     n_bins=None,
#     alpha=0.7,
#     s=1,
#     log_bin_y=False,
#     jitter=0.0,
#     c=None,
# ):
#     """
#     Plot expression values of a gene over circadian phase.

#     Parameters
#     ----------
#     adata : AnnData
#         Annotated data matrix.
#     phis : array-like
#         Array of circadian phases (in radians).
#     g : str
#         Gene name to plot.
#     ax : matplotlib.axes.Axes or None
#         Axis to plot on. If None, a new figure and axis are created.
#     layer : str
#         Layer in `adata.layers` to extract expression from.
#     n_bins : int or None
#         Number of bins for aggregating expression. If None, raw points are plotted.
#     alpha : float
#         Transparency of scatter points.
#     s : float
#         Size of scatter points.
#     log_bin_y : bool
#         If True, log-transform the binned expression values.
#     jitter : float
#         Standard deviation of Gaussian noise added to phase values (in hours) for visual separation.
#     """
#     if ax is None:
#         fig, ax = plt.subplots()

#     phi_x = np.linspace(0, 2 * np.pi, 100)

#     # Get expression values
#     if layer is None:
#         expression = adata[:, g].X
#     else:
#         expression = adata[:, g].layers[layer]

#     try:
#         expression = expression.toarray().squeeze()
#     except AttributeError:
#         expression = expression.squeeze()

#     if n_bins is not None:
#         # Bin the data
#         bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
#         bin_indices = np.digitize(phis, bin_edges) - 1
#         # Wrap around for circular data
#         bin_indices[bin_indices == n_bins] = 0

#         # Calculate the center of each bin
#         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

#         # Calculate mean expression for each bin
#         binned_expr = np.zeros(n_bins)
#         for i in range(n_bins):
#             # Check if there are data points in this bin
#             if np.sum(bin_indices == i) > 0:
#                 binned_expr[i] = np.mean(expression[bin_indices == i])
#             else:
#                 binned_expr[i] = np.nan  # No data in this bin

#         # Plot binned data
#         valid_bins = ~np.isnan(binned_expr)
#         if log_bin_y:
#             binned_expr[valid_bins] = np.log(binned_expr[valid_bins])
#         ax.scatter(
#             bin_centers[valid_bins] * rh,
#             binned_expr[valid_bins],
#             s=s,
#             alpha=alpha,
#             label="binned data",
#             marker="o",
#         )
#     else:
#         x = (phis * rh) + np.random.normal(0, jitter, size=phis.shape)
#         # Plot original data points
#         ax.scatter(
#             x,
#             expression,
#             s=s,
#             label="data",
#             alpha=alpha,
#             c=c,
#         )

#     # Plot details
#     plt.title(f"{g}")
#     ax.set_xlabel("Circadian phase")
#     ax.set_ylabel("Normalized expression")
#     ax.legend()

#     return ax

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
    c=None,
    boxplot=False,  # New parameter
    rasterized=True,  # New parameter for rasterization
    remove_outliers=False,  # New parameter to remove outliers
    label="data",
):
    if ax is None:
        fig, ax = plt.subplots()

    # Get expression values
    expression = adata[:, g].layers[layer] if layer else adata[:, g].X
    try:
        expression = expression.toarray().squeeze()
    except AttributeError:
        expression = expression.squeeze()

    # Factor to convert radians to hours (assuming 24h cycle)
    rh = 24 / (2 * np.pi)

    if boxplot:
        # 1. Group expression by unique phase values
        df = pd.DataFrame({"phi": phis * rh, "expr": expression})
        groups = df.groupby("phi")["expr"].apply(list)

        # 2. Extract positions (x-values) and data (y-values)
        positions = groups.index.values
        data = groups.values

        # 3. Plot using the actual numerical values as positions
        ax.boxplot(data, positions=positions, widths=0.5, manage_ticks=False)

        # 4. Clean up the ticks: Show only a few representative hours (e.g., every 4h)
        ax.set_xticks(np.arange(0, 25, 4))
        ax.set_xlim(-0.5, 24.5)  # Ensure the full cycle is visible

    elif n_bins is not None:
        bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
        bin_indices = np.digitize(phis, bin_edges) - 1
        bin_indices[bin_indices == n_bins] = 0
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        binned_expr = np.array(
            [
                (
                    np.mean(expression[bin_indices == i])
                    if np.any(bin_indices == i)
                    else np.nan
                )
                for i in range(n_bins)
            ]
        )

        valid_bins = ~np.isnan(binned_expr)
        y_vals = (
            np.log(binned_expr[valid_bins]) if log_bin_y else binned_expr[valid_bins]
        )
        x_vals = bin_centers[valid_bins] * rh

        # ax.plot(
        #     bin_centers[valid_bins] * rh,
        #     y_vals,
        #     s=s,
        #     alpha=alpha,
        #     label=label,
        # )
        ax.plot(
            x_vals,
            y_vals,
            marker="o",
            markersize=np.sqrt(s),
            alpha=alpha,
            label=label,
            c=c,
        )
    else:
        if remove_outliers:
            # remove 1% of large outliers for better visualization
            threshold = np.percentile(expression, 99)
            mask = expression <= threshold
            expression = expression[mask]
            phis = phis[mask]

        x = (phis * rh) + np.random.normal(0, jitter, size=phis.shape)
        ax.scatter(
            x, expression, s=s, alpha=alpha, c=c, label=label, rasterized=rasterized
        )

    ax.set_title(f"{g}")
    ax.set_xlabel("Circadian phase (hours)")
    ax.set_ylabel("Normalized expression")
    if not boxplot:
        ax.legend()

    return ax


def plot_circadian_data_and_fit(
    adata,
    phis,
    g,
    params_g,
    layer,
    ax=None,
    n_bins=None,
    alpha=0.7,
    line_color="red",
    log_bin_y=False,
    exp=True,
    columns_names=["amp", "phase", "a_0"],
    s=10,
    jitter=0.0,
    c=None,
    boxplot=False,  # New parameter
    rasterized=True,  # New parameter for rasterization
    remove_outliers=False,  # New parameter to remove outliers
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
        c=c,
        boxplot=boxplot,
        rasterized=rasterized,
        remove_outliers=remove_outliers,
    )

    # Now add the GLM fit
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
        label="Fit",
    )

    # Update legend since we've added a new line
    ax.legend()

    return ax


def get_binned_stats(x, y, n_bins=10, log_bin_y=False):
    """
    Bins y values based on x-axis intervals and returns bin centers
    and mean y values per bin.
    """
    # Define edges and calculate centers
    bin_edges = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Assign each x to a bin index
    # -1 to make it 0-indexed; clip to ensure max value stays in last bin
    bin_indices = np.digitize(x, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Calculate mean per bin
    binned_y = np.array(
        [
            y[bin_indices == i].mean() if np.any(bin_indices == i) else np.nan
            for i in range(n_bins)
        ]
    )

    # Filter out empty bins
    valid_mask = ~np.isnan(binned_y)
    final_x = bin_centers[valid_mask]
    final_y = binned_y[valid_mask]

    if log_bin_y:
        final_y = np.log(final_y)

    return final_x, final_y
