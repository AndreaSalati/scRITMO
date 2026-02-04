import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns


def scatter_with_hist_marginals(
    x,
    y,
    labels,
    bins_x=20,
    bins_y=20,
    scatter_alpha=0.7,
    hist_alpha=0.5,
    palette=None,
    figsize=(6, 6),
    edgecolor="w",
    xlabel=None,
    ylabel=None,
    title=None,
):
    """
    Draw a scatter plot of (x, y) colored by 'labels', with separate
    histogram marginals (one per group) on the top and right. You can
    control:
      - bins_x:    number of bins (or bin‐edges) for the x‐marginal
      - bins_y:    number of bins (or bin‐edges) for the y‐marginal
      - scatter_alpha: transparency of the scatter points
      - hist_alpha:    transparency of the marginal histograms

    Parameters
    ----------
    x : array‐like, shape (n,)
        X‐coordinates for all points.
    y : array‐like, shape (n,)
        Y‐coordinates for all points.
    labels : array‐like, shape (n,)
        Categorical labels (e.g. strings or ints) for grouping.
    bins_x : int or sequence, default=20
        Number of bins (or bin‐edges) for the x‐marginal histograms.
    bins_y : int or sequence, default=20
        Number of bins (or bin‐edges) for the y‐marginal histograms.
    scatter_alpha : float, default=0.7
        Alpha (opacity) for the scatter‐plot points.
    hist_alpha : float, default=0.5
        Alpha (opacity) for the marginal histograms.
    palette : dict or list or None, default=None
        If dict: maps each unique label to a color, e.g. {'A':'C0','B':'C1',…}.
        If list: must match number of unique labels in length; assigned in sorted order of labels.
        If None: seaborn’s “tab10” palette is used.
    figsize : tuple, default=(6,6)
        Size (in inches) of the JointGrid.
    edgecolor : color, default='w'
        Edgecolor for scatter markers (helps separate overlapping points).
    xlabel, ylabel, title : str or None
        Axis labels for the central plot (and overall title).

    Returns
    -------
    g : seaborn.axisgrid.JointGrid
        The JointGrid object, so you can tweak afterwards if needed.
    """

    # 1. Build a DataFrame for convenience
    df = pd.DataFrame({"x": x, "y": y, "label": labels})
    unique_labels = np.unique(labels)

    # 2. Choose or validate a palette
    if palette is None:
        base_colors = sns.color_palette("tab10", n_colors=len(unique_labels))
        palette = {lab: base_colors[i] for i, lab in enumerate(unique_labels)}
    else:
        if isinstance(palette, (list, tuple)):
            if len(palette) != len(unique_labels):
                raise ValueError(
                    "If palette is a list, its length must match the number of unique labels."
                )
            palette = {lab: palette[i] for i, lab in enumerate(unique_labels)}
        # if it’s a dict, assume it already covers all labels

    # 3. Create a JointGrid
    g = sns.JointGrid(data=df, x="x", y="y", height=figsize[0], ratio=4)

    # 4. Scatter each group with its own color and scatter_alpha
    for lab in unique_labels:
        subset = df[df["label"] == lab]
        g.ax_joint.scatter(
            subset["x"],
            subset["y"],
            color=palette[lab],
            label=str(lab),
            alpha=scatter_alpha,
            edgecolor=edgecolor,
            linewidth=0.5,
        )

    # 5. Compute common bin edges so that all groups use identical bins
    #    for the x‐marginal and for the y‐marginal, respectively.
    if isinstance(bins_x, int):
        bins_x_edges = np.histogram_bin_edges(df["x"], bins=bins_x)
    else:
        # if user already passed explicit bin edges
        bins_x_edges = np.asarray(bins_x)

    if isinstance(bins_y, int):
        bins_y_edges = np.histogram_bin_edges(df["y"], bins=bins_y)
    else:
        bins_y_edges = np.asarray(bins_y)

    # 6. Draw histograms on the top margin (x) and right margin (y)
    #    using the same bins, one histogram per group.
    for lab in unique_labels:
        subset = df[df["label"] == lab]
        # top marginal (x‐distribution)
        g.ax_marg_x.hist(
            subset["x"],
            bins=bins_x_edges,
            color=palette[lab],
            alpha=hist_alpha,
            edgecolor="k",
            linewidth=0.5,
            label=str(lab),
        )
        # right marginal (y‐distribution; horizontal orientation)
        g.ax_marg_y.hist(
            subset["y"],
            bins=bins_y_edges,
            orientation="horizontal",
            color=palette[lab],
            alpha=hist_alpha,
            edgecolor="k",
            linewidth=0.5,
        )

    # 7. Labels, legend, and layout tweaks
    if xlabel is not None or ylabel is not None:
        g.set_axis_labels(xlabel or "x", ylabel or "y")
    else:
        g.set_axis_labels("x", "y")

    if title:
        g.fig.suptitle(title)
        g.fig.tight_layout()
        g.fig.subplots_adjust(top=0.95)

    g.ax_joint.legend(title="group")
    sns.despine(ax=g.ax_joint, trim=True)
    sns.despine(ax=g.ax_marg_x, left=True)
    sns.despine(ax=g.ax_marg_y, bottom=True)

    return g
