import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scritmo.linear_regression import polar_genes_pandas
from scritmo.basics import w, rh, ind2
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from adjustText import adjust_text
from scritmo.beta import Beta

import seaborn as sns
from scritmo.linear_regression import fit_periodic_spline
from scipy.stats import vonmises


def amp_inflation_plot(
    params_g,
    par_refit,
    mask_inf,
    full_name_titles="",
    annotate_names=False,
    show_marginals=True,
    max_val=4.0,
    add_kde=True,
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
    if add_kde:
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
