import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns


def rad_to_hours(x, pos=None):
    return f"{(x / (2 * np.pi) * 24):.0f}h"


def plot_bias_phase(res, lines_h=[9.5, 22], ax=None):
    """
    Plot the bias mean(phase inferred - true phase) for each phase and number of circadian counts
    --------------
    res: result of the get_simulation_results function
    """
    # Data
    phi_true = res["phi_sim"]
    phi_pred = res["res_trained"]["post_mean_c"]
    counts = res["circadian_counts"].astype(int)

    # Circular displacement
    def circular_diff(pred, true):
        return np.angle(np.exp(1j * (pred - true)))

    displacement = circular_diff(phi_pred, phi_true)

    # Binning
    n_phi_bins = 24
    phi_bins = np.linspace(0, 2 * np.pi, n_phi_bins + 1)
    unique_counts = np.unique(counts)
    count_bins = (
        np.arange(unique_counts.min(), unique_counts.max() + 2) - 0.5
    )  # bin edges for discrete counts

    # Compute 2D binned statistics
    stat, x_edges, y_edges, binnumber = binned_statistic_2d(
        phi_true, counts, displacement, statistic="mean", bins=[phi_bins, count_bins]
    )

    # Also compute bin counts for alpha
    bin_counts, _, _, _ = binned_statistic_2d(
        phi_true, counts, None, statistic="count", bins=[phi_bins, count_bins]
    )

    # Normalize bin_counts to [0, 1] for alpha
    # alpha_mask = (bin_counts - np.min(bin_counts,axis=0) )/ (np.max(bin_counts,axis=0) - np.min(bin_counts,axis=0)) * 0 +1
    normed_stat = (stat + np.pi) / (2 * np.pi)  # Normalize to [0,1] for colormap
    stat_masked = np.ma.masked_invalid(stat)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    fig = ax.get_figure()

    norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
    cmap = plt.get_cmap("twilight")

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        stat_masked.T,
        cmap=cmap,
        norm=norm,
        shading="auto",
        # alpha=alpha_mask.T
    )

    # Customize x-axis ticks and labels to show hours instead of radians

    labels_to_hours(ax, modify_x=True)

    ax.set_xlabel("True Phase")
    ax.set_ylabel("Circadian Counts")
    ax.set_title(
        "Phase Bias in Function of \nTrue Phase and Number of Circadian Counts",
        fontsize=8,
    )

    # Colorbar with ticks converted to hours
    cbar = fig.colorbar(mesh, ax=ax)

    # Set colorbar ticks at -pi, -pi/2, 0, pi/2, pi (same as x ticks but shifted by -pi to pi)
    cbar_ticks = np.linspace(-np.pi, np.pi, 5)

    def cbar_tick_formatter(x, pos=None, negative=False):
        # Convert from [-pi, pi] to [0, 24] hours scale
        hours = (x + np.pi) / (2 * np.pi) * 24
        if negative:
            hours -= 12
        return f"{hours:.0f}h"

    for h in lines_h:
        plt.plot(
            (h / 12 * np.pi, h / 12 * np.pi),
            (0, 23),
            linestyle="dashed",
            c="tan",
        )

    cbar.set_ticks(cbar_ticks)
    cbar.ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: cbar_tick_formatter(x, negative=True))
    )
    cbar.set_label("Mean Phase Displacement [h]")

    return ax


def plot_fourier_coefficients_change(phi, amp, phi_original, amp_original):
    # Plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    # Set 0h (midnight) at the top, and make the clock go clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Define hour labels and corresponding angles (0 to 2π)
    hour_labels = [str(h) for h in range(0, 24, 2)]  # Every 2 hours
    hour_angles = [
        2 * np.pi * h / 24 for h in range(0, 24, 2)
    ]  # Convert hours to radians

    ax.set_xticks(hour_angles)
    ax.set_xticklabels(hour_labels)

    ax.scatter(phi, amp, s=50)
    ax.scatter(phi_original, amp_original, marker="x", s=50)
    for i in range(len(phi)):
        ax.annotate(
            "",
            xy=(phi[i], amp[i]),  # Arrow points **to** this location
            xytext=(phi_original[i], amp_original[i]),  # Arrow starts **from** here
            arrowprops=dict(arrowstyle="-", color="gray", lw=1.5),
        )
    ax.set_title("Fourier coefficients")


def plot_clock_genes(
    dfs,
    labels,
    label_column="gene",
    title="Core Clock Genes",
    add_gene_names_last=True,
    plotting_order=None,
):
    """
    Plots clock genes on a polar plot.

    Parameters:
    - dfs: list of pandas DataFrames
    - labels: list of labels corresponding to each DataFrame
    - label_column: column to use for gene labeling
    - title: plot title
    - add_gene_names_last: add the gene names for the last df
    """
    if isinstance(dfs, dict):
        dfs = list(dfs.values())

    for df in dfs:
        if "a_1" in df.columns:
            df.rename({"a_1": "a_g", "b_1": "b_g", "a_0": "m_g"}, axis=1, inplace=True)

    def compute_metrics(dataframe, source_label):
        phases, amplitudes_log2, mean_expr, gene_names, source_labels = (
            [],
            [],
            [],
            [],
            [],
        )
        for i, row in dataframe.iterrows():
            a, b, m = row["a_g"], row["b_g"], row["m_g"]
            amplitude = np.sqrt(a**2 + b**2)
            log2_amp = np.log2(np.exp(amplitude))  # convert from log to log2
            phase = np.arctan2(b, a) % (2 * np.pi)
            mean = np.exp(m)

            phases.append(phase)
            amplitudes_log2.append(log2_amp)
            mean_expr.append(mean)
            if label_column:
                gene_names.append(row[label_column])
            else:
                gene_names.append(str(i))
            source_labels.append(source_label)

        return phases, amplitudes_log2, mean_expr, gene_names, source_labels

    # Store data per label so we can control plot and legend order
    data_by_label = {}
    for df, label in zip(dfs, labels):
        data_by_label[label] = compute_metrics(df, label)

    # Concatenate data in **reverse order** so the first label plots on top
    plot_order = plotting_order or list(reversed(labels))
    all_phases, all_amplitudes, all_expr, all_genes, all_labels = [], [], [], [], []
    for label in plot_order:
        phases, amps, expr, genes, lbls = data_by_label[label]
        all_phases += phases
        all_amplitudes += amps
        all_expr += expr
        all_genes += genes
        all_labels += lbls

    all_phases = np.array(all_phases)
    all_amplitudes = np.array(all_amplitudes)
    all_expr = np.array(all_expr)
    sizes = 400 * (all_expr / all_expr.max()) + 50

    # Create color map in user-specified order
    colormap = plt.get_cmap("tab10")
    color_map = {label: colormap(i) for i, label in enumerate(labels)}
    colors = [color_map[label] for label in all_labels]

    # Plotting
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    ax.scatter(all_phases, all_amplitudes, s=sizes, c=colors, alpha=0.8)
    if add_gene_names_last:
        # Label genes from the last label
        label_for_annotation = labels[-1]
        for phase, amp, gene, label in zip(
            all_phases, all_amplitudes, all_genes, all_labels
        ):
            if label == label_for_annotation:
                ax.text(phase, amp, gene, fontsize=16, ha="center", va="bottom")

    # Polar plot config
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    hour_labels = [str(h) for h in range(0, 24, 2)]
    hour_angles = [2 * np.pi * h / 24 for h in range(0, 24, 2)]
    ax.set_xticks(hour_angles)
    ax.set_xticklabels(hour_labels)

    # Legend
    size_levels = [0.01, 0.1, 0.3, 1.0]
    size_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            color="gray",
            markersize=np.sqrt(400 * s + 50),
            label=f"{s * all_expr.max():.2E}",
        )
        for s in size_levels
    ]
    color_handles = [
        Line2D(
            [], [], marker="o", color=color_map[label], linestyle="None", label=label
        )
        for label in labels
    ]

    plt.legend(
        handles=color_handles + size_handles,
        title="Gene Source & Mean Expression",
        loc="upper right",
        bbox_to_anchor=(1.1, 1.1),
    )

    plt.title(title)
    plt.tight_layout()
    plt.show()


def labels_to_hours(
    ax, modify_x=False, modify_y=False, min=0, max=2 * np.pi, n_points=5, no_label=False
):
    def rad_to_hours(x, pos=None):
        return f"{(x / (2 * np.pi) * 24):.0f}h"

    if modify_x:
        xticks = np.linspace(min, max, n_points)
        ax.set_xticks(xticks)
        if no_label:
            ax.set_xticklabels(["" for tick in xticks])
        else:
            ax.set_xticklabels([rad_to_hours(tick) for tick in xticks])
    if modify_y:
        yticks = np.linspace(min, max, n_points)
        ax.set_yticks(yticks)
        if no_label:
            ax.set_yticklabels(["" for tick in yticks])
        else:
            ax.set_yticklabels([rad_to_hours(tick) for tick in yticks])


def plot_bias_context(df_plot_no_context, df_plot_context, df_fit_indep):
    fig, axs = plt.subplots(figsize=(12, 6), nrows=1, ncols=3)

    sns.lineplot(
        df_plot_no_context,
        y="circular_deviation",
        x="phi_sim_binned",
        hue="population",
        errorbar=None,
        # palette=sns.color_palette("Greys", 5)[1:],
        ax=axs[2],
        legend=None,
    )
    sns.lineplot(
        df_plot_context,
        y="circular_deviation",
        x="phi_sim_binned",
        hue="population",
        errorbar=None,
        ax=axs[0],
    )
    sns.lineplot(
        df_fit_indep,
        y="circular_deviation",
        x="phi_sim_binned",
        hue="population",
        errorbar=None,
        # linestyle="--",
        # palette=sns.color_palette("Greys", 5)[1:],
        ax=axs[1],
        legend=None,
    )
    ax = axs[0]
    sns.move_legend(ax, "lower right")
    ax.set_ylim((-np.pi, np.pi))
    ax.set_ylabel("Bias")
    ax.set_xlabel("Phase Binned")
    labels_to_hours(ax, modify_x=True)
    labels_to_hours(ax, modify_y=True, min=-np.pi, max=np.pi)
    for ax in axs[1:]:
        ax.set_ylim((-np.pi, np.pi))
        ax.set_ylabel("")
        ax.set_xlabel("Phase Binned")
        labels_to_hours(ax, modify_x=True)
        labels_to_hours(ax, modify_y=True, min=-np.pi, max=np.pi, no_label=True)

    axs[0].set_title("With Context")
    axs[1].set_title("Indep. Fit")
    axs[2].set_title("Without Context")
