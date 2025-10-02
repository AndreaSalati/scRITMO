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


def get_high_contrast_colors(cmap_name, num_colors):
    """
    Generates a list of high-contrast colors by reordering a colormap.
    """
    cmap = plt.get_cmap(cmap_name)

    # Generate initial indices
    indices = np.arange(num_colors)

    # Reorder indices by interleaving the first and second half
    # e.g., for 10 colors, indices [0,1,2,3,4,5,6,7,8,9] -> [0,5,1,6,2,7,3,8,4,9]
    half = (num_colors + 1) // 2
    reordered_indices = np.empty(num_colors, dtype=int)
    reordered_indices[0::2] = indices[:half]
    reordered_indices[1::2] = indices[half:]

    # Get colors from the colormap using the reordered indices
    color_fractions = reordered_indices / num_colors
    colors = cmap(color_fractions)

    return colors


def plot_phase_polar_single_ct(
    ct,
    cell_type_list,
    time,
    phase,
    sample_name=None,
    amplitude=None,
    plot_type="histogram",
    cmap_name="twilight",
    bins=30,
    ylim=None,
    color_order="linear",
):
    """
    same thing as before, jsut returns the ax and fig
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

    if color_order == "linear":
        base_colors = cmap(np.linspace(0, 1, len(zts), endpoint=False))
    else:
        base_colors = get_high_contrast_colors(cmap_name, len(zts))
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
