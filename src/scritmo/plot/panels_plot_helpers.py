import matplotlib.transforms as transforms
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib import pyplot as plt
from scritmo.plot import panels_constants as constants
import math
from functools import reduce

import subprocess
import os


def labels_panels_mosaic(axs):
    for i, (key, ax) in enumerate(sorted(axs.items())):
        # You can customize the letter list if keys aren't 'A', 'B', 'C'
        label = key  # Or string.ascii_uppercase[i]
        ax.text(
            -0.05,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            va="bottom",
            ha="right",
        )


def update_paper_figure(
    fig, filename, repo_path="/home/maxine/Documents/andrea/scRitmo-paper/"
):
    repo_path = os.path.expanduser(repo_path)

    # Use environment variable to prevent Git from asking for input
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"

    try:
        # 1. Pull changes
        subprocess.run(
            ["git", "-C", repo_path, "pull", "--rebase"],
            check=True,
            env=env,
            capture_output=True,
        )

        # 2. Save figure (Metric: 15cm width)
        filepath = repo_path + "/figures/" + filename
        fig.savefig(filepath, format="pdf", bbox_inches="tight")

        # 3. Add and Commit
        subprocess.run(
            ["git", "-C", repo_path, "add", "figures/" + filename], check=True, env=env
        )
        # Only commit if something changed
        status = subprocess.run(
            ["git", "-C", repo_path, "status", "--porcelain"],
            capture_output=True,
            text=True,
            env=env,
        ).stdout
        if status:
            subprocess.run(
                ["git", "-C", repo_path, "commit", "-m", f"update {filename}"],
                check=True,
                env=env,
            )
            subprocess.run(["git", "-C", repo_path, "push"], check=True, env=env)
            print(f"✅ Figure '{filename}' pushed to Overleaf repo.")
        else:
            print("ℹ️ No changes to the figure. Nothing to push.")

    except subprocess.CalledProcessError as e:
        print(f"❌ Git error: {e.stderr.decode() if e.stderr else e}")


# def label_panels_mosaic(
#     fig, axes, mosaic, xloc=None, yloc=None, size=constants.BIGGER_SIZE, to_skip=[]
# ):
#     """
#     Labels the panels in a mosaic plot.

#     Parameters:
#         - fig: The figure object.
#         - axes: A dictionary of axes objects representing the panels.
#         - xloc: The x-coordinate for the label position (default: 0).
#         - yloc: The y-coordinate for the label position (default: 1.0).
#         - size: The font size of the labels (default: constants.BIGGER_SIZE).
#     """
#     if xloc is not None:
#         assert len(mosaic) == len(xloc), "One xloc value per row required"
#     else:
#         xloc = [0] * len(mosaic)
#     if yloc is not None:
#         assert len(mosaic) == len(yloc), "One yloc value per row required"
#     else:
#         yloc = [1.0] * len(mosaic)

#     for i, key in enumerate(mosaic):
#         mosaic_row = mosaic[i]
#         for key in mosaic_row:
#             if key in to_skip:
#                 continue
#             # label physical distance to the left and up:
#             ax = axes[key]
#             if isinstance(ax, list):
#                 ax = ax[0]
#             trans = transforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
#             ax.text(
#                 xloc[i],
#                 yloc[i],
#                 key,
#                 transform=ax.transAxes + trans,
#                 fontsize=size,
#                 va="bottom",
#             )


# def make_blank_panel(ax):
#     """
#     Makes a panel blank by turning off the axis and setting aspect ratio to 'auto'.

#     Parameters:
#         - ax: The axis object representing the panel.

#     Returns:
#         The modified axis object.
#     """
#     ax.axis("off")
#     ax.set_aspect("auto")
#     return ax


# def save_figure(
#     figure,
#     axs,
#     fig_name,
#     folder=constants.PLOT_FOLDER,
#     uncompressed_rows=[],
#     mosaic=None,
#     wpad=0.05,
# ):
#     """
#     Saves the figure as an SVG file.

#     Parameters:
#         - figure: The figure object to be saved.
#         - fig_name: The figure name
#         - folder: The folder path to save the figure (default: constants.PLOT_FOLDER).
#     """
#     figure.tight_layout()
#     for ax in axs.values():
#         if isinstance(ax, list):
#             compress_axes_horizontal(ax)
#     if len(uncompressed_rows):  # in case label outside plot, compress all other rows
#         assert mosaic is not None, "mosaic is required to compress rows"
#         compressed_rows = [i for i in range(len(mosaic)) if i not in uncompressed_rows]
#         compress_axes_horizontal_rows(mosaic, axs, compressed_rows, wpad)
#     figure.savefig(folder + str(fig_name) + ".svg", dpi=figure.dpi)
#     print(f"Figure saved to: {folder + str(fig_name) + '.svg'}")
#     return figure


# def compress_axes_horizontal_rows(mosaic, axs, rows, wpad=0.05):
#     """
#     Reduce horizontal spacing between axes in specified rows of a mosaic layout.

#     Parameters
#     ----------
#     mosaic : list of lists
#         The mosaic layout where each inner list represents a row of plot identifiers.
#     axs : dict
#         A dictionary of axes objects corresponding to the mosaic layout.
#     rows : list of int
#         The indices of the rows in the mosaic to compress horizontally.
#     wpad : float
#         Horizontal padding between axes (fraction of figure width).
#     """
#     for row_idx in rows:
#         row_keys = mosaic[row_idx]
#         ax_list = []
#         for key in row_keys:
#             ax = axs[key]
#             if isinstance(ax, list):
#                 ax_list.extend(ax)
#             else:
#                 ax_list.append(ax)
#         compress_axes_horizontal(ax_list, wpad=wpad)


# def prepare_mosaic_layout(simple_mosaic):
#     """
#     Calculates a scaled mosaic layout and height ratios for square plots.

#     Given a simple layout like [['A', 'B'], ['C', 'D', 'E']], this function
#     finds the least common multiple (LCM) of the row lengths (LCM of 2 and 3 is 6).
#     It then scales the mosaic layout to this LCM.

#     It also calculates the `height_ratios` required by subplot_mosaic
#     to make each individual plot (A, B, C...) approximately square.

#     Args:
#         simple_mosaic: A list of lists, where each inner list represents a
#                        row of plot identifiers.
#                        Example: [['A', 'B'], ['C', 'D', 'E']]

#     Returns:
#         A tuple (scaled_mosaic, height_ratios):

#         - scaled_mosaic: A new list of lists, scaled by the LCM.
#                          Example: [['A', 'A', 'A', 'B', 'B', 'B'],
#                                    ['C', 'C', 'D', 'D', 'E', 'E']]

#         - height_ratios: A list of relative row heights.
#                          Example: [3, 2]
#     """

#     # --- Helper function to find LCM of two numbers ---
#     def _lcm(a, b):
#         if a == 0 or b == 0:
#             return 0
#         return abs(a * b) // math.gcd(a, b)

#     # 1. Get original row lengths
#     row_lengths = [len(row) for row in simple_mosaic]

#     if not row_lengths:
#         return [], []  # Handle empty input

#     # 2. Calculate the LCM of all row lengths
#     # This will be the new total width (in columns) of all rows
#     total_lcm_width = reduce(_lcm, row_lengths)

#     # 3. Calculate height ratios
#     # To make a plot square, its height must be proportional to its width.
#     # The width of a single plot in a row is (total_lcm_width / num_plots_in_row).
#     # We use this value as the height ratio for the whole row.
#     height_ratios = [total_lcm_width // length for length in row_lengths]

#     # 4. Build the new scaled mosaic
#     scaled_mosaic = []
#     for row, length in zip(simple_mosaic, row_lengths):
#         new_row = []

#         # Calculate how many times to repeat each item in this row
#         repeat_factor = total_lcm_width // length

#         for plot_id in row:
#             # Add the plot_id 'repeat_factor' times
#             new_row.extend([plot_id] * repeat_factor)

#         scaled_mosaic.append(new_row)

#     return scaled_mosaic, height_ratios


# def create_pannels(
#     mosaic=[["A", "B", "C"], ["D", "E", "F"], ["G", "H", "I"]],
#     height_ratios=None,
#     width_ratios=None,
#     h_pad=20,  # vertical padding in points (1/72 inch)
#     w_pad=0,  # horizontal padding in points (1/72 inch)
# ):
#     """
#     Plots the whole figure.
#     """
#     mosaic, height_ratios_scaling = prepare_mosaic_layout(mosaic)
#     if height_ratios is None:
#         height_ratios = height_ratios_scaling
#     else:
#         height_ratios = [
#             hr * hrs for hr, hrs in zip(height_ratios_scaling, height_ratios)
#         ]
#     total_width = constants.FIG_WIDTH
#     # ensure squares by default
#     # Account for vertical padding between subplots
#     num_rows = len(height_ratios_scaling)
#     padding_height = h_pad * (num_rows - 1) / 72  # Convert points to inches
#     print("height_ratios_scaling:", height_ratios_scaling)
#     total_length = (
#         sum(height_ratios_scaling) * (total_width / len(mosaic[0])) + padding_height
#     )

#     # Create the figure and axes objects using the subplot_mosaic function
#     fig, ax_dict = plt.subplot_mosaic(
#         mosaic,  # Specify the layout of subplots using the mosaic parameter
#         figsize=(total_width, total_length),  # Set the size of the figure in inches
#         dpi=constants.DPI,  # Set the resolution of the figure in dots per inch
#         # layout="constrained",  # Enable constrained layout for automatic adjustment
#         gridspec_kw={
#             "height_ratios": height_ratios,
#             "width_ratios": width_ratios,
#         },
#     )  # Set the relative widths of the columns

#     # Set padding for constrained layout
#     fig.set_constrained_layout_pads(h_pad=h_pad / 72, w_pad=w_pad / 72)

#     return fig, ax_dict


# def make_polar(axs, label):
#     """
#     Convert axs[label] to a polar plot.
#     """
#     original_ax = axs[label]
#     sspec = original_ax.get_subplotspec()
#     fig = original_ax.figure
#     original_ax.remove()

#     # Create new polar axis
#     ax_polar = fig.add_subplot(sspec, projection="polar")

#     # Replace entry in axs dict
#     axs[label] = ax_polar

#     return axs


# def replace_mosaic_cell_with_grid(fig, axs, label, nrows, ncols):
#     """
#     Replace axs[label] with a grid of (nrows x ncols) subplots.
#     Returns the updated axs dict where axs[label] is now a list of axes.
#     """
#     # Extract subplotspec from the original axes
#     original_ax = axs[label]
#     sspec = original_ax.get_subplotspec()
#     fig = original_ax.figure
#     original_ax.remove()

#     # Create new inner grid
#     inner = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=sspec)

#     # Create new axes and store them in a list
#     subaxes = []
#     for r in range(nrows):
#         for c in range(ncols):
#             ax = fig.add_subplot(inner[r, c])
#             subaxes.append(ax)

#     # Replace entry in axs dict
#     axs[label] = subaxes

#     return axs


# def compress_axes_horizontal(ax_list, wpad=0.02):
#     """
#     Reduce horizontal spacing between a list of Axes objects (side-by-side).

#     Parameters
#     ----------
#     ax_list : list of matplotlib.axes.Axes
#         The axes to compress horizontally. Assumes they are aligned side-by-side.
#     wpad : float
#         Horizontal padding between axes (fraction of figure width).
#     """

#     # Get leftmost x0 and rightmost x1 in figure coords
#     left = ax_list[0].get_position().x0
#     right = ax_list[-1].get_position().x1
#     total_width = right - left

#     n = len(ax_list)
#     total_gap = wpad * (n - 1)
#     width_each = (total_width - total_gap) / n

#     # Reassign positions
#     for i, ax in enumerate(ax_list):
#         pos = ax.get_position()
#         new_x0 = left + i * (width_each + wpad)
#         new_pos = [new_x0, pos.y0, width_each, pos.height]
#         ax.set_position(new_pos)
