import matplotlib.transforms as transforms
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib import pyplot as plt
from scritmo.plot import panels_constants as constants
import math
from functools import reduce

import subprocess
import os


def labels_panels_mosaic(axs, unbold=False):
    for i, (key, ax) in enumerate(sorted(axs.items())):
        # You can customize the letter list if keys aren't 'A', 'B', 'C'
        label = key  # Or string.ascii_uppercase[i]
        if unbold:
            label = label.lower()
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
    fig, filename, repo_path="/home/maxine/Documents/andrea/paper-scritmo/"
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


def replace_mosaic_cell_with_grid(fig, axs, label, nrows, ncols):
    """
    Replace axs[label] with a grid of (nrows x ncols) subplots.
    Returns the updated axs dict where axs[label] is now a list of axes.
    """
    # Extract subplotspec from the original axes
    original_ax = axs[label]
    sspec = original_ax.get_subplotspec()
    fig = original_ax.figure
    original_ax.remove()

    # Create new inner grid
    inner = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=sspec)

    # Create new axes and store them in a list
    subaxes = []
    for r in range(nrows):
        for c in range(ncols):
            ax = fig.add_subplot(inner[r, c])
            subaxes.append(ax)

    # Replace entry in axs dict
    axs[label] = subaxes

    return axs


import matplotlib


def save_files(
    fig, figure_name, path="/home/maxine/Documents/andrea/context_repo/plot/FIGURES/"
):
    matplotlib.rcParams["svg.fonttype"] = "none"
    # optionally be explicit about font:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

    fig.savefig(path + figure_name + ".svg", format="svg")
    # fig.savefig(path + figure_name + ".png", format="png", dpi=300)
