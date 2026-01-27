import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import pandas as pd
import colorsys
import numpy as np
from matplotlib.lines import Line2D
from statannotations.Annotator import Annotator


def plot_annotated_comparison(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    plot_type: str = "bar",
    show_points: bool = False,
    estimator=np.mean,
    test: str = "t-test_ind",
    text_format: str = "star",
    loc: str = "outside",
    rotation: int = 45,
    ax=None,
    verbose_annoations: bool = False,
    annotate_values: bool = False,
    **plot_kwargs,
):
    """
    Creates a seaborn barplot or boxplot comparing two groups within a hue category
    and adds statistical annotations between them.
    """

    # --- 1. Validate Hue and Define Orders ---
    hue_values = sorted(data[hue].unique())
    if len(hue_values) != 2:
        raise ValueError(
            f"Hue column '{hue}' must have exactly 2 unique values, found: {hue_values}"
        )
    if hue == x:
        x = hue + " "
        data[x] = "value"

    # We explicitly define the order to ensure consistency between the plot and the annotator
    order = sorted(data[x].unique())
    hue_order = hue_values

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # --- 2. Create the Plot ---
    if plot_type == "bar":
        ax = sns.barplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            estimator=estimator,
            ax=ax,
            order=order,
            hue_order=hue_order,
            **plot_kwargs,
        )

        if annotate_values:
            for bar in ax.patches:
                height = bar.get_height()
                if np.isnan(height):
                    continue
                ax.annotate(
                    f"{height:.2f}",
                    (bar.get_x() + bar.get_width() / 2, height),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    xytext=(0, 3),
                    textcoords="offset points",
                )

    elif plot_type == "box":
        ax = sns.boxplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            order=order,
            hue_order=hue_order,
            **plot_kwargs,
        )

        if show_points:
            sns.stripplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                ax=ax,
                dodge=True,
                color="black",
                alpha=0.6,
                jitter=True,
                size=4,
                order=order,
                hue_order=hue_order,
            )

            # Clean up legend (remove duplicates from stripplot)
            handles, labels = ax.get_legend_handles_labels()
            if len(labels) > 2:
                ax.legend(handles[:2], labels[:2], title=hue)

    else:
        raise ValueError("plot_type must be 'bar' or 'box'")

    plt.setp(ax.get_xticklabels(), rotation=rotation)

    # --- 3. Define Pairs for Annotation ---
    # The pairs must be formatted as ((x_cat, hue_val1), (x_cat, hue_val2))
    pairs = [
        ((category, hue_values[0]), (category, hue_values[1])) for category in order
    ]

    # --- 4. Add Statistical Annotations ---
    # CRITICAL FIX: We must pass 'plot' to Annotator so it handles the hue nesting correctly
    sa_plot_type = "barplot" if plot_type == "bar" else "boxplot"

    annotator = Annotator(
        ax,
        pairs=pairs,
        data=data,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        plot=sa_plot_type,
    )

    annotator.configure(
        test=test, text_format=text_format, loc=loc, verbose=verbose_annoations
    )

    try:
        annotator.apply_and_annotate()
    except Exception as e:
        print(f"Error during annotation: {e}")
        raise e

    return ax


def adjust_lightness(color, amount=0.5):
    """Helper to modify color lightness (amount > 1 lightens, < 1 darkens)"""
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def update_xtick_labels(ax, data, x_col, label_col, rotation=45):
    """
    Renames x-axis tick labels using a mapping from the dataframe.
    """
    if label_col is None:
        return

    # 1. Create Lookup: Current X Value -> New Label
    # drop_duplicates ensures we have a unique mapping
    mapping = (
        data[[x_col, label_col]].drop_duplicates().set_index(x_col)[label_col].to_dict()
    )

    # 2. Get current labels and map them to new ones
    current_labels = [lbl.get_text() for lbl in ax.get_xticklabels()]
    new_labels = [mapping.get(lbl, lbl) for lbl in current_labels]

    # 3. Update the axes
    ax.set_xticklabels(new_labels, rotation=rotation, ha="center")


def plot_dual_layer(
    data,
    x,
    y,
    hue,
    color_col,
    palette,
    kind="box",
    ax=None,
    hue_order=None,
    order=None,
    light_factor=1.2,
    dark_factor=0.6,
    xticks_col=None,
    rotation=45,
    show_color_legend=True,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    # 1. Enforce Order
    if order is None:
        order = sorted(data[x].unique())
    if hue_order is None:
        hue_order = sorted(data[hue].unique())

    # 2. Draw the Base Plot
    plot_func = sns.boxplot if kind == "box" else sns.barplot

    plot_func(
        data=data, x=x, y=y, hue=hue, order=order, hue_order=hue_order, ax=ax, **kwargs
    )

    # 3. Create Lookup Maps for Coloring
    x_to_color_cat = (
        data[[x, color_col]].drop_duplicates().set_index(x)[color_col].to_dict()
    )

    # 4. Iterate and Repaint (Coloring Logic)
    for patch in ax.patches:
        # Determine X Center (Rectangle vs PathPatch)
        if hasattr(patch, "get_x"):
            x_center = patch.get_x() + patch.get_width() / 2
        else:
            extents = patch.get_path().get_extents()
            x_center = (extents.xmin + extents.xmax) / 2

        x_idx = int(round(x_center))

        if 0 <= x_idx < len(order):
            current_x_cat = order[x_idx]
            current_color_cat = x_to_color_cat.get(current_x_cat)
            base_color = palette.get(current_color_cat, "grey")

            # Left (Hue 0) vs Right (Hue 1)
            if x_center < x_idx:
                new_color = adjust_lightness(base_color, light_factor)
            else:
                new_color = adjust_lightness(base_color, dark_factor)

            patch.set_facecolor(new_color)
            if kind == "bar":
                patch.set_edgecolor("black")
                patch.set_linewidth(1)

    # 5. Handle Custom Legend
    if ax.get_legend():
        ax.get_legend().remove()

    legend_elements = [
        Line2D([0], [0], color="gray", lw=4, label=f"{hue_order[0]}"),
        Line2D([0], [0], color="black", lw=4, label=f"{hue_order[1]}"),
        # Line2D([0], [0], color="white", label=""),
    ]
    if show_color_legend:
        for cat_name, col in palette.items():
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor=col,
                    markersize=10,
                    label=cat_name,
                )
            )
    ax.legend(handles=legend_elements, loc="best")

    # 6. Rename X-Ticks (New Feature)
    if xticks_col is not None:
        # We reuse the logic from the standalone function here
        mapping = (
            data[[x, xticks_col]].drop_duplicates().set_index(x)[xticks_col].to_dict()
        )
        current_labels = [lbl.get_text() for lbl in ax.get_xticklabels()]
        new_labels = [mapping.get(lbl, lbl) for lbl in current_labels]
        ax.set_xticklabels(new_labels, ha="center")

    # remvove x label
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)

    return ax


# def plot_dual_layer(
#     data,
#     x,
#     y,
#     hue,
#     color_col,
#     palette,
#     kind="box",
#     ax=None,
#     hue_order=None,
#     order=None,
#     light_factor=1.4,
#     dark_factor=0.8,
#     **kwargs,
# ):
#     """
#     Plots a seaborn box or bar plot where:
#     - x, y: Standard axes
#     - hue: Binary variable (controls Left/Right position & Light/Dark intensity)
#     - color_col: Categorical variable (controls the Base Hue/Color)
#     """
#     if ax is None:
#         ax = plt.gca()

#     # 1. Enforce Order
#     # We must know the order of X and Hue to map the patches correctly
#     if order is None:
#         order = sorted(data[x].unique())
#     if hue_order is None:
#         hue_order = sorted(data[hue].unique())

#     # Validate hue is binary
#     if len(hue_order) != 2:
#         raise ValueError(
#             f"The 'hue' column must have exactly 2 categories. Found: {hue_order}"
#         )

#     # 2. Draw the Base Plot
#     plot_func = sns.boxplot if kind == "box" else sns.barplot

#     # We pass hue=hue to get the dodging (spacing), but we ignore the default colors
#     plot_func(
#         data=data, x=x, y=y, hue=hue, order=order, hue_order=hue_order, ax=ax, **kwargs
#     )

#     # 3. Create Lookup Maps
#     # specific_map: Maps 'ContextName' -> 'Organ'
#     # We drop duplicates to get a unique mapping for the x-axis
#     x_to_color_cat = (
#         data[[x, color_col]].drop_duplicates().set_index(x)[color_col].to_dict()
#     )

#     # 4. Iterate and Repaint
#     # Supported patch types: Rectangle (Bar/Box old) or PathPatch (Box new)
#     for patch in ax.patches:

#         # A. Get X Center (Robust to patch type)
#         if hasattr(patch, "get_x"):  # Rectangle
#             x_center = patch.get_x() + patch.get_width() / 2
#         else:  # PathPatch
#             extents = patch.get_path().get_extents()
#             x_center = (extents.xmin + extents.xmax) / 2

#         # B. Identify X-Axis Category (Integer Index)
#         x_idx = int(round(x_center))

#         # Safety check: ensure we are within bounds of the data we plotted
#         if 0 <= x_idx < len(order):
#             current_x_cat = order[x_idx]

#             # C. Retrieve Base Color
#             current_color_cat = x_to_color_cat.get(current_x_cat)
#             base_color = palette.get(current_color_cat, "grey")

#             # D. Apply Logic: Left (Hue 0) vs Right (Hue 1)
#             # If center is to the left of the integer tick -> First Hue Category
#             if x_center < x_idx:
#                 new_color = adjust_lightness(base_color, light_factor)
#             else:
#                 new_color = adjust_lightness(base_color, dark_factor)

#             patch.set_facecolor(new_color)

#             # Optional: For barplots, you might want to set edgecolor to match or black
#             if kind == "bar":
#                 patch.set_edgecolor("black")
#                 patch.set_linewidth(1)

#     # 5. Generate Custom Legend
#     # Clear existing legend (which shows wrong colors)
#     if ax.get_legend():
#         ax.get_legend().remove()

#     legend_elements = [
#         Line2D([0], [0], color="gray", lw=4, label=f"{hue_order[0]}"),
#         Line2D([0], [0], color="black", lw=4, label=f"{hue_order[1]}"),
#         # Line2D([0], [0], color="white", label=""),  # Spacer
#     ]

#     for cat_name, col in palette.items():
#         legend_elements.append(
#             Line2D(
#                 [0],
#                 [0],
#                 marker="s",
#                 color="w",
#                 markerfacecolor=col,
#                 markersize=10,
#                 label=cat_name,
#             )
#         )

#     ax.legend(handles=legend_elements, loc="best")

#     return ax
