import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import pandas as pd
import colorsys
import numpy as np
from matplotlib.lines import Line2D
from statannotations.Annotator import Annotator
from matplotlib.patches import Patch


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
    OLD! use plot_dual_layer instead.
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


def update_xtick_labels(ax, data, x_col, label_col, rotation=45, xticks_fontsize=12):
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
    ax.set_xticklabels(
        new_labels, rotation=rotation, ha="center", fontsize=xticks_fontsize
    )


def plot_dual_layer(
    data,
    x,
    y,
    hue,
    color_col=None,  # Now optional: None disables dual-layer coloring
    palette=None,  # Optional unless color_col is used
    kind="box",
    ax=None,
    hue_order=None,
    order=None,
    light_factor=1.6,
    dark_factor=1,
    xticks_col=None,
    xticks_fontsize=12,
    rotation=45,
    show_color_legend=True,
    add_pvalues=False,
    test="t-test_ind",
    text_format="star",
    loc="outside",
    verbose_annotations=False,
    annotate_values=False,
    show_points=False,
    **kwargs,
):
    """
    Universal plotting function: box/bar plots with optional dual-layer coloring
    and optional statistical annotations.

    Parameters
    ----------
    color_col : str or None
        If str, enables dual-layer coloring (base color by color_col, lightness by hue).
        If None, uses standard seaborn coloring (palette applies to hue).
    palette : dict or seaborn palette
        If color_col is set: mapping of color_col categories to colors.
        If color_col is None: passed directly to seaborn (e.g., 'Set2', dict, list).
    add_pvalues : bool
        If True, adds significance brackets between hue groups within each x category.
    Other parameters : see plot_annotated_comparison and previous plot_dual_layer
    """
    if ax is None:
        ax = plt.gca()

    # 1. Enforce Order
    if order is None:
        order = sorted(data[x].unique())
    if hue_order is None:
        hue_order = sorted(data[hue].unique())

    # Validation for statistical annotations
    if add_pvalues and len(hue_order) != 2:
        raise ValueError(
            f"When add_pvalues=True, hue column '{hue}' must have exactly 2 unique values, "
            f"found: {hue_order}"
        )

    # Handle edge case where x == hue
    original_x = x
    modified_data = data.copy()
    if hue == x:
        x = hue + "_x"
        modified_data[x] = modified_data[original_x].copy()

    # 2. Draw the Base Plot
    plot_func = sns.boxplot if kind == "box" else sns.barplot

    # When color_col is None, pass palette to seaborn normally
    # When color_col is set, we handle colors manually, so don't pass palette to seaborn
    plot_palette = None if color_col is not None else palette

    ax = plot_func(
        data=modified_data,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        ax=ax,
        palette=plot_palette,  # Only used when color_col is None
        **kwargs,
    )

    # Add strip plot for boxplots if requested
    if kind == "box" and show_points:
        sns.stripplot(
            data=modified_data,
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
            legend=False,
        )

    # Annotate values for bar plots
    if kind == "bar" and annotate_values:
        for bar in ax.patches:
            height = bar.get_height()
            if np.isnan(height):
                continue
            ax.annotate(
                f"{height:.2f}",
                (bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )

    # 3. DUAL-LAYER COLORING (Only if color_col is specified)
    if color_col is not None:
        if palette is None:
            raise ValueError("palette must be provided when color_col is specified")

        # Handle edge case: x is the same as color_col
        if color_col == original_x:
            x_to_color_cat = {cat: cat for cat in order}
        else:
            x_to_color_cat = (
                modified_data[[x, color_col]]
                .drop_duplicates()
                .set_index(x)[color_col]
                .to_dict()
            )

        # Iterate and Repaint
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

    # 4. STATISTICAL ANNOTATIONS
    if add_pvalues:
        pairs = [
            ((category, hue_order[0]), (category, hue_order[1])) for category in order
        ]

        sa_plot_type = "barplot" if kind == "bar" else "boxplot"

        annotator = Annotator(
            ax,
            pairs=pairs,
            data=modified_data,
            x=x,
            y=y,
            hue=hue,
            order=order,
            hue_order=hue_order,
            plot=sa_plot_type,
        )

        annotator.configure(
            test=test, text_format=text_format, loc=loc, verbose=verbose_annotations
        )

        try:
            annotator.apply_and_annotate()
        except Exception as e:
            print(f"Error during statistical annotation: {e}")
            raise e

    # 5. LEGEND HANDLING
    if ax.get_legend():
        ax.get_legend().remove()

    legend_elements = []

    if color_col is not None:
        # Dual-layer mode: Show hue indicators (light/dark) + color categories
        legend_elements.extend(
            [
                Line2D([0], [0], color="gray", lw=4, label=f"{hue_order[0]}"),
                Line2D([0], [0], color="black", lw=4, label=f"{hue_order[1]}"),
            ]
        )
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
    else:
        # Standard mode: Reconstruct standard hue legend
        # Try to get colors from the plot or use default
        try:
            # For standard plots, recreate hue legend manually or let seaborn handle it
            # We'll create simple color patches for each hue category
            if isinstance(plot_palette, dict):
                for h in hue_order:
                    legend_elements.append(
                        Line2D(
                            [0],
                            [0],
                            marker="s",
                            color="w",
                            markerfacecolor=plot_palette.get(h),
                            markersize=10,
                            label=h,
                        )
                    )
            else:
                # Use default seaborn colors or provided palette name
                for i, h in enumerate(hue_order):
                    legend_elements.append(
                        Line2D(
                            [0],
                            [0],
                            marker="s",
                            color="w",
                            markerfacecolor=f"C{i}",
                            markersize=10,
                            label=h,
                        )
                    )
        except:
            # Fallback: just use text labels if color detection fails
            for h in hue_order:
                legend_elements.append(Line2D([0], [0], color="gray", lw=0, label=h))

    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc="best",
            title=hue if color_col is None else None,
        )

    if xticks_col is not None:
        mapping = (
            modified_data[[x, xticks_col]]
            .drop_duplicates()
            .set_index(x)[xticks_col]
            .to_dict()
        )
        current_labels = [lbl.get_text() for lbl in ax.get_xticklabels()]
        base_labels = [mapping.get(lbl, lbl) for lbl in current_labels]
    else:
        base_labels = [lbl.get_text() for lbl in ax.get_xticklabels()]

    # Apply smart breakline strategy or standard rotation
    if rotation == "breakline":
        # Every second label (indices 1, 3, 5...) gets pushed to second line
        final_labels = [
            f"\n{lbl}" if i % 2 == 1 else lbl for i, lbl in enumerate(base_labels)
        ]
        ax.set_xticklabels(
            final_labels, ha="center", fontsize=xticks_fontsize, rotation=0
        )
    else:
        ax.set_xticklabels(
            base_labels, ha="center", fontsize=xticks_fontsize, rotation=rotation
        )

    ax.set_xlabel("")

    return ax


def plot_dual_layer_hatch(
    data,
    x,
    y,
    hue,
    color_col=None,  # Now optional: None disables dual-layer coloring
    palette=None,  # Optional unless color_col is used
    kind="box",
    ax=None,
    hue_order=None,
    order=None,
    xticks_col=None,
    xticks_fontsize=12,
    rotation=45,
    show_color_legend=True,
    add_pvalues=False,
    test="t-test_ind",
    text_format="star",
    loc="outside",
    verbose_annotations=False,
    annotate_values=False,
    show_points=False,
    hatch_density="/",  # Diagonal stripes (density: /, //, or ///)
    **kwargs,
):
    """
    Universal plotting function: box/bar plots with optional dual-layer coloring
    and optional statistical annotations.

    Parameters
    ----------
    color_col : str or None
        If str, enables dual-layer coloring (base color by color_col, lightness by hue).
        If None, uses standard seaborn coloring (palette applies to hue).
    palette : dict or seaborn palette
        If color_col is set: mapping of color_col categories to colors.
        If color_col is None: passed directly to seaborn (e.g., 'Set2', dict, list).
    add_pvalues : bool
        If True, adds significance brackets between hue groups within each x category.
    Other parameters : see plot_annotated_comparison and previous plot_dual_layer
    """
    if ax is None:
        ax = plt.gca()

    # 1. Enforce Order
    if order is None:
        order = sorted(data[x].unique())
    if hue_order is None:
        hue_order = sorted(data[hue].unique())

    # Validation for statistical annotations
    if add_pvalues and len(hue_order) != 2:
        raise ValueError(
            f"When add_pvalues=True, hue column '{hue}' must have exactly 2 unique values, "
            f"found: {hue_order}"
        )

    # Handle edge case where x == hue
    original_x = x
    modified_data = data.copy()
    if hue == x:
        x = hue + "_x"
        modified_data[x] = modified_data[original_x].copy()

    # 2. Draw the Base Plot
    plot_func = sns.boxplot if kind == "box" else sns.barplot

    # When color_col is None, pass palette to seaborn normally
    # When color_col is set, we handle colors manually, so don't pass palette to seaborn
    plot_palette = None if color_col is not None else palette

    ax = plot_func(
        data=modified_data,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        ax=ax,
        palette=plot_palette,  # Only used when color_col is None
        **kwargs,
    )

    # Add strip plot for boxplots if requested
    if kind == "box" and show_points:
        sns.stripplot(
            data=modified_data,
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
            legend=False,
        )

    # Annotate values for bar plots
    if kind == "bar" and annotate_values:

        for bar in ax.patches:
            height = bar.get_height()
            if np.isnan(height):
                continue
            ax.annotate(
                f"{height:.2f}",
                (bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )

    # 3. DUAL-LAYER COLORING (Only if color_col is specified)
    if color_col is not None:
        if palette is None:
            raise ValueError("palette must be provided when color_col is specified")

        # Handle edge case: x is the same as color_col
        if color_col == original_x:
            x_to_color_cat = {cat: cat for cat in order}
        else:
            x_to_color_cat = (
                modified_data[[x, color_col]]
                .drop_duplicates()
                .set_index(x)[color_col]
                .to_dict()
            )

        # 4. Iterate and Apply Hatching (instead of lightness)
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

                # Apply base color to both
                patch.set_facecolor(base_color)

                # Left (Hue 0) = Solid, Right (Hue 1) = Hatched
                if x_center < x_idx:
                    patch.set_hatch("")  # No pattern
                    patch.set_edgecolor("black" if kind == "bar" else base_color)
                else:
                    patch.set_hatch(
                        hatch_density
                    )  # Diagonal stripes (density: /, //, or ///)
                    patch.set_edgecolor("black")  # Ensure hatch lines are visible

                if kind == "bar":
                    patch.set_linewidth(1)

    # 4. STATISTICAL ANNOTATIONS
    if add_pvalues:
        pairs = [
            ((category, hue_order[0]), (category, hue_order[1])) for category in order
        ]

        sa_plot_type = "barplot" if kind == "bar" else "boxplot"

        annotator = Annotator(
            ax,
            pairs=pairs,
            data=modified_data,
            x=x,
            y=y,
            hue=hue,
            order=order,
            hue_order=hue_order,
            plot=sa_plot_type,
        )

        annotator.configure(
            test=test, text_format=text_format, loc=loc, verbose=verbose_annotations
        )

        try:
            annotator.apply_and_annotate()
        except Exception as e:
            print(f"Error during statistical annotation: {e}")
            raise e

    # 5. LEGEND HANDLING
    if ax.get_legend():
        ax.get_legend().remove()

    legend_elements = []

    if color_col is not None:
        # Dual-layer mode: Show hue indicators (solid vs hatched) + color categories
        # Use light gray to show the pattern difference clearly
        legend_elements.extend(
            [
                Patch(
                    facecolor="lightgray",
                    edgecolor="black",
                    hatch="",
                    label=f"{hue_order[0]}",
                ),
                Patch(
                    facecolor="lightgray",
                    edgecolor="black",
                    hatch="///",
                    label=f"{hue_order[1]}",
                ),
            ]
        )
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
    else:
        # Standard mode: Reconstruct standard hue legend
        # Try to get colors from the plot or use default
        try:
            # For standard plots, recreate hue legend manually or let seaborn handle it
            # We'll create simple color patches for each hue category
            if isinstance(plot_palette, dict):
                for h in hue_order:
                    legend_elements.append(
                        Line2D(
                            [0],
                            [0],
                            marker="s",
                            color="w",
                            markerfacecolor=plot_palette.get(h),
                            markersize=10,
                            label=h,
                        )
                    )
            else:
                # Use default seaborn colors or provided palette name
                for i, h in enumerate(hue_order):
                    legend_elements.append(
                        Line2D(
                            [0],
                            [0],
                            marker="s",
                            color="w",
                            markerfacecolor=f"C{i}",
                            markersize=10,
                            label=h,
                        )
                    )
        except:
            # Fallback: just use text labels if color detection fails
            for h in hue_order:
                legend_elements.append(Line2D([0], [0], color="gray", lw=0, label=h))

    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc="best",
            title=hue if color_col is None else None,
        )

    if xticks_col is not None:
        mapping = (
            modified_data[[x, xticks_col]]
            .drop_duplicates()
            .set_index(x)[xticks_col]
            .to_dict()
        )
        current_labels = [lbl.get_text() for lbl in ax.get_xticklabels()]
        base_labels = [mapping.get(lbl, lbl) for lbl in current_labels]
    else:
        base_labels = [lbl.get_text() for lbl in ax.get_xticklabels()]

    # Apply smart breakline strategy or standard rotation
    if rotation == "breakline":
        # Every second label (indices 1, 3, 5...) gets pushed to second line
        final_labels = [
            f"\n{lbl}" if i % 2 == 1 else lbl for i, lbl in enumerate(base_labels)
        ]
        ax.set_xticklabels(
            final_labels, ha="center", fontsize=xticks_fontsize, rotation=0
        )
    else:
        ax.set_xticklabels(
            base_labels, ha="center", fontsize=xticks_fontsize, rotation=rotation
        )

    ax.set_xlabel("")

    # # 6. X-TICKS RENAMING
    # if xticks_col is not None:
    #     mapping = (
    #         modified_data[[x, xticks_col]]
    #         .drop_duplicates()
    #         .set_index(x)[xticks_col]
    #         .to_dict()
    #     )
    #     current_labels = [lbl.get_text() for lbl in ax.get_xticklabels()]
    #     new_labels = [mapping.get(lbl, lbl) for lbl in current_labels]
    #     ax.set_xticklabels(new_labels, ha="center", fontsize=xticks_fontsize)

    # # Cleanup
    # ax.set_xlabel("")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)

    return ax
