import numpy as np
from .basics import BIC
import scipy.stats as stats
import pandas as pd
import scanpy as sc
import anndata as AnnData

from typing import Any, Hashable, Iterable, Union, Sequence


def pseudo_bulk_time(
    labels: Iterable[Hashable],
    values: Iterable[Any],
    n_groups: int = 1,
    return_dict: bool = False,
) -> Union[pd.Series, dict, np.ndarray]:
    """
    Map each unique label to its unique value.

    Parameters
    ----------
    labelsn
        An iterable of labels (e.g. sample names). Must be hashable.
    values
        An iterable of values; for each label, all corresponding entries
        in `values` must be identical.
    n_groups
        If >1, returns a flat numpy array of each label’s value repeated
        `n_groups` times (length = num_unique_labels * n_groups).
    return_dict
        If True (and n_groups == 1), returns a dict[label→value] instead of a Series.

    Returns
    -------
    pd.Series by default (index=unique labels, values=unique assigned value),
    or dict if return_dict=True, or np.ndarray if n_groups>1.

    Raises
    ------
    ValueError
        If any label is associated with more than one distinct value.
    """
    labels_arr = np.asarray(labels)
    values_arr = np.asarray(values)
    if labels_arr.shape != values_arr.shape:
        raise ValueError(
            f"labels and values must have the same length; "
            f"got {labels_arr.shape} vs {values_arr.shape}"
        )

    unique_labels = np.unique(labels_arr)
    mapping = {}
    for lab in unique_labels:
        mask = labels_arr == lab
        vals = np.unique(values_arr[mask])
        if vals.size != 1:
            raise ValueError(
                f"Label {lab!r} has multiple distinct values: {vals.tolist()}"
            )
        mapping[lab] = vals.item()

    # n_groups > 1: return flat numpy array of repeated values
    if n_groups > 1:
        # list(mapping.values()) preserves order of unique_labels
        rep = np.repeat(list(mapping.values()), n_groups)
        return rep

    # n_groups == 1
    if return_dict:
        return mapping

    # default: pandas Series
    return pd.Series(
        data=list(mapping.values()), index=list(mapping.keys()), name="value"
    )


def pseudobulk(
    adata: AnnData,
    groupby_obs_list: Sequence[str],
    n_replicates: int = 1,
    pseudobulk_layer: str = "spliced",
    keep_obs: Sequence[str] = ("ZT", "ZTmod"),
) -> AnnData:
    """
    Creates a pseudobulk AnnData object, with an option to create pseudo-replicates.

    This function sums counts within specified groups and aggregates metadata.
    It can also subdivide each group into a specified number of pseudo-replicates
    by randomly assigning cells to subgroups before aggregation.

    Args:
        adata: The annotated data matrix.
        groupby_obs_list: List of columns in `adata.obs` to group by.
        n_replicates: The number of pseudo-replicates to create within each group.
                      If 1, no pseudo-replicates are made. Defaults to 1.
        pseudobulk_layer: The layer in `adata` to use for aggregation.
        keep_obs: Columns from `adata.obs` to keep in the pseudobulk object.
                  The first value for each group is retained.

    Returns:
        A new AnnData object with pseudobulked data.
    """
    if isinstance(groupby_obs_list, str):
        groupby_obs_list = [groupby_obs_list]

    grouping_vars = list(groupby_obs_list)
    replicate_col = "_pseudo_replicate"

    # --- Create pseudo-replicates if requested ---
    if n_replicates > 1:
        # Prevent conflicts if the column name already exists
        if replicate_col in adata.obs.columns:
            raise ValueError(
                f"Column '{replicate_col}' already exists in adata.obs. "
                "Please remove it before creating pseudo-replicates."
            )

        adata.obs[replicate_col] = -1
        grouped = adata.obs.groupby(grouping_vars, observed=True)

        for _, group_indices in grouped.groups.items():
            n_cells = len(group_indices)
            assignments = np.random.randint(0, n_replicates, size=n_cells)
            adata.obs.loc[group_indices, replicate_col] = assignments

        adata.obs[replicate_col] = adata.obs[replicate_col].astype("category")
        grouping_vars.append(replicate_col)

    try:
        # 1. Create the pseudobulk counts object
        pb = sc.get.aggregate(
            adata,
            by=grouping_vars,
            func="sum",
            layer=pseudobulk_layer,
        )

        if "sum" in pb.layers:
            pb.X = pb.layers["sum"].copy()
        if hasattr(pb.X, "toarray"):
            pb.X = pb.X.toarray()

        # 2. Aggregate metadata using a robust merge strategy
        if keep_obs:
            # Get the columns needed for metadata aggregation
            cols_to_process = list(dict.fromkeys(grouping_vars + list(keep_obs)))

            # Group by and get the first entry for each metadata column
            aggregated_metadata = (
                adata.obs[cols_to_process]
                .groupby(grouping_vars, observed=True)
                .first()
                .reset_index()  # Crucially, move grouping vars from index to columns
            )

            # Merge metadata into the pseudobulk object's obs DataFrame.
            # This is more robust than manual index string matching. It joins
            # based on the content of the grouping_vars columns.
            # We reset and then set the index to preserve the original AnnData index.
            pb.obs = (
                pb.obs.reset_index()
                .merge(aggregated_metadata, on=grouping_vars, how="left")
                .set_index("index")
            )
            pb.obs.index.name = None  # Clean up index name

        # 4. Add total counts
        pb.obs["n_counts"] = pb.X.sum(axis=1)

    finally:
        # --- Cleanup: remove temporary column ---
        if n_replicates > 1 and replicate_col in adata.obs:
            del adata.obs[replicate_col]

    return pb


