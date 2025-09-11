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
    labels
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


def normalize_log_PB(n_ygt, eps=None, base=2.0):
    """
    This function normalizes the pseudobulk data and applies a log2 transformation
    To be run after the change_shape function
    Parameters:
    n_ygt : 3D numpy array
        The pseudobulk data, with shape (NZ, NG, NS)
    eps : float the pseudocount to add before log2 transformation
        by default, it's an adaptive value based on the median of the data
    base : float the base of the log transformation
    """
    # number of counts per celltype and sample
    N_yt = n_ygt.sum(axis=1)
    if eps is None:
        cc = np.nanmedian(N_yt)
        eps = 1 / cc
    f_ygt = n_ygt / N_yt[:, None, :]
    # replace nan values with 0
    f_ygt = np.nan_to_num(f_ygt)
    print(" eliminating nan values")
    gamma_ygt = np.log(f_ygt + eps) / np.log(base)
    return f_ygt, gamma_ygt


# def pseudobulk_new(
#     adata,
#     groupby_obs_list,
#     pseudobulk_layer="spliced",
#     n_groups=1,
#     keep_obs=["ZT", "ZTmod"],
#     add_layers=False,
# ):
#     """
#     This function implements the megacell pseudobulk strategy
#     in a much cleaner way, leveraging the .groupby method of pandas
#     It takes in the adata object and outputs another adata object that
#     has been pseudobulked. Only some relevant .obs columns are retained.
#     warning:
#     The output as a 2-dim anndata is not compatible yet with functions
#     as harmonic_regression_loop. It needs to be tranformed in a 3-d
#     tensor
#     Inputs:
#     - adata: anndata object
#     - groupby_obs_list: list of the 2 obs columns to group the cells
#         One should be the sample column, discriminating between biological replicates
#         The other is whatever you want (usually celltype)
#     - pseudobulk_layer: layer of the adata object to be used
#     - n_groups: number of groups to split the cells in each timepoint, it's a pseudo-pseudo-bulk
#     - keep_obs: list of the obs columns to keep in the output, be careful that all cells
#         in the same group have the same value for these columns!
#     - add_layers: Adds the normalized and log layers to the output, use only
#         if passign all genes when pseudobulking
#     """

#     groups = adata.obs.groupby(groupby_obs_list).groups
#     group_names = list(groups.keys())
#     # now do pseudobulk using these indices, sum the counts

#     PBS = []

#     for i, gr in enumerate(group_names):

#         idx = np.array(groups[gr])
#         np.random.shuffle(idx)
#         bb = np.array_split(idx, n_groups)

#         for j, gr2 in enumerate(bb):

#             counts = adata[gr2, :].layers[pseudobulk_layer].sum(axis=0).reshape(1, -1)
#             xx = sc.AnnData(
#                 X=counts,
#                 var=adata.var.index.values,
#             )
#             xx.obs[groupby_obs_list[0]] = gr[0]
#             xx.obs[groupby_obs_list[1]] = gr[1]

#             for k in keep_obs:
#                 xx.obs[k] = adata.obs.loc[idx, k].values[0]
#             PBS.append(xx)

#     PB = sc.concat(PBS)
#     PB.X = np.array(PB.X)
#     PB.var_names = adata.var_names
#     PB.obs["counts"] = PB.X.sum(axis=1)

#     if add_layers:
#         PB.layers["norm"] = PB.X.copy()
#         PB.layers["norm"] = PB.layers["norm"] / PB.obs["counts"].values[:, None]

#     return PB


# def pseudo_bulk_time(adata, sample_obs="sample_name", ZT_obs="ZTmod", n_groups=1):
#     """
#     It produces the sample_ZT vector for the pseudobulk harmonic regression
#     It basically returns the ZT for each sample
#     """
#     samples = adata.obs[sample_obs]
#     # samples are ordered by sample name
#     samples_u = np.unique(samples)

#     # create poabda series with the sample_u as index
#     sample_ZT = pd.Series(index=samples_u, dtype=int)

#     for i, s in enumerate(samples_u):
#         time_sample = adata.obs[ZT_obs][samples == s].values
#         ts = int(np.unique(time_sample)[0])
#         sample_ZT[s] = ts

#     if n_groups > 1:
#         ZT_vec = np.zeros(len(sample_ZT) * n_groups)
#         for i, t in enumerate(sample_ZT):
#             for j in range(n_groups):
#                 ZT_vec[i * n_groups + j] = t
#         return ZT_vec

#     return sample_ZT
