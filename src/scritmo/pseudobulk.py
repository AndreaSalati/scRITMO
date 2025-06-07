import numpy as np
from .basics import BIC
import scipy.stats as stats
import pandas as pd
import scanpy as sc

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
    adata,
    groupby_obs_list: Sequence[str],
    pseudobulk_layer: str = "spliced",
    n_groups: int = 1,
    keep_obs: Sequence[str] = ("ZT", "ZTmod"),
    add_layers: bool = False,
) -> sc.AnnData:
    """
    Fast pseudobulk via sc.get.aggregate, then annotate with extra obs columns.

    Parameters
    ----------
    adata
        Input AnnData.
    groupby_obs_list
        Two obs columns to group by, e.g. ["celltype", "sample_name"].
    pseudobulk_layer
        Which layer to sum.
    n_groups
        Number of sub-splits per group (currently only 1 is supported).
    keep_obs
        Other obs-columns (must be constant within each `sample_name`)
        to carry over into the pseudobulk.
    add_layers
        If True, add a `"norm"` layer (counts divided by row-sum).

    Returns
    -------
    A new AnnData with
      - X: summed counts (dense `ndarray`),
      - obs: the grouping columns + your `keep_obs` + a `"counts"` column,
      - optionally a `"norm"` layer.
    """
    if n_groups != 1:
        raise ValueError("n_groups>1 is not yet supported in this fast implementation")

    if type(groupby_obs_list) is not list:
        groupby_obs_list = [groupby_obs_list]

    # 1) fast aggregate
    pb = sc.get.aggregate(
        adata,
        groupby_obs_list,
        func="sum",
        layer=pseudobulk_layer,
    )

    pb.X = pb.layers["sum"]

    # 2) densify X
    if hasattr(pb.X, "toarray"):
        pb.X = pb.X.toarray()

    # 3) counts per pseudobulk
    pb.obs["counts"] = pb.X.sum(axis=1)

    # 4) optional normalized layer
    if add_layers:
        pb.layers["norm"] = pb.X.copy()
        pb.layers["norm"] /= pb.obs["counts"].values[:, None]

    # 5) pull through any extra obs via your pseudo_bulk helper
    #    assume groupby_obs_list[0] is the "sample" axis
    sample_col = groupby_obs_list[0]
    for col in keep_obs:
        # build a Series mapping sample → col value
        mapping: pd.Series = pseudo_bulk_time(
            labels=adata.obs[sample_col],
            values=adata.obs[col],
            return_dict=False,  # get a Series
        )
        # align to pb.obs[sample_col]
        pb.obs[col] = mapping.loc[pb.obs[sample_col]].values

    return pb


def change_shape(PB, groupby_obs_list, n_groups=1):
    """
    This function changes the shape of the pseudobulk data
    into a 3D array, with shape (NZ, NG, NS*n_groups)
    Where : NZ is the number of unique categories in the first element of groupby_obs_list
            NG is the number of genes
            NS is the number of samples (second element in groupby_obs_list)

    Parameters:
    PB : AnnData pseudobulked object
    groupby_obs_list : list of strings of 2 elements
        The first element is the observation key to group by
        The second element is the observation key to group by
    n_groups : int (default=1)

    Returns:
    n_ygt : 3D numpy array
        The pseudobulk data, with shape (NZ, NG, NS*n)
        In case there are no cells for a given combination of z_obs and sample_obs
        the data is padded with np.nan
    """
    # Create the 3D array
    _, NG = PB.shape
    z_obs, sample_obs = groupby_obs_list

    z_u = np.unique(PB.obs[z_obs])
    NZ = z_u.shape[0]

    sample_u = np.unique(PB.obs[sample_obs])
    NS = sample_u.shape[0]

    # n_ygt = np.zeros((NZ, NG, NS * n_groups))
    n_ygt = np.full((NZ, NG, NS * n_groups), np.nan)

    for i, ct in enumerate(z_u):
        cluster_indices = PB.obs[z_obs] == ct

        for j, sample in enumerate(sample_u):
            sample_indices = PB.obs[sample_obs] == sample
            selected_indices = cluster_indices & sample_indices

            if np.any(selected_indices):
                # Add the actual data for existing z_obs and sample_obs combinations
                n_ygt[i, :, j] = PB[selected_indices, :].X.T.squeeze()
            else:
                # The combination of z_obs and sample_obs does not exist, so keep zeros
                print(
                    f"Missing data for cluster {ct} and sample {sample}, padding with np.nan."
                )

    return n_ygt


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
