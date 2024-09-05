import numpy as np
from .basics import BIC
import scipy.stats as stats
import pandas as pd
import scanpy as sc


def pseudobulk_new(
    adata,
    groupby_obs_list,
    pseudobulk_layer="spliced",
    n_groups=1,
    keep_obs=["ZT", "ZTmod"],
):
    """
    This function implements the megacell pseudobulk strategy
    in a much cleaner way, leveraging the .groupby method of pandas
    It takes in the adata object and outputs another adata object that
    has been pseudobulked. Only some relevant .obs columns are retained.
    warning:
    The output as a 2-dim anndata is not compatible yet with functions
    as harmonic_regression_loop. It needs to be tranformed in a 3-d
    tensor
    Inputs:
    - adata: anndata object
    - groupby_obs_list: list of the 2 obs columns to group the cells
        One should be the sample column, discriminating between biological replicates
        The other is whatever you want (usually celltype)
    - pseudobulk_layer: layer of the adata object to be used
    - n_groups: number of groups to split the cells in each timepoint, it's a pseudo-pseudo-bulk
    - keep_obs: list of the obs columns to keep in the output, be careful that all cells
        in the same group have the same value for these columns!
    """

    groups = adata.obs.groupby(groupby_obs_list).groups
    group_names = list(groups.keys())
    # now do pseudobulk using these indices, sum the counts

    PBS = []

    for i, gr in enumerate(group_names):

        idx = np.array(groups[gr])
        np.random.shuffle(idx)
        bb = np.array_split(idx, n_groups)

        for j, gr2 in enumerate(bb):

            counts = adata[gr2, :].layers[pseudobulk_layer].sum(axis=0)
            xx = sc.AnnData(
                X=counts,
                var=adata.var.index.values,
            )
            xx.obs[groupby_obs_list[0]] = gr[0]
            xx.obs[groupby_obs_list[1]] = gr[1]

            for k in keep_obs:
                xx.obs[k] = adata.obs.loc[idx, k].values[0]
            PBS.append(xx)

    PB = sc.concat(PBS)
    PB.X = np.array(PB.X)
    PB.var_names = adata.var_names

    return PB


def pseudo_bulk_time(adata, sample_obs="sample_name", ZT_obs="ZTmod", n_groups=1):
    """
    It produces the sample_ZT vector for the pseudobulk harmonic regression
    It basically returns the ZT for each sample
    """
    samples = adata.obs[sample_obs]
    # samples are ordered by sample name
    samples_u = np.unique(samples)

    # create poabda series with the sample_u as index
    sample_ZT = pd.Series(index=samples_u, dtype=int)

    for i, s in enumerate(samples_u):
        time_sample = adata.obs[ZT_obs][samples == s].values
        ts = int(np.unique(time_sample)[0])

        sample_ZT[s] = ts

    if n_groups > 1:
        ZT_vec = np.zeros(len(sample_ZT) * n_groups)
        for i, t in enumerate(sample_ZT):
            for j in range(n_groups):
                ZT_vec[i * n_groups + j] = t
        return ZT_vec

    return sample_ZT


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
    """
    # Create the 3D array
    _, NG = PB.shape
    z_obs, sample_obs = groupby_obs_list

    z_u = np.unique(PB.obs[z_obs])
    NZ = z_u.shape[0]

    sample_u = np.unique(PB.obs[sample_obs])
    NS = sample_u.shape[0]

    n_ygt = np.zeros((NZ, NG, NS * n_groups))

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
                    f"Missing data for cluster {ct} and sample {sample}, padding with zeros."
                )

    return n_ygt


def normalize_log_PB(n_ygt, eps=None, base=2.0):
    """
    This function normalizes the pseudobulk data and applies a log2 transformation
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
        cc = np.median(N_yt)
        eps = 1 / cc
    f_ygt = n_ygt / N_yt[:, None, :]
    # replace nan values with 0
    f_ygt = np.nan_to_num(f_ygt)
    print(" eliminating nan values")
    gamma_ygt = np.log(f_ygt + eps) / np.log(base)
    return f_ygt, gamma_ygt


# def pseudo_bulk_time(adata, sample_obs="sample_name", ZT_obs="ZT"):
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

#     return sample_ZT


# def circadian_celltype(
#     adata, celltype_list, celltype, ccg, circadian_time, layer="norm"
# ):
#     """
#     returns the mean expression of the core clock genes
#     for a given celltype and timepoint. It is a pseudobulk approach

#     You can pass the sample list instead of the circadian time to
#     distinguish between different replicates at the same timepoint

#     Input:
#     - adata: anndata object
#     - celltype_list: list of celltypes, same length as adata.obs
#     - celltype: string, celltype of interest
#     - ccg: list of genes that will be avraged i the pseudobulk
#     - circadian_time: list of timepoints, same length as adata.obs
#     - layer: layer of the adata object to be used
#     """
#     tu = np.unique(circadian_time)
#     out = np.zeros((len(ccg), len(tu)))

#     for j, t in enumerate(tu):
#         mask_timepoint = circadian_time == t
#         mask_ct = (celltype_list == celltype).squeeze()
#         mask = mask_timepoint & mask_ct
#         out[:, j] = adata[mask, ccg].layers[layer].mean(axis=0)

#     return out


# def circadian_celltype_megacell(
#     adata, celltype_list, celltype, ccg, circadian_time, layer="spliced", n_groups=1
# ):
#     """
#     returns the mean expression of the core clock genes
#     for a given celltype and timepoint. It is a pseudobulk approach

#     You can pass the sample list instead of the circadian time to
#     distinguish between different replicates at the same timepoint

#     Input:
#     - adata: anndata object
#     - celltype_list: list of celltypes, same length as adata.obs
#     - celltype: string, celltype of interest
#     - ccg: list of genes that will be avraged i the pseudobulk
#     - circadian_time: list of timepoints, same length as adata.obs
#     - layer: layer of the adata object to be used
#     - n_groups: number of groups to split the cells in each timepoint, it's a pseudo-pseudo-bulk
#     """
#     tu = np.unique(circadian_time)
#     out = np.zeros((len(ccg), len(tu), n_groups))

#     for j, t in enumerate(tu):
#         mask_timepoint = circadian_time == t
#         mask_ct = (celltype_list == celltype).squeeze()
#         mask = mask_timepoint & mask_ct

#         # Indices where mask is True
#         indices = np.where(mask)[0]
#         # np.random.shuffle(indices)
#         # subdivide the indices into n_groups
#         split_indices = np.array_split(indices, n_groups)

#         for k, idx in enumerate(split_indices):
#             out[:, j, k] = (
#                 adata[idx, ccg].layers[layer].sum(axis=0)
#                 / adata[idx, :].layers[layer].sum()
#             )

#     # reshaping safely the output, in roder to be compatible with the pseudobulk_loop
#     out2 = np.zeros((len(ccg), len(tu) * n_groups))
#     for j, t in enumerate(tu):
#         out2[:, j * n_groups : (j + 1) * n_groups] = out[:, j, :]

#     return out2


# def circadian_celltype_counts_g(
#     adata, celltype_list, celltype, ccg, circadian_time, layer="spliced", n_groups=1
# ):
#     """
#     This function gives the counts of the genes in the pseudobulk. No division
#     by the total counts is performed.
#     """
#     tu = np.unique(circadian_time)
#     out = np.zeros((len(ccg), len(tu), n_groups))

#     for j, t in enumerate(tu):
#         mask_timepoint = circadian_time == t
#         mask_ct = (celltype_list == celltype).squeeze()
#         mask = mask_timepoint & mask_ct

#         # Indices where mask is True
#         indices = np.where(mask)[0]
#         # np.random.shuffle(indices)
#         # subdivide the indices into n_groups
#         split_indices = np.array_split(indices, n_groups)

#         for k, idx in enumerate(split_indices):
#             out[:, j, k] = adata[idx, ccg].layers[layer].sum(axis=0)

#     # reshaping safely the output, in roder to be compatible with the pseudobulk_loop
#     out2 = np.zeros((len(ccg), len(tu) * n_groups))
#     for j, t in enumerate(tu):
#         out2[:, j * n_groups : (j + 1) * n_groups] = out[:, j, :]

#     return out2


# def pseudobulk_loop(
#     adata,
#     celltype_list,
#     genes,
#     circadian_time,
#     layer,
#     megacell=False,
#     log=False,
#     log2=True,
#     eps=1,
#     n_groups=1,
#     cpm=False,
# ):
#     """
#     It creates a pseudobulk datset froma  single cell dataset.
#     It returns a 3D array with dimensions celltype, gene, timepoint. The pseudosamples
#     are found by intersecting the celltype and the timepoint. With n_groups you can
#     further split the cells in each timepoint, it's a pseudo-pseudo-bulk.

#     Input:
#     adata: anndata object
#     celltype_list: list of celltypes, same length as adata.obs
#     genes: list of genes
#     circadian_time: list of timepoints
#     layer: layer of the adata object
#     output: 3D array with dimensions celltype, gene, timepoint
#     megacell: if True, it will normalize the expression by the total
#         expression of the cell, you need to use layer=counts
#     n_groups: number of groups to split the cells in each timepoint, it's a pseudo-pseudo-bulk
#     cmp: if True, it will return the expression in counts per million

#     Output:
#     3D array with dimensions (celltype, gene, timepoint * n_groups)
#     """
#     out = np.zeros(
#         (
#             len(np.unique(celltype_list)),
#             len(genes),
#             len(np.unique(circadian_time)) * n_groups,
#         )
#     )
#     if not megacell:
#         for i, ct in enumerate(np.unique(celltype_list)):
#             out[i, :, :] = circadian_celltype(
#                 adata, celltype_list, ct, genes, circadian_time, layer=layer
#             )
#     else:
#         for i, ct in enumerate(np.unique(celltype_list)):
#             out[i, :, :] = circadian_celltype_megacell(
#                 adata,
#                 celltype_list,
#                 ct,
#                 genes,
#                 circadian_time,
#                 layer=layer,
#                 n_groups=n_groups,
#             )

#     if log:
#         if cpm:
#             out = np.log(eps + 1e6 * out)
#         else:
#             out = np.log(eps + out)
#         # log base 2
#         if log2:
#             out = out / np.log(2)
#     return out


# # same of before, no log possibility, but with the count mode
# def pseudobulk_loop2(
#     adata,
#     celltype_list,
#     genes,
#     circadian_time,
#     layer,
#     # megacell=False,
#     log=False,
#     eps=1,
#     n_groups=1,
#     cpm=False,
#     count_mode=True,
# ):
#     """
#     It creates a pseudobulk datset froma  single cell dataset.
#     It returns a 3D array with dimensions celltype, gene, timepoint. The pseudosamples
#     are found by intersecting the celltype and the timepoint. With n_groups you can
#     further split the cells in each timepoint, it's a pseudo-pseudo-bulk.

#     Input:
#     adata: anndata object
#     celltype_list: list of celltypes, same length as adata.obs
#     genes: list of genes
#     circadian_time: list of timepoints
#     layer: layer of the adata object
#     megacell: if True, it will normalize the expression by the total
#         expression of the cell, you need to use layer=counts
#     n_groups: number of groups to split the cells in each timepoint, it's a pseudo-pseudo-bulk
#     cmp: if True, it will return the expression in counts per million

#     Output:
#     3D array with dimensions (celltype, gene, timepoint * n_groups)
#     """
#     out = np.zeros(
#         (
#             len(np.unique(celltype_list)),
#             len(genes),
#             len(np.unique(circadian_time)) * n_groups,
#         )
#     )
#     # this is the megacell regime, where we normalize by th elibrary size
#     if not count_mode:
#         for i, ct in enumerate(np.unique(celltype_list)):
#             out[i, :, :] = circadian_celltype_megacell(
#                 adata,
#                 celltype_list,
#                 ct,
#                 genes,
#                 circadian_time,
#                 layer=layer,
#                 n_groups=n_groups,
#             )
#     # this is the integer counts regime, no normalization
#     if count_mode:
#         for i, ct in enumerate(np.unique(celltype_list)):
#             out[i, :, :] = circadian_celltype_counts_g(
#                 adata,
#                 celltype_list,
#                 ct,
#                 genes,
#                 circadian_time,
#                 layer=layer,
#                 n_groups=n_groups,
#             )
#     return out


# ###############
# # these functions is used for taking the sums, the megacell regime is implied
# ###############


# def circadian_celltype_counts(
#     adata, celltype_list, celltype, circadian_time, layer="spliced", n_groups=1
# ):
#     tu = np.unique(circadian_time)
#     counts_sample = np.zeros((len(tu), n_groups))

#     for j, t in enumerate(tu):
#         mask_timepoint = circadian_time == t
#         mask_ct = (celltype_list == celltype).squeeze()
#         mask = mask_timepoint & mask_ct

#         # Indices where mask is True
#         indices = np.where(mask)[0]
#         # Optional: shuffle to randomize group assignments
#         # np.random.shuffle(indices)
#         # subdivide the indices into n_groups
#         split_indices = np.array_split(indices, n_groups)

#         for k, idx in enumerate(split_indices):
#             counts_sample[j, k] = adata[idx, :].layers[layer].sum()

#     # reshaping safely the output, in order to be compatible with the pseudobulk_loop
#     counts_sample2 = np.zeros((len(tu) * n_groups))
#     for j, t in enumerate(tu):
#         counts_sample2[j * n_groups : (j + 1) * n_groups] = counts_sample[j, :]

#     return counts_sample2


# def pseudobulk_loop_counts(
#     adata,
#     celltype_list,
#     circadian_time,
#     layer,
#     n_groups=1,
# ):
#     """
#     It takes the som of the counts (library size) for each pseudosample
#     """
#     out = np.zeros(
#         (
#             len(np.unique(celltype_list)),
#             len(np.unique(circadian_time)) * n_groups,
#         )
#     )
#     for i, ct in enumerate(np.unique(celltype_list)):
#         out[i, :] = circadian_celltype_counts(
#             adata,
#             celltype_list,
#             ct,
#             circadian_time,
#             layer=layer,
#             n_groups=n_groups,
#         )

#     return out
