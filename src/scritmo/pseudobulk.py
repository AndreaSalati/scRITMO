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
    add_layers=False,
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
    - add_layers: Adds the normalized and log layers to the output, use only
        if passign all genes when pseudobulking
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

            counts = adata[gr2, :].layers[pseudobulk_layer].sum(axis=0).reshape(1, -1)
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
    PB.obs["counts"] = PB.X.sum(axis=1)

    if add_layers:
        PB.layers["norm"] = PB.X.copy()
        PB.layers["norm"] = PB.layers["norm"] / PB.obs["counts"].values[:, None]

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
