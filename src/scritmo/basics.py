import numpy as np
import pandas as pd

w = 2 * np.pi / 24
rh = w**-1


def LL(y_true, y_pred):
    return -np.sum((y_true - y_pred) ** 2 / y_true.var())


def BIC(y_true, y_pred, n_params):
    return -2 * LL(y_true, y_pred) + n_params * np.log(len(y_true))


def gene_pos(adata, gene_name, obs_field="x"):
    """
    avarages the expression of a gene across all cells in a given assigned field
    """
    v = adata[:, gene_name].X
    pos = adata.obs[obs_field]
    pos_un = pos.unique()
    out = np.zeros(pos_un.shape)

    for i, p in enumerate(pos_un):
        out[i] = np.mean(v[pos == p])

    return out


def ind(list, sublist):
    """
    list: list of strings
    return the index of the sublist in the list
    """
    # if input is a string and not a list, make it a list
    if isinstance(sublist, str):
        sublist = [sublist]
    return np.array([np.where(list == i)[0][0] for i in sublist])


def ind2(large_list, small_list, verbose=False):
    """
    large_list: list of strings
    small_list: list of strings
    return the index of the small_list in the large_list
    """

    if isinstance(small_list, str):
        small_list = [small_list]
    large_array = np.array(large_list)
    small_array = np.array(small_list)
    indices = []

    element_to_index = {element: idx for idx, element in enumerate(large_array)}

    for element in small_array:
        if element in element_to_index:
            indices.append(element_to_index[element])
        else:
            # Optionally handle or notify when an element is not found
            if verbose:
                print(f"Warning: '{element}' not found in the large list.")

    return indices


def vectorized_subsample_matrix_units(matrix, p):
    """
    Subsample the counts of a matrix by a given probability for each unit.
    Implemented in a vectorized fashion for speed.
    Inputs:
    matrix: np.ndarray
        The matrix to be subsampled. Needs to have integer values.
    p: float
        The probability of retaining each unit.
    """
    # Create a random array for each unit
    random_array = np.random.rand(np.max(matrix), *matrix.shape) < p
    # Initialize the subsampled matrix with zeros of the same shape as the original matrix
    subsampled_matrix = np.zeros(matrix.shape, dtype=int)

    # Iterate over each possible count value
    for k in range(1, np.max(matrix) + 1):
        # Create a mask for the positions where the current count is greater than or equal to k
        mask = matrix >= k
        # Add the survived units to the subsampled matrix
        subsampled_matrix += mask * random_array[k - 1]

    return subsampled_matrix


def dict2df(d, index=None):
    """
    converts the dictionary d to a pandas dataframe.
    Dictionary d is the output of inference functions in numpyro_models_handles.py
    """

    for k in d.keys():
        d[k] = np.array(d[k]).squeeze()

    df = pd.DataFrame(d)

    if index is not None:
        df.index = index
    return df


def df2dict(df, return_index=False):
    """
    converts a pandas dataframe to a dictionary.
    """

    index = df.index.values
    d = df.to_dict(orient="list")

    for k in d.keys():
        d[k] = np.array(d[k])[None, :]

    if return_index:
        return d, index
    else:
        return d


def fold_change(log_amp, base=np.e):
    """
    convert log-amplitude to fold change
    """
    return base ** (2 * log_amp)


def mean_disp_to_np(mean, dispersion):
    """
    Convert mean and dispersion to n and p parameters of the negative binomial distribution
    WARNING: This works with the scipy parameterization of the negative binomial distribution
        such param. can vary between packages, BE CAREFUL
    Input:
    mean: the mean of the negative binomial distribution
    dispersion: the dispersion of the negative binomial distribution
    """
    p = mean / (mean + dispersion * mean**2)
    n = 1 / dispersion
    return n, p


def add_ccg_back(ccg, listt, verbose=False):
    diff = set(ccg) - set(listt)
    if verbose:
        print(f"ccg removed in the selection {diff}")
    added = set(listt) | set(diff)
    return np.array(list(added))


def length_normalized_library_size(adata, layer="spliced", length_var="gene_length"):
    """
    Function used to get the length-normalized library size
    for bulk data. it returns gamma_cg such that
    f_cg = n_cg / gamma_cg sums to 1 over genes for each sample
    """
    n_cg = adata.layers[layer]
    l_g = adata.var[length_var].values[None, :]
    # try to convert n_cg to dense if sparse
    if hasattr(n_cg, "toarray"):
        n_cg = n_cg.toarray()

    gamma_cg = l_g * np.sum(n_cg / l_g, axis=1, keepdims=True)
    # dataframe with same index and columns as n_cg
    gamma_cg = pd.DataFrame(gamma_cg, index=adata.obs_names, columns=adata.var_names)
    return gamma_cg
