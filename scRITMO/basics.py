import numpy as np


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
