import numpy as np
import pandas as pd
import itertools


# Generate all partitions of groups to determine sharing structure
# [[0], [1, 2]] -> Group 0 independent; Groups 1 & 2 share parameters
def generate_partitions(collection):
    if len(collection) == 1:
        yield [collection]
        return
    first = collection[0]
    for smaller in generate_partitions(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1 :]
        yield [[first]] + smaller


def create_matrix_list(time, group, n_groups, period=24):
    """
    Generates the list of design matrices for rhythmicity analysis.
    Mimics the 'create_matrix_list' logic from R, adapted for Python.

    Args:
        time: array-like of time points
        group: array-like of group labels
        n_groups: integer, total number of unique groups
        period: oscillation period (default 24)

    Returns:
        List of pandas DataFrames (Design Matrices)
    """
    t = np.array(time)
    c_term = np.cos(2 * np.pi * t / period)
    s_term = np.sin(2 * np.pi * t / period)

    unique_groups = np.unique(group)
    group_indices = range(n_groups)

    partitions = list(generate_partitions(list(group_indices)))

    models = []

    for p in partitions:
        num_clusters = len(p)
        # For each cluster in this partition, decide if it is rhythmic (True) or Flat (False)
        rhythm_combinations = list(
            itertools.product([False, True], repeat=num_clusters)
        )

        for rhythm_mask in rhythm_combinations:
            mat_dict = {}

            # 1. Intercepts (u) - Always one per group
            for g_idx in range(n_groups):
                col_name = f"u.{g_idx + 1}"
                col_vals = np.zeros(len(t))
                col_vals[group == unique_groups[g_idx]] = 1
                mat_dict[col_name] = col_vals

            # 2. Rhythmic terms (a, b) - Added only if cluster is rhythmic
            for cluster_idx, indices in enumerate(p):
                is_rhythmic = rhythm_mask[cluster_idx]

                if is_rhythmic:
                    # Name parameters based on the first group in the cluster
                    primary_id = indices[0] + 1
                    a_col_name = f"a.{primary_id}"
                    b_col_name = f"b.{primary_id}"

                    a_vals = np.zeros(len(t))
                    b_vals = np.zeros(len(t))

                    # Apply sine/cosine to all groups in this shared cluster
                    for g_idx in indices:
                        mask = group == unique_groups[g_idx]
                        a_vals[mask] = c_term[mask]
                        b_vals[mask] = s_term[mask]

                    mat_dict[a_col_name] = a_vals
                    mat_dict[b_col_name] = b_vals

            # Convert to DataFrame
            df = pd.DataFrame(mat_dict, index=range(len(t)))

            # Filter out empty columns if any (though logic above should be clean)
            df = df.loc[:, (df != 0).any(axis=0)]
            models.append(df)

    return models


def create_model_labels(group, n_groups):
    """
    Generates human-readable labels corresponding to the models created
    by create_matrix_list.

    Args:
        group: array-like of group labels (used to get names)
        n_groups: integer, number of unique groups

    Returns:
        List of strings, e.g.,
        ['WT=Flat, KO=Flat', 'WT=Rhythmic, KO=Flat', ..., 'Shared(WT,KO)=Rhythmic']
    """
    unique_groups = np.unique(group)  # Sorted alphabetically
    group_indices = range(n_groups)

    partitions = list(generate_partitions(list(group_indices)))

    labels = {}
    counter = 1

    # 2. Iterate exactly as the matrix generator does
    for p in partitions:
        num_clusters = len(p)
        rhythm_combinations = list(
            itertools.product([False, True], repeat=num_clusters)
        )

        for rhythm_mask in rhythm_combinations:
            cluster_labels = []

            # Describe each cluster in this specific model configuration
            for cluster_idx, indices in enumerate(p):
                # Get the names of the groups in this cluster
                # e.g., indices [0] -> "KO", indices [0,1] -> "KO", "WT"
                names = [str(unique_groups[i]) for i in indices]

                # Determine state
                state = "Rhythmic" if rhythm_mask[cluster_idx] else "Flat"

                # Format the string
                if len(names) == 1:
                    # Independent: "WT=Rhythmic"
                    cluster_labels.append(f"{names[0]}={state}")
                else:
                    # Shared: "Shared(WT,KO)=Rhythmic"
                    joined_names = "+".join(names)
                    cluster_labels.append(f"Shared({joined_names})={state}")

            # Combine all parts of the model
            # e.g. "WT=Rhythmic, KO=Flat"
            labels[counter] = ", ".join(cluster_labels)
            counter += 1

    return labels


def compute_BICW(bic_values):
    """
    Computes Schwarz weights (BICW) from a list/array of BIC values.
    Returns normalized probabilities (0-1).
    """
    bic_arr = np.array(bic_values)
    # Handle NaNs effectively
    valid_mask = ~np.isnan(bic_arr)
    if not np.any(valid_mask):
        return np.full(bic_arr.shape, np.nan)

    min_bic = np.nanmin(bic_arr)
    delta_bic = bic_arr - min_bic

    # Weight calculation
    weights = np.exp(-0.5 * delta_bic)
    # Normalize
    return weights / np.nansum(weights)
