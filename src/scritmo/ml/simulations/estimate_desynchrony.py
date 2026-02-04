import torch
import numpy as np
import pandas as pd
from ..utils import df2dict, nmp
import seaborn as sns
import anndata
from scipy.stats import circstd, circmean
from scritmo import median_dispersion, cSTD
from scipy.sparse import csr_matrix
import scritmo as sr
from scritmo import w, rh, ccg

from context_repo import trainer
from tqdm import tqdm
from torch import tensor as tt
from functools import partial

from .simulations import simulate_data_no_context
from .utils import (
    kappa2circular_std,
    circular_std2kappa,
    get_df_clock,
    get_ext_time,
    results_to_df,
    assemble_mp_simulation,
    get_df_var_inference,
    assign_replicates,
)

circSTD = partial(cSTD, adjust=True)


def _simulate_and_infer_for_group(
    cmodel,
    df_obs_group: pd.DataFrame,
    library_size_group: np.ndarray,
    fourier_coefficients_context: pd.DataFrame,
    context_label: str,
    kappa: float,
    n_cells: int,
    device: str = "cuda",
    post_estimator: str = "post_mode",
    n_theta: int = 48,
):
    """
    Simulates data and infers phases for a SINGLE group of cells.
    This is the core, lean function called repeatedly by the search algorithm.

    Args:
        cmodel: The main context model.
        df_obs_group: The observation metadata for the specific group.
        library_size_group: Library sizes for cells in this group.
        fourier_coefficients_context: Fourier coefficients for the group's context.
        context_label: The context identifier (e.g., 'liver').
        kappa: The von Mises concentration parameter for this simulation run.
        n_cells: The number of cells to simulate for this group.
        device: The compute device.

    Returns:
        np.ndarray: An array of inferred phases for the simulated cells.
    """
    # 1. Prepare simulation inputs
    # Use the real external time of this group as the mean for the von Mises distribution
    ext_time_mean = df_obs_group["ext_time_rad"].values[0]

    # Generate true phases with desynchrony
    if np.isfinite(kappa):
        true_phases = (ext_time_mean + np.random.vonmises(0.0, kappa, size=n_cells)) % (
            2 * np.pi
        )
    else:
        true_phases = np.full(n_cells, ext_time_mean)

    # Sample library sizes with replacement from the real cells in this group
    simulated_library_sizes = np.random.choice(
        library_size_group, size=n_cells, replace=True
    )

    # 2. Simulate gene expression data
    generated_data = simulate_data_no_context(
        phases=true_phases,
        seq_depths=simulated_library_sizes,
        fourier_coefficients=fourier_coefficients_context,
        context_label=context_label,
        noise_model=cmodel.noise_model,
        dispersion=cmodel.disp,
    )

    # 3. Perform phase inference on the simulated data
    N_genes_ct = generated_data.shape[1]
    data_c = torch.tensor(generated_data, dtype=torch.float32, device=device)
    data_c = data_c.unsqueeze(0).expand(cmodel.Nx, n_cells, N_genes_ct)

    # Prepare a minimal model parameter object for this context
    mp_y = {
        "params_g": fourier_coefficients_context,
        "counts": tt(simulated_library_sizes[:, None]),
        "context": None,  # No context needed for this sub-model
    }
    from ..context_model import ContextModel

    model_y = ContextModel(
        mp_y,
        data_c,
        context_mode="none",
        noise_model=cmodel.noise_model,
        fix_phase=cmodel.fix_phase,
        fix_disp_val=cmodel.fix_disp_val,
    )
    model_y.to(device)

    _ = model_y.get_inferred_phases(y=data_c, n_theta=n_theta)

    if post_estimator == "post_mode":
        phi = model_y.post_mode_c
    elif post_estimator == "post_mean":
        phi = model_y.post_mean_c

    return phi


def estimate_desynchrony(
    cmodel,
    adata: anndata.AnnData,
    inferred_phases_real: np.ndarray,
    context_col: str,
    sample_label: str = "sample_name",
    ext_time_label: str = "ZT",
    layer_to_use: str = "spliced",
    n_cells_per_group: int = 300,
    max_iter: int = 30,
    epsilon_h: float = 0.5,
    period: int = 24,
    max_kappa: float = 1000.0,
    device: str = "cuda",
    post_estimator: str = "post_mode",
    circ_dispersion_func=circSTD,
    n_theta: int = 48,
    n_replicates: int | None = None,
    seed: int = 42,
):
    """
    Finds the desynchrony parameter (kappa) for each cell group that best
    matches the circular dispersion of real inferred phases.

    This function uses an efficient, per-group bisection search to avoid
    redundant simulations.

    Args:
        cmodel: The main context model.
        adata: AnnData object containing the data.
        inferred_phases_real: Real inferred phases from the data.
        context_col: Column in adata.obs that defines the context (e.g., 'celltype').
        sample_label: Column in adata.obs that defines the sample (e.g., 'sample_name').
        ext_time_label: Column in adata.obs that contains the external time (e.g., 'ZT').
        layer_to_use: The layer in adata to use for library sizes (e.g., 'spliced').
        n_cells_per_group: Number of cells to simulate for each group.
        max_iter: Maximum number of iterations for the bisection search.
        epsilon_h: Convergence threshold in hours.
        period: The period of the circadian rhythm (default is 24 hours).
        max_kappa: Maximum kappa value to consider for desynchrony.
        device: The compute device to use ('cuda' or 'cpu').
        circ_dispersion_func: Function to compute circular dispersion/std. Default is circSTD.
    """
    # --- 1. Initial Setup and Data Preparation ---

    # Work on a copy of obs to avoid modifying the original AnnData object
    obs = adata.obs.copy()
    obs[context_col] = obs[context_col].astype(str)
    obs[sample_label] = obs[sample_label].astype(str)

    obs["inferred_phases_real"] = inferred_phases_real
    obs["library_size"] = csr_matrix(adata.layers[layer_to_use]).sum(axis=1).A1
    obs["ext_time_rad"] = get_ext_time(
        obs[ext_time_label], period=period, convert_rad=True
    )

    # Define the base groups
    base_group_by_cols = [context_col, sample_label]

    # --- NEW REPLICATE LOGIC ---
    if n_replicates is not None:
        # Assign cells to reproducible subsets
        obs["replicate"] = assign_replicates(
            obs, base_group_by_cols, n_replicates, seed
        )
        # The new, unique groups are defined by the replicates
        group_by_cols = [context_col, sample_label, "replicate"]
    else:
        # Standard behavior, no replicates
        group_by_cols = base_group_by_cols
    # --- END NEW LOGIC ---

    # Define the final list of all unique groups (now includes replicates)
    unique_groups = obs[group_by_cols].drop_duplicates().to_dict("records")

    # Pre-calculate target cStd for *each replicate subset*
    target_cStds = {}
    for group in unique_groups:
        # Build a mask for the specific subset
        mask = pd.Series(True, index=obs.index)
        for col, val in group.items():
            mask &= obs[col] == val

        target_cStds[tuple(group.values())] = circ_dispersion_func(
            obs.loc[mask, "inferred_phases_real"].values
        )

    # Get Fourier coefficients for each context
    fourier_coefficients_all_contexts = cmodel.get_parameter_dataframe_context(
        np.arange(cmodel.Ng)
    )

    # --- 2. Bisection Search Setup ---
    print(f"Initializing bisection search for {len(unique_groups)} total groups...")

    # Convergence threshold in radians
    epsilon_rad = epsilon_h / period * (2 * np.pi)

    # State tracking for each group (now per-replicate)
    search_state = {}
    for group in unique_groups:
        group_key = tuple(group.values())
        search_state[group_key] = {
            "low_kappa": circular_std2kappa(target_cStds[group_key]),
            "high_kappa": max_kappa,
            "final_kappa": None,
            "final_cStd": None,
            "converged": False,
        }

    # --- 3. Iterative Bisection Search ---
    pbar = tqdm(range(max_iter), desc="Bisection search progress")
    for i in pbar:
        converged_count = 0

        for group in unique_groups:  # 'group' is now the per-replicate group
            group_key = tuple(group.values())
            state = search_state[group_key]

            if state["converged"]:
                converged_count += 1
                continue

            # Build a mask for the *specific replicate subset*
            mask = pd.Series(True, index=obs.index)
            for col, val in group.items():
                mask &= obs[col] == val

            df_obs_group = obs.loc[mask]
            # Use library sizes *only from this subset*
            library_size_group = df_obs_group["library_size"].values

            if len(library_size_group) == 0:
                print(f"Warning: Group {group} has 0 cells. Skipping.")
                state["converged"] = True  # Mark as "converged" to skip
                state["final_kappa"] = np.nan
                state["final_cStd"] = np.nan
                converged_count += 1
                continue

            # Get fourier coefficients for the group's context
            fourier_coeffs = fourier_coefficients_all_contexts[group[context_col]]

            # Bisection step
            mid_kappa = (state["low_kappa"] + state["high_kappa"]) / 2

            # Simulate and infer using the subset's library sizes
            inferred_phases_sim = _simulate_and_infer_for_group(
                cmodel,
                df_obs_group,
                library_size_group,
                fourier_coeffs,
                group[context_col],
                mid_kappa,
                n_cells_per_group,
                device,
                post_estimator=post_estimator,
                n_theta=n_theta,
            )
            current_cStd = circ_dispersion_func(inferred_phases_sim)

            # Check for convergence against the *subset's target*
            target_cStd = target_cStds[group_key]
            if abs(current_cStd - target_cStd) < epsilon_rad:
                state["converged"] = True
                state["final_kappa"] = mid_kappa
                state["final_cStd"] = current_cStd
                converged_count += 1
                continue

            # Update bounds
            if current_cStd > target_cStd:
                state["low_kappa"] = mid_kappa
            else:
                state["high_kappa"] = mid_kappa

        pbar.set_description(
            f"Bisection search progress (Converged: {converged_count}/{len(unique_groups)})"
        )
        if converged_count == len(unique_groups):
            print("\nAll groups converged.")
            break

    if i == max_iter - 1:
        print("\nMax iterations reached. Some groups may not have converged.")

    # --- 4. Finalize and Return Results ---
    results = []
    rad_unit = period / (2 * np.pi)

    for group in unique_groups:  # 'group' is the per-replicate group
        group_key = tuple(group.values())
        state = search_state[group_key]

        final_kappa = (
            state["final_kappa"]
            if state["converged"]
            else (state["low_kappa"] + state["high_kappa"]) / 2
        )

        # Get external time from this subset
        mask = pd.Series(True, index=obs.index)
        for col, val in group.items():
            mask &= obs[col] == val

        ext_time_rad = obs.loc[mask, "ext_time_rad"].values[0]
        ext_time_hours = (ext_time_rad * period) / (2 * np.pi)

        # --- Build Replicate Name ---
        context = group[context_col]
        original_sample_name = group[sample_label]

        if n_replicates is not None:
            replicate_num = group["replicate"] + 1
            output_sample_name = f"{original_sample_name}_{replicate_num}"
        else:
            output_sample_name = original_sample_name
        # --- End Build Name ---

        res_row = {
            "context": context,
            sample_label: output_sample_name,  # Use the original col name, but new value
            "ext_time_hours": ext_time_hours,
            "Data_cSTD": target_cStds[group_key] * rad_unit,  # The replicate's target
            "estimated_kappa": final_kappa,
            "cStd_inferred": (state["final_cStd"] or np.nan) * rad_unit,
            "Bio_cSTD": (
                kappa2circular_std(final_kappa) * rad_unit if final_kappa > 0 else np.inf
            ),
            "converged": state["converged"],
        }
        results.append(res_row)

    return pd.DataFrame(results)

def quadrature_desynchrony_estimate(sigma_data, sigma_tech):
    return np.sqrt(sigma_data**2 - sigma_tech**2)