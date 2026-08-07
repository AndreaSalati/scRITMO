import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from ..misc.power_spherical.torch_distribution import PowerSpherical
from ..utils import df2dict, nmp
import seaborn as sns
import anndata
from scipy.stats import circstd, circmean

from scipy.sparse import csr_matrix
import scritmo as sr
from scritmo import w, rh, ccg
import seaborn as sns
from .. import trainer
from tqdm import tqdm
import numpy as np
from torch import tensor as tt
from ..trainer import train_ritmo
from matplotlib import pyplot as plt
from .simulations import simulate_data_no_context

from .utils import (
    circular_std,
    get_df_clock,
    get_ext_time,
    results_to_df,
    assemble_mp_simulation,
    get_df_var_inference,
    assign_replicates,
)


def _infer_phases_for_context(
    generated_data: np.ndarray,
    library_sizes: np.ndarray,
    fourier_coefficients: pd.DataFrame,
    cmodel,
    device: str,
    context_label: str,
    simulated_sample_names: list,
    s_ext_time: list,
    s_true_time: list,
    n_epochs_training: int = 0,
    posterior_cell_chunk: int | None = None,
):
    """
    Performs phase inference and returns structured results as a list of dictionaries.
    """
    from ..context_model import ContextModel

    N_cell_ct, N_genes_ct = generated_data.shape
    data_c = torch.tensor(generated_data, dtype=torch.float32, device=device)
    data_c = data_c.unsqueeze(0).expand(cmodel.Nx, N_cell_ct, N_genes_ct)

    # Prepare model parameters for this specific context
    mp_y = {}
    # Assumes 'tt' is available in the scope
    mp_y["counts"] = tt(library_sizes[:, None])
    mp_y["context"] = None
    mp_y["params_g"] = fourier_coefficients
    mp_y["disp"] = cmodel.disp

    model_y = ContextModel(
        mp_y,
        data_c,
        context_mode="none",
        fix_phase=cmodel.fix_phase,
        noise_model=cmodel.noise_model,
        fix_disp_val=cmodel.fix_disp_val,
        log_amp_fn=cmodel.log_amp_fn,
    )
    model_y.to(device)

    if n_epochs_training > 0:
        losses = train_ritmo(
            model=model_y,
            data=data_c,
            # data_u=data_u_c,
            n_epochs=n_epochs_training,
            batch_size=128,
        )
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    _ = model_y.get_inferred_phases(
        data_c, n_theta=100, cell_chunk=posterior_cell_chunk
    )

    # Capture both mean and standard deviation
    post_mean_c = model_y.post_mean_c
    post_std_c = model_y.post_std_c
    post_mode_c = model_y.post_mode_c

    # Build the list of dictionaries
    # myabe call the utils function here create_results_dataframe
    results = []
    for i in range(len(post_mean_c)):
        results.append(
            {
                "post_mean": post_mean_c[i],
                "post_std": post_std_c[i],
                "context": context_label,
                "sample_name": simulated_sample_names[i],
                "ext_time": s_ext_time[i],
                "true_time": s_true_time[i],
                "post_mode": post_mode_c[i],
            }
        )

    # Explicit GPU cleanup to avoid accumulation across context iterations
    del model_y, data_c
    torch.cuda.empty_cache()

    return results


def simulate_cell_populations(
    cmodel,
    adata,
    context_col: str | None = None,
    n_cells: int | None = None,  # If None, defaults to exact number of cells in group
    layer_to_use="spliced",
    ext_time_label="ZT",
    sample_label="sample_name",
    kappa=np.inf,
    period=24,
    device="cuda",
    return_sim_data=False,
    n_epochs_training=0,
    n_replicates: int | None = None,
    seed_replicates: int = 42,
    seed_sim: int | None = None,
    library_size_vec: np.ndarray | None = None,
    n_sim_runs: int = 5,  # NEW: Number of "Twin" simulations to run
    use_circular_mean: bool = False,
    posterior_cell_chunk: int | None = None,
):
    # --- 1. Initial Setup ---
    fourier_coefficients_y = cmodel.get_parameter_dataframe_context(
        np.arange(cmodel.Ng)
    )
    obs = adata.obs.copy()

    if context_col is None:
        context_col = "context"
        context_val = list(cmodel.get_parameter_dataframe_context().keys())[0]
        obs[context_col] = context_val

    obs[context_col] = obs[context_col].astype(str)
    obs[sample_label] = obs[sample_label].astype(str)

    if library_size_vec is None:
        obs["library_size"] = csr_matrix(adata.layers[layer_to_use]).sum(axis=1).A1
    else:
        obs["library_size"] = library_size_vec

    obs["ext_time_rad"] = get_ext_time(
        adata.obs[ext_time_label], period=period, convert_rad=True
    )

    if use_circular_mean:
        obs["inferred_phase"] = cmodel.post_mode_c

    if seed_sim is not None:
        torch.manual_seed(seed_sim)
        np.random.seed(seed_sim)

    # Replicate assignment logic
    base_group_by_cols = [context_col, sample_label]
    if n_replicates is not None:
        obs["replicate"] = assign_replicates(
            obs, base_group_by_cols, n_replicates, seed_replicates
        )
        group_by_cols = base_group_by_cols + ["replicate"]
    else:
        group_by_cols = base_group_by_cols

    all_results_list = []
    unique_contexts = obs[context_col].unique()

    # --- 2. NESTED LOOP: Context (Outer) -> Samples (Inner) ---
    for context_label in unique_contexts:
        print(f"Simulating context: {context_label}")
        # Containers for all samples within THIS context
        ctx_gen_data = []
        ctx_lib_sizes = []
        ctx_sample_names = []
        ctx_ext_times = []
        ctx_true_times = []

        # Get all sample groups belonging to this context
        context_mask = obs[context_col] == context_label
        unique_groups_in_ctx = (
            obs[context_mask][group_by_cols].drop_duplicates().to_dict("records")
        )

        for group in unique_groups_in_ctx:
            # Mask for this specific sample/replicate
            mask = pd.Series(True, index=obs.index)
            for col, val in group.items():
                mask &= obs[col] == val

            df_obs_group = obs.loc[mask]
            library_size_group = df_obs_group["library_size"].values
            if use_circular_mean:
                ext_time_mean = circmean(
                    df_obs_group["inferred_phase"].values, high=2 * np.pi, low=0
                )
            else:
                ext_time_mean = df_obs_group["ext_time_rad"].values[0]

            if len(library_size_group) == 0:
                continue

            ############################

            # 1. Determine Sample Size & Library Sizes
            real_lib_sizes = df_obs_group["library_size"].values
            n_real = len(real_lib_sizes)

            # Decide on N for simulation
            current_n_cells = n_cells if n_cells is not None else n_real

            # 2. REPEAT SIMULATION LOOP
            for run_idx in range(n_sim_runs):

                # Library Size Handling: Exact Match vs Sampling
                if current_n_cells == n_real and library_size_vec is None:
                    # EXACT MATCH: Best for "Twin" comparison
                    sim_lib_sizes = real_lib_sizes
                else:
                    # BOOTSTRAP: Fallback if N differs
                    sim_lib_sizes = np.random.choice(
                        real_lib_sizes, size=current_n_cells, replace=True
                    )

                if np.all(np.isfinite(kappa)):
                    true_phases = (
                        ext_time_mean
                        + np.random.vonmises(0.0, kappa, size=current_n_cells)
                    ) % (2 * np.pi)
                else:
                    true_phases = np.full(current_n_cells, ext_time_mean)

                fourier_coeffs = fourier_coefficients_y[context_label]
                generated_data = simulate_data_no_context(
                    phases=true_phases,
                    seq_depths=sim_lib_sizes,
                    fourier_coefficients=fourier_coeffs,
                    context_label=context_label,
                    noise_model=cmodel.noise_model,
                    dispersion=cmodel.disp,
                )

                # Metadata name
                if n_replicates is not None:
                    base_name = f"{group[sample_label]}_{group['replicate'] + 1}"
                else:
                    base_name = group[sample_label]

                output_sample_name = f"{base_name}_run{run_idx}"

                # Append to context lists
                ctx_gen_data.append(generated_data)
                ctx_lib_sizes.append(sim_lib_sizes)
                ctx_sample_names.extend([output_sample_name] * current_n_cells)
                ctx_ext_times.extend([ext_time_mean] * current_n_cells)
                ctx_true_times.extend(list(true_phases))

        # --- 3. Batch Inference for the whole Context ---
        if ctx_gen_data:
            context_results = _infer_phases_for_context(
                generated_data=np.concatenate(ctx_gen_data, axis=0),
                library_sizes=np.concatenate(ctx_lib_sizes, axis=0),
                fourier_coefficients=fourier_coefficients_y[context_label],
                cmodel=cmodel,
                device=device,
                context_label=context_label,
                simulated_sample_names=ctx_sample_names,
                s_ext_time=ctx_ext_times,
                s_true_time=ctx_true_times,
                n_epochs_training=n_epochs_training,
                posterior_cell_chunk=posterior_cell_chunk,
            )
            all_results_list.extend(context_results)

    # --- 4. Finalize ---
    if not all_results_list:
        return pd.DataFrame(
            columns=["post_mean", "post_mode", "post_std", "context", "sample_name"]
        )

    return pd.DataFrame(all_results_list)


def simulate_technical_grid(
    cmodel,
    adata,
    context_col: str | None = None,
    layer_to_use="spliced",
    n_grid: int = 12,
    n_cells_per_gridpoint: int = 1000,
    period=24,
    device="cuda",
    n_sim_runs: int = 1,
    library_size_vec: np.ndarray | None = None,
    seed_sim: int | None = None,
    posterior_cell_chunk: int | None = None,
):
    """Twin-population grid for the phase-resolved ("harmonic") technical floor.

    σ_tech is itself phase-dependent (large near the Bmal1 trough, small at high
    expression). This builds a perfectly-synchronized ("twin") population at each of
    ``n_grid`` common phases evenly spaced over [0, 2π), re-infers phases, and returns the
    inferred phases per grid point so the caller can fit σ_tech²(φ) (see
    :func:`scritmo.ml.analysis_utils.aggregate_technical_harmonic`).

    Every twin population is common-phase (σ_bio = 0) -- that is what makes its inferred
    spread a pure noise floor. Library sizes are POOLED across all cells of ``adata`` (all
    biological replicates of the celltype) and resampled at each grid point, so the floor's
    phase shape reflects the dataset's overall depth distribution. Reuses
    :func:`simulate_data_no_context` for generation and :func:`_infer_phases_for_context`
    for inference -- the same primitives as :func:`simulate_cell_populations`.

    Parameters mirror :func:`simulate_cell_populations` where shared. ``n_grid`` grid points,
    ``n_cells_per_gridpoint`` twin cells per (grid point, run), ``n_sim_runs`` independent
    runs per grid point (more runs -> more fit points for the 2-harmonic OLS).

    Returns
    -------
    pandas.DataFrame
        One row per simulated twin cell, columns ``context``, ``grid_idx``, ``grid_phase``
        (injected common phase φ_k, rad), ``run_id``, ``post_mode``, ``post_mean``,
        ``post_std``, ``sample_name`` (``f"grid{k}_run{r}"``).
    """
    fourier_coefficients_y = cmodel.get_parameter_dataframe_context(np.arange(cmodel.Ng))
    obs = adata.obs.copy()

    if context_col is None:
        context_col = "context"
        context_val = list(fourier_coefficients_y.keys())[0]
        obs[context_col] = context_val

    obs[context_col] = obs[context_col].astype(str)

    # Library-size pool for the twins. The size factor MUST be the genome-wide library (what
    # a_0 was calibrated against), NOT the modelled-gene (e.g. clock-gene) sum -- the latter is
    # tiny (~3-30 counts) and phase-dependent, so generating twins with it produces ~0 counts ->
    # near-uniform posteriors -> an artificially small/biased technical floor. (This is a
    # recurrent foot-gun; see CLAUDE.md.) Preference order:
    #   1. explicit library_size_vec (callers pass the constant generative seq_depth for the
    #      subset-of-genome sims, exactly like simulate_cell_populations);
    #   2. cmodel.counts -- the size factor the model was actually fit with, genome-wide and
    #      self-consistent with a_0 (same source the Cramér-Rao twin uses), when it aligns 1:1
    #      with adata;
    #   3. fallback: adata.layers[layer].sum(1) -- correct ONLY if adata holds the full genome.
    if library_size_vec is not None:
        obs["library_size"] = library_size_vec
    else:
        model_counts = nmp(cmodel.counts).reshape(-1).astype(float)
        if model_counts.shape[0] == adata.n_obs:
            obs["library_size"] = model_counts
        else:
            obs["library_size"] = csr_matrix(adata.layers[layer_to_use]).sum(axis=1).A1

    if seed_sim is not None:
        torch.manual_seed(seed_sim)
        np.random.seed(seed_sim)

    # grid of common phases over the full cycle
    grid_phases = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)

    all_results_list = []
    unique_contexts = obs[context_col].unique()

    for context_label in unique_contexts:
        print(f"Simulating technical grid for context: {context_label}")
        # library-size pool for THIS context (all its cells = all replicates)
        lib_pool = obs.loc[obs[context_col] == context_label, "library_size"].values
        if len(lib_pool) == 0:
            continue

        fourier_coeffs = fourier_coefficients_y[context_label]

        ctx_gen_data = []
        ctx_lib_sizes = []
        ctx_sample_names = []
        ctx_grid_phase = []  # carried via s_ext_time (injected common phase)
        ctx_true_times = []

        for k, phi_k in enumerate(grid_phases):
            for run_idx in range(n_sim_runs):
                sim_lib_sizes = np.random.choice(
                    lib_pool, size=n_cells_per_gridpoint, replace=True
                )
                # common-phase twin: every cell shares phi_k (sigma_bio = 0)
                true_phases = np.full(n_cells_per_gridpoint, phi_k)

                generated_data = simulate_data_no_context(
                    phases=true_phases,
                    seq_depths=sim_lib_sizes,
                    fourier_coefficients=fourier_coeffs,
                    context_label=context_label,
                    noise_model=cmodel.noise_model,
                    dispersion=cmodel.disp,
                )

                output_sample_name = f"grid{k}_run{run_idx}"
                ctx_gen_data.append(generated_data)
                ctx_lib_sizes.append(sim_lib_sizes)
                ctx_sample_names.extend([output_sample_name] * n_cells_per_gridpoint)
                ctx_grid_phase.extend([phi_k] * n_cells_per_gridpoint)
                ctx_true_times.extend(list(true_phases))

        if not ctx_gen_data:
            continue

        # single batched inference for the whole context (reuses the sim inference path)
        context_results = _infer_phases_for_context(
            generated_data=np.concatenate(ctx_gen_data, axis=0),
            library_sizes=np.concatenate(ctx_lib_sizes, axis=0),
            fourier_coefficients=fourier_coeffs,
            cmodel=cmodel,
            device=device,
            context_label=context_label,
            simulated_sample_names=ctx_sample_names,
            s_ext_time=ctx_grid_phase,  # injected common phase phi_k
            s_true_time=ctx_true_times,
            n_epochs_training=0,
            posterior_cell_chunk=posterior_cell_chunk,
        )
        all_results_list.extend(context_results)

    if not all_results_list:
        return pd.DataFrame(
            columns=[
                "context", "grid_idx", "grid_phase", "run_id",
                "post_mode", "post_mean", "post_std", "sample_name",
            ]
        )

    df_grid = pd.DataFrame(all_results_list)
    # 'ext_time' holds the injected common phase phi_k; 'sample_name' encodes grid/run
    df_grid = df_grid.rename(columns={"ext_time": "grid_phase"})
    df_grid["grid_idx"] = df_grid["sample_name"].str.extract(r"grid(\d+)_").astype(int)
    df_grid["run_id"] = df_grid["sample_name"].str.extract(r"(run\d+)$")
    return df_grid
