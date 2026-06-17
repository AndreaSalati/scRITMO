import pandas as pd
import numpy as np
import scritmo as sr
import anndata
from scritmo import w, rh
from .simulations.utils import assign_replicates
from scritmo import cstd2R, R2cstd
import matplotlib.pyplot as plt
import seaborn as sns


def create_results_dataframe(
    cmodel,
    adata: anndata.AnnData,
    ext_phase: None | np.ndarray = None,
    context_col: None = None,
    sample_col: str = "sample_name",
    ext_time_col: str = "ZTmod",
    post_estimator: str = "post_mode",
    layer="spliced",
    other_obs_cols: list = [],
    allow_flip: bool = False,
):
    """
    Creates the main results DataFrame (df_res) from a trained ContextModel.
    (This function is unchanged)
    """

    # Create the DataFrame
    df_res = pd.DataFrame()
    df_res.index = adata.obs.index

    if post_estimator == "post_mode":
        phi = cmodel.post_mode_c
    else:
        phi = cmodel.post_mean_c

    post_std_c = cmodel.post_std_c

    if ext_phase is not None:
        # Align phases and calculate MAE (cad)
        df_res["true_phase"] = ext_phase
        phi_aligned, best_mad = sr.optimal_shift(phi, ext_phase, allow_flip=allow_flip)
        cad = sr.circular_deviation(ext_phase, phi_aligned, period=2 * np.pi) * rh
        df_res["MAE"] = cad
        print(f"Created df_res. Median MAE: {best_mad*rh:.2f} hours")

    # Re-compute ccounts
    # Use intersection to be safe
    ccg_genes = np.intersect1d(cmodel.genes, adata.var_names)
    # ccounts = np.array(adata[:, ccg_genes].layers[layer].sum(axis=1)).squeeze()

    if context_col is not None and context_col in adata.obs.columns:
        df_res["context"] = adata.obs[context_col].values
    else:
        print("WARNING: context column not present")

    df_res["pred_phase"] = phi
    df_res["pred_phase_h"] = phi * rh
    df_res["post_mode"] = cmodel.post_mode_c
    df_res["post_mean"] = cmodel.post_mean_c
    df_res["post_std_c"] = post_std_c
    # df_res["counts"] = np.array(adata.layers[layer].sum(1)).squeeze()
    # df_res["ccounts"] = ccounts
    if ext_time_col in adata.obs.columns:
        df_res["ext_time_hours"] = adata.obs[ext_time_col].values
    if sample_col in adata.obs.columns:
        df_res["sample_name"] = adata.obs[sample_col].values

    for col in other_obs_cols:
        df_res[col] = adata.obs[col].values

    return df_res


def desync_results(
    df_real,
    df_sim,
    group_cols: list = None,
    disp_function=sr.cSTD,
    post_estimator: str = "post_mode",
    # real arguments
    metrics: dict = None,
    n_replicates: int | None = None,
    seed: int = 42,
    # sim arguments
    ext_time_col: str = "ext_time",
    # weighting
    weight_col: str | None = None,
    # precomputed technical floor (Cramér–Rao branch): if given, df_sim is ignored and this
    # per-(context, sample) table supplies Technical_cSTD/Technical_R directly.
    sim_agg: pd.DataFrame | None = None,
):
    """
    First it aggregates data by calling aggregate_real_results and aggregate_simulated_results,
    then fuses the 2 in one dataframe. Finally it computes the
    biological desynchrony with the quadrature difference.

    `sim_agg` lets a caller bypass the simulation twin: pass a precomputed technical table (same
    schema as `aggregate_simulated_results`: context, sample_name, Technical_cSTD[rad], Technical_R)
    and `df_sim` is not used. This is the analytic Cramér–Rao path (`aggregate_technical_rao`).

    TO BE FIXED: Currently the group cols can only be two: context and sample_name.
    columns with another names will create problems
    """
    # 0. Handle Defaults
    if group_cols is None:
        group_cols = ["context", "sample_name"]
    else:
        print(
            "Warning: Currently only ['context', 'sample_name'] are supported as group_cols."
        )
    # 1. Aggregate Data
    real_agg = aggregate_real_results(
        df_real,
        group_cols=group_cols,
        disp_function=disp_function,
        post_estimator=post_estimator,
        metrics=metrics,
        n_replicates=n_replicates,
        seed=seed,
        weight_col=weight_col,
    )

    if sim_agg is None:
        sim_agg = aggregate_simulated_results(
            df_sim,
            # group_cols=group_cols,
            disp_function=disp_function,
            post_estimator=post_estimator,
            ext_time_col=ext_time_col,
            weight_col="post_std" if weight_col is not None else None,
        )

    # 2. Initialize Mixed DataFrame
    df_mixed = real_agg.copy()

    # 3. Create Lookup Logic
    # We index the simulated data by the grouping columns for fast mapping
    try:
        sim_lookup = sim_agg.set_index(group_cols)
    except KeyError:
        print(f"Error: 'sim_agg' is missing one of the group_cols: {group_cols}")
        return df_mixed

    # Create the mapping index from the target dataframe (df_mixed)
    # This ensures alignment even if the sort order differs
    try:
        map_index = pd.MultiIndex.from_frame(df_mixed[group_cols])
    except KeyError:
        print(f"Error: 'df_mixed' is missing one of the group_cols: {group_cols}")
        return df_mixed

    # 4. Map Technical Component
    # We pull 'Technical_cSTD' from the sim results into the real results
    if "Technical_cSTD" in sim_lookup.columns:
        df_mixed["Technical_cSTD"] = map_index.map(sim_lookup["Technical_cSTD"])
    if "Technical_R" in sim_lookup.columns:
        df_mixed["Technical_R"] = map_index.map(sim_lookup["Technical_R"])

    # multiply by rh both Technical and Data cSTD to convert to hours
    df_mixed["Technical_cSTD"] = df_mixed["Technical_cSTD"] * rh
    df_mixed["Data_cSTD"] = df_mixed["Data_cSTD"] * rh

    # 5. Compute Biological Desynchrony (Quadrature Difference)

    df_mixed["Bio_cSTD"] = np.sqrt(
        df_mixed["Data_cSTD"] ** 2 - df_mixed["Technical_cSTD"] ** 2
    )
    df_mixed["Bio_R"] = cstd2R(df_mixed["Bio_cSTD"] / rh)

    return df_mixed


def _weighted_se_bio_cSTD(bio_vars, weights, final_bio_var):
    """
    Compute the standard error of the biological cSTD (circular standard deviation)
    from per-replicate biological variance estimates, using:
    1. Weighted SEM on the variance (with Kish's effective sample size)
    2. Delta method to propagate to the sqrt (cSTD) scale

    Parameters
    ----------
    bio_vars : np.ndarray
        Per-replicate biological variances (Data_cSTD^2 - Technical_cSTD^2), clipped >= 0.
    weights : np.ndarray
        Per-replicate weights (e.g. group_size).
    final_bio_var : float
        The weighted mean of bio_vars (i.e. the point estimate of the biological variance).

    Returns
    -------
    float
        Standard error of sqrt(final_bio_var), or NaN if not computable.
    """
    n_reps = len(bio_vars)
    if n_reps <= 1 or final_bio_var <= 0:
        return np.nan

    # Weighted variance of the per-replicate bio variances
    weighted_mean = np.average(bio_vars, weights=weights)
    weighted_var = np.average((bio_vars - weighted_mean) ** 2, weights=weights)

    # Kish's effective sample size
    n_eff = np.sum(weights) ** 2 / np.sum(weights**2)

    # Weighted SEM on the variance
    se_var = np.sqrt(weighted_var / n_eff)

    # Delta method: SE(sqrt(V)) = SE(V) / (2 * sqrt(V))
    se_bio_cSTD = se_var / (2 * np.sqrt(final_bio_var))

    return se_bio_cSTD


def desync_means(df_desync):
    context_u = df_desync["context"].unique()
    df_desync["Technical_cSTD2"] = df_desync["Technical_cSTD"] ** 2
    df_desync["Data_cSTD2"] = df_desync["Data_cSTD"] ** 2
    df_desync["Bio_Var"] = np.maximum(
        df_desync["Data_cSTD2"] - df_desync["Technical_cSTD2"], 0
    )

    results = []
    for ct in context_u:
        df_ct = df_desync[df_desync["context"] == ct]
        if "organ" in df_ct.columns:
            organ = df_ct["organ"].iloc[0]
        else:
            organ = None
        if "celltype" in df_ct.columns:
            celltype = df_ct["celltype"].iloc[0]
        else:
            celltype = None
        if "condition" in df_ct.columns:
            condition = df_ct["condition"].iloc[0]
        else:
            condition = None

        weights = df_ct["group_size"].values

        weighted_mean_technical_var = np.average(
            df_ct["Technical_cSTD2"], weights=weights
        )
        weighted_mean_data_var = np.average(df_ct["Data_cSTD2"], weights=weights)

        final_bio_var = np.maximum(
            weighted_mean_data_var - weighted_mean_technical_var, 0
        )
        final_bio_cSTD = np.sqrt(final_bio_var)
        final_technical_cSTD = np.sqrt(weighted_mean_technical_var)
        final_data_cSTD = np.sqrt(weighted_mean_data_var)

        se_bio_cSTD = _weighted_se_bio_cSTD(
            df_ct["Bio_Var"].values, weights, final_bio_var
        )

        results.append(
            {
                "ct": ct,
                "Technical_cSTD": final_technical_cSTD,
                "Bio_cSTD": final_bio_cSTD,
                "Bio_cSTD_SE": se_bio_cSTD,
                "Data_cSTD": final_data_cSTD,
                "organ": organ,
                "celltype": celltype,
                "condition": condition,
            }
        )

    df_summary = pd.DataFrame(results).set_index("ct")[
        [
            "Technical_cSTD",
            "Bio_cSTD",
            "Bio_cSTD_SE",
            "Data_cSTD",
            "organ",
            "celltype",
            "condition",
        ]
    ]
    return df_summary


def aggregate_real_results(
    df_res: pd.DataFrame,
    group_cols: list = None,
    disp_function=sr.cSTD,
    post_estimator: str = "post_mode",
    metrics: dict = None,
    n_replicates: int | None = None,
    seed: int = 42,
    weight_col: str | None = None,
):
    """
    Aggregates the 'df_res' DataFrame.
    If n_replicates is provided, it first splits the data into
    reproducible subsets and aggregates over them.
    """
    if group_cols is None:
        group_cols = ["context", "sample_name"]

    if metrics is None:
        metrics = {
            "MAE": "median",
            "post_std_c": "median",
            post_estimator: disp_function,
            "ext_time_hours": "first",
        }

    # Only aggregate columns that are present in the DataFrame
    metrics = {k: v for k, v in metrics.items() if k in df_res.columns}

    # --- NEW REPLICATE LOGIC ---
    df_to_agg = df_res.copy()

    # Ensure columns are string type for grouping
    for col in group_cols:
        df_to_agg[col] = df_to_agg[col].astype(str)

    if n_replicates is not None:
        # Use the same reproducible subsetting
        df_to_agg["replicate"] = assign_replicates(
            df_to_agg, group_cols, n_replicates, seed
        )
        # Add 'replicate' to the grouping
        group_cols_with_rep = group_cols + ["replicate"]
    else:
        group_cols_with_rep = group_cols

    # Aggregate the metrics
    agg_df = df_to_agg.groupby(group_cols_with_rep).agg(metrics).reset_index()
    # change name from pred_phase to Data_cSTD if present
    if post_estimator in agg_df.columns:
        agg_df = agg_df.rename(columns={post_estimator: "Data_cSTD"})
        agg_df["Data_R"] = cstd2R(agg_df["Data_cSTD"])

    # Override Data_cSTD with weighted version (w = 1/post_std_c) if requested
    if weight_col is not None and weight_col in df_to_agg.columns:
        def _weighted_cstd(group_df):
            phases = group_df[post_estimator].values
            w = 1.0 / np.maximum(group_df[weight_col].values, 1e-10)
            w /= w.sum()
            R_val = float(np.abs(np.sum(w * np.exp(1j * phases))))
            return float(np.sqrt(-2 * np.log(R_val + 1e-10)))

        wcstd_series = df_to_agg.groupby(group_cols_with_rep).apply(_weighted_cstd)
        wcstd_series.name = "Data_cSTD"
        wcstd_df = wcstd_series.reset_index()
        agg_df = agg_df.drop(columns=["Data_cSTD", "Data_R"], errors="ignore")
        agg_df = agg_df.merge(wcstd_df, on=group_cols_with_rep)
        agg_df["Data_R"] = cstd2R(agg_df["Data_cSTD"])

    # Get group sizes
    group_sizes = (
        df_to_agg.groupby(group_cols_with_rep).size().reset_index(name="group_size")
    )

    # Merge metrics and group sizes
    agg_df = agg_df.merge(group_sizes, on=group_cols_with_rep)

    # Create the replicate name in the output
    if n_replicates is not None:
        sample_col = group_cols[-1]  # Assumes sample_name is the last
        agg_df[sample_col] = (
            agg_df[sample_col].astype(str) + "_" + (agg_df["replicate"] + 1).astype(str)
        )
        agg_df = agg_df.drop(columns=["replicate"])

    return agg_df


def aggregate_simulated_results(
    df_sim: pd.DataFrame,
    disp_function=sr.cSTD,
    post_estimator: str = "post_mode",
    ext_time_col: str = "ext_time",
    weight_col: str | None = None,
):
    # Extract base sample name (removing _runX) and run id
    df_sim["base_sample"] = df_sim["sample_name"].str.replace(
        r"_run\d+$", "", regex=True
    )
    df_sim["run_id"] = df_sim["sample_name"].str.extract(r"(run\d+)$")

    run_groups = ["context", "base_sample", "run_id"]

    # 2. First Aggregation: Calculate Variance PER RUN
    if weight_col is not None and weight_col in df_sim.columns:
        def _weighted_cstd_run(group_df):
            phases = group_df[post_estimator].values
            w = 1.0 / np.maximum(group_df[weight_col].values, 1e-10)
            w /= w.sum()
            R_val = float(np.abs(np.sum(w * np.exp(1j * phases))))
            return float(np.sqrt(-2 * np.log(R_val + 1e-10)))

        wcstd_series = df_sim.groupby(run_groups).apply(_weighted_cstd_run)
        wcstd_series.name = "cSTD_run"
        post_std_mean = df_sim.groupby(run_groups)["post_std"].mean()
        run_level_stats = pd.concat([wcstd_series, post_std_mean], axis=1).reset_index()
    else:
        run_level_stats = (
            df_sim.groupby(run_groups)
            .agg(
                {
                    post_estimator: disp_function,  # This computes cSTD for one run
                    "post_std": "mean",
                }
            )
            .reset_index()
        )
        run_level_stats = run_level_stats.rename(columns={post_estimator: "cSTD_run"})

    # Convert to Variance (Statistics must be done on Variance, not STD)
    run_level_stats["Var_run"] = run_level_stats["cSTD_run"] ** 2

    # 3. Second Aggregation: Average the VARIANCES across runs
    # This collapses N runs -> 1 final technical value
    final_stats = (
        run_level_stats.groupby(["context", "base_sample"])
        .agg(
            {
                "Var_run": "mean",  # Mean of Variances
                "cSTD_run": "mean",  # (Optional) Mean of STDs for reference, but don't use for math
                "post_std": "mean",
            }
        )
        .reset_index()
    )

    # 4. Convert back to cSTD for the final output (if needed for the dataframe format)
    # But remember the "Twin" logic: we want the squared term.
    final_stats["Technical_cSTD"] = np.sqrt(final_stats["Var_run"])

    # Cleanup for merging
    final_stats = final_stats.rename(columns={"base_sample": "sample_name"})

    # Add the R conversion if needed
    final_stats["Technical_R"] = cstd2R(final_stats["Technical_cSTD"])

    return final_stats


def aggregate_technical_rao(
    df_real: pd.DataFrame,
    group_cols: list = None,
    sigma_col: str = "sigma_tech_rao",
):
    """Aggregate the per-cell analytic Cramér–Rao σ_tech into a per-(context, sample) technical
    floor, with the SAME output schema as `aggregate_simulated_results` (context, sample_name,
    Technical_cSTD[rad], Technical_R) so `desync_results(..., sim_agg=...)` consumes it unchanged.

    Sample-level Technical_cSTD is the circular-mixture spread of the per-cell wrapped-normal MAP
    estimators (`cramer_rao.technical_cstd_rao`): R̄ = mean_i exp(−σ_i²/2), cSTD = √(−2 ln R̄).
    """
    from .cramer_rao import technical_cstd_rao

    if group_cols is None:
        group_cols = ["context", "sample_name"]

    final_stats = (
        df_real.groupby(group_cols)[sigma_col]
        .apply(lambda s: technical_cstd_rao(s.values))
        .reset_index(name="Technical_cSTD")
    )
    final_stats["Technical_R"] = cstd2R(final_stats["Technical_cSTD"])
    return final_stats


def cSTD_posterior_mixture(posterior_xc: np.ndarray) -> float:
    """
    Compute the circular standard deviation of the mixture of posterior distributions.

    Parameters
    ----------
    posterior_xc : np.ndarray, shape (Nx, Nc)
        Normalized posterior distributions. Each column is a probability
        distribution over the phase grid [0, 2π) for one cell.

    Returns
    -------
    float
        Circular standard deviation of the population mixture (in radians).
    """
    # Average over cells → mixture distribution, shape (Nx,)
    mixture = posterior_xc.mean(axis=1)
    mixture = mixture / mixture.sum()

    Nx = len(mixture)
    # Grid endpoint=False, consistent with how posteriors are computed
    phis = np.linspace(0, 2 * np.pi, Nx, endpoint=False)

    R = np.abs(np.sum(mixture * np.exp(1j * phis)))
    cstd = float(np.sqrt(-2 * np.log(R + 1e-10)))
    return cstd


def aggregate_real_results_posterior(
    posterior_xc: np.ndarray,
    df_real: pd.DataFrame,
    group_cols: list = None,
    n_replicates: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Aggregate real data desynchrony using full posterior distributions.

    Parameters
    ----------
    posterior_xc : np.ndarray, shape (Nx, Nc)
        Full posteriors from get_inferred_phases; columns aligned with df_real rows.
    df_real : pd.DataFrame
        Results DataFrame from create_results_dataframe.
    group_cols : list
        Columns to group by (default: ["context", "sample_name"]).
    n_replicates : int, optional
        If set, bootstrap into this many sub-samples per group.
    seed : int
        Random seed for replicate assignment.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with Data_cSTD (radians) and Data_R per group.
    """
    if group_cols is None:
        group_cols = ["context", "sample_name"]

    df_to_agg = df_real.copy().reset_index(drop=True)
    df_to_agg["_pos"] = np.arange(len(df_to_agg))

    for col in group_cols:
        df_to_agg[col] = df_to_agg[col].astype(str)

    if n_replicates is not None:
        df_to_agg["replicate"] = assign_replicates(
            df_to_agg, group_cols, n_replicates, seed
        )
        group_cols_with_rep = group_cols + ["replicate"]
    else:
        group_cols_with_rep = group_cols

    results = []
    for group_key, group_df in df_to_agg.groupby(group_cols_with_rep):
        positions = group_df["_pos"].values
        group_posterior = posterior_xc[:, positions]
        data_cstd = cSTD_posterior_mixture(group_posterior)

        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_cols_with_rep, group_key))
        row["Data_cSTD"] = data_cstd
        row["Data_R"] = float(cstd2R(data_cstd))
        row["group_size"] = len(positions)
        if "ext_time_hours" in group_df.columns:
            row["ext_time_hours"] = group_df["ext_time_hours"].iloc[0]
        if "post_std_c" in group_df.columns:
            row["post_std_c"] = float(group_df["post_std_c"].median())
        results.append(row)

    agg_df = pd.DataFrame(results)

    if n_replicates is not None:
        sample_col = group_cols[-1]
        agg_df[sample_col] = (
            agg_df[sample_col].astype(str) + "_" + (agg_df["replicate"] + 1).astype(str)
        )
        agg_df = agg_df.drop(columns=["replicate"])

    return agg_df


def aggregate_simulated_results_posterior(
    posteriors_dict: dict,
    df_sim: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate simulated data using full posterior distributions.

    Parameters
    ----------
    posteriors_dict : dict
        Dict with keys 'posterior_xc' (Nx, N_total), 'sample_name' (list, N_total),
        'context' (list, N_total). Columns of posterior_xc align with df_sim rows.
    df_sim : pd.DataFrame
        Simulation results DataFrame (from simulate_cell_populations).

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with Technical_cSTD and Technical_R per (context, sample_name).
    """
    posterior_xc = posteriors_dict["posterior_xc"]  # (Nx, N_total)

    df_work = df_sim.copy().reset_index(drop=True)
    df_work["_pos"] = np.arange(len(df_work))
    df_work["base_sample"] = df_work["sample_name"].str.replace(
        r"_run\d+$", "", regex=True
    )
    df_work["run_id"] = df_work["sample_name"].str.extract(r"(run\d+)$")

    # First aggregation: compute mixture cSTD per (context, base_sample, run_id)
    run_level = []
    for (ctx, base_samp, run_id), group_df in df_work.groupby(
        ["context", "base_sample", "run_id"]
    ):
        positions = group_df["_pos"].values
        group_posterior = posterior_xc[:, positions]
        cstd_run = cSTD_posterior_mixture(group_posterior)
        run_level.append(
            {
                "context": ctx,
                "base_sample": base_samp,
                "run_id": run_id,
                "cSTD_run": cstd_run,
                "Var_run": cstd_run**2,
            }
        )

    run_level_df = pd.DataFrame(run_level)

    # Second aggregation: average variances across runs
    final_stats = (
        run_level_df.groupby(["context", "base_sample"])
        .agg({"Var_run": "mean", "cSTD_run": "mean"})
        .reset_index()
    )

    final_stats["Technical_cSTD"] = np.sqrt(final_stats["Var_run"])
    final_stats["Technical_R"] = cstd2R(final_stats["Technical_cSTD"])
    final_stats = final_stats.rename(columns={"base_sample": "sample_name"})

    return final_stats


def desync_results_posterior(
    df_real: pd.DataFrame,
    real_posterior_xc: np.ndarray,
    df_sim: pd.DataFrame,
    sim_posteriors_dict: dict,
    group_cols: list = None,
    n_replicates: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute desynchrony using full posterior distributions (posterior_mixture mode).

    Replaces desync_results when mode='posterior_mixture'. Uses the mixture of
    per-cell posteriors to compute circular std rather than point estimates.

    Parameters
    ----------
    df_real : pd.DataFrame
        Real data results from create_results_dataframe (needed for group metadata).
    real_posterior_xc : np.ndarray, shape (Nx, Nc)
        Full posteriors from the real data inference pass.
    df_sim : pd.DataFrame
        Simulated data results from simulate_cell_populations.
    sim_posteriors_dict : dict
        Posterior dict returned by simulate_cell_populations(return_posteriors=True).
    group_cols : list
        Grouping columns (default: ["context", "sample_name"]).
    n_replicates : int, optional
        Bootstrap replicates for real data aggregation.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with Data_cSTD, Technical_cSTD, Bio_cSTD (all in hours).
    """
    if group_cols is None:
        group_cols = ["context", "sample_name"]

    real_agg = aggregate_real_results_posterior(
        real_posterior_xc,
        df_real,
        group_cols=group_cols,
        n_replicates=n_replicates,
        seed=seed,
    )

    sim_agg = aggregate_simulated_results_posterior(sim_posteriors_dict, df_sim)

    df_mixed = real_agg.copy()

    try:
        sim_lookup = sim_agg.set_index(group_cols)
    except KeyError:
        print(f"Error: sim_agg missing one of: {group_cols}")
        return df_mixed

    try:
        map_index = pd.MultiIndex.from_frame(df_mixed[group_cols])
    except KeyError:
        print(f"Error: df_mixed missing one of: {group_cols}")
        return df_mixed

    if "Technical_cSTD" in sim_lookup.columns:
        df_mixed["Technical_cSTD"] = map_index.map(sim_lookup["Technical_cSTD"])
    if "Technical_R" in sim_lookup.columns:
        df_mixed["Technical_R"] = map_index.map(sim_lookup["Technical_R"])

    # Convert radians → hours
    df_mixed["Technical_cSTD"] = df_mixed["Technical_cSTD"] * rh
    df_mixed["Data_cSTD"] = df_mixed["Data_cSTD"] * rh

    df_mixed["Bio_cSTD"] = np.sqrt(
        np.maximum(df_mixed["Data_cSTD"] ** 2 - df_mixed["Technical_cSTD"] ** 2, 0)
    )
    df_mixed["Bio_R"] = cstd2R(df_mixed["Bio_cSTD"] / rh)

    return df_mixed


def append_first_timepoint_periodic(df_desync, time_col: str = "ext_time_hours"):
    # (This function is unchanged)
    # Find the minimum ext_time_hours (first timepoint)
    zt_min = df_desync[time_col].min()
    # Select all rows corresponding to the first timepoint
    first_rows = df_desync[df_desync.ext_time_hours == zt_min].copy()
    # Set ext_time_hours to 24 + zt_min for these rows
    first_rows.ext_time_hours = 24 + zt_min
    # Concatenate to the original dataframe
    df_periodic = pd.concat([df_desync, first_rows], ignore_index=True)
    return df_periodic


def summarize_desync_results_one_ct(
    df_desync, plot=True, context: str = "", palette="Set1"
):
    weighted_mean_technical_R = np.average(
        df_desync["Technical_R"], weights=df_desync["group_size"]
    )
    weighted_mean_data_R = np.average(
        df_desync["Data_R"], weights=df_desync["group_size"]
    )

    final_technical_cSTD = sr.R2cstd(weighted_mean_technical_R) * rh
    final_data_cSTD = sr.R2cstd(weighted_mean_data_R) * rh
    final_bio_cSTD = np.sqrt(final_data_cSTD**2 - final_technical_cSTD**2)

    print("Final Technical cSTD (hours):", f"{final_technical_cSTD:.2f}h")
    print("Final Data cSTD (hours):", f"{final_data_cSTD:.2f}h")
    print("Final Bio cSTD (hours):", f"{final_bio_cSTD:.2f}h")

    if plot:
        equation_string2 = (
            "\n$\\sigma_{Bio} = \\sqrt{\\sigma_{Data}^2 - \\sigma_{Technical}^2}$"
        )
        plt.figure(figsize=(6, 5))
        bar_labels = [r"$\sigma_{Technical}$", r"$\sigma_{Bio}$", r"$\sigma_{Data}$"]
        bar_values = [final_technical_cSTD, final_bio_cSTD, final_data_cSTD]
        ax = sns.barplot(x=bar_labels, y=bar_values, hue=bar_labels, palette=palette)

        # Remove top frame
        ax.spines["top"].set_visible(False)

        # Add value labels on top of bars
        for i, v in enumerate(bar_values):
            ax.text(i, v + 0.05, f"{v:.2f}h", ha="center", va="bottom", fontsize=12)

        plt.ylabel("Circular STD [h]")
        plt.title(f"Desynchrony summary {context} \n" + equation_string2)
        plt.tight_layout()
        plt.show()

    return final_technical_cSTD, final_bio_cSTD, final_data_cSTD
