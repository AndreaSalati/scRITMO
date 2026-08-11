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


def _bio_variance(data_var, tech_var, clamp: bool):
    """sigma_data^2 - sigma_tech^2, either clamped at 0 or left to go NaN under sqrt.

    Single source of truth for the over-subtraction policy, shared by `desync_results`
    (per-timepoint Bio_cSTD) and `desync_means` (per-batch Bio_Var + the aggregate).
    With clamp=False the negative entries are mapped to NaN explicitly rather than left
    for `np.sqrt` to warn about.
    """
    diff = np.asarray(data_var, dtype=float) - np.asarray(tech_var, dtype=float)
    if clamp:
        return np.maximum(diff, 0.0)
    return np.where(diff < 0.0, np.nan, diff)


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
    clamp_bio_variance: bool = True,
):
    """
    First it aggregates data by calling aggregate_real_results and aggregate_simulated_results,
    then fuses the 2 in one dataframe. Finally it computes the
    biological desynchrony with the quadrature difference.

    `sim_agg` lets a caller bypass the simulation twin: pass a precomputed technical table (same
    schema as `aggregate_simulated_results`: context, sample_name, Technical_cSTD[rad], Technical_R)
    and `df_sim` is not used. This is the analytic Cramér–Rao path (`aggregate_technical_rao`).

    `clamp_bio_variance` controls what happens when the technical floor EXCEEDS the observed
    spread, i.e. sigma_data^2 - sigma_tech^2 < 0:
      True  (default) -- clamp the difference to 0, so Bio_cSTD is 0 there. Hides the
             over-subtraction but keeps every timepoint plottable/fittable.
      False -- leave it negative, so Bio_cSTD is NaN there. The over-corrected timepoints
             become visible and are EXCLUDED from downstream fits instead of being pulled
             to 0, which biases a slope fit less than a floor of zeros does.
    Use False to audit how often the floor over-corrects (see `run_fig2g_ablation.py`).

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
        _bio_variance(
            df_mixed["Data_cSTD"] ** 2,
            df_mixed["Technical_cSTD"] ** 2,
            clamp_bio_variance,
        )
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


def desync_means(df_desync, clamp_bio_variance: bool = True):
    """Aggregate a per-batch desync table to one row per context.

    `clamp_bio_variance` has the same meaning as in `desync_results`: True clamps a
    negative sigma_data^2 - sigma_tech^2 to 0, False lets it become NaN. With False the
    over-corrected batches drop out of the weighted means (via `np.average` on masked
    weights) instead of entering them as zeros.
    """
    context_u = df_desync["context"].unique()
    df_desync["Technical_cSTD2"] = df_desync["Technical_cSTD"] ** 2
    df_desync["Data_cSTD2"] = df_desync["Data_cSTD"] ** 2
    df_desync["Bio_Var"] = _bio_variance(
        df_desync["Data_cSTD2"], df_desync["Technical_cSTD2"], clamp_bio_variance
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

        # With clamp_bio_variance=False the over-corrected batches carry a NaN Bio_Var.
        # Drop them from the weighted means rather than let one NaN poison the whole
        # context (clamp=True never produces NaN, so this mask is all-True there and the
        # historical result is unchanged).
        bio_var_b = df_ct["Bio_Var"].values.astype(float)
        valid = np.isfinite(bio_var_b)
        weights = df_ct["group_size"].values[valid]
        n_dropped = int((~valid).sum())
        if weights.size == 0:
            results.append(
                {
                    "ct": ct,
                    "Technical_cSTD": np.nan,
                    "Bio_cSTD": np.nan,
                    "Bio_cSTD_SE": np.nan,
                    "Data_cSTD": np.nan,
                    "n_batches_dropped": n_dropped,
                    "organ": organ,
                    "celltype": celltype,
                    "condition": condition,
                }
            )
            continue

        weighted_mean_technical_var = np.average(
            df_ct["Technical_cSTD2"].values[valid], weights=weights
        )
        weighted_mean_data_var = np.average(
            df_ct["Data_cSTD2"].values[valid], weights=weights
        )

        final_bio_var = float(
            _bio_variance(
                weighted_mean_data_var, weighted_mean_technical_var, clamp_bio_variance
            )
        )
        final_bio_cSTD = np.sqrt(final_bio_var)
        final_technical_cSTD = np.sqrt(weighted_mean_technical_var)
        final_data_cSTD = np.sqrt(weighted_mean_data_var)

        se_bio_cSTD = _weighted_se_bio_cSTD(
            bio_var_b[valid], weights, final_bio_var
        )

        results.append(
            {
                "ct": ct,
                "Technical_cSTD": final_technical_cSTD,
                "Bio_cSTD": final_bio_cSTD,
                "Bio_cSTD_SE": se_bio_cSTD,
                "Data_cSTD": final_data_cSTD,
                "n_batches_dropped": n_dropped,
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
            "n_batches_dropped",
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

    # match aggregate_real_results: stringify the group keys so the desync_results merge aligns
    # (real_agg stringifies; without this, numeric sample ids like smFISH ZT mismatch -> NaN floor).
    df_real = df_real.copy()
    for col in group_cols:
        df_real[col] = df_real[col].astype(str)

    final_stats = (
        df_real.groupby(group_cols)[sigma_col]
        .apply(lambda s: technical_cstd_rao(s.values))
        .reset_index(name="Technical_cSTD")
    )
    final_stats["Technical_R"] = cstd2R(final_stats["Technical_cSTD"])
    return final_stats


def _harmonic_design(x_phase, orders):
    """Design matrix [1, cos(kφ), sin(kφ) for k in orders] for the floor OLS."""
    x = np.asarray(x_phase, dtype=float)
    cols = [np.ones_like(x)]
    for k in orders:
        cols += [np.cos(k * x), np.sin(k * x)]
    return np.column_stack(cols)


def fit_harmonic_floor_multi(x_phase, y_var, orders=(1, 2, 3)):
    """OLS fit of σ_tech²(φ) = m + Σ_k [a_k·cos(kφ) + b_k·sin(kφ)] over `orders`.

    Generalises :func:`fit_harmonic_floor`, which is the ``orders=(2,)`` special case (the
    12h-only form that used to be the default). 12h-only was chosen because a single
    sinusoidal gene's Fisher information is 12h-periodic — but with many genes at different
    acrophases the total information also carries 24h (k=1) and 8h (k=3) components, and
    which one dominates depends on the panel. Measured R² on the raw twin grid
    (``review/scripts/run_harmonic_floor_fit.py``, 2026-08-11):

        basis        15-gene clock sim   4-gene SABER-FISH
        (2,)                     0.013               0.754
        (1,2)                    0.766               0.959
        (1,2,3)                  0.922               0.969

    i.e. 12h-only explained essentially *nothing* on the 15-gene template (it collapsed to a
    near-flat line and mis-corrected every sample), while ``(1,2,3)`` is where both panels
    saturate — hence the default. Needs ``n_grid >= 8`` to avoid over-parametrising the 7
    coefficients; the pipeline default is now ``n_grid=24``.

    Parameters
    ----------
    x_phase : array-like
        Grid phases (radians), in the same frame F will be evaluated at.
    y_var : array-like
        σ_tech² at each grid phase (variance, i.e. cSTD²).
    orders : tuple of int, default (1, 2, 3)
        Harmonic orders to include. ``(2,)`` reproduces the legacy 12h-only fit exactly.

    Returns
    -------
    dict
        ``{"m", "a": {k: a_k}, "b": {k: b_k}, "orders", "r2", "rmse"}``. Consume it with
        :func:`eval_harmonic_floor_multi`.
    """
    orders = tuple(int(k) for k in orders)
    x = np.asarray(x_phase, dtype=float)
    y = np.asarray(y_var, dtype=float)
    D = _harmonic_design(x, orders)
    coeffs, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ coeffs
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "m": float(coeffs[0]),
        "a": {k: float(coeffs[1 + 2 * i]) for i, k in enumerate(orders)},
        "b": {k: float(coeffs[2 + 2 * i]) for i, k in enumerate(orders)},
        "orders": orders,
        "r2": (float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else np.nan),
        "rmse": float(np.sqrt(np.mean(resid**2))),
    }


def eval_harmonic_floor_multi(theta, coef):
    """Evaluate a :func:`fit_harmonic_floor_multi` result at `theta` (σ_tech², rad²).

    Clipped at 0 so fit noise can't yield a negative variance.
    """
    theta = np.asarray(theta, dtype=float)
    F = np.full(np.shape(theta), float(coef["m"]), dtype=float)
    for k in coef["orders"]:
        F = F + coef["a"][k] * np.cos(k * theta) + coef["b"][k] * np.sin(k * theta)
    return np.clip(F, 0.0, None)


def harmonic_floor_peaks_hours(coef, n=720):
    """Local maxima of the fitted floor, in hours, tallest first (max 2 returned).

    For the pure 12h form the two maxima are analytic (2θ* = atan2(b, a)); for a general
    `orders` there is no closed form, so they are located on a dense grid.
    """
    if tuple(coef["orders"]) == (2,):
        theta_star = np.arctan2(coef["b"][2], coef["a"][2]) / 2.0
        return sorted(
            float((p % (2 * np.pi)) * rh) for p in (theta_star, theta_star + np.pi)
        )
    grid = np.linspace(0, 2 * np.pi, n, endpoint=False)
    F = eval_harmonic_floor_multi(grid, coef)
    is_max = (F > np.roll(F, 1)) & (F > np.roll(F, -1))
    idx = np.flatnonzero(is_max)
    if idx.size == 0:
        idx = np.array([int(np.argmax(F))])
    idx = idx[np.argsort(F[idx])[::-1]][:2]
    return sorted(float(grid[i] * rh) for i in idx)


def fit_harmonic_floor(x_phase, y_var):
    """OLS fit of the 12h (2nd-harmonic) technical floor  y(φ) = m + a·cos(2φ) + b·sin(2φ).

    Linear in (m, a, b), so an ordinary least-squares fit on the design matrix
    [1, cos(2φ), sin(2φ)] denoises the noisy per-gridpoint Monte-Carlo variances into 3
    parameters and captures the expected 12h structure.

    Parameters
    ----------
    x_phase : array-like
        Grid phases (radians), in the same frame as the phases F will be evaluated at.
    y_var : array-like
        σ_tech² at each grid phase (variance, i.e. cSTD²).

    Returns
    -------
    (m, a, b) : tuple of float
        Mesor and 2nd-harmonic cosine/sine coefficients of the variance curve.
    """
    x_phase = np.asarray(x_phase, dtype=float)
    y_var = np.asarray(y_var, dtype=float)
    D = np.column_stack(
        [np.ones_like(x_phase), np.cos(2 * x_phase), np.sin(2 * x_phase)]
    )
    coeffs, *_ = np.linalg.lstsq(D, y_var, rcond=None)
    m, a, b = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    return m, a, b


def eval_harmonic_floor(theta, m, a, b):
    """Evaluate the fitted floor F(θ) = m + a·cos(2θ) + b·sin(2θ) (σ_tech², rad²).

    Clipped at 0 so fit noise can't yield a negative variance.
    """
    theta = np.asarray(theta, dtype=float)
    F = m + a * np.cos(2 * theta) + b * np.sin(2 * theta)
    return np.clip(F, 0.0, None)


def aggregate_technical_harmonic(
    df_grid: pd.DataFrame,
    df_real: pd.DataFrame,
    group_cols: list = None,
    post_estimator: str = "post_mode",
    n_replicates: int | None = None,
    harmonic_orders=(1, 2, 3),
):
    """Phase-resolved ("harmonic") technical floor, with the SAME output schema as
    `aggregate_simulated_results` (context, sample_name, Technical_cSTD[rad], Technical_R) so
    `desync_results(..., sim_agg=...)` consumes it unchanged (df_sim then unused).

    Pipeline (all averaging in VARIANCE = cSTD², root only at the very end):
      1. Per context, per (grid_idx, run_id) of the twin grid (`df_grid` from
         :func:`scritmo.ml.simulations.simulate_technical_grid`): x_k = the injected common
         phase `grid_phase`, y_k = sr.cSTD(post_mode)² (variance). Fit the floor via
         :func:`fit_harmonic_floor_multi` over `harmonic_orders` (default ``(1, 2, 3)``;
         pass ``(2,)`` for the legacy 12h-only form, which under-fits badly — see that
         function's docstring for the measured R²). (The injected φ_k is the right x-axis: generation and
         re-inference share the model's acrophases, so the inferred frame coincides with the
         injected frame -- and a uniform grid keeps the OLS design orthogonal. The real cells
         in step 2 are inferred with the same template, so F is evaluated in the same frame.)
      2. Evaluate F(θ_c) at each REAL cell's inferred phase (`post_estimator` column) using
         that cell's context coefficients.
      3. Per (context, sample_name): Technical_cSTD = sqrt(mean_c F(θ_c)); Technical_R = cstd2R.
      4. If `n_replicates` is set, broadcast each sample's floor to its `_1.._n` splits to
         match the renaming `aggregate_real_results` does (else `desync_results`' map -> NaN).

    Returns
    -------
    (final_stats, coeffs) : (pandas.DataFrame, dict)
        `final_stats`: the per-(context, sample_name) technical table.
        `coeffs`: {context: {"m","a","b","coef","orders","r2","rmse","grid_phase",
        "grid_var","peak_hours"}} for diagnostics. `grid_phase`/`grid_var` are the RAW
        Monte-Carlo grid points the fit was made to — enough to plot data vs fit and judge
        whether the functional form is adequate; `coef` feeds
        :func:`eval_harmonic_floor_multi`. `"a"`/`"b"` are the k=2 coefficients (NaN when 2
        is not in `harmonic_orders`).
    """
    if group_cols is None:
        group_cols = ["context", "sample_name"]

    # --- 1. Fit the floor per context from the twin grid ---
    coeffs = {}
    for context_label, df_ctx in df_grid.groupby("context"):
        x_list, y_list = [], []
        for _, df_pt in df_ctx.groupby(["grid_idx", "run_id"]):
            # x = injected common phase phi_k (uniform grid, same frame as the real cells);
            # y = circular variance of the inferred phases at that grid point.
            x_list.append(float(df_pt["grid_phase"].iloc[0]))
            y_list.append(sr.cSTD(df_pt[post_estimator].values) ** 2)  # variance
        coef = fit_harmonic_floor_multi(x_list, y_list, orders=harmonic_orders)
        peak_hours = harmonic_floor_peaks_hours(coef)
        # "m"/"a"/"b" stay flat scalars for the legacy 12h-only form so existing readers
        # (printing, saved diagnostics) keep working; `coef` carries the general fit.
        coeffs[str(context_label)] = {
            "m": coef["m"],
            "a": coef["a"].get(2, np.nan),
            "b": coef["b"].get(2, np.nan),
            "coef": coef,
            "orders": coef["orders"],
            "r2": coef["r2"],
            "rmse": coef["rmse"],
            "grid_phase": np.asarray(x_list),
            "grid_var": np.asarray(y_list),
            "peak_hours": peak_hours,
        }

    # --- 2. Evaluate the floor at each real cell's inferred phase ---
    df_real = df_real.copy()
    for col in group_cols:
        df_real[col] = df_real[col].astype(str)

    floor_var = np.empty(len(df_real), dtype=float)
    ctx_vals = df_real["context"].values
    theta_vals = df_real[post_estimator].values
    for context_label, c in coeffs.items():
        mask = ctx_vals == context_label
        floor_var[mask] = eval_harmonic_floor_multi(theta_vals[mask], c["coef"])
    df_real = df_real.assign(_floor_var=floor_var)

    # --- 3. Per-(context, sample) floor: mean variance -> sqrt at the very end ---
    final_stats = (
        df_real.groupby(group_cols)["_floor_var"]
        .mean()
        .reset_index(name="_floor_var")
    )
    final_stats["Technical_cSTD"] = np.sqrt(final_stats["_floor_var"])
    final_stats = final_stats.drop(columns=["_floor_var"])

    # --- 4. Broadcast to n_replicates splits (matches aggregate_real_results renaming) ---
    # aggregate_real_results renames each sample to f"{sample}_{rep+1}" via
    # `(replicate + 1).astype(str)`, where `replicate` comes from assign_replicates (float64),
    # so the suffix is "1.0", "2.0", ... -- mirror that exact float formatting here, else the
    # desync_results map misses (-> NaN Bio_cSTD).
    if n_replicates is not None:
        sample_col = group_cols[-1]
        rows = []
        for i in range(n_replicates):
            sub = final_stats.copy()
            suffix = str(float(i + 1))  # "1.0", "2.0", ... (matches float64 replicate ids)
            sub[sample_col] = sub[sample_col].astype(str) + "_" + suffix
            rows.append(sub)
        final_stats = pd.concat(rows, ignore_index=True)

    final_stats["Technical_R"] = cstd2R(final_stats["Technical_cSTD"])
    return final_stats, coeffs


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
