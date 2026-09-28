"""
Phase-resolved technical floor: fit sigma_tech^2(phi) on a synchronized twin grid
with a few harmonics, then evaluate it per group (``sigma_tech_method="harmonic"``).
"""

import numpy as np
import pandas as pd
from scipy.stats import circmean

import scritmo as sr
from scritmo import rh, cstd2R
from .results import aggregate_real_results


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


def aggregate_technical_harmonic(
    df_grid: pd.DataFrame,
    df_real: pd.DataFrame,
    group_cols: list = None,
    post_estimator: str = "post_mode",
    n_replicates: int | None = None,
    harmonic_orders=(1, 2, 3),
    use_circular_mean: bool = False,
    ext_time_col: str = "ext_time_hours",
    harmonic_eval: str = "sample",
    period: float = 24.0,
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
      2. Evaluate F at ONE phase per (context, sample_name) group -- the same phase the
         simulation twin would be generated at, selected by `use_circular_mean` exactly as in
         :func:`simulate_cell_populations`:
           False (default) -> the sample's external time (`ext_time_col`, hours -> rad);
           True            -> the circular mean of that sample's inferred phases.
         Set `harmonic_eval="per_cell"` for the LEGACY behaviour (evaluate F at every cell's
         inferred phase and average). That is biased -- see the note below.
      3. Per (context, sample_name): Technical_cSTD = sqrt(F(φ_sample)); Technical_R = cstd2R.
      4. If `n_replicates` is set, broadcast each sample's floor to its `_1.._n` splits to
         match the renaming `aggregate_real_results` does (else `desync_results`' map -> NaN).

    Why NOT per-cell (changed 2026-08-11)
    -------------------------------------
    F is CURVED, so by Jensen's inequality ``mean_c F(θ_c) != F(mean θ)``: averaging over a
    spread of phases inflates the floor in a convex region (a trough) and deflates it near a
    peak. Because F is roughly sinusoidal, that flattens the fitted floor toward its own mean --
    and the width of the averaging window is σ_tech ITSELF, so the damage grows exactly where a
    phase-resolved floor is supposed to help. Measured on the Fig-1 sim (12 populations,
    `review/scripts/run_harmonic_floor_diagnose.py`): at 3k UMI per-cell averaging collapsed the
    floor's range across populations from **2.11 h to 0.83 h**, and its RMS deviation from the
    per-group twin floor was **0.44 h vs 0.14 h** when read at the sample's known external time.
    Per-population σ_bio error over 3k/10k/100k: 0.399 h (per-cell) -> 0.248 h (per-sample);
    the twin itself gets 0.193 h. Per-cell evaluation is only defensible when the within-sample
    spread is genuinely BIOLOGICAL (cells really at different phases), and the inferred spread
    cannot distinguish that from technical noise -- hence the per-sample default.

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

    # --- 2. Evaluate the floor, then 3. reduce to one value per (context, sample) ---
    df_real = df_real.copy()
    for col in group_cols:
        df_real[col] = df_real[col].astype(str)

    if harmonic_eval not in ("sample", "per_cell"):
        raise ValueError(
            f"harmonic_eval must be 'sample' or 'per_cell', got {harmonic_eval!r}"
        )

    if harmonic_eval == "per_cell":
        # LEGACY (pre-2026-08-11): F at every cell's inferred phase, averaged. Biased by
        # Jensen -- see the docstring. Kept only to reproduce older results.
        floor_var = np.empty(len(df_real), dtype=float)
        ctx_vals = df_real["context"].values
        theta_vals = df_real[post_estimator].values
        for context_label, c in coeffs.items():
            mask = ctx_vals == context_label
            floor_var[mask] = eval_harmonic_floor_multi(theta_vals[mask], c["coef"])
        final_stats = (
            df_real.assign(_floor_var=floor_var)
            .groupby(group_cols)["_floor_var"]
            .mean()
            .reset_index(name="_floor_var")
        )
    else:
        # ONE phase per sample, chosen exactly as simulate_cell_populations chooses the phase
        # its twin is generated at, so the two floors are computed at the same place.
        have_ext = ext_time_col in df_real.columns
        if not use_circular_mean and not have_ext:
            print(
                f"  WARNING: use_circular_mean=False needs '{ext_time_col}' in the results "
                "frame (it comes from create_results_df's ext_time_col); falling back to the "
                "circular mean of the inferred phases."
            )
        rows = []
        for keys, grp in df_real.groupby(group_cols):
            keys = keys if isinstance(keys, tuple) else (keys,)
            ctx = str(dict(zip(group_cols, keys))["context"])
            if use_circular_mean or not have_ext:
                phi = float(circmean(grp[post_estimator].values, high=2 * np.pi, low=0))
            else:
                # hours -> rad, matching simulations.utils.get_ext_time(convert_rad=True)
                phi = float(
                    (float(grp[ext_time_col].iloc[0]) % period) / period * 2 * np.pi
                )
            rows.append(
                dict(
                    zip(group_cols, keys),
                    _floor_var=float(
                        eval_harmonic_floor_multi(np.array([phi]), coeffs[ctx]["coef"])[0]
                    ),
                    _floor_phase=phi,
                )
            )
        final_stats = pd.DataFrame(rows).drop(columns=["_floor_phase"])

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
