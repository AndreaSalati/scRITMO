"""
Biological phase desynchrony, corrected for the technical (inference) floor.

:func:`estimate_phase_desynchrony` is the end-to-end pipeline; the model method
``Scritmo.estimate_phase_desynchrony`` and :meth:`Scritmo.desynchrony` call it.
"""

import numpy as np

from scritmo import cSTD, cstd2R, rh
from ..utils import resolve_device
from .results import create_results_dataframe, desync_results
from .harmonic_floor import aggregate_technical_harmonic
from .deconvolution import aggregate_technical_deconvolution
from .technical_sim import simulate_cell_populations, simulate_technical_grid


def estimate_phase_desynchrony(
    model,
    adata,
    ext_phase: None | np.ndarray = None,
    # --- Shared Data/Column Arguments ---
    context_col: str | None = None,
    sample_col: str = "sample_name",
    ext_time_col: str = "ZTmod",
    layer: str = "spliced",
    post_estimator: str = "post_mode",
    # --- Simulation Arguments (simulate_cell_populations) ---
    n_cells: int | None = None,
    period: float = 24.0,
    device: str = "cuda",
    n_epochs_training: int = 0,
    n_replicates_sim: int | None = None,
    library_size_vec=None,
    n_sim_runs: int = 5,
    posterior_cell_chunk: int | None = None,
    # --- Real Data Arguments (create_results_df) ---
    other_obs_cols: list = [],
    allow_flip: bool = False,
    # --- Desynchrony Calculation Arguments (desync_results) ---
    group_cols: list | None = None,
    disp_function=cSTD,
    metrics: dict | None = None,
    n_replicates_real: int | None = None,
    seed_real: int = 42,
    seed_sim: int | None = None,
    # --- Technical floor method ---
    sigma_tech_method: str = "simulation",
    # --- Harmonic floor arguments ---
    n_grid: int = 24,
    n_cells_per_gridpoint: int = 1000,
    return_harmonic_diagnostics: bool = False,
    harmonic_orders=(1, 2, 3),
    harmonic_eval: str = "sample",
    # --- Deconvolution floor arguments ---
    deconv_form: str = "exact",
    return_deconv_diagnostics: bool = False,
    tech_grid=None,
    # --- Cell filtering / weighting ---
    post_std_threshold: float = np.inf,
    weight_by_post_std: bool = False,
    # --- Simulation mean estimation ---
    use_circular_mean: bool = False,
    # --- Over-subtraction policy ---
    clamp_bio_variance: bool = True,
):
    """
    Estimate biological phase desynchrony, correcting for the technical floor.

    The observed spread of per-cell phases within a sample mixes true
    biological desynchrony with technical (estimation) noise. This method
    separates the two by comparing the real spread against a "technical twin"
    whose only spread is estimation noise, then subtracting in quadrature
    inside :func:`desync_results`. Available as the model method
    ``Scritmo.estimate_phase_desynchrony`` (same arguments) and, with explicit
    ``adata.obs`` keys, :meth:`Scritmo.desynchrony`. End to end it:

    1. Builds a per-cell results DataFrame from the real data
       (:func:`create_results_dataframe`), optionally filtering cells by posterior
       phase uncertainty (``post_std_threshold``).
    2. Estimates the technical floor with one of two methods (``sigma_tech_method``):
         - "simulation": simulate a perfectly-synchronized population
           (``kappa=inf``) with this model and re-infer phases, so the recovered
           spread is purely technical (:func:`simulate_cell_populations`).
         - "harmonic": run the synchronized twin across a GRID of common phases
           (:func:`simulate_technical_grid`), fit the 12h structure of σ_tech²(φ)
           with a 2-harmonic model, then evaluate that fitted floor at each real
           cell's inferred phase and average within each replicate. Corrects the
           single-φ "simulation" floor for its known phase dependence (large near
           the Bmal1 trough, small at high expression).
         - "deconvolution": run the same σ=0 twin grid, take f(φ_k) = mean over
           runs of cSTD², expand it in its FULL Fourier series (all harmonics up to
           Nyquist, no fit), and solve per group for the σ_bio that makes the
           bump-averaged floor consistent with the data:
           V_b = T_b(σ) + σ², T_b(σ) = f_0 + Σ_j e^{−j²σ²/2}[a_j cos jμ_b + b_j sin jμ_b]
           (``deconv_form="exact"``, brentq), or its first-order closed form
           σ̂² = (V_b − f(μ_b)) / (1 + ½ f''(μ_b)) (``deconv_form="taylor"``).
           Removes the "one point for the technical term" error of the twin
           (≈ ½ σ²_bio f''(μ_b)). See :mod:`scritmo.ml.deconvolution`.
    3. Computes desynchrony per group by comparing the real dispersion to the
       technical floor (:func:`desync_results`).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix; must contain ``model.genes``.
    ext_phase : np.ndarray, optional
        External (reference) phase per cell, in radians. If None, the cell
        posterior estimates are used as the external frame.
    context_col : str, optional
        Column in ``adata.obs`` used as the outer grouping level for the
        desynchrony tables (e.g. celltype/condition/genotype). Unrelated to the
        legacy ``context_mode`` model terms. If None, falls back to "context";
        allowed to be None only when the model has a single context (it is then
        auto-assigned).
    sample_col : str, default "sample_name"
        Column in ``adata.obs`` identifying the biological replicate/sample.
    ext_time_col : str, default "ZTmod"
        Column for external time; passed as ``ext_time_label`` to the simulation.
    layer : str, default "spliced"
        AnnData layer used for real-data processing and simulation.
    post_estimator : str, default "post_mode"
        Per-cell posterior point estimator ("post_mode" / "post_mean") used for
        both the results DataFrame and the dispersion computation.
    n_cells : int, optional
        Number of cells to simulate (simulation method). None -> match the data.
    period : float, default 24.0
        Circadian period, in hours.
    device : str, default "cuda"
        Torch device for the simulation re-inference. Resolved with
        :func:`scritmo.ml.utils.resolve_device`, so the default falls back to
        CPU on a machine without a GPU.
    n_epochs_training : int, default 0
        Re-training epochs on the simulated twin (0 = reuse current gene params).
    n_replicates_sim : int, optional
        Number of simulated replicates per group.
    library_size_vec : array-like, optional
        Per-cell library sizes for the simulated population (defaults to data).
    n_sim_runs : int, default 5
        Number of independent simulation runs to average the technical floor over.
    posterior_cell_chunk : int, optional
        Chunk size for posterior inference on simulated cells (caps memory).
    other_obs_cols : list, default []
        Extra ``adata.obs`` columns to carry into the results DataFrame.
    allow_flip : bool, default False
        Allow a global 180° flip when aligning to the external frame.
    group_cols : list, optional
        Grouping for desynchrony aggregation. Defaults to ["context", "sample_name"].
    disp_function : callable, default cSTD
        Circular dispersion estimator applied per group (e.g. circular STD).
    metrics : dict, optional
        Custom metric callables forwarded to :func:`desync_results`.
    n_replicates_real : int, optional
        Bootstrap replicates for the real-data aggregation.
    seed_real : int, default 42
        RNG seed for the real-data bootstrap.
    seed_sim : int, optional
        RNG seed for the simulation.
    sigma_tech_method : {"simulation", "harmonic", "deconvolution"}, default "simulation"
        How to estimate the technical floor (see step 2).
    n_grid : int, default 24
        (harmonic / deconvolution) Number of common phases on the twin grid, evenly spaced
        over [0, 2π). Raised from 12 on 2026-08-11 together with the wider
        ``harmonic_orders`` default, so the 7-coefficient fit is not over-parametrised.
    n_cells_per_gridpoint : int, default 1000
        (harmonic / deconvolution) Twin cells simulated per (grid point, run). For
        deconvolution, match it to the typical group size n_b: f(φ_k) is a mean of
        per-run cSTD², so its finite-n bias then matches the data's V_b.
    return_harmonic_diagnostics : bool, default False
        (harmonic method) If True, store the fitted coefficients, raw grid points and
        implied peak locations on ``model.harmonic_floor_diag`` for plotting/checking.
        ``grid_phase``/``grid_var`` there are the RAW Monte-Carlo σ_tech² points the fit
        was made to, so data-vs-fit adequacy can be judged directly; ``coef`` evaluates
        via :func:`scritmo.ml.analysis_utils.eval_harmonic_floor_multi`.
    harmonic_orders : tuple of int, default (1, 2, 3)
        (harmonic method) Harmonic orders in the floor
        σ_tech²(φ) = m + Σ_k [a_k·cos(kφ) + b_k·sin(kφ)]. Was ``(2,)`` (12h only, from the
        single-gene Fisher-information argument) until 2026-08-11; that basis explained
        R²=0.013 of the raw twin grid on a 15-gene template — it collapsed to a near-flat
        line and mis-corrected every sample. ``(1, 2, 3)`` is where both the 15-gene sim
        and the 4-gene SABER-FISH panel saturate. Pass ``(2,)`` to reproduce older results.
        Fits 1 + 2·len(orders) coefficients, so keep ``n_grid`` comfortably above that.
    harmonic_eval : {"sample", "per_cell"}, default "sample"
        (harmonic method) Where the fitted floor σ_tech²(φ) is evaluated.
        ``"sample"`` uses ONE phase per (context, sample) — the same phase the simulation
        twin would be generated at, selected by ``use_circular_mean`` (False → the sample's
        external time, True → the circular mean of its inferred phases).
        ``"per_cell"`` is the pre-2026-08-11 behaviour: evaluate at every cell's inferred
        phase and average. That is biased — the floor is curved, so averaging over a spread
        of phases flattens it (Jensen), and the width of that spread is σ_tech itself, so
        the error grows exactly where a phase-resolved floor should help. Kept only for
        reproducibility; see
        :func:`scritmo.ml.analysis_utils.aggregate_technical_harmonic` for the measured cost.
    deconv_form : {"exact", "taylor", "vector"}, default "exact"
        (deconvolution method) "exact" solves T_b(σ) + σ² = V_b on [0, π] with brentq
        and returns NaN with ``deconv_flag`` ∈ {"below_floor", "no_root",
        "non_monotone"} when σ is not identified (h − V_b must cross zero exactly
        once, with dh/dσ > 0 at the root). "taylor" is the closed form
        (V_b − f(μ_b)) / (1 + ½ f''(μ_b)) with flags "below_floor", "denominator",
        "negative_tech". "vector" works on the complex mean resultant instead of the
        variance: ρ(φ_k) = mean of exp(i·post_mode) over all grid cells at φ_k, its
        complex Fourier series smoothed by the bump, ρ̄_b(σ) = Σ_j c_j e^{−j²σ²/2}
        e^{ijμ_b}, and σ solves |ρ̄_b(σ)|² + (1 − |ρ̄_b(σ)|²)/n_b = |z̄_b|². It keeps the
        direction of ρ, so it contains the attractor-bias and circular log-Jensen terms
        that "exact"/"taylor" ignore; its implied Technical_cSTD is
        √(Data_cSTD² − σ̂²) (NaN if σ̂ > Data_cSTD, flagged by ``deconv_neg_tech``).
        The group phase μ_b is chosen by ``use_circular_mean`` exactly
        as for the other methods. The output gains ``deconv_flag``,
        ``deconv_f_mu_h`` (√f(μ_b), h), ``deconv_f2_mu`` (f''(μ_b), dimensionless),
        ``deconv_mu_h`` and ``Technical_cSTD_floor`` (√f(μ_b), h, the single-point
        twin value read from the series). ``Technical_cSTD`` is the IMPLIED term
        √T_b(σ̂_b), so Data² = Technical² + Bio² per group by construction and
        :func:`desync_means` pools exactly the weighted mean of σ̂²_b over the
        identified groups (the unidentified ones carry NaN and are dropped/counted).
        With ``clamp_bio_variance=True`` the "below_floor" groups are set to
        Bio_cSTD = 0 and Technical_cSTD = Data_cSTD (the other flags stay NaN).
    tech_grid : pandas.DataFrame, optional
        (harmonic / deconvolution) A precomputed twin grid from
        :meth:`simulate_technical_grid` to reuse instead of simulating a new one (e.g.
        one grid shared by several estimators). The grid used is kept on
        ``model.last_tech_grid``.
    return_deconv_diagnostics : bool, default False
        (deconvolution method) Store {context: {"curve", "per_run", "coef"}} (the grid
        f(φ_k), its per-run values and the Fourier coefficients) on
        ``model.deconv_diag``.
    post_std_threshold : float, default inf
        Drop cells whose posterior phase std exceeds this (radians) before
        computing desynchrony. Default keeps all cells.
    weight_by_post_std : bool, default False
        If True, weight the desynchrony aggregation by ``post_std_c``.
    use_circular_mean : bool, default False
        Use the circular mean (vs. point estimate) for the simulated population means.
    clamp_bio_variance : bool, default True
        What to do where the technical floor exceeds the observed spread
        (``sigma_data^2 - sigma_tech^2 < 0``). True clamps the difference to 0, so
        ``Bio_cSTD`` is 0 there; False leaves it NaN, which marks the over-corrected
        groups instead of pulling them to zero (they then drop out of the aggregate
        in :func:`desync_means` rather than entering it as zeros). See
        :func:`desync_results`.

    Returns
    -------
    pandas.DataFrame
        Per-group desynchrony table from :func:`desync_results`, with the
        technical floor removed (biological desynchrony, plus the intermediate
        real / technical dispersion columns). Also stores the intermediate
        real-data frame on ``model.result_df``. With
        ``sigma_tech_method="deconvolution"`` it also carries the ``deconv_*``
        columns and ``Technical_cSTD_floor`` (see ``deconv_form``); ``Bio_cSTD`` is
        then σ̂_b in hours (NaN where not identified).
    """

    device = resolve_device(device)

    if getattr(model, "phase_range", None) is not None:
        raise NotImplementedError(
            "estimate_phase_desynchrony simulates its technical twin on the full "
            "circle; it is not supported for models fit with phase_range."
        )

    if sigma_tech_method not in ("simulation", "harmonic", "deconvolution"):
        raise ValueError(
            f"Unknown sigma_tech_method '{sigma_tech_method}'. Use 'simulation', "
            "'harmonic' or 'deconvolution' ('cramer_rao' was removed)."
        )
    if sigma_tech_method == "deconvolution" and deconv_form not in (
        "exact", "taylor", "vector"
    ):
        raise ValueError(
            f"deconv_form must be 'exact', 'taylor' or 'vector', got {deconv_form!r}"
        )

    if context_col is None:
        context_col = "context"
        if len(model.context_u) == 1:
            context_val = model.context_u[0]
            adata.obs[context_col] = context_val
        else:
            raise ValueError(
                "context_col cannot be None as there are multiple possible contexts. "
                "Please specify the context_col argument."
            )

    # 1. Generate Real Results DataFrame
    df_real = create_results_dataframe(
        cmodel=model,
        adata=adata,
        ext_phase=ext_phase,
        context_col=context_col,
        sample_col=sample_col,
        ext_time_col=ext_time_col,
        post_estimator=post_estimator,
        layer=layer,
        other_obs_cols=other_obs_cols,
        allow_flip=allow_flip,
    )
    model.result_df = df_real

    # 1b. Filter cells by posterior uncertainty
    if post_std_threshold < np.inf and "post_std_c" in df_real.columns:
        mask = df_real["post_std_c"] <= post_std_threshold
        n_before = len(df_real)
        df_real = df_real[mask].copy()
        print(
            f"  post_std_threshold={post_std_threshold:.3f} rad: "
            f"kept {len(df_real)}/{n_before} cells"
        )
        if len(df_real) == 0:
            raise ValueError(
                f"post_std_threshold={post_std_threshold} filtered out all cells."
            )

    deconv_table = None
    if sigma_tech_method in ("harmonic", "deconvolution"):
        # the sigma=0 twin grid of common phases; reused as-is when the caller passes a
        # precomputed one (e.g. one grid shared by deconv_form="exact" and "taylor")
        if tech_grid is not None:
            df_grid = tech_grid
        else:
            df_grid = simulate_technical_grid(
                cmodel=model,
                adata=adata,
                context_col=context_col,
                layer_to_use=layer,
                n_grid=n_grid,
                n_cells_per_gridpoint=n_cells_per_gridpoint,
                period=period,
                device=device,
                n_sim_runs=n_sim_runs,
                library_size_vec=library_size_vec,
                seed_sim=seed_sim,
                posterior_cell_chunk=posterior_cell_chunk,
            )
        model.last_tech_grid = df_grid

    if sigma_tech_method == "deconvolution":
        # 2a'''. Deconvolved floor: full Fourier series of f(phi_k), then solve
        # V_b = T_b(sigma) + sigma^2 per group (see scritmo.ml.deconvolution).
        _gcols = group_cols if group_cols is not None else ["context", "sample_name"]
        deconv_table, deconv_diag = aggregate_technical_deconvolution(
            df_grid,
            df_real,
            group_cols=_gcols,
            post_estimator=post_estimator,
            disp_function=disp_function,
            n_replicates=n_replicates_real,
            seed=seed_real,
            weight_col="post_std_c" if weight_by_post_std else None,
            use_circular_mean=use_circular_mean,
            period=period,
            deconv_form=deconv_form,
        )
        if return_deconv_diagnostics:
            model.deconv_diag = deconv_diag
        tech_agg = deconv_table[_gcols + ["Technical_cSTD", "Technical_R"]]
        df_sim = None
        print(
            f"  deconvolution floor ({deconv_form}): flags "
            f"{deconv_table['deconv_flag'].value_counts().to_dict()}"
        )
    elif sigma_tech_method == "harmonic":
        # 2a''. Phase-resolved floor: twin grid of common phases -> fit sigma_tech^2(phi),
        # evaluate at each real cell's inferred phase, average within replicate.
        tech_agg, harmonic_coeffs = aggregate_technical_harmonic(
            df_grid,
            df_real,
            group_cols=group_cols if group_cols is not None
            else ["context", "sample_name"],
            post_estimator=post_estimator,
            n_replicates=n_replicates_real,
            harmonic_orders=harmonic_orders,
            # evaluate the fitted floor at the SAME phase the simulation twin would be
            # generated at, so the two methods differ only in how the floor is obtained
            use_circular_mean=use_circular_mean,
            harmonic_eval=harmonic_eval,
            period=period,
        )
        # sanity output: terms actually fitted, plus how well they explain the raw grid
        for ctx, c in harmonic_coeffs.items():
            terms = "".join(
                f" + {c['coef']['a'][k]:.4f}*cos({k}phi)"
                f" + {c['coef']['b'][k]:.4f}*sin({k}phi)"
                for k in c["coef"]["orders"]
            )
            peaks = " / ".join(f"{p:.1f}h" for p in c["peak_hours"])
            print(
                f"  harmonic floor [{ctx}]: sigma_tech^2(phi) = "
                f"{c['m']:.4f}{terms}  -> peaks at {peaks}  "
                f"(R^2={c['r2']:.3f} on the raw grid)"
            )
        if return_harmonic_diagnostics:
            model.harmonic_floor_diag = harmonic_coeffs
        df_sim = None
    else:
        # 2a. Simulate the technical twin (point estimates only)
        df_sim = simulate_cell_populations(
            cmodel=model,
            adata=adata,
            context_col=context_col,
            n_cells=n_cells,
            layer_to_use=layer,
            ext_time_label=ext_time_col,
            sample_label=sample_col,
            kappa=np.inf,
            period=period,
            device=device,
            return_sim_data=True,
            n_epochs_training=n_epochs_training,
            n_replicates=n_replicates_sim,
            seed_sim=seed_sim,
            library_size_vec=library_size_vec,
            n_sim_runs=n_sim_runs,
            use_circular_mean=use_circular_mean,
            posterior_cell_chunk=posterior_cell_chunk,
        )
        tech_agg = None

    # 3a. Compute desynchrony from point estimates (sim_agg short-circuits the twin)
    df_final = desync_results(
        df_real=df_real,
        df_sim=df_sim,
        sim_agg=tech_agg,
        group_cols=group_cols,
        disp_function=disp_function,
        post_estimator=post_estimator,
        metrics=metrics,
        n_replicates=n_replicates_real,
        seed=seed_real,
        weight_col="post_std_c" if weight_by_post_std else None,
        clamp_bio_variance=clamp_bio_variance,
    )

    if deconv_table is not None:
        df_final = _attach_deconvolution(
            df_final,
            deconv_table,
            group_cols if group_cols is not None else ["context", "sample_name"],
            clamp_bio_variance,
        )

    return df_final



def _attach_deconvolution(df_final, deconv_table, group_cols, clamp_bio_variance):
    """Merge the deconvolution columns into the desync table and set Bio_cSTD = σ̂.

    `desync_results` already computed Bio from Data and the implied Technical term, which
    equals σ̂ up to the root tolerance; it is overwritten with σ̂ itself so the reported
    value is exactly the solution. Unidentified groups get NaN Bio and Technical, except
    "below_floor" under clamp_bio_variance=True -> Bio 0 and Technical = Data.
    """
    keep = group_cols + [
        c for c in deconv_table.columns
        if c.startswith("deconv_") and c not in ("deconv_V", "deconv_T_hat")
    ]
    dt = deconv_table[keep].copy()
    out = df_final.copy()
    for col in group_cols:
        dt[col] = dt[col].astype(str)
        out[col] = out[col].astype(str)
    out = out.merge(dt, on=group_cols, how="left")
    ok = (out["deconv_flag"] == "ok").values
    out["Bio_cSTD"] = np.where(ok, out["deconv_sigma"].values * rh, np.nan)
    out.loc[~ok, "Technical_cSTD"] = np.nan
    if clamp_bio_variance:
        low = (out["deconv_flag"] == "below_floor").values
        out.loc[low, "Bio_cSTD"] = 0.0
        out.loc[low, "Technical_cSTD"] = out.loc[low, "Data_cSTD"]
    out["Bio_R"] = cstd2R(out["Bio_cSTD"] / rh)
    out["Technical_R"] = cstd2R(out["Technical_cSTD"] / rh)
    # the single-point twin floor read from the series, plus the solution, in hours
    out["Technical_cSTD_floor"] = np.sqrt(out["deconv_f_mu"]) * rh
    out["deconv_f_mu_h"] = out["Technical_cSTD_floor"]
    out["deconv_mu_h"] = out["deconv_mu"] * rh
    out["deconv_sigma_h"] = out["deconv_sigma"] * rh
    return out

