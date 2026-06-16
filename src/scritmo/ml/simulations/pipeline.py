"""pipeline.py -- clean, decoupled simulation -> inference pipeline for the Fig 1 / S1
circadian simulations (and reusable for any sim driver).

Motivation (vs the older simulations.py / utils.py path):
  * `results_to_df(trained=False)` silently returned the UNTRAINED (oracle true-template)
    result, so figures never reflected a refit; and `get_simulation_results` exposed BOTH
    `res_not_trained` and `res_trained` -- a confusing two-switch design.
  * The old path trained via `trainer.train_ritmo` directly, bypassing the canonical
    `warmup_and_train` (no warmup intercept-fit, no final inference) used by every real run.

This module fixes all three by:
  1. DECOUPLING generation from inference: `simulate_adata(...)` produces a canonical
     AnnData; `infer_phases(...)` consumes it. The AnnData is the single carrier of the
     generated counts + ground truth.
  2. ONE switch, ONE result: `infer_phases` always returns a single `res` dict; whether the
     gene template is refit is controlled solely by `n_epochs` (0 = posterior-only on the
     provided template; >0 = refit). There is no res_not_trained/res_trained split.
  3. The canonical `warmup_and_train` with the production soft-prior regime
     (`fix_phase=False, k_beta=2`, `kill_amps` auto-coupled to `n_epochs>0`).

Gauge: a refit lands in an arbitrary rotation+reflection gauge, so phases/AE are aligned to
truth with `optimal_shift(allow_flip=True)` (CLAUDE.md gotcha 10) BEFORE anything is saved --
the older rotate-only path corrupts refit panels.

All phases are radians internally; the `*_posterior_phase` columns emitted by `cells_to_df`
are in HOURS (matching the FIGURE_1 `sim_seq_depths.csv` contract: the notebook multiplies
them by `w` when it recomputes AE), while `true_phase` stays in radians.
"""

import numpy as np
import pandas as pd
import torch

import scritmo as sr
from scritmo import rh  # 24 / (2*pi): radians -> hours
from scritmo.ml import warmup_and_train

from .simulations import simulate_data


def _pop_means_hours(n_pop):
    """Population mean phases (hours) on the linspace grid simulate_data uses."""
    return np.linspace(0, 2 * np.pi, n_pop + 1)[:-1] * rh


def simulate_adata(
    genes_parameters,
    *,
    n_cells,
    seq_depth,
    dispersion=0.071,
    population_kappa=None,
    n_pop=None,
    phi_sim=None,
    n_theta=24,
    context="sim",
    device="cuda",
    seed=None,
    gene_names=None,
):
    """Generate one simulated condition and wrap it into a canonical AnnData that both
    `warmup_and_train` and `cmodel.estimate_phase_desynchrony` accept.

    Mirrors the validated `review/scripts/_sim_common.build_sim_adata` mapping:
      layers['spliced'] = counts          (warmup_and_train reads this; also the desync twin's
                                           library sizes)
      obs['true_phase'] = phi_sim          (radians)
      obs['sample_name'] = population id   ('0' when there is no population structure)
      obs['ZTmod']      = external time h  (pop-mean phase; per-cell true phase if single-pop)
      obs['context']    = context label
      obs['counts'] / obs['circadian_counts'] = library size per cell
    `uns['seq_depth']` records the depth so `infer_phases` can propagate it.

    population_kappa/n_pop semantics are simulate_data's: None+None -> single linspace of
    phases over the circle (the attractor sim); kappa+n_pop -> VonMises populations (the
    response-curve sim); kappa=None+n_pop -> delta-populations.
    """
    import scanpy as sc

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    par = sr.Beta(genes_parameters)
    if gene_names is None:
        gene_names = list(par.index)

    _mp, data_c, phi_sim_out, population = simulate_data(
        n_cells=n_cells,
        seq_depth=seq_depth,
        genes_parameters=genes_parameters,
        device=device,
        n_theta=n_theta,
        dispersion=dispersion,
        population_kappa=population_kappa,
        n_pop=n_pop,
        phi_sim=phi_sim,
    )

    X = data_c[0].cpu().numpy()
    phi_np = (
        phi_sim_out.cpu().numpy() if hasattr(phi_sim_out, "cpu")
        else np.asarray(phi_sim_out, dtype=float)
    )

    adata = sc.AnnData(X)
    adata.var_names = list(gene_names)
    adata.layers["spliced"] = X.copy()
    adata.obs["true_phase"] = phi_np.astype(float)

    if population is None:
        adata.obs["sample_name"] = "0"
        adata.obs["ZTmod"] = (phi_np * rh).astype(float)
    else:
        pop = np.asarray(population).astype(str)
        pop_means_h = _pop_means_hours(n_pop)
        adata.obs["sample_name"] = pop
        adata.obs["ZTmod"] = np.array([pop_means_h[int(p)] for p in pop], dtype=float)

    adata.obs["context"] = str(context)
    lib = np.asarray(X.sum(1)).reshape(-1).astype(float)
    adata.obs["counts"] = lib
    adata.obs["circadian_counts"] = lib
    adata.uns["seq_depth"] = int(seq_depth)
    return adata


def _apply_gauge(post_mode, post_mean, true_phase):
    """Align the posterior mode to truth with the full rotation+reflection gauge, and apply
    the SAME shift+flip to the mean. optimal_shift returns the shift but not the flip flag,
    so we recover the flip by checking which formula reproduces its aligned output."""
    aligned_mode, best_mad, shift = sr.optimal_shift(
        post_mode, true_phase, return_shift=True, allow_flip=True, verbose=False
    )
    aligned_mode = np.asarray(aligned_mode, dtype=float) % (2 * np.pi)
    no_flip = np.allclose((post_mode - shift) % (2 * np.pi), aligned_mode, atol=1e-6)
    sign = 1.0 if no_flip else -1.0
    aligned_mean = ((sign * post_mean) - shift) % (2 * np.pi)
    return aligned_mode, aligned_mean, float(best_mad), sign, float(shift)


def infer_phases(
    adata,
    genes_parameters,
    *,
    n_epochs=300,
    fix_phase=False,
    k_beta=2.0,
    kill_amps=None,
    init_mean=True,
    fix_disp_val="gene",
    n_theta=24,
    n_theta_post=100,
    batch_size=256,
    context_mode="none",
    device="cuda",
    align=True,
):
    """Fit `adata` with the canonical `warmup_and_train` (production soft-prior regime) and
    return ONE result dict. `n_epochs` is the single training switch:
      n_epochs == 0 -> no refit, posterior inference on the provided template (oracle);
      n_epochs  > 0 -> refit phi_g (soft Von-Mises prior) + amplitudes.
    `kill_amps` defaults to (n_epochs > 0): at n_epochs=0 the template amplitudes are kept
    (otherwise they'd be zeroed and never relearned).

    Posterior mode/mean are gauge-aligned to truth (allow_flip) before returning, so the
    saved phases/AE are flip-correct even when the refit lands in the mirror gauge.
    """
    if kill_amps is None:
        kill_amps = n_epochs > 0

    context = adata.obs["context"].values
    true_phase = adata.obs["true_phase"].values.astype(float)
    n_obs = adata.n_obs

    cmodel, losses, _mad = warmup_and_train(
        adata,
        params_g=sr.Beta(genes_parameters).copy(),
        context=np.asarray(context),
        context_mode=context_mode,
        fix_phase=fix_phase,
        k_beta=k_beta,
        kill_amps=kill_amps,
        init_mean=init_mean,
        fix_disp_val=fix_disp_val,
        n_epochs=n_epochs,
        n_theta=n_theta,
        n_theta_post=n_theta_post,
        batch_size=min(batch_size, n_obs),
        true_phase=true_phase,
        device=device,
    )

    post_mode = np.asarray(cmodel.post_mode_c, dtype=float).reshape(-1)
    post_mean = np.asarray(cmodel.post_mean_c, dtype=float).reshape(-1)
    post_std = np.asarray(cmodel.post_std_c, dtype=float).reshape(-1)

    if align:
        post_mode, post_mean, best_mad, _sign, _shift = _apply_gauge(
            post_mode, post_mean, true_phase
        )
        mad_h = best_mad * rh
    else:
        mad_h = float(np.median(sr.circular_deviation(true_phase, post_mode, period=2 * np.pi)) * rh)

    cad_h = np.asarray(
        sr.circular_deviation(true_phase, post_mode, period=2 * np.pi), dtype=float
    ).reshape(-1) * rh

    sample = adata.obs["sample_name"].values
    population = None if (np.unique(sample).size == 1 and sample[0] == "0") else np.asarray(sample)

    return {
        "cmodel": cmodel,
        "adata": adata,
        "post_mode_c": post_mode,          # radians, gauge-aligned
        "post_mean_c": post_mean,          # radians, gauge-aligned
        "post_std_c": post_std,            # radians
        "cad": cad_h,                      # hours, gauge-aligned AE (mode)
        "mad": mad_h,                      # hours
        "true_phase": true_phase,          # radians
        "population": population,           # array or None
        "circadian_counts": adata.obs["circadian_counts"].values.astype(float),
        "seq_depth": int(adata.uns.get("seq_depth", 0)),
        "n_epochs": int(n_epochs),
    }


def cells_to_df(res_or_list):
    """Per-cell long DataFrame matching the FIGURE_1 `sim_seq_depths.csv` contract (single,
    gauge-aligned `res` -- replaces the old `results_to_df`). Accepts one `res` or a list
    (e.g. one per sequencing depth) and concatenates. Phase columns are HOURS except
    `true_phase` (radians); `population` is NaN when there is no population structure."""
    results = res_or_list if isinstance(res_or_list, (list, tuple)) else [res_or_list]
    frames = []
    for res in results:
        n = len(res["true_phase"])
        pop = res["population"]
        frames.append(pd.DataFrame({
            "cad": res["cad"],
            "seq_depth": [res["seq_depth"]] * n,
            "circadian_counts": res["circadian_counts"],
            "true_phase": res["true_phase"],                       # radians
            "posterior_std": res["post_std_c"] * rh,               # hours
            "mean_posterior_phase": res["post_mean_c"] * rh,       # hours, aligned
            "mode_posterior_phase": res["post_mode_c"] * rh,       # hours, aligned
            "population": (np.nan if pop is None else pop),
        }))
    df = pd.concat(frames, ignore_index=True)
    df["seq_depth"] = df["seq_depth"].astype("category")
    df["true_phase_binned"] = (
        pd.cut(df["true_phase"], bins=np.linspace(0, 2.01 * np.pi, 24 + 1),
               include_lowest=True).apply(lambda x: x.mid * rh)
    )
    return df


def estimate_desync(
    cmodel,
    adata,
    *,
    ext_phase=None,
    n_sim_runs=10,
    n_epochs_training=0,
    device="cuda",
    **kwargs,
):
    """sigma_bio decomposition via the canonical `cmodel.estimate_phase_desynchrony` with the
    sim-appropriate defaults (the production model-twin floor -- NOT a kappa-reference floor).
    Returns one row per (context, sample) with Data_cSTD / Technical_cSTD / Bio_cSTD (hours)."""
    return cmodel.estimate_phase_desynchrony(
        adata=adata,
        ext_phase=None if ext_phase is None else np.asarray(ext_phase, dtype=float),
        context_col="context",
        sample_col="sample_name",
        ext_time_col="ZTmod",
        post_estimator="post_mode",
        allow_flip=True,
        n_sim_runs=n_sim_runs,
        n_epochs_training=n_epochs_training,
        device=device,
        **kwargs,
    )
