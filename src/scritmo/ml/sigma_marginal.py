"""Marginal-likelihood desynchrony estimator (Felix's method).

This module implements a cell-level marginal likelihood where each cell has a
latent phase offset δ_i ~ N(0, σ_k), integrated out with Gauss-Hermite quadrature.

It can be used in two modes:

1. **Full fit** (:func:`fit_felix`): two-stage MLE of all parameters (gene + σ),
   identical to Felix's original ``felix_method.core.fit_felix``.

2. **σ-only fit** (:func:`fit_sigma_only`): given fixed gene parameters from a
   trained scRITMO ContextModel, optimize only the per-timepoint σ_k. This is
   the "post-hoc σ estimator" (Level A integration).

Integration with scRITMO
------------------------
::

    from scritmo.ml.sigma_marginal import estimate_sigma_from_data

    # After training a ContextModel:
    result = estimate_sigma_from_data(
        Y=counts,              # (N, G) integer counts
        L=library_sizes,       # (N,)  total UMIs per cell
        time_index=time_idx,   # (N,)  integer index into timepoints
        timepoints=timepoints, # (K,)  hours
    )
    sigma_hr = result["sigma_phi_hr"]  # scalar estimate in hours

Or, for direct integration with an AnnData + ContextModel::

    result = estimate_sigma_from_adata(
        adata=adata,
        cmodel=trained_cmodel,
        ext_time_col="ZTmod",
    )

This module uses **numpy / scipy only** (no JAX, no PyTorch) — it is a
standalone post-hoc estimator that takes pre-computed gene parameters and
optimises only the desynchrony σ.

References
----------
- Hollis et al. 2026 (ORPHEUS): law-of-total-variance desynchrony estimator.
- Felix's reimplementation: cell-level marginal likelihood with Gauss-Hermite
  quadrature over the latent per-cell phase offset.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from scipy.sparse import issparse

from scritmo import rh

OMEGA = 2.0 * np.pi / 24.0


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------


def nb_logpmf_r(y, mu, r):
    """NB log-pmf with E[Y] = mu, Var[Y] = mu + mu² / r (so disp = 1/r).

    Parameters
    ----------
    y : ndarray
        Observed counts.
    mu : ndarray
        Mean of the NB distribution.
    r : ndarray
        Overdispersion parameter (r = 1/disp).

    Returns
    -------
    ndarray
        Log-probability of y under NB(r, p) with p = r/(r+mu).
    """
    mu = np.maximum(mu, 1e-12)
    r = np.maximum(r, 1e-12)
    return (
        gammaln(y + r)
        - gammaln(r)
        - gammaln(y + 1)
        + r * (np.log(r) - np.log(r + mu))
        + y * (np.log(mu) - np.log(r + mu))
    )


def marginal_loglik(Y, L, time_index, timepoints, params, gh_x, gh_w):
    """Cell-level marginal log-likelihood, summed over cells.

    Parameters
    ----------
    Y : ndarray, shape (N, G)
        Integer counts per cell per gene.
    L : ndarray, shape (N,)
        Library size (total UMI count) per cell.
    time_index : ndarray, shape (N,)
        Integer index mapping each cell to a timepoint in ``timepoints``.
    timepoints : ndarray, shape (K,)
        Nominal collection times in hours (one per timepoint).
    params : dict
        Must contain keys:
          - ``alpha``    (G,)  log-baseline expression
          - ``beta_cos`` (G,)  cosine amplitude
          - ``beta_sin`` (G,)  sine amplitude
          - ``disp``     (G,)  NB dispersion (Var = mu + disp * mu²)
          - ``sigma``    (K,)  per-timepoint phase spread (radians)
    gh_x : ndarray, shape (n_gh,)
        Gauss-Hermite quadrature nodes.
    gh_w : ndarray, shape (n_gh,)
        Gauss-Hermite quadrature weights.

    Returns
    -------
    float
        Sum of marginal log-likelihood over all cells.
    """
    alpha = params["alpha"]
    beta_cos = params["beta_cos"]
    beta_sin = params["beta_sin"]
    disp = params["disp"]
    sigma = params["sigma"]

    t_i = timepoints[time_index]
    sigma_i = sigma[time_index]

    # Gauss-Hermite: δ_iq = sqrt(2) * σ_i * x_q
    delta_iq = np.sqrt(2.0) * sigma_i[:, None] * gh_x[None, :]
    # φ_iq = ω * t_i + δ_iq
    x_iq = OMEGA * t_i[:, None] + delta_iq

    # μ_iqg = L_i * exp(α_g + β^c_g cos(φ_iq) + β^s_g sin(φ_iq))
    rhythmic = (
        beta_cos[None, None, :] * np.cos(x_iq[:, :, None])
        + beta_sin[None, None, :] * np.sin(x_iq[:, :, None])
    )
    mu = L[:, None, None] * np.exp(alpha[None, None, :] + rhythmic)

    r = 1.0 / disp

    # log P(y_ig | φ_iq, ...)  summed over genes -> shape (N, n_gh)
    log_nb = nb_logpmf_r(Y[:, None, :], mu, r[None, None, :]).sum(axis=2)

    # Quadrature: ∫ ... dδ ≈ (1/√π) Σ w_q * ...
    cell_ll = logsumexp(np.log(gh_w)[None, :] + log_nb, axis=1) - 0.5 * np.log(np.pi)
    return cell_ll.sum()


# ---------------------------------------------------------------------------
# Parameter packing / unpacking (for scipy.optimize)
# ---------------------------------------------------------------------------


def _to_unconstrained(name, x):
    return np.log(x) if name in ("sigma", "disp") else x


def _from_unconstrained(name, z):
    return np.exp(z) if name in ("sigma", "disp") else z


def _bounds_for(name, size):
    if name == "sigma":
        return [(np.log(0.01), np.log(np.pi))] * size
    if name == "disp":
        return [(np.log(0.005), np.log(5.0))] * size
    if name == "alpha":
        return [(-20.0, -2.0)] * size
    if name in ("beta_cos", "beta_sin"):
        return [(-5.0, 5.0)] * size
    return [(None, None)] * size


def _pack(params, free):
    """Flatten free parameters into a vector on the unconstrained scale."""
    vec, meta, bounds = [], [], []
    for name, value in params.items():
        if not free.get(name, False):
            continue
        value = np.asarray(value, dtype=float)
        z = _to_unconstrained(name, value)
        meta.append((name, value.shape, z.size))
        vec.append(z.ravel())
        bounds.extend(_bounds_for(name, z.size))
    if not vec:
        return np.array([]), meta, bounds
    return np.concatenate(vec), meta, bounds


def _unpack(vec, template, meta):
    """Restore full params dict from a flat unconstrained vector."""
    out = {k: np.array(v, copy=True) for k, v in template.items()}
    pos = 0
    for name, shape, size in meta:
        z = vec[pos:pos + size].reshape(shape)
        out[name] = _from_unconstrained(name, z)
        pos += size
    return out


# ---------------------------------------------------------------------------
# Initialisation: pseudobulk cosinor regression
# ---------------------------------------------------------------------------


def pseudobulk_init(Y, L, time_index, timepoints, fixed_disp=0.1, sigma0=0.05):
    """Initialise (alpha, beta_cos, beta_sin) by per-timepoint mean-fraction
    cosinor regression; dispersion and sigma are set to stable scalars.

    Parameters
    ----------
    Y : ndarray, shape (N, G)
        Integer counts.
    L : ndarray, shape (N,)
        Library sizes.
    time_index : ndarray, shape (N,)
        Per-cell timepoint index.
    timepoints : ndarray, shape (K,)
        Timepoint values in hours.
    fixed_disp : float
        Initial dispersion value (copied for all genes).
    sigma0 : float
        Initial sigma value (copied for all timepoints).

    Returns
    -------
    dict
        Initial parameter dict with keys ``alpha``, ``beta_cos``, ``beta_sin``,
        ``disp``, ``sigma``.
    """
    K = len(timepoints)
    G = Y.shape[1]

    # Per-timepoint mean expression fraction
    mean_frac = np.zeros((K, G))
    for k in range(K):
        idx = time_index == k
        if idx.any():
            mean_frac[k] = np.mean(Y[idx] / L[idx, None], axis=0)

    # Cosinor regression: log(frac) = a_0 + a_1*cos(ωt) + b_1*sin(ωt)
    X = np.column_stack([
        np.ones(K),
        np.cos(OMEGA * timepoints),
        np.sin(OMEGA * timepoints),
    ])
    coef = np.zeros((G, 3))
    for g in range(G):
        coef[g], *_ = np.linalg.lstsq(
            X,
            np.log(np.maximum(mean_frac[:, g], 1e-8)),
            rcond=None,
        )
    return dict(
        alpha=coef[:, 0],
        beta_cos=coef[:, 1],
        beta_sin=coef[:, 2],
        disp=np.full(G, fixed_disp),
        sigma=np.full(K, sigma0),
    )


def init_from_scritmo_params(gene_params, n_timepoints, dispersion=None,
                             sigma0=0.05):
    """Initialise Felix-style params from scRITMO's fitted gene parameters.

    scRITMO stores gene params as ``a_0`` (MESOR = alpha), ``a_1`` (cos
    amplitude = beta_cos), ``b_1`` (sin amplitude = beta_sin).  This function
    converts them to the dict Felix's methods expect.

    Parameters
    ----------
    gene_params : pd.DataFrame
        Gene parameter DataFrame from a trained ContextModel.  Must have columns
        ``a_0``, ``a_1``, ``b_1``. Indexed by gene name.
    n_timepoints : int
        Number of timepoints (K).  ``sigma`` will be initialised to ``sigma0``
        for each timepoint.
    dispersion : ndarray, optional
        Per-gene dispersion values.  If None, uses ``gene_params["disp"]`` if
        present, otherwise defaults to 0.1 for all genes.
    sigma0 : float
        Initial value for per-timepoint sigma (radians).

    Returns
    -------
    dict
        Parameter dict suitable for :func:`fit_sigma_only` or :func:`fit_felix`.
    """
    alpha = gene_params["a_0"].values.astype(float)
    beta_cos = gene_params["a_1"].values.astype(float)
    beta_sin = gene_params["b_1"].values.astype(float)

    if dispersion is not None:
        disp = np.asarray(dispersion, dtype=float).ravel()
    elif "disp" in gene_params.columns:
        disp = gene_params["disp"].values.astype(float).ravel()
    else:
        disp = np.full(len(gene_params), 0.1)

    return dict(
        alpha=alpha,
        beta_cos=beta_cos,
        beta_sin=beta_sin,
        disp=disp,
        sigma=np.full(n_timepoints, sigma0),
    )


# ---------------------------------------------------------------------------
# MLE driver
# ---------------------------------------------------------------------------


def fit_mle(Y, L, time_index, timepoints, init_params, free,
            n_gh=25, maxiter=300):
    """Maximise the marginal log-likelihood (L-BFGS-B) over the parameters
    flagged ``True`` in ``free``.

    Parameters
    ----------
    Y, L, time_index, timepoints : ndarray
        See :func:`marginal_loglik`.
    init_params : dict
        Initial parameter values (used as template for shape/dtype).
    free : dict
        Boolean mask: ``{"alpha": True, "sigma": True, ...}``.
    n_gh : int
        Number of Gauss-Hermite quadrature nodes.
    maxiter : int
        Maximum iterations for L-BFGS-B.

    Returns
    -------
    result : scipy.OptimizeResult or None
        The scipy optimization result (None if no free params).
    params_hat : dict
        Fitted parameters.
    loglik : float
        Marginal log-likelihood at the optimum.
    """
    gh_x, gh_w = hermgauss(n_gh)
    x0, meta, bounds = _pack(init_params, free)

    if len(x0) == 0:
        ll = float(marginal_loglik(
            Y, L, time_index, timepoints, init_params, gh_x, gh_w,
        ))
        return None, init_params, ll

    def objective(vec):
        params = _unpack(vec, init_params, meta)
        ll = marginal_loglik(
            Y, L, time_index, timepoints, params, gh_x, gh_w,
        )
        return -float(ll)

    res = minimize(
        objective, x0, method="L-BFGS-B", bounds=bounds,
        options=dict(maxiter=maxiter),
    )

    params_hat = _unpack(res.x, init_params, meta)
    return res, params_hat, -float(res.fun)


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------


def fit_felix(Y, L, time_index, timepoints, n_gh=25, maxiter=300,
              fixed_disp=0.1, sigma0=0.05, verbose=False):
    """Full two-stage MLE (Felix's method): fit all gene params + σ.

    Stage 1: clamp dispersion, fit (alpha, beta_cos, beta_sin, sigma).
    Stage 2: release dispersion, fit everything jointly.

    Parameters
    ----------
    Y : ndarray, shape (N, G)
        Integer counts.
    L : ndarray, shape (N,)
        Library sizes (total UMI per cell).
    time_index : ndarray, shape (N,)
        Per-cell timepoint index (0, ..., K-1).
    timepoints : ndarray, shape (K,)
        Timepoint values in hours.
    n_gh : int
        Gauss-Hermite quadrature nodes.
    maxiter : int
        Max iterations per stage.
    fixed_disp : float
        Initial dispersion (copied per gene, clamped in stage 1).
    sigma0 : float
        Initial sigma (radians).
    verbose : bool
        If True, print stage info.

    Returns
    -------
    dict
        With keys: ``init``, ``stage1``, ``stage2``, ``stage1_loglik``,
        ``stage2_loglik``, ``sigma_rad`` (per-timepoint), ``sigma_hr``,
        ``sigma_phi_rad`` (median scalar), ``sigma_phi_hr`` (median scalar, hours).
    """
    init = pseudobulk_init(Y, L, time_index, timepoints,
                           fixed_disp=fixed_disp, sigma0=sigma0)

    free1 = dict(alpha=True, beta_cos=True, beta_sin=True, disp=False, sigma=True)
    res1, fit1, ll1 = fit_mle(Y, L, time_index, timepoints, init, free1,
                               n_gh=n_gh, maxiter=maxiter)
    if verbose:
        print(f"  stage1: success={getattr(res1, 'success', None)} ll={ll1:.1f}")

    free2 = dict(alpha=True, beta_cos=True, beta_sin=True, disp=True, sigma=True)
    res2, fit2, ll2 = fit_mle(Y, L, time_index, timepoints, fit1, free2,
                               n_gh=n_gh, maxiter=maxiter)
    if verbose:
        print(f"  stage2: success={getattr(res2, 'success', None)} ll={ll2:.1f}")

    sigma_rad = np.asarray(fit2["sigma"])
    sigma_summary_rad = float(np.median(sigma_rad))

    return dict(
        init=init,
        stage1=fit1, stage1_loglik=ll1, stage1_result=res1,
        stage2=fit2, stage2_loglik=ll2, stage2_result=res2,
        sigma_rad=sigma_rad,
        sigma_hr=sigma_rad * rh,
        sigma_phi_rad=sigma_summary_rad,
        sigma_phi_hr=sigma_summary_rad * rh,
    )


def fit_sigma_only(Y, L, time_index, timepoints, fixed_params,
                   n_gh=25, maxiter=200, verbose=False):
    """Optimise only σ given fixed gene parameters (Level A integration).

    This is the key integration point with scRITMO: after training a
    ContextModel, extract its fitted gene parameters (alpha, beta_cos,
    beta_sin, disp) and pass them as ``fixed_params``.  Only the per-timepoint
    σ values are optimised.

    Parameters
    ----------
    Y : ndarray, shape (N, G)
        Integer counts.
    L : ndarray, shape (N,)
        Library sizes (total UMI per cell).
    time_index : ndarray, shape (N,)
        Per-cell timepoint index (0, ..., K-1).
    timepoints : ndarray, shape (K,)
        Timepoint values in hours.
    fixed_params : dict
        Fixed gene parameters with keys ``alpha`` (G,), ``beta_cos`` (G,),
        ``beta_sin`` (G,), ``disp`` (G,).  These are held constant.
    n_gh : int
        Gauss-Hermite quadrature nodes.
    maxiter : int
        Max iterations.
    verbose : bool
        Print optimisation progress.

    Returns
    -------
    dict
        With keys: ``sigma_rad`` (per-timepoint), ``sigma_hr``,
        ``sigma_phi_rad`` (median), ``sigma_phi_hr`` (median, hours),
        ``loglik``, ``result``, ``fixed_params``.
    """
    K = len(timepoints)
    init_sigma = np.full(K, 0.05)

    # Combine fixed params with optimisable sigma
    params = dict(fixed_params)
    params["sigma"] = init_sigma

    free = dict(alpha=False, beta_cos=False, beta_sin=False, disp=False, sigma=True)
    res, params_hat, loglik = fit_mle(Y, L, time_index, timepoints, params, free,
                                      n_gh=n_gh, maxiter=maxiter)

    sigma_rad = np.asarray(params_hat["sigma"])
    sigma_summary_rad = float(np.median(sigma_rad))

    if verbose:
        print(f"  sigma_only: success={getattr(res, 'success', None)} "
              f"ll={loglik:.1f}  sigma_hr={sigma_summary_rad * rh:.3f}")

    return dict(
        sigma_rad=sigma_rad,
        sigma_hr=sigma_rad * rh,
        sigma_phi_rad=sigma_summary_rad,
        sigma_phi_hr=sigma_summary_rad * rh,
        loglik=loglik,
        result=res,
        fixed_params=fixed_params,
    )


# ---------------------------------------------------------------------------
# Data adapters
# ---------------------------------------------------------------------------


def data_dict_from_arrays(Y, L, time_index, timepoints):
    """Build the data dict Felix's methods expect.

    Parameters
    ----------
    Y : ndarray, shape (N, G)
        Integer counts (gene subset for rhythm inference).
    L : ndarray, shape (N,)
        Library sizes — should be *total* UMIs across ALL genes, not just
        the gene subset, for correct size-factor normalisation.
    time_index : ndarray, shape (N,)
        Integer index mapping each cell to a timepoint.
    timepoints : ndarray, shape (K,)
        Unique timepoint values in hours.

    Returns
    -------
    dict
        ``{"Y": ..., "L": ..., "time_index": ..., "timepoints": ...}``
    """
    return {
        "Y": np.asarray(Y, dtype=np.int64),
        "L": np.asarray(L, dtype=np.float64),
        "time_index": np.asarray(time_index, dtype=np.int64),
        "timepoints": np.asarray(timepoints, dtype=np.float64),
    }


def data_dict_from_adata(adata, gene_names=None, layer="spliced",
                         ext_time_col="ZTmod", use_total_lib=True):
    """Build Felix data dict from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Single-cell data.  Must contain ``layer`` and ``adata.obs[ext_time_col]``.
    gene_names : list, optional
        Gene subset to use for ``Y``.  If None, uses all genes.
    layer : str
        Layer name containing counts.
    ext_time_col : str
        Column in ``adata.obs`` with external time in hours (e.g. ZTmod).
    use_total_lib : bool
        If True, ``L`` = total UMIs across ALL genes (recommended).  If False,
        ``L`` = total UMIs across the gene subset only.

    Returns
    -------
    dict
        Felix data dict, or ``None`` if no genes overlap.
    int
        Number of genes selected.
    """
    sp = adata.layers[layer] if layer in adata.layers else adata.X

    if issparse(sp):
        total_lib = np.asarray(sp.sum(axis=1)).ravel().astype(np.float64)
    else:
        total_lib = np.asarray(sp.sum(axis=1)).ravel().astype(np.float64)

    if gene_names is not None:
        var_idx = adata.var_names
        present = [g for g in gene_names if g in var_idx]
        if len(present) == 0:
            return None, 0
        n_genes = len(present)
        if issparse(sp):
            sub = sp[:, var_idx.get_indexer(present)]
            Y = sub.toarray()
        else:
            Y = sp[:, var_idx.get_indexer(present)]
        Y = np.rint(Y).astype(np.int64)
        if not use_total_lib:
            total_lib = np.asarray(Y.sum(axis=1)).ravel().astype(np.float64)
    else:
        n_genes = sp.shape[1]
        if issparse(sp):
            Y = sp.toarray()
        else:
            Y = sp
        Y = np.rint(Y).astype(np.int64)

    total_lib = np.maximum(total_lib, 1.0)

    zt = adata.obs[ext_time_col].to_numpy().astype(float)
    timepoints = np.sort(np.unique(zt))
    tp_to_idx = {float(t): k for k, t in enumerate(timepoints)}
    time_index = np.array([tp_to_idx[float(t)] for t in zt], dtype=np.int64)

    return dict(Y=Y, L=total_lib, time_index=time_index, timepoints=timepoints), n_genes


def data_dict_from_scritmo_model(adata, cmodel, ext_time_col="ZTmod",
                                 layer="spliced", use_total_lib=True):
    """Build Felix data dict from a trained scRITMO ContextModel.

    Uses the same gene subset that ``cmodel`` was trained on.

    Parameters
    ----------
    adata : AnnData
        The data the model was trained on.
    cmodel : ContextModel
        Trained scRITMO model.
    ext_time_col : str
        Column in ``adata.obs`` with external time.
    layer : str
        Layer with counts.
    use_total_lib : bool
        If True, ``L`` is total UMIs across all genes.

    Returns
    -------
    dict
        Felix data dict, or None if no genes.
    int
        Number of genes.
    """
    return data_dict_from_adata(
        adata,
        gene_names=cmodel.genes,
        layer=layer,
        ext_time_col=ext_time_col,
        use_total_lib=use_total_lib,
    )


def params_dict_from_scritmo_model(cmodel, context_key=None):
    """Extract Felix-style fixed parameters from a trained ContextModel.

    Parameters
    ----------
    cmodel : ContextModel
        Trained scRITMO model.
    context_key : str, optional
        If the model has multiple contexts, which one to extract.
        If None and the model has a single context, extracts that one.

    Returns
    -------
    dict
        Fixed params dict with keys ``alpha``, ``beta_cos``, ``beta_sin``,
        ``disp``, suitable for :func:`fit_sigma_only`.
    """
    param_df = cmodel.get_parameter_dataframe_context(
        np.arange(cmodel.Ng)
    )
    if context_key is None:
        # Single context
        keys = list(param_df.keys())
        if len(keys) == 1:
            context_key = keys[0]
        else:
            raise ValueError(
                f"Model has {len(keys)} contexts; specify which one "
                f"via context_key=."
            )

    gp = param_df[context_key]
    alpha = gp["a_0"].values.astype(float)
    beta_cos = gp["a_1"].values.astype(float)
    beta_sin = gp["b_1"].values.astype(float)

    # Dispersion: cmodel.disp is typically a scalar or per-gene tensor
    disp = cmodel.disp.detach().cpu().numpy().ravel()
    if len(disp) == 1:
        disp = np.full(len(alpha), float(disp[0]))
    elif len(disp) != len(alpha):
        raise ValueError(
            f"disp length ({len(disp)}) doesn't match n_genes ({len(alpha)})"
        )

    return dict(alpha=alpha, beta_cos=beta_cos, beta_sin=beta_sin, disp=disp)


# ---------------------------------------------------------------------------
# High-level estimation
# ---------------------------------------------------------------------------


def estimate_sigma_from_data(Y, L, time_index, timepoints,
                             fixed_params=None, n_gh=25, maxiter=300,
                             verbose=False):
    """Estimate desynchrony σ from count data.

    Two modes:
    - If ``fixed_params`` is provided, runs σ-only optimisation (Level A).
    - Otherwise, runs the full two-stage MLE (Felix's method).

    Parameters
    ----------
    Y, L, time_index, timepoints : ndarray
        See :func:`marginal_loglik`.
    fixed_params : dict, optional
        Fixed gene parameters from scRITMO.  If None, runs full fit.
    n_gh, maxiter, verbose :
        Passed to the optimiser.

    Returns
    -------
    dict
        With keys ``sigma_phi_hr`` (scalar median, hours), ``sigma_hr``
        (per-timepoint, hours), ``sigma_phi_rad``, ``sigma_rad``, and
        method-specific keys.
    """
    if fixed_params is not None:
        return fit_sigma_only(
            Y, L, time_index, timepoints, fixed_params,
            n_gh=n_gh, maxiter=maxiter, verbose=verbose,
        )
    else:
        return fit_felix(
            Y, L, time_index, timepoints,
            n_gh=n_gh, maxiter=maxiter, verbose=verbose,
        )


def estimate_sigma_from_adata(adata, cmodel=None,
                              gene_names=None, layer="spliced",
                              ext_time_col="ZTmod", use_total_lib=True,
                              context_key=None,
                              n_gh=25, maxiter=300,
                              verbose=False):
    """Estimate desynchrony σ from an AnnData object, optionally using a
    trained ContextModel for fixed gene parameters.

    If ``cmodel`` is provided, uses scRITMO's fitted gene params (Level A).
    Otherwise runs the full Felix method independently.

    Parameters
    ----------
    adata : AnnData
        Single-cell data with ``layer`` and ``obs[ext_time_col]``.
    cmodel : ContextModel, optional
        Trained scRITMO model.  If provided, its genes and parameters are used.
    gene_names : list, optional
        Gene subset for Y.  Ignored if ``cmodel`` is provided.
    layer : str
        Counts layer.
    ext_time_col : str
        External time column.
    use_total_lib : bool
        Whether to use total UMIs as library size.
    context_key : str, optional
        Which context's parameters to use.  Required if cmodel has multiple
        contexts and no ``context_key`` is given.
    n_gh, maxiter, verbose :
        Passed to the optimiser.

    Returns
    -------
    dict
        Estimation result, plus keys ``data`` (the Felix data dict) and
        ``n_genes``.
    """
    if cmodel is not None:
        data, n_genes = data_dict_from_scritmo_model(
            adata, cmodel,
            ext_time_col=ext_time_col,
            layer=layer,
            use_total_lib=use_total_lib,
        )
        if data is None:
            return {"sigma_phi_hr": np.nan, "n_genes": 0, "data": None}
        fixed_params = params_dict_from_scritmo_model(cmodel, context_key)
        result = fit_sigma_only(
            data["Y"], data["L"], data["time_index"], data["timepoints"],
            fixed_params,
            n_gh=n_gh, maxiter=maxiter, verbose=verbose,
        )
    else:
        data, n_genes = data_dict_from_adata(
            adata, gene_names=gene_names,
            layer=layer, ext_time_col=ext_time_col,
            use_total_lib=use_total_lib,
        )
        if data is None:
            return {"sigma_phi_hr": np.nan, "n_genes": 0, "data": None}
        result = fit_felix(
            data["Y"], data["L"], data["time_index"], data["timepoints"],
            n_gh=n_gh, maxiter=maxiter, verbose=verbose,
        )

    result["data"] = data
    result["n_genes"] = n_genes
    return result
