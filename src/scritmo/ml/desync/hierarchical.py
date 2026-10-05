"""
Hierarchical likelihood estimator of sigma_bio for ONE group of cells (no twin).

Note: Obsidian `Review_2/important_notes/hierarchical_likelihood_estimator.md`
(Hogg, Myers & Bovy 2010 style). The population of true phases is WN(mu, sigma); each cell
contributes the integral of its likelihood over that population,

    log L(mu, sigma) = sum_c log sum_x l_c(phi_x) w_x(mu, sigma),

with w the wrapped normal on the phase grid (normalized to sum 1). The posterior scritmo
returns is the normalized likelihood (no phase prior), so it is used directly.

Implementation: mu is scanned on the phase grid points and sigma on a fine grid, so the
whole surface is one matrix product per sigma. Cost is N_sigma matmuls of (N, N) x (N, n_b).
For sigma smaller than the grid spacing the weights collapse to a delta on a grid point.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from scritmo import rh

CHI2_1_95 = 1.92  # half of the chi2_1 95% quantile (3.84)


def wn_kernel(phi, sigma, n_wrap=3):
    """K[x, m] = wrapped normal weight of grid point x for centre phi[m], columns sum to 1."""
    N = len(phi)
    if sigma <= 0:
        return np.eye(N)
    d = phi[:, None] - phi[None, :]
    logk = np.full((N, N), -np.inf)
    for k in range(-n_wrap, n_wrap + 1):
        z = -0.5 * ((d + 2 * np.pi * k) / sigma) ** 2
        logk = np.logaddexp(logk, z)
    logk -= logk.max(0, keepdims=True)
    K = np.exp(logk)
    return K / K.sum(0, keepdims=True)


@lru_cache(maxsize=512)
def _kernel_cached(N, sigma):
    return wn_kernel(np.arange(N) * 2 * np.pi / N, sigma)


def _sigma_grid(sigma_max, n_sigma):
    return np.concatenate([[0.0], sigma_max * (np.arange(1, n_sigma) / (n_sigma - 1)) ** 2])


def loglik_surface(post_xc, phi_x, sigma_max=np.pi, n_sigma=200, floor=1e-300):
    """log L[sigma_i, mu_m] of one group (up to a constant), plus the sigma grid."""
    P = np.asarray(post_xc, dtype=np.float64)
    Pn = P / np.maximum(P.max(0, keepdims=True), floor)  # constant per cell: no effect on argmax
    sig = _sigma_grid(sigma_max, n_sigma)
    LL = np.empty((len(sig), len(phi_x)))
    for i, s in enumerate(sig):
        M = _kernel_cached(len(phi_x), float(s)).T @ Pn  # (N_mu, n_b)
        LL[i] = np.log(np.maximum(M, floor)).sum(1)
    return sig, LL


def _summarize(sig, prof, LL=None, sigma_max=np.pi, phi=None):
    """Maximum (parabolic refinement) and 95% profile interval of a sigma profile."""
    i_hat = int(np.argmax(prof))
    s_hat = sig[i_hat]
    if 0 < i_hat < len(sig) - 1:
        x0, x1, x2 = sig[i_hat - 1:i_hat + 2]
        y0, y1, y2 = prof[i_hat - 1:i_hat + 2]
        den = (x0 - x1) * (x0 - x2) * (x1 - x2)
        A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
        B = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / den
        if A < 0:
            s_hat = float(np.clip(-B / (2 * A), x0, x2))
    thr = prof[i_hat] - CHI2_1_95
    inside = prof >= thr
    lo = i_hat
    while lo > 0 and inside[lo - 1]:
        lo -= 1
    hi = i_hat
    while hi < len(sig) - 1 and inside[hi + 1]:
        hi += 1

    def interp(a, b):  # sigma where prof crosses thr between index a (outside) and b (inside)
        ya, yb = prof[a], prof[b]
        return float(sig[a] + (thr - ya) / (yb - ya) * (sig[b] - sig[a])) if yb != ya else sig[b]

    ci_lo = 0.0 if lo == 0 else interp(lo - 1, lo)
    ci_hi = sigma_max if hi == len(sig) - 1 else interp(hi + 1, hi)
    mu = np.nan if LL is None else float(phi[int(np.argmax(LL[i_hat]))] % (2 * np.pi))
    return dict(sigma=float(s_hat), mu=mu, loglik_max=float(prof[i_hat]), ci_lo=float(ci_lo),
                ci_hi=float(ci_hi), flag="at_boundary" if s_hat < 1e-3 else "ok")


def _bad(post_xc, phi):
    P = np.asarray(post_xc)
    return (P.ndim != 2 or P.shape[0] != len(phi) or P.shape[1] < 2
            or not np.all(np.isfinite(P)))


def solve_hierarchical(post_xc, phi_x, sigma_max=np.pi, n_sigma=200, floor=1e-300):
    """
    post_xc : (N_theta, n_b) posterior (normalized likelihood) of the cells of one group
    phi_x   : (N_theta,) uniform phase grid on [0, 2pi)
    Returns dict(sigma, mu, loglik_max, ci_lo, ci_hi, flag), sigma/mu/ci in radians.
    flag in {ok, at_boundary, fail}.
    """
    phi = np.asarray(phi_x, dtype=np.float64)
    if _bad(post_xc, phi):
        return dict(sigma=np.nan, mu=np.nan, loglik_max=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    flag="fail")
    sig, LL = loglik_surface(post_xc, phi, sigma_max, n_sigma, floor)
    return _summarize(sig, LL.max(1), LL, sigma_max, phi)


def solve_hierarchical_shared(posts, phi_x, sigma_max=np.pi, n_sigma=200, floor=1e-300):
    """
    ONE sigma for several groups, each with its own free mu_b: the profile log-likelihoods
    (max over mu_b) of the groups are summed over sigma, then maximized. posts is a list of
    (N_theta, n_b) arrays. Same return as solve_hierarchical (mu is NaN).
    """
    phi = np.asarray(phi_x, dtype=np.float64)
    if not posts or any(_bad(p, phi) for p in posts):
        return dict(sigma=np.nan, mu=np.nan, loglik_max=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    flag="fail")
    prof = 0.0
    for p in posts:
        sig, LL = loglik_surface(p, phi, sigma_max, n_sigma, floor)
        prof = prof + LL.max(1)
    return _summarize(sig, prof, None, sigma_max, phi)


def cell_posteriors(model, adata, layer="spliced", n_theta=240, library_size_vec=None,
                    cell_chunk=None):
    """Normalized per cell posteriors on a fine phase grid: (n_theta, n_cells), and the grid.

    The posterior is the likelihood (``get_phase_posteriors`` adds no phase prior). The grid
    is finer than the one used for the fit, since sigma_bio of about 1 h needs well under 1 h
    spacing. ``library_size_vec`` must be the genome wide library size, as for the fit;
    None uses ``model.counts`` (the fit's own size factor), never the sum over the modelled
    genes.
    """
    import torch

    dev = model.m_g.device
    X = adata[:, list(model.genes)].layers[layer]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    y = torch.tensor(X, dtype=torch.float32, device=dev).unsqueeze(0)
    if library_size_vec is None:
        counts = model.counts
        if counts.shape[0] != adata.n_obs:
            raise ValueError(
                "model.counts does not match adata; pass library_size_vec (the genome wide "
                "library size per cell)."
            )
        counts = counts.to(dev)
    else:
        counts = torch.tensor(np.asarray(library_size_vec, dtype=np.float32).reshape(-1, 1),
                              device=dev)
    post = model.get_phase_posteriors(y, counts=counts, n_theta=n_theta, cell_chunk=cell_chunk)
    return np.asarray(post), np.arange(n_theta) * 2 * np.pi / n_theta


def aggregate_hierarchical(post, phi, df_real, group_cols, shared=False, cell_mask=None,
                           sigma_max=np.pi, n_sigma=200):
    """Per group (``group_cols``) hierarchical sigma_bio from the cell posteriors.

    post : (n_theta, n_cells) in the cell order of the AnnData; ``df_real`` is the per cell
    table in that order (after the optional ``cell_mask``, a boolean over the AnnData cells,
    that dropped cells). Returns one row per group with ``hier_sigma`` (rad), ``hier_mu``
    (rad), ``hier_ci_lo``/``hier_ci_hi`` (rad), ``hier_flag``, ``hier_loglik``, ``hier_n``.
    With ``shared=True`` the rows also carry ``hier_shared_sigma`` / ``_ci_lo`` / ``_ci_hi``
    (rad): one sigma per level of ``group_cols[0]`` (e.g. per context), each group keeping a
    free mu_b.
    """
    if cell_mask is not None:
        post = post[:, np.asarray(cell_mask, dtype=bool)]
    if post.shape[1] != len(df_real):
        raise ValueError("posteriors and df_real have different numbers of cells")
    keys = df_real[group_cols].astype(str).reset_index(drop=True)
    rows = []
    for key, idx in keys.groupby(group_cols, sort=False).groups.items():
        key = key if isinstance(key, tuple) else (key,)
        e = solve_hierarchical(post[:, np.asarray(idx)], phi, sigma_max, n_sigma)
        rows.append(dict(zip(group_cols, key), hier_sigma=e["sigma"], hier_mu=e["mu"],
                         hier_ci_lo=e["ci_lo"], hier_ci_hi=e["ci_hi"], hier_flag=e["flag"],
                         hier_loglik=e["loglik_max"], hier_n=len(idx)))
    out = pd.DataFrame(rows)
    if shared:
        sh = {}
        for lvl, idxs in keys.groupby(group_cols[0], sort=False).groups.items():
            sub = keys.loc[idxs]
            posts = [post[:, np.asarray(ix)]
                     for ix in sub.groupby(group_cols, sort=False).groups.values()]
            sh[lvl] = solve_hierarchical_shared(posts, phi, sigma_max, n_sigma)
        for c, k in (("hier_shared_sigma", "sigma"), ("hier_shared_ci_lo", "ci_lo"),
                     ("hier_shared_ci_hi", "ci_hi")):
            out[c] = out[group_cols[0]].map(lambda g, k=k: sh[g][k])
    return out
