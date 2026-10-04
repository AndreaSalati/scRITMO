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
