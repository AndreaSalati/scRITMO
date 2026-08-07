"""cramer_rao.py -- analytic Cramér–Rao (CRB) floor for the phase-estimation noise σ_tech.

σ_tech is the phase-estimation noise of a perfectly-synchronized population. Under the NB
single-harmonic model its variance floor is 1 / Fisher-information-of-phase, so it has a closed
form computable directly from the fitted gene parameters -- no simulation twin needed:

    μ_g(θ)   = s · exp(a_0_g + A_g·cos(θ − φ_g))                  # the model's own NB mean
    I(θ, s)  = Σ_g  A_g² · sin²(θ − φ_g) · μ_g / (1 + μ_g·disp_g)
    σ_rao    = 1 / sqrt(I(θ, s))                                  # radians

Convention (verified against context_model.model_formula + compute_nb_params):
  * NB mean μ = counts·exp(a_0 + A·cos(θ−φ)) ⇒ b_g = a_0 (log-mesor), s = counts (size factor).
  * compute_nb_params uses r = 1/disp and Var = μ + disp·μ², i.e. scritmo's `disp` is the
    overdispersion α, so the Fisher weight μ/(1+μ/r_g) = μ/(1 + μ·disp_g). Getting r wrong
    silently breaks the high-depth floor.
The Fisher information is gauge-invariant (rotation θ→θ+Δ,φ→φ+Δ and reflection θ→−θ,φ→−φ both
leave Σ A²sin²(θ−φ) unchanged), so the model-frame MAP phase and acrophase are already consistent
-- no optimal_shift alignment is needed.

This module is the analytic counterpart of the simulation twin behind
`ContextModel.estimate_phase_desynchrony` (`simulate_cell_populations(kappa=∞)`); validated against
it in context_repo/experiments/sigma_tech_rao.py (twin/CRB ≈ 1.1–1.3, the twin riding just above
the CRB lower bound).
"""

import numpy as np

from scritmo import cstd2R, R2cstd


def sigma_tech_rao_per_cell(A, phi, a0, disp, theta, s, kappa_theta=0.0):
    """Analytic Cramér–Rao σ_tech (radians), per cell, for the NB single-harmonic model.

    Parameters
    ----------
    A, phi, a0, disp : array-like, shape (G,)
        Per-gene amplitude, acrophase [rad], log-mesor (a_0), and NB overdispersion
        (Var = μ + disp·μ², i.e. r_g = 1/disp_g).
    theta : array-like, shape (...,)
        Phase(s) [rad] at which to evaluate the floor (broadcast against `s`).
    s : array-like, shape (...,)
        Size factor(s) -- the model's per-cell `counts`.
    kappa_theta : float, default 0.0
        Concentration of an optional per-cell von-Mises prior on θ (posterior precision ≈
        likelihood info + kappa_theta). 0 ⇒ the bare CRB (the uniform-grid MAP). The φ_g soft
        prior does NOT enter here.

    Returns
    -------
    np.ndarray, shape (...,)
        σ_tech per cell in radians (∞ at exact zero-information phases).
    """
    A = np.asarray(A, dtype=float)
    phi = np.asarray(phi, dtype=float)
    a0 = np.asarray(a0, dtype=float)
    disp = np.asarray(disp, dtype=float)
    theta = np.asarray(theta, dtype=float)[..., None]            # (..., 1)
    s = np.asarray(s, dtype=float)[..., None]                    # (..., 1)

    d = theta - phi                                              # (..., G)
    mu = s * np.exp(a0 + A * np.cos(d))
    weight = mu / (1.0 + mu * disp)                              # = μ/(1+μ/r_g)
    info = np.sum(A ** 2 * np.sin(d) ** 2 * weight, axis=-1) + kappa_theta
    with np.errstate(divide="ignore"):
        return 1.0 / np.sqrt(info)                               # radians


def technical_cstd_rao(sigma_rad):
    """Aggregate per-cell CRB σ's into a sample-level Technical_cSTD (radians).

    The sample-level technical floor is the circular spread of the ensemble of MAP estimates.
    Modeling each cell's MAP as a wrapped-normal of width σ_i, the spread that matches `circstd`
    is the resultant of the mixture: R̄ = mean_i exp(−σ_i²/2), Technical_cSTD = √(−2·ln R̄). This
    is bounded as the information → 0 (trough cells: σ_i→∞ ⇒ R_i→0, contributing maximal spread
    without blowing up), and reduces to the quadrature/RMS mean √(mean σ_i²) in the small-angle
    limit (the same mixture-of-posteriors logic the simulation twin's circstd follows).
    """
    sigma_rad = np.asarray(sigma_rad, dtype=float)
    # cstd2R(∞)=0 ⇒ trough/zero-info cells correctly contribute maximal spread (not dropped);
    # nanmean only guards genuine NaNs.
    R_bar = float(np.nanmean(cstd2R(sigma_rad)))                # exp(−σ²/2) per cell, then mean
    return float(R2cstd(R_bar))                                # √(−2 ln R̄); ∞ if R̄→0
