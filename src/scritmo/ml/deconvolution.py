"""Deconvolution estimator of the biological phase desynchrony σ_bio.

Background (the "two shells" note, Review_2/variance_decomposition_two_shells.md).
Inside one group b (one population / sample), the law of total variance is exact:

    V_b = T_b + B_b,    T_b = E_{θ ~ p_b}[ f(θ) ],

where V_b is the observed spread of the inferred phases (cSTD², rad²) and f(θ) is the
technical variance of a perfectly synchronized population at true phase θ. The
per-group "twin" (Eq. 14, ``sigma_tech_method="simulation"``) reads f at ONE point,

    σ̂²_bio,b = V_b − f(μ_b),

i.e. it replaces the bump of true phases p_b by a spike at its centre. Because f is
curved, T_b ≠ f(μ_b) and the error is ≈ ½ σ²_bio,b f''(μ_b) (approximation A1).

This module removes A1 by deconvolving the bump explicitly:

1. Measure f on a uniform grid of n_grid phases φ_k with the σ=0 twin grid
   (:func:`scritmo.ml.simulations.simulate_technical_grid`):
   f(φ_k) = mean over runs of cSTD(post_mode)² (mean of per-run variances, so the
   finite-n bias matches a data group of the same size).
2. Real DFT on the uniform grid, keeping ALL harmonics up to Nyquist (J = n_grid // 2):

       f(θ) = f_0 + Σ_{j=1}^{J} [a_j cos jθ + b_j sin jθ],
       f_0 = (1/N) Σ_k f_k,
       a_j = (2/N) Σ_k f_k cos jφ_k,   b_j = (2/N) Σ_k f_k sin jφ_k   (j < N/2),
       a_{N/2} = (1/N) Σ_k f_k cos(N/2 φ_k), b_{N/2} = (1/N) Σ_k f_k sin(N/2 φ_k)
       (Nyquist, even N: half weight).

   This is the trigonometric interpolant: it reproduces f_k exactly at every grid phase.
3. Wrapped-normal bump of width σ (rad): κ_j(σ) = exp(−j²σ²/2), and

       T_b(σ) = f_0 + Σ_j κ_j(σ) [a_j cos jμ_b + b_j sin jμ_b],   T_b(0) = f(μ_b).

4. ``exact`` form: solve h(σ) = T_b(σ) + σ² = V_b for σ ∈ [0, σ_max] (brentq).
   dh/dσ = σ [2 − Σ_j j² κ_j(σ) c_j(μ_b)],  c_j(μ) = a_j cos jμ + b_j sin jμ.
   Identified ("ok") iff h − V_b has exactly one sign change on a dense scan of
   [0, σ_max] and dh/dσ > 0 at the root. Otherwise σ̂ = NaN with a flag:
     "below_floor"   V_b < h(σ) everywhere (V_b < f(μ_b), no crossing),
     "no_root"       V_b > h(σ_max),
     "non_monotone"  several crossings, or dh/dσ ≤ 0 at the root.
   No clamping is done here.
5. ``taylor`` form (closed, first order in σ²):

       σ̂² = (V_b − f(μ_b)) / (1 + ½ f''(μ_b)),   f''(μ) = −Σ_j j² c_j(μ).

   NaN + flag "denominator" if 1 + ½ f'' ≤ eps, "below_floor" if V_b < f(μ_b),
   "negative_tech" if σ̂² > V_b (implied technical term would be negative).
6. Implied technical term: exact → T_b(σ̂_b); taylor → V_b − σ̂²_b. In both cases
   V_b = Tech_b + σ̂²_b per group by construction, so the Shell-2 pool in
   :func:`scritmo.ml.analysis_utils.desync_means` (mean V − mean Tech) equals the
   weighted mean of σ̂²_b over the identified groups.

VECTOR form (``deconv_form="vector"``) — the forms above model only the smoothing of f
(approximation A1). The twin's real per-group error also contains the attractor-bias terms
(A2: σ²(2β'+β'²), β = m(θ) − θ), the placement term (A3) and the circular log-Jensen term
(−¼ f'² σ²). All of them are exact consequences of ONE first-moment identity (law of total
expectation, note §6a):

    E_b[e^{iθ̂}] = E_{θ~p_b}[ρ(θ)],     ρ(θ) = E[e^{iθ̂} | θ] = r(θ) e^{i m(θ)}.

1. From the grid: ρ(φ_k) = plain mean of exp(i·post_mode) over ALL twin cells at φ_k (runs
   pooled) — unbiased at any n, and it keeps the DIRECTION m(θ) (the attractor bias).
2. Complex DFT on the uniform grid: ρ(θ) = Σ_{j=-J..J} c_j e^{ijθ}; for even N the Nyquist
   coefficient is split half/half between j = ±N/2 so the series interpolates ρ_k exactly
   and is smoothed symmetrically.
3. ρ̄_b(σ) = Σ_j c_j κ_{|j|}(σ) e^{ijμ_b}  (wrapped-normal bump around μ_b).
4. Data: z̄_b = mean over the n_b cells of exp(i·post_mode), R̄²_b = |z̄_b|²
   (= exp(−Data_cSTD²), since cSTD = √(−2 ln R̄)).
5. Solve  L(σ) = |ρ̄_b(σ)|² + (1 − |ρ̄_b(σ)|²)/n_b = R̄²_b  for σ ∈ [0, σ_max]. The 1/n_b term
   is the exact finite-n bias of |z̄|² for i.i.d. cells, E|z̄|² = |E z|² + (1 − |E z|²)/n.
   Identified iff L − R̄² changes sign exactly once on a dense scan and dL/dσ < 0 at the root.
   "below_floor": R̄² > L(0) (data tighter than the σ = 0 prediction); "no_root":
   R̄² < L(σ_max); "non_monotone": several crossings / dL/dσ ≥ 0 at the root.
6. Implied technical term: Tech² = Data_cSTD² − σ̂² (NaN when σ̂ > Data_cSTD, which can happen
   because −2 ln R̄ is not additive when r(θ) varies; σ̂ itself is kept).

All angles are radians, all variances rad² (convert with ``scritmo.rh`` at the end).
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import circmean

import scritmo as sr
from scritmo import cstd2R

DECONV_FLAGS = (
    "ok", "below_floor", "non_monotone", "no_root",  # exact / vector
    "denominator", "negative_tech",                   # taylor only
)
DECONV_FORMS = ("exact", "taylor", "vector")


# ---------------------------------------------------------------------------
# 1-2. grid -> Fourier coefficients
# ---------------------------------------------------------------------------
def grid_fourier_coefficients(grid_phase, f_grid, n_harmonics=None):
    """Real Fourier coefficients of f sampled on a UNIFORM grid (trigonometric interpolant).

    Parameters
    ----------
    grid_phase : array-like, shape (N,)
        Grid phases (rad), uniformly spaced over one period (any offset, any order).
    f_grid : array-like, shape (N,)
        f(φ_k), the technical variance (rad²) at each grid phase.
    n_harmonics : int, optional
        Highest harmonic kept. None (default) keeps all harmonics up to Nyquist
        (J = N // 2). Only meant for diagnostics; truncating is a smoothing choice.

    Returns
    -------
    dict
        ``{"f0", "j", "a", "b", "n_grid"}`` with ``j = 1..J`` (int array) and ``a``/``b``
        the cosine/sine coefficients (arrays of length J). With even N the Nyquist
        coefficients (j = N/2) carry half weight, so the series interpolates f_k exactly.
    """
    phi = np.asarray(grid_phase, dtype=float)
    f = np.asarray(f_grid, dtype=float)
    if phi.shape != f.shape or phi.ndim != 1:
        raise ValueError("grid_phase and f_grid must be 1-D arrays of equal length")
    n = phi.size
    if n < 3:
        raise ValueError("need at least 3 grid phases")
    if not np.all(np.isfinite(f)):
        raise ValueError("f_grid contains non-finite values")
    # uniform-spacing check (mod 2π, any rotation/order)
    srt = np.sort(np.mod(phi, 2 * np.pi))
    gaps = np.diff(np.concatenate([srt, [srt[0] + 2 * np.pi]]))
    if not np.allclose(gaps, 2 * np.pi / n, atol=1e-6):
        raise ValueError("grid phases are not uniformly spaced over [0, 2π)")

    j_max = n // 2
    if n_harmonics is not None:
        j_max = min(int(n_harmonics), j_max)
    j = np.arange(1, j_max + 1)
    cos_jk = np.cos(np.outer(j, phi))
    sin_jk = np.sin(np.outer(j, phi))
    weight = np.full(j.size, 2.0 / n)
    if n % 2 == 0 and j_max == n // 2:
        weight[-1] = 1.0 / n  # Nyquist: half weight
    return {
        "f0": float(f.mean()),
        "j": j,
        "a": weight * (cos_jk @ f),
        "b": weight * (sin_jk @ f),
        "n_grid": int(n),
    }


def kappa_wrapped_normal(j, sigma):
    """Fourier coefficients of a wrapped-normal kernel of width σ (rad): exp(−j²σ²/2)."""
    j = np.asarray(j, dtype=float)
    return np.exp(-0.5 * j**2 * float(sigma) ** 2)


def _c_j(mu, coef):
    """c_j(μ) = a_j cos jμ + b_j sin jμ (array over j)."""
    j = coef["j"]
    return coef["a"] * np.cos(j * mu) + coef["b"] * np.sin(j * mu)


def eval_fourier(theta, coef):
    """f(θ) from the Fourier coefficients (array-valued in θ)."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    jt = np.outer(theta, coef["j"])
    out = coef["f0"] + np.cos(jt) @ coef["a"] + np.sin(jt) @ coef["b"]
    return out


def f_at(mu, coef):
    """f(μ) = T(μ, σ=0): the single-point twin floor read from the series."""
    return float(coef["f0"] + np.sum(_c_j(mu, coef)))


def f_second_derivative(mu, coef):
    """f''(μ) = −Σ_j j² c_j(μ)."""
    return float(-np.sum(coef["j"] ** 2 * _c_j(mu, coef)))


def technical_term(mu, sigma, coef):
    """T(μ, σ) = f_0 + Σ_j κ_j(σ) c_j(μ): f averaged under a wrapped normal N(μ, σ)."""
    return float(coef["f0"] + np.sum(kappa_wrapped_normal(coef["j"], sigma) * _c_j(mu, coef)))


def _h_and_dh(mus, weights, coef):
    """Build h(σ) = Σ_b w_b T(μ_b, σ) + σ² and its derivative, vectorised in σ.

    With a single (μ, w=1) this is the per-group h of the exact form; with several it is
    the pooled ("common-σ") version Σ_b w_b V_b = Σ_b w_b T_b(σ) + σ².
    """
    mus = np.atleast_1d(np.asarray(mus, dtype=float))
    w = np.atleast_1d(np.asarray(weights, dtype=float))
    w = w / w.sum()
    j = coef["j"].astype(float)
    # weighted c_j over groups: C_j = Σ_b w_b c_j(μ_b)
    C = (w[:, None] * (coef["a"][None, :] * np.cos(np.outer(mus, j))
                       + coef["b"][None, :] * np.sin(np.outer(mus, j)))).sum(0)

    def h(s):
        s = np.asarray(s, dtype=float)
        k = np.exp(-0.5 * np.multiply.outer(s**2, j**2))
        return coef["f0"] + k @ C + s**2

    def dh(s):
        s = np.asarray(s, dtype=float)
        k = np.exp(-0.5 * np.multiply.outer(s**2, j**2))
        return s * (2.0 - k @ (j**2 * C))

    return h, dh


def _solve_monotone(h, dh, V, sigma_max, n_scan):
    """Unique root of h(σ) = V on [0, σ_max]. Returns (σ̂, flag, n_crossings, monotone)."""
    s = np.linspace(0.0, sigma_max, int(n_scan))
    g = h(s) - V
    monotone = bool(np.all(np.diff(h(s)) > 0))
    pos = g >= 0
    idx = np.flatnonzero(pos[1:] != pos[:-1])
    n_cross = int(idx.size)
    if n_cross == 0:
        if g[0] == 0.0:
            return 0.0, "ok", 0, monotone
        return np.nan, ("below_floor" if g[0] > 0 else "no_root"), 0, monotone
    if n_cross > 1 or g[0] > 0:
        # several crossings, or h starts ABOVE V and still crosses (so it dips below V)
        return np.nan, "non_monotone", n_cross, monotone
    i = int(idx[0])
    root = brentq(lambda x: float(h(x) - V), s[i], s[i + 1], xtol=1e-14, rtol=1e-14)
    if not float(dh(root)) > 0.0:
        return np.nan, "non_monotone", n_cross, monotone
    return float(root), "ok", n_cross, monotone


def solve_exact(V, mu, coef, sigma_max=np.pi, n_scan=4001):
    """Exact deconvolution for one group: solve T(μ, σ) + σ² = V for σ ≥ 0.

    Parameters
    ----------
    V : float
        Observed variance of the group's inferred phases (cSTD², rad²).
    mu : float
        Group phase (rad) — the same phase the twin would be generated at.
    coef : dict
        Output of :func:`grid_fourier_coefficients`.
    sigma_max : float, default π
        Upper end of the bracket (rad; π = 12 h).
    n_scan : int, default 4001
        Dense-scan resolution used to count crossings (identifiability check).

    Returns
    -------
    dict
        ``sigma`` (rad, NaN unless flag == "ok"), ``flag``, ``n_crossings``,
        ``monotone`` (h strictly increasing on the whole scan), ``T_hat`` = T(μ, σ̂).
    """
    h, dh = _h_and_dh([mu], [1.0], coef)
    sig, flag, n_cross, mono = _solve_monotone(h, dh, float(V), sigma_max, n_scan)
    return {
        "sigma": sig,
        "flag": flag,
        "n_crossings": n_cross,
        "monotone": mono,
        "T_hat": technical_term(mu, sig, coef) if flag == "ok" else np.nan,
    }


def solve_taylor(V, mu, coef, eps=1e-3):
    """First-order closed form σ̂² = (V − f(μ)) / (1 + ½ f''(μ)).

    Returns
    -------
    dict
        ``sigma`` (rad, NaN unless "ok"),
        ``flag`` ∈ {"ok", "below_floor", "denominator", "negative_tech"},
        ``denominator`` = 1 + ½ f''(μ), ``T_hat`` = V − σ̂² (implied technical term).
    """
    num = float(V) - f_at(mu, coef)
    den = 1.0 + 0.5 * f_second_derivative(mu, coef)
    if not den > eps:
        return {"sigma": np.nan, "flag": "denominator", "denominator": den, "T_hat": np.nan}
    if num < 0:
        return {"sigma": np.nan, "flag": "below_floor", "denominator": den, "T_hat": np.nan}
    s2 = num / den
    if float(V) - s2 < 0:
        # den < 1 (f'' < 0, near a peak of f) can push σ̂² above V itself: the implied
        # technical term would be negative, so the split is not interpretable.
        return {"sigma": np.nan, "flag": "negative_tech", "denominator": den,
                "T_hat": np.nan}
    return {"sigma": float(np.sqrt(s2)), "flag": "ok", "denominator": den,
            "T_hat": float(V) - s2}


def solve_common_sigma(V, mus, weights, coef, sigma_max=np.pi, n_scan=4001):
    """Pooled (Shell-2) deconvolution with ONE σ for all groups:

        Σ_b w_b V_b = Σ_b w_b T(μ_b, σ) + σ².

    ``V``, ``mus``, ``weights`` are per-group arrays (weights need not be normalised).
    Returns the same dict as :func:`solve_exact` (``T_hat`` = Σ_b w_b T(μ_b, σ̂)).
    """
    V = np.asarray(V, dtype=float)
    w = np.asarray(weights, dtype=float)
    Vbar = float(np.sum(w * V) / np.sum(w))
    h, dh = _h_and_dh(mus, w, coef)
    sig, flag, n_cross, mono = _solve_monotone(h, dh, Vbar, sigma_max, n_scan)
    return {
        "sigma": sig,
        "flag": flag,
        "n_crossings": n_cross,
        "monotone": mono,
        "T_hat": float(h(sig) - sig**2) if flag == "ok" else np.nan,
        "V_bar": Vbar,
    }


# ---------------------------------------------------------------------------
# VECTOR form: complex mean resultant ρ(θ) = E[e^{iθ̂} | θ]
# ---------------------------------------------------------------------------
def grid_complex_fourier(grid_phase, rho_grid):
    """Complex Fourier series of ρ sampled on a UNIFORM grid (exact interpolant).

    ρ(θ) = Σ_{j=-J..J} c_j e^{ijθ}, c_j = (1/N) Σ_k ρ_k e^{-ijφ_k}. For even N the Nyquist
    coefficient is split in half between j = +N/2 and j = −N/2 (each gets
    ½·(1/N) Σ_k ρ_k e^{∓i(N/2)φ_k}), which reproduces ρ_k exactly at every grid phase and keeps
    the smoothing symmetric in |j|.

    Returns
    -------
    dict
        ``{"j": int array (−J..J), "c": complex array, "n_grid"}``.
    """
    phi = np.asarray(grid_phase, dtype=float)
    rho = np.asarray(rho_grid, dtype=complex)
    if phi.shape != rho.shape or phi.ndim != 1:
        raise ValueError("grid_phase and rho_grid must be 1-D arrays of equal length")
    n = phi.size
    if n < 3:
        raise ValueError("need at least 3 grid phases")
    if not np.all(np.isfinite(rho)):
        raise ValueError("rho_grid contains non-finite values")
    srt = np.sort(np.mod(phi, 2 * np.pi))
    gaps = np.diff(np.concatenate([srt, [srt[0] + 2 * np.pi]]))
    if not np.allclose(gaps, 2 * np.pi / n, atol=1e-6):
        raise ValueError("grid phases are not uniformly spaced over [0, 2π)")
    J = n // 2
    j = np.arange(-J, J + 1)
    c = (np.exp(-1j * np.outer(j, phi)) @ rho) / n
    if n % 2 == 0:  # (odd N has no Nyquist term: J = (N-1)/2)
        c[0] *= 0.5   # j = -N/2
        c[-1] *= 0.5  # j = +N/2
    return {"j": j, "c": c, "n_grid": int(n)}


def eval_complex_fourier(theta, cc):
    """ρ(θ) from :func:`grid_complex_fourier` coefficients (complex, array in θ)."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    return np.exp(1j * np.outer(theta, cc["j"])) @ cc["c"]


def resultant_under_bump(mu, sigma, cc):
    """ρ̄(μ, σ) = Σ_j c_j κ_{|j|}(σ) e^{ijμ}: E[e^{iθ̂}] for θ ~ WrappedNormal(μ, σ)."""
    k = kappa_wrapped_normal(np.abs(cc["j"]), sigma)
    return complex(np.sum(cc["c"] * k * np.exp(1j * cc["j"] * mu)))


def _vector_L_and_dL(mu, cc, n):
    """L(σ) = (1 − 1/n)|ρ̄(σ)|² + 1/n and dL/dσ, vectorised in σ."""
    j = cc["j"].astype(float)
    base = cc["c"] * np.exp(1j * j * mu)  # c_j e^{ijμ}
    inv_n = 0.0 if not np.isfinite(n) else 1.0 / float(n)

    def rho_bar(s):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        return np.exp(-0.5 * np.multiply.outer(s**2, j**2)) @ base

    def L(s):
        rb = rho_bar(s)
        return (1.0 - inv_n) * np.abs(rb) ** 2 + inv_n

    def dL(s):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        k = np.exp(-0.5 * np.multiply.outer(s**2, j**2))
        rb = k @ base
        drb = (k * (-(j**2))[None, :] * s[:, None]) @ base
        return (1.0 - inv_n) * 2.0 * np.real(np.conj(rb) * drb)

    return L, dL, rho_bar


def solve_vector(R2, n, mu, cc, sigma_max=np.pi, n_scan=4001):
    """Vector-form deconvolution for one group:
    solve |ρ̄(σ)|² + (1 − |ρ̄(σ)|²)/n = R̄² for σ ≥ 0.

    Parameters
    ----------
    R2 : float
        |z̄_b|², squared mean resultant length of the group's inferred phases.
    n : int
        Number of cells in the group (finite-n bias term). ``np.inf`` drops it.
    mu : float
        Group phase (rad), the phase the twin would be generated at.
    cc : dict
        Output of :func:`grid_complex_fourier`.

    Returns
    -------
    dict
        ``sigma`` (rad, NaN unless "ok"), ``flag``, ``n_crossings``, ``rho0_abs`` = |ρ̄(0)|,
        ``L0`` = L(0), ``pred_dir`` = arg ρ̄(σ̂) (rad in [0, 2π); NaN unless "ok").
    """
    L, dL, rho_bar = _vector_L_and_dL(mu, cc, n)
    s = np.linspace(0.0, sigma_max, int(n_scan))
    g = L(s) - float(R2)  # decreasing in σ for an identified group: g(0) >= 0 > g(σ*)
    rho0 = float(np.abs(rho_bar(0.0)[0]))
    out = {"rho0_abs": rho0, "L0": float(g[0] + R2), "n_crossings": 0,
           "sigma": np.nan, "pred_dir": np.nan}
    if abs(g[0]) <= 1e-12:  # data exactly at the σ = 0 prediction
        out.update(sigma=0.0, flag="ok",
                   pred_dir=float(np.angle(rho_bar(0.0)[0]) % (2 * np.pi)))
        return out
    neg = g < 0
    idx = np.flatnonzero(neg[1:] != neg[:-1])
    out["n_crossings"] = int(idx.size)
    if idx.size == 0:
        out["flag"] = "below_floor" if g[0] < 0 else "no_root"
        return out
    if idx.size > 1 or g[0] < 0:
        out["flag"] = "non_monotone"
        return out
    i = int(idx[0])
    root = brentq(lambda x: float(L(x)[0] - R2), s[i], s[i + 1], xtol=1e-14, rtol=1e-14)
    if not float(dL(root)[0]) < 0.0:
        out["flag"] = "non_monotone"
        return out
    out.update(sigma=float(root), flag="ok",
               pred_dir=float(np.angle(rho_bar(root)[0]) % (2 * np.pi)))
    return out


def grid_resultant_curve(df_grid, post_estimator="post_mode"):
    """Per context and grid phase: ρ(φ_k) = mean of exp(i·post_estimator) over ALL twin cells
    at φ_k (runs pooled). A plain mean, so unbiased at any n.

    Returns
    -------
    pandas.DataFrame
        context, grid_idx, grid_phase, rho_re, rho_im, rho_abs, rho_dir (rad), n_cells.
    """
    rows = []
    for (ctx, k), df_pt in df_grid.groupby(["context", "grid_idx"]):
        z = np.mean(np.exp(1j * df_pt[post_estimator].values.astype(float)))
        rows.append(dict(
            context=str(ctx), grid_idx=int(k),
            grid_phase=float(df_pt["grid_phase"].iloc[0]),
            rho_re=float(z.real), rho_im=float(z.imag), rho_abs=float(np.abs(z)),
            rho_dir=float(np.angle(z) % (2 * np.pi)), n_cells=int(len(df_pt)),
        ))
    return pd.DataFrame(rows).sort_values(["context", "grid_idx"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# grid table -> f(φ_k)
# ---------------------------------------------------------------------------
def grid_variance_curve(df_grid, post_estimator="post_mode"):
    """Per context: f(φ_k) = mean over runs of cSTD(post_estimator)² on the twin grid.

    Returns
    -------
    (curve, per_run) : (pandas.DataFrame, pandas.DataFrame)
        ``curve``: context, grid_idx, grid_phase, f (rad², mean over runs), n_runs.
        ``per_run``: context, grid_idx, grid_phase, run_id, var (rad²), n_cells.
    """
    rows = []
    for (ctx, k, run), df_pt in df_grid.groupby(["context", "grid_idx", "run_id"]):
        rows.append(
            dict(
                context=str(ctx),
                grid_idx=int(k),
                grid_phase=float(df_pt["grid_phase"].iloc[0]),
                run_id=str(run),
                var=float(sr.cSTD(df_pt[post_estimator].values) ** 2),
                n_cells=int(len(df_pt)),
            )
        )
    per_run = pd.DataFrame(rows)
    curve = (
        per_run.groupby(["context", "grid_idx"])
        .agg(grid_phase=("grid_phase", "first"), f=("var", "mean"), n_runs=("var", "size"))
        .reset_index()
        .sort_values(["context", "grid_idx"])
    )
    return curve, per_run


# ---------------------------------------------------------------------------
# aggregation (drop-in for aggregate_technical_harmonic)
# ---------------------------------------------------------------------------
def aggregate_technical_deconvolution(
    df_grid: pd.DataFrame,
    df_real: pd.DataFrame,
    group_cols: list = None,
    post_estimator: str = "post_mode",
    disp_function=sr.cSTD,
    n_replicates: int | None = None,
    seed: int = 42,
    weight_col: str | None = None,
    use_circular_mean: bool = False,
    ext_time_col: str = "ext_time_hours",
    period: float = 24.0,
    deconv_form: str = "exact",
    sigma_max: float = np.pi,
    n_scan: int = 4001,
    taylor_eps: float = 1e-3,
    n_harmonics: int | None = None,
):
    """Deconvolution technical floor, same schema as `aggregate_simulated_results`
    (context, sample_name, Technical_cSTD [rad], Technical_R) plus the deconvolution
    columns, so `desync_results(..., sim_agg=...)` consumes it unchanged.

    Steps (see the module docstring for the formulas):
      1. f(φ_k) per context from the σ=0 twin grid (:func:`grid_variance_curve`), then its
         full Fourier series (:func:`grid_fourier_coefficients`).
      2. V_b per output group = the SAME Data_cSTD² that `desync_results` computes
         (:func:`scritmo.ml.analysis_utils.aggregate_real_results` with identical
         arguments), so the rows line up exactly, including the ``n_replicates`` splits.
      3. μ_b per (context, sample): the sample's external time (use_circular_mean=False)
         or the circular mean of its inferred phases (True) — the phase the simulation
         twin is generated at. With ``n_replicates`` the sample-level μ is broadcast to
         its ``_1.._n`` splits (as in `aggregate_technical_harmonic`); each split keeps
         its own V_b, so σ̂ is solved per split.
      4. Solve per row with :func:`solve_exact`, :func:`solve_taylor` or, for
         ``deconv_form="vector"``, :func:`solve_vector` on the complex resultant of the grid
         (R̄² = exp(−Data_cSTD²), n = group_size; see the module docstring).

    Returns
    -------
    (table, diag) : (pandas.DataFrame, dict)
        ``table`` columns: group cols, Technical_cSTD (rad; implied, NaN if not
        identified), Technical_R, deconv_sigma (rad, NaN if not identified),
        deconv_flag, deconv_V, deconv_mu (rad), deconv_f_mu (= T(μ,0), rad²),
        deconv_f2_mu (f''(μ)), deconv_T_hat (rad²), deconv_denominator (taylor) /
        deconv_n_crossings (exact, vector). Vector adds deconv_R2, deconv_n,
        deconv_rho0_abs (|ρ̄(0)|), deconv_L0, deconv_pred_dir (arg ρ̄(σ̂)),
        deconv_data_dir (circmean of the group's inferred phases, sample level) and
        deconv_neg_tech (σ̂ > Data_cSTD, implied Tech undefined).
        ``diag``: {context: {"curve", "per_run", "coef"}}.
    """
    from .analysis_utils import aggregate_real_results  # local: avoid import cycle

    if group_cols is None:
        group_cols = ["context", "sample_name"]
    if deconv_form not in DECONV_FORMS:
        raise ValueError(f"deconv_form must be one of {DECONV_FORMS}, got {deconv_form!r}")

    # --- 1. grid -> f(φ_k) -> Fourier coefficients, per context ---
    curve, per_run = grid_variance_curve(df_grid, post_estimator=post_estimator)
    diag = {}
    for ctx, c in curve.groupby("context"):
        coef = grid_fourier_coefficients(
            c["grid_phase"].values, c["f"].values, n_harmonics=n_harmonics
        )
        diag[str(ctx)] = {
            "curve": c.reset_index(drop=True),
            "per_run": per_run[per_run.context == ctx].reset_index(drop=True),
            "coef": coef,
        }
    if deconv_form == "vector":
        rcurve = grid_resultant_curve(df_grid, post_estimator=post_estimator)
        for ctx, c in rcurve.groupby("context"):
            diag[str(ctx)]["rho_curve"] = c.reset_index(drop=True)
            diag[str(ctx)]["cc"] = grid_complex_fourier(
                c["grid_phase"].values, c["rho_re"].values + 1j * c["rho_im"].values
            )

    # --- 2. V_b exactly as desync_results will compute Data_cSTD ---
    real_agg = aggregate_real_results(
        df_real,
        group_cols=group_cols,
        disp_function=disp_function,
        post_estimator=post_estimator,
        metrics={post_estimator: disp_function},
        n_replicates=n_replicates,
        seed=seed,
        weight_col=weight_col,
    )

    # --- 3. μ per (context, sample) ---
    df_r = df_real.copy()
    for col in group_cols:
        df_r[col] = df_r[col].astype(str)
    have_ext = ext_time_col in df_r.columns
    if not use_circular_mean and not have_ext:
        print(
            f"  WARNING: use_circular_mean=False needs '{ext_time_col}' in the results "
            "frame; falling back to the circular mean of the inferred phases."
        )
    mu_of, dir_of = {}, {}
    for keys, grp in df_r.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        cm = float(circmean(grp[post_estimator].values, high=2 * np.pi, low=0))
        dir_of[keys] = cm
        if use_circular_mean or not have_ext:
            mu = cm
        else:
            mu = float((float(grp[ext_time_col].iloc[0]) % period) / period * 2 * np.pi)
        mu_of[keys] = mu

    # --- 4. solve per output row ---
    rows = []
    for _, r in real_agg.iterrows():
        keys = tuple(str(r[c]) for c in group_cols)
        if n_replicates is not None:
            # "<sample>_<k.0>" -> "<sample>" (aggregate_real_results' split renaming)
            base = keys[-1].rsplit("_", 1)[0]
            mu_key = keys[:-1] + (base,)
        else:
            mu_key = keys
        ctx = str(dict(zip(group_cols, keys))["context"])
        coef = diag[ctx]["coef"]
        mu = mu_of[mu_key]
        V = float(r["Data_cSTD"]) ** 2
        f_mu = f_at(mu, coef)
        f2 = f_second_derivative(mu, coef)
        if deconv_form == "exact":
            sol = solve_exact(V, mu, coef, sigma_max=sigma_max, n_scan=n_scan)
            extra = {"deconv_n_crossings": sol["n_crossings"],
                     "deconv_monotone": sol["monotone"]}
        elif deconv_form == "vector":
            n_b = float(r["group_size"])
            R2 = float(np.exp(-V))  # cSTD = sqrt(-2 ln R)  ->  R^2 = exp(-cSTD^2)
            sol = solve_vector(R2, n_b, mu, diag[ctx]["cc"], sigma_max=sigma_max,
                               n_scan=n_scan)
            sig = sol["sigma"]
            sol["T_hat"] = (V - sig**2) if sol["flag"] == "ok" else np.nan
            extra = {"deconv_n_crossings": sol["n_crossings"],
                     "deconv_R2": R2, "deconv_n": n_b,
                     "deconv_rho0_abs": sol["rho0_abs"], "deconv_L0": sol["L0"],
                     "deconv_pred_dir": sol["pred_dir"],
                     "deconv_data_dir": dir_of[mu_key],
                     "deconv_neg_tech": bool(sol["flag"] == "ok" and V - sig**2 < 0)}
        else:
            sol = solve_taylor(V, mu, coef, eps=taylor_eps)
            extra = {"deconv_denominator": sol["denominator"]}
        T_hat = sol["T_hat"]
        rows.append(
            dict(
                zip(group_cols, keys),
                Technical_cSTD=(np.sqrt(T_hat) if np.isfinite(T_hat) and T_hat >= 0
                                else np.nan),
                deconv_sigma=sol["sigma"],
                deconv_flag=sol["flag"],
                deconv_V=V,
                deconv_mu=mu,
                deconv_f_mu=f_mu,
                deconv_f2_mu=f2,
                deconv_T_hat=T_hat,
                deconv_form=deconv_form,
                **extra,
            )
        )
    table = pd.DataFrame(rows)
    table["Technical_R"] = cstd2R(table["Technical_cSTD"])
    return table, diag
