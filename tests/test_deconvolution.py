"""Tests for the deconvolution σ_bio estimator (scritmo.ml.deconvolution)."""

import numpy as np
import pandas as pd
import pytest

from scritmo import rh
from scritmo.ml.analysis_utils import desync_results, desync_means
from scritmo.ml.context_model import ContextModel
from scritmo.ml.deconvolution import (
    aggregate_technical_deconvolution,
    eval_fourier,
    f_at,
    f_second_derivative,
    grid_fourier_coefficients,
    solve_common_sigma,
    solve_exact,
    solve_taylor,
    technical_term,
)

N_GRID = 24
GRID = np.linspace(0, 2 * np.pi, N_GRID, endpoint=False)

# a known, strictly positive f with several harmonics (all below Nyquist)
F0 = 0.30
A = {1: 0.08, 2: -0.03, 3: 0.015, 5: 0.004}
B = {1: -0.05, 2: 0.02, 4: -0.006}


def f_true(theta):
    theta = np.asarray(theta, dtype=float)
    out = np.full(theta.shape, F0)
    for j, a in A.items():
        out = out + a * np.cos(j * theta)
    for j, b in B.items():
        out = out + b * np.sin(j * theta)
    return out


def wrapped_normal_pdf(theta, mu, sigma, n_wrap=12):
    k = np.arange(-n_wrap, n_wrap + 1)
    d = theta[:, None] - mu + 2 * np.pi * k[None, :]
    return np.exp(-0.5 * (d / sigma) ** 2).sum(1) / (sigma * np.sqrt(2 * np.pi))


def V_numerical(mu, sigma, f=f_true, n=200_000):
    """E_{θ~WN(μ,σ)}[f(θ)] + σ² by brute-force quadrature on the circle."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = wrapped_normal_pdf(th, mu, sigma)
    dth = 2 * np.pi / n
    return float(np.sum(p * f(th)) * dth) + sigma**2


@pytest.fixture(scope="module")
def coef():
    return grid_fourier_coefficients(GRID, f_true(GRID))


def test_fourier_recovers_bandlimited_coefficients(coef):
    assert coef["f0"] == pytest.approx(F0, abs=1e-12)
    assert len(coef["j"]) == N_GRID // 2
    for j in range(1, N_GRID // 2 + 1):
        assert coef["a"][j - 1] == pytest.approx(A.get(j, 0.0), abs=1e-12)
        assert coef["b"][j - 1] == pytest.approx(B.get(j, 0.0), abs=1e-12)
    th = np.linspace(0, 2 * np.pi, 97)
    np.testing.assert_allclose(eval_fourier(th, coef), f_true(th), atol=1e-12)


def test_fourier_interpolates_nyquist_and_noise():
    # arbitrary (noisy) grid values incl. a Nyquist component: the series must
    # reproduce every grid point exactly (half-weight Nyquist term)
    rng = np.random.default_rng(0)
    vals = 0.5 + 0.1 * rng.standard_normal(N_GRID) + 0.07 * np.cos(12 * GRID)
    c = grid_fourier_coefficients(GRID, vals)
    np.testing.assert_allclose(eval_fourier(GRID, c), vals, atol=1e-12)
    # and a pure Nyquist cosine is recovered with its full amplitude
    c2 = grid_fourier_coefficients(GRID, 0.2 + 0.07 * np.cos(12 * GRID))
    assert c2["a"][-1] == pytest.approx(0.07, abs=1e-12)
    # offset / shuffled uniform grid works as well
    off = (GRID + 0.1)[::-1]
    c3 = grid_fourier_coefficients(off, f_true(off))
    np.testing.assert_allclose(eval_fourier(GRID, c3), f_true(GRID), atol=1e-12)


def test_non_uniform_grid_raises():
    with pytest.raises(ValueError):
        grid_fourier_coefficients(np.sort(np.random.default_rng(1).uniform(0, 6, 24)),
                                  np.ones(24))


def test_technical_term_matches_quadrature(coef):
    for mu in (0.0, 1.3, 4.0):
        for sigma in (0.0, 0.2, 0.7):
            if sigma == 0:
                assert technical_term(mu, 0.0, coef) == pytest.approx(f_true(mu), abs=1e-12)
                continue
            T_num = V_numerical(mu, sigma) - sigma**2
            assert technical_term(mu, sigma, coef) == pytest.approx(T_num, abs=1e-9)


@pytest.mark.parametrize("mu", [0.0, 0.9, 2.5, 3.7, 5.6])
@pytest.mark.parametrize("sigma", [0.05, 0.26, 0.6, 1.2])
def test_exact_recovers_sigma(coef, mu, sigma):
    V = V_numerical(mu, sigma)
    sol = solve_exact(V, mu, coef)
    assert sol["flag"] == "ok"
    assert sol["sigma"] == pytest.approx(sigma, abs=1e-6)
    # implied split is exact: V = T(σ̂) + σ̂²
    assert sol["T_hat"] + sol["sigma"] ** 2 == pytest.approx(V, abs=1e-10)


def test_taylor_accurate_at_small_sigma(coef):
    for mu in (0.0, 2.5, 5.6):
        sigma = 0.05
        V = V_numerical(mu, sigma)
        sol = solve_taylor(V, mu, coef)
        assert sol["flag"] == "ok"
        assert sol["sigma"] == pytest.approx(sigma, rel=1e-3)
        # and it is strictly better than the plain twin sqrt(V - f(mu))
        twin = np.sqrt(max(V - f_at(mu, coef), 0.0))
        assert abs(sol["sigma"] - sigma) <= abs(twin - sigma) + 1e-12
    # at large sigma the first-order form degrades but stays finite here
    V = V_numerical(0.9, 1.2)
    assert np.isfinite(solve_taylor(V, 0.9, coef)["sigma"])


def test_second_derivative(coef):
    mu, h = 1.1, 1e-4
    fd = (f_true(mu + h) - 2 * f_true(mu) + f_true(mu - h)) / h**2
    assert f_second_derivative(mu, coef) == pytest.approx(float(fd), abs=1e-5)


def test_flags(coef):
    mu = 1.0
    f_mu = f_at(mu, coef)
    # below the floor: V < f(mu)
    assert solve_exact(f_mu - 0.01, mu, coef)["flag"] == "below_floor"
    assert np.isnan(solve_exact(f_mu - 0.01, mu, coef)["sigma"])
    assert solve_taylor(f_mu - 0.01, mu, coef)["flag"] == "below_floor"
    # exactly at the floor -> sigma 0
    assert solve_exact(f_mu, mu, coef)["sigma"] == pytest.approx(0.0, abs=1e-12)
    # above h(sigma_max)
    assert solve_exact(20.0, mu, coef)["flag"] == "no_root"

    # a sharply peaked f: f''(0) = -9*0.4 < -2 -> h decreases first (non-monotone),
    # and the Taylor denominator 1 + f''/2 is negative
    peak = grid_fourier_coefficients(GRID, 0.6 + 0.4 * np.cos(3 * GRID))
    assert 1 + 0.5 * f_second_derivative(0.0, peak) < 0
    assert solve_taylor(1.2, 0.0, peak)["flag"] == "denominator"
    s = np.linspace(0, np.pi, 4001)
    h = np.array([technical_term(0.0, x, peak) + x**2 for x in s])
    V_mid = 0.5 * (h[0] + h.min())  # between the dip and h(0): two roots
    sol = solve_exact(V_mid, 0.0, peak)
    assert sol["flag"] == "non_monotone" and np.isnan(sol["sigma"])
    assert not sol["monotone"]


def test_common_sigma(coef):
    mus = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    w = np.array([1.0, 2.0, 1.0, 3.0, 1.0, 1.0])
    sigma = 0.4
    V = np.array([V_numerical(m, sigma) for m in mus])
    sol = solve_common_sigma(V, mus, w, coef)
    assert sol["flag"] == "ok"
    assert sol["sigma"] == pytest.approx(sigma, abs=1e-6)


# ---------------------------------------------------------------------------
# aggregation plumbing
# ---------------------------------------------------------------------------
def _wn(rng, loc, sd, n):
    return np.mod(loc + sd * rng.standard_normal(n), 2 * np.pi)


def _synthetic_frames(rng, n_grid_cells=20_000, n_real=30_000, sigma_bio=0.3):
    rows = []
    for k, phi in enumerate(GRID):
        for r in range(2):
            pm = _wn(rng, phi, np.sqrt(f_true(phi)), n_grid_cells)
            rows.append(pd.DataFrame(dict(
                context="c", grid_idx=k, grid_phase=phi, run_id=f"run{r}",
                post_mode=pm, sample_name=f"grid{k}_run{r}",
            )))
    df_grid = pd.concat(rows, ignore_index=True)
    real = []
    mus_h = np.array([0.0, 8.0, 16.0])
    for i, mh in enumerate(mus_h):
        mu = mh / rh
        th = _wn(rng, mu, sigma_bio, n_real)
        pm = np.mod(th + np.sqrt(f_true(th)) * rng.standard_normal(n_real), 2 * np.pi)
        real.append(pd.DataFrame(dict(
            context="c", sample_name=f"s{i}", post_mode=pm, ext_time_hours=mh,
            MAE=0.0, post_std_c=0.1,
        )))
    return df_grid, pd.concat(real, ignore_index=True)


@pytest.mark.parametrize("form", ["exact", "taylor"])
def test_aggregation_and_attach(form):
    rng = np.random.default_rng(3)
    sigma_bio = 0.3
    df_grid, df_real = _synthetic_frames(rng, sigma_bio=sigma_bio)
    table, diag = aggregate_technical_deconvolution(
        df_grid, df_real, deconv_form=form
    )
    assert set(table["deconv_flag"]) == {"ok"}
    # composite noise model is only approximately V = E f + σ² in -2lnR units
    np.testing.assert_allclose(table["deconv_sigma"], sigma_bio, atol=0.03)

    gcols = ["context", "sample_name"]
    df_d = desync_results(df_real=df_real, df_sim=None,
                          sim_agg=table[gcols + ["Technical_cSTD", "Technical_R"]],
                          clamp_bio_variance=False)
    df_d = ContextModel._attach_deconvolution(df_d, table, gcols, False)
    # per-group identity Data² = Tech² + Bio² (hours)
    np.testing.assert_allclose(
        df_d["Data_cSTD"] ** 2, df_d["Technical_cSTD"] ** 2 + df_d["Bio_cSTD"] ** 2,
        rtol=1e-9,
    )
    # Shell 2: desync_means pools the weighted mean of σ̂²
    m = desync_means(df_d.copy(), clamp_bio_variance=False)
    w = df_d["group_size"].values
    expect = np.sqrt(np.average(df_d["Bio_cSTD"].values ** 2, weights=w))
    assert float(m["Bio_cSTD"].iloc[0]) == pytest.approx(expect, rel=1e-9)


def test_aggregation_n_replicates_broadcast():
    rng = np.random.default_rng(4)
    df_grid, df_real = _synthetic_frames(rng, n_grid_cells=2000, n_real=6000)
    table, _ = aggregate_technical_deconvolution(df_grid, df_real, n_replicates=3)
    assert len(table) == 9
    assert table["sample_name"].str.endswith(("_1.0", "_2.0", "_3.0")).all()
    df_d = desync_results(df_real=df_real, df_sim=None, n_replicates=3,
                          sim_agg=table[["context", "sample_name", "Technical_cSTD",
                                         "Technical_R"]],
                          clamp_bio_variance=False)
    assert df_d["Technical_cSTD"].notna().sum() == (table["deconv_flag"] == "ok").sum()


def test_attach_clamp_below_floor():
    table = pd.DataFrame(dict(
        context=["c", "c"], sample_name=["a", "b"],
        Technical_cSTD=[0.5, np.nan], Technical_R=[np.nan, np.nan],
        deconv_sigma=[0.2, np.nan], deconv_flag=["ok", "below_floor"],
        deconv_V=[0.29, 0.2], deconv_mu=[0.0, 1.0], deconv_f_mu=[0.25, 0.3],
        deconv_f2_mu=[0.1, 0.1], deconv_T_hat=[0.25, np.nan], deconv_form="exact",
    ))
    df_d = pd.DataFrame(dict(context=["c", "c"], sample_name=["a", "b"],
                             Data_cSTD=[np.sqrt(0.29) * rh, np.sqrt(0.2) * rh],
                             Technical_cSTD=[0.5 * rh, np.nan], group_size=[10, 10]))
    gc = ["context", "sample_name"]
    out_nc = ContextModel._attach_deconvolution(df_d, table, gc, False)
    assert np.isnan(out_nc["Bio_cSTD"].iloc[1])
    out_c = ContextModel._attach_deconvolution(df_d, table, gc, True)
    assert out_c["Bio_cSTD"].iloc[1] == 0.0
    assert out_c["Technical_cSTD"].iloc[1] == pytest.approx(out_c["Data_cSTD"].iloc[1])
    assert out_c["Bio_cSTD"].iloc[0] == pytest.approx(0.2 * rh)


# ---------------------------------------------------------------------------
# VECTOR form
# ---------------------------------------------------------------------------
from scritmo.ml.deconvolution import (  # noqa: E402
    eval_complex_fourier,
    grid_complex_fourier,
    resultant_under_bump,
    solve_vector,
)

# a band-limited complex resultant with an identity-like j=1 term plus terms that create
# a non-trivial attractor bias m(θ) − θ and a phase-dependent r(θ); |ρ| < 1 everywhere
C_RHO = {1: 0.72, 0: 0.05, 2: 0.08j, -1: 0.04, 3: -0.05, -2: 0.03 + 0.02j}


def rho_true(theta):
    theta = np.asarray(theta, dtype=float)
    return sum(c * np.exp(1j * j * theta) for j, c in C_RHO.items())


def z_numerical(mu, sigma, rho=rho_true, n=200_000):
    """E_{θ~WN(μ,σ)}[ρ(θ)] by quadrature."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = wrapped_normal_pdf(th, mu, sigma)
    return complex(np.sum(p * rho(th)) * 2 * np.pi / n)


@pytest.fixture(scope="module")
def cc():
    return grid_complex_fourier(GRID, rho_true(GRID))


def test_rho_is_nontrivial():
    th = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    r = rho_true(th)
    assert np.abs(r).max() < 1
    bias = np.angle(r * np.exp(-1j * th))
    assert np.ptp(bias) > 0.2 and np.ptp(np.abs(r)) > 0.1


def test_complex_fourier_recovers_coefficients(cc):
    for j, cj in zip(cc["j"], cc["c"]):
        assert cj == pytest.approx(C_RHO.get(int(j), 0.0), abs=1e-12)
    th = np.linspace(0, 2 * np.pi, 91)
    np.testing.assert_allclose(eval_complex_fourier(th, cc), rho_true(th), atol=1e-12)


def test_complex_fourier_nyquist_interpolates():
    rng = np.random.default_rng(7)
    vals = 0.6 * np.exp(1j * GRID) + 0.05 * (rng.standard_normal(24)
                                             + 1j * rng.standard_normal(24))
    vals = vals + 0.03 * np.exp(12j * GRID)
    c = grid_complex_fourier(GRID, vals)
    np.testing.assert_allclose(eval_complex_fourier(GRID, c), vals, atol=1e-12)
    # the Nyquist term is split evenly between j = -12 and j = +12
    assert abs(c["c"][0]) == pytest.approx(abs(c["c"][-1]), abs=1e-12)


def test_resultant_under_bump_matches_quadrature(cc):
    for mu in (0.3, 2.0, 4.5):
        for sigma in (0.1, 0.5, 1.1):
            assert resultant_under_bump(mu, sigma, cc) == pytest.approx(
                z_numerical(mu, sigma), abs=1e-9)


@pytest.mark.parametrize("mu", [0.0, 1.1, 2.6, 4.0, 5.5])
@pytest.mark.parametrize("sigma", [0.05, 0.3, 0.7, 1.2])
@pytest.mark.parametrize("n", [1000, np.inf])
def test_vector_recovers_sigma(cc, mu, sigma, n):
    z = z_numerical(mu, sigma)
    R2 = abs(z) ** 2 + (0.0 if not np.isfinite(n) else (1 - abs(z) ** 2) / n)
    sol = solve_vector(R2, n, mu, cc)
    assert sol["flag"] == "ok"
    assert sol["sigma"] == pytest.approx(sigma, abs=1e-6)
    assert sol["pred_dir"] == pytest.approx(np.angle(z) % (2 * np.pi), abs=1e-6)


def test_variance_exact_biased_where_vector_is_not(cc):
    # the variance form sees only f = -2 ln r on the grid: the infinite-n twin floor
    f_grid = -2 * np.log(np.abs(rho_true(GRID)))
    coef_f = grid_fourier_coefficients(GRID, f_grid)
    sigma = 0.6
    err_exact, err_vec = [], []
    for mu in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        z = z_numerical(mu, sigma)
        V = -2 * np.log(abs(z))  # Data cSTD^2 at infinite n
        se = solve_exact(V, mu, coef_f)
        sv = solve_vector(abs(z) ** 2, np.inf, mu, cc)
        err_exact.append(se["sigma"] - sigma if se["flag"] == "ok" else np.inf)
        err_vec.append(sv["sigma"] - sigma)
    assert np.max(np.abs(err_vec)) < 1e-6
    assert np.max(np.abs(err_exact)) > 0.05


def test_vector_flags(cc):
    mu = 1.0
    L0 = abs(resultant_under_bump(mu, 0.0, cc)) ** 2
    assert solve_vector(min(L0 + 0.01, 1.0), np.inf, mu, cc)["flag"] == "below_floor"
    assert solve_vector(L0, np.inf, mu, cc)["sigma"] == pytest.approx(0.0, abs=1e-12)
    assert solve_vector(0.0, np.inf, mu, cc)["flag"] == "no_root"


@pytest.mark.parametrize("n_rep", [None, 3])
def test_vector_aggregation(n_rep):
    rng = np.random.default_rng(5)
    sigma_bio = 0.3
    df_grid, df_real = _synthetic_frames(rng, sigma_bio=sigma_bio)
    table, diag = aggregate_technical_deconvolution(
        df_grid, df_real, deconv_form="vector", n_replicates=n_rep)
    assert set(table["deconv_flag"]) == {"ok"}
    tol = 0.03 if n_rep is None else 0.06
    np.testing.assert_allclose(table["deconv_sigma"], sigma_bio, atol=tol)
    # direction check: the model predicts the data's mean direction
    d = np.angle(np.exp(1j * (table["deconv_pred_dir"] - table["deconv_data_dir"])))
    assert np.max(np.abs(d)) < 0.02
    # implied split Data^2 = Tech^2 + sigma^2
    np.testing.assert_allclose(
        table["deconv_V"], table["Technical_cSTD"] ** 2 + table["deconv_sigma"] ** 2,
        rtol=1e-9)
