"""Tests for the hierarchical likelihood sigma_bio estimator (scritmo.ml.desync.hierarchical)."""

import numpy as np

from scritmo import rh
from scritmo.ml.desync import solve_hierarchical, solve_hierarchical_shared

N = 240
PHI = np.arange(N) * 2 * np.pi / N


def _toy(sig_h, n=600, rng=None):
    """Posteriors = wrapped normals at y_c = theta_c + noise, width s_c in [0.5, 3] h."""
    th = rng.normal(2.0, sig_h / rh, n)
    s = rng.uniform(0.5, 3, n) / rh
    y = th + rng.normal(0, 1, n) * s
    d = (PHI[:, None] - y[None, :] + np.pi) % (2 * np.pi) - np.pi
    P = np.exp(-0.5 * (d / s) ** 2)
    return P / P.sum(0), th


def test_recovers_sigma_not_total_spread():
    rng = np.random.default_rng(0)
    for sg in (1.0, 2.4):
        est = np.mean([solve_hierarchical(_toy(sg, rng=rng)[0], PHI)["sigma"] * rh
                       for _ in range(5)])
        assert abs(est - sg) < 0.2


def test_zero_sigma_is_finite():
    rng = np.random.default_rng(1)
    e = solve_hierarchical(_toy(0.0, rng=rng)[0], PHI)
    assert np.isfinite(e["sigma"]) and e["sigma"] * rh < 0.5
    assert e["flag"] in ("ok", "at_boundary")


def test_rotation_and_flip_invariance():
    P, _ = _toy(1.5, rng=np.random.default_rng(5))
    base = solve_hierarchical(P, PHI)["sigma"]
    assert np.isclose(solve_hierarchical(np.roll(P, 37, 0), PHI)["sigma"], base, atol=1e-6)
    assert np.isclose(solve_hierarchical(P[::-1], PHI)["sigma"], base, atol=1e-6)


def test_shared_sigma():
    rng = np.random.default_rng(9)
    e = solve_hierarchical_shared([_toy(1.5, n=300, rng=rng)[0] for _ in range(6)], PHI)
    assert abs(e["sigma"] * rh - 1.5) < 0.2
    assert e["ci_lo"] <= e["sigma"] <= e["ci_hi"]
