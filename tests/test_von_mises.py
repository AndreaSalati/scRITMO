import numpy as np
import pytest

import scritmo as sr
from scritmo.circular.von_mises import (
    circular_std,
    kappa2circular_std,
    circular_std2kappa,
)
import scritmo.ml.simulations.utils as ml_sim_utils


def _old_radian_power_law(cstd):
    """The old (radian-units) fitted power law that used to live in
    ml/simulations/utils.py: cstd = a * kappa^b, a=1.0142, b=-0.4952."""
    a = 1.0142
    b = -0.4952
    cstd_safe = np.maximum(np.asarray(cstd, dtype=float), 1e-9)
    return np.power(cstd_safe / a, 1 / b)


def test_roundtrip_kappa_to_cstd_to_kappa():
    kappas = np.logspace(-3, 3, 50)
    for k in kappas:
        cstd = circular_std(k)
        k_back = circular_std2kappa(cstd)
        # skip degenerate saturation regions (cstd == 0 for huge kappa)
        if cstd == 0:
            continue
        assert np.isclose(k_back, k, rtol=1e-6, atol=1e-6), (k, cstd, k_back)


def test_roundtrip_cstd_to_kappa_to_cstd():
    cstds = np.linspace(1e-3, 3, 50)
    for c in cstds:
        kappa = circular_std2kappa(c)
        c_back = kappa2circular_std(kappa)
        if c_back == 0:
            # circular_std (left untouched, as required) has a pre-existing
            # hard cutoff that returns exactly 0 for kappa > 700. Very small
            # cstd values invert to kappa > 700 (cstd(700) ~= 0.0378), so the
            # round trip saturates to 0 there -- this is a limitation of the
            # unmodified circular_std, not of circular_std2kappa (which is
            # itself the exact inverse of the underlying A1(kappa) relation).
            assert c < 0.04, (c, kappa, c_back)
            continue
        assert np.isclose(c_back, c, rtol=1e-8), (c, kappa, c_back)


def test_scalar_input_returns_float():
    result = circular_std2kappa(0.5)
    assert isinstance(result, float)


def test_array_input_returns_ndarray():
    result = circular_std2kappa(np.array([0.1, 0.5, 1.0]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


def test_list_input_returns_ndarray():
    result = circular_std2kappa([0.1, 0.5, 1.0])
    assert isinstance(result, np.ndarray)


def test_edge_case_zero_returns_inf():
    assert circular_std2kappa(0.0) == np.inf


def test_edge_case_inf_returns_zero():
    assert circular_std2kappa(np.inf) == 0.0


def test_edge_case_large_cstd_saturates_to_zero():
    # cstd = sqrt(2*ln(2)) ~= 1.1774 is the max value circular_std can
    # produce (as kappa -> 0, R -> 0, cstd -> sqrt(-2 ln 0) = inf actually;
    # but well above the practically relevant range, R underflows to 0 and
    # the exact inverse is kappa -> 0).
    result = circular_std2kappa(50.0)
    assert result == 0.0


def test_edge_case_negative_raises():
    with pytest.raises(ValueError):
        circular_std2kappa(-0.1)

    with pytest.raises(ValueError):
        circular_std2kappa(np.array([-0.1, 0.5]))


def test_ml_simulations_utils_reexports_same_object():
    assert ml_sim_utils.circular_std2kappa is sr.circular_std2kappa
    assert ml_sim_utils.kappa2circular_std is sr.kappa2circular_std
    assert ml_sim_utils.circular_std is sr.circular_std


def test_top_level_namespace_exposes_von_mises_functions():
    assert sr.circular_std2kappa is circular_std2kappa
    assert sr.kappa2circular_std is kappa2circular_std
    assert sr.ml.kappa2circular_std is kappa2circular_std

    assert sr.circular_std2kappa.__module__ == "scritmo.circular.von_mises"
    assert sr.kappa2circular_std.__module__ == "scritmo.circular.von_mises"
    assert sr.ml.kappa2circular_std.__module__ == "scritmo.circular.von_mises"


def test_old_radian_power_law_agrees_with_exact_for_large_kappa():
    # For large kappa (well-concentrated distributions), the old fitted
    # power law (already expressed in radians, a=1.0142, b=-0.4952) should
    # be close (~5%) to the exact numerical inverse -- confirming the fix
    # captures the same underlying relation, just made exact.
    # kappa=1000 is excluded: circular_std (left untouched, as required)
    # hard-cutoffs to exactly 0 for kappa > 700, which would make the
    # comparison degenerate (circular_std2kappa(0) == inf).
    kappas = np.array([10, 30, 100, 300], dtype=float)
    cstds = np.array([circular_std(k) for k in kappas])

    exact_kappas = circular_std2kappa(cstds)
    approx_kappas = _old_radian_power_law(cstds)

    rel_err = np.abs(exact_kappas - approx_kappas) / exact_kappas
    assert np.all(rel_err < 0.1), rel_err
