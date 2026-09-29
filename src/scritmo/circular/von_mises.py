import numpy as np
from scipy.stats import circmean, circstd, circvar, vonmises
from scipy.special import i0, i1, i0e, i1e
from scipy.optimize import brentq
import pandas as pd
from .circular import circular_deviation


def circular_std(kappa):
    """
    Compute the circular standard deviation (cStd) of a von Mises distribution
    given the concentration parameter kappa.

    Parameters:
        kappa (float): The concentration parameter (kappa > 0)

    Returns:
        float: Circular standard deviation
    """
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    if kappa > 700:
        return 0
    R = i1(kappa) / i0(kappa)
    cstd = np.sqrt(-2 * np.log(R))
    return cstd


def kappa2circular_std(kappa):
    """return circular standard deviation given kappa in radians"""
    return circular_std(kappa)


def _A1(kappa):
    """
    Ratio I1(kappa) / I0(kappa), computed with the exponentially scaled
    Bessel functions (i1e / i0e) so it is stable for large kappa (no
    overflow, unlike i0/i1).
    """
    return i1e(kappa) / i0e(kappa)


def _circular_std2kappa_scalar(cstd):
    if cstd < 0:
        raise ValueError("cstd must be non-negative")
    if cstd == 0:
        return np.inf
    if not np.isfinite(cstd):
        return 0.0

    # R = exp(-cstd^2 / 2) = A1(kappa); solve A1(kappa) - R = 0 for kappa.
    R = np.exp(-(cstd**2) / 2)
    if R <= 0:
        return 0.0
    if R >= 1:
        return np.inf

    lo, hi = 1e-8, 1e8
    f_lo = _A1(lo) - R
    f_hi = _A1(hi) - R
    # A1 is monotonically increasing in kappa (0 at kappa=0, ->1 as kappa->inf)
    if f_lo >= 0:
        return lo
    if f_hi <= 0:
        return hi

    kappa = brentq(lambda k: _A1(k) - R, lo, hi, xtol=1e-14, rtol=1e-14)
    return float(kappa)


def circular_std2kappa(cstd):
    """
    Compute the concentration parameter (kappa) of a von Mises distribution
    given the circular standard deviation (cStd), as the EXACT numerical
    inverse of ``circular_std``.

    Units: ``cstd`` must be expressed in **radians** (same convention as
    ``circular_std``/``kappa2circular_std``). If you have a circular
    standard deviation expressed in hours (e.g. on a 24h period), convert
    it to radians first by dividing by ``scritmo.rh`` (``rh = 24 / (2*pi)``),
    i.e. ``cstd_rad = cstd_hours / scritmo.rh``.

    This inverts ``R = exp(-cstd**2 / 2) = A1(kappa) = I1(kappa) / I0(kappa)``
    numerically (via ``scipy.special.i1e``/``i0e`` for numerical stability,
    and ``scipy.optimize.brentq`` for root finding), so, unlike a fitted
    power-law approximation, it is exact to floating point tolerance over
    the whole range of kappa.

    Parameters:
        cstd (float or array-like): Circular standard deviation, in radians.
            Must be non-negative.

    Returns:
        float or np.ndarray: The kappa value(s) satisfying
        ``circular_std(kappa) == cstd``. Returns ``np.inf`` for ``cstd == 0``
        and ``0.0`` for ``cstd == np.inf`` (or any cstd large enough that the
        implied concentration is negligible).

    Raises:
        ValueError: If any element of ``cstd`` is negative.
    """
    cstd_arr = np.asarray(cstd, dtype=float)

    if np.any(cstd_arr < 0):
        raise ValueError("cstd must be non-negative")

    is_scalar = cstd_arr.ndim == 0
    flat = cstd_arr.reshape(-1)
    out = np.empty(flat.shape, dtype=float)
    for i, c in enumerate(flat):
        out[i] = _circular_std2kappa_scalar(float(c))

    if is_scalar:
        return float(out[0])
    return out.reshape(cstd_arr.shape)
