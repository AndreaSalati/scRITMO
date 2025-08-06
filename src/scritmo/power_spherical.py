import numpy as np
import scipy


def powershperical2beta(k, d):
    alpha = (d - 1) / 2 + k
    beta = (d - 1) / 2
    return alpha, beta


def power_spherical_2d(x, mu, k):
    """
    Compute the power spherical distribution
    """
    d = 2
    alpha, beta = powershperical2beta(k, d)
    N = (
        2 ** (alpha + beta)
        * np.pi ** (beta)
        * scipy.special.gamma(alpha)
        / scipy.special.gamma(alpha + beta)
    )
    return (1 + np.cos(x - mu)) ** (k) / N
