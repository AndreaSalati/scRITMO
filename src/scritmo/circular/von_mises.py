import numpy as np
from scipy.stats import circmean, circstd, circvar, vonmises
from scipy.special import i0, i1
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


def circular_std2kappa(cstd):
    """
    Compute the approximate concentration parameter (kappa) given
    the circular standard deviation using a fitted Power Law inverse.

    Model: cstd = a * kappa^b
    Inverse: kappa = (cstd / a)^(1/b)

    Fitted Parameters:
      a = 3.8741
      b = -0.4952

    Parameters:
        cstd (float or np.array): The circular standard deviation.

    Returns:
        float or np.array: The estimated kappa.
    """
    # Hardcoded fitted parameters
    a = 3.8741
    b = -0.4952

    # Avoid division by zero errors if cstd is 0 (implies infinite kappa)
    # Using numpy's maximum to prevent negative inputs if working with noisy data
    cstd_safe = np.maximum(cstd, 1e-9)

    return np.power(cstd_safe / a, 1 / b)
