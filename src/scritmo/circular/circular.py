import numpy as np
from scipy.stats import circmean, circstd, circvar
import pandas as pd


# error metrics for circular data
def circular_deviation(x, y, period):
    """
    It computes the circular absolute deviation between two vectors x and y
    PASS SQUEEZED ARRAYS
    Inputs:
    x: phase array
    y: phase array
    period: period of the circular variable
    """
    if x.ndim > 1 or y.ndim > 1:
        print(
            "WARNING: Both x and y must be 1D arrays, or output of the function will be wrong"
        )
    x, y = x % period, y % period
    v1 = np.abs(x.squeeze() - y.squeeze())
    v2 = period - v1

    return np.minimum(v1, v2)


def circular_square_error(x, y, period):
    """
    It computes the circular absolute deviation between two vectors x and y
    PASS SQUEEZED ARRAYS
    Inputs:
    x: phase array
    y: phase array
    period: period of the circular variable
    """
    if x.ndim > 1 or y.ndim > 1:
        print(
            "WARNING: Both x and y must be 1D arrays, or output of the function will be wrong"
        )
    x, y = x % period, y % period
    v1 = (x.squeeze() - y.squeeze()) ** 2
    v2 = (period - v1) ** 2

    return np.minimum(v1, v2)


def median_SE(x, y, period):
    """
    It computes the circular mean absolute deviation between two vectors x and y
    PASS SQUEEZED ARRAYS
    """
    # throw error if x and y are not squeezed

    v = circular_square_error(x, y, period)
    return np.median(v)


def mean_SE(x, y, period):
    """
    It computes the circular mean absolute deviation between two vectors x and y
    PASS SQUEEZED ARRAYS
    """
    v = circular_square_error(x, y, period)
    return np.mean(v)


def median_AE(x, y, period):
    """
    It computes the circular mean absolute deviation between two vectors x and y
    PASS SQUEEZED ARRAYS
    """
    # throw error if x and y are not squeezed

    v = circular_deviation(x, y, period)
    return np.median(v)


def mean_AE(x, y, period):
    """
    It computes the circular mean absolute deviation between two vectors x and y
    PASS SQUEEZED ARRAYS
    """
    v = circular_deviation(x, y, period)
    return np.mean(v)


def circular_sqrt_mean_squared_error(x, y, period):
    """
    It computes the srt of sum of squared errors absolute deviation between two vectors x and y
    PASS SQUEEZED ARRAYS
    """
    v = circular_deviation(x, y, period)
    return np.sqrt(np.mean(v**2))


# dispersion estimators for circular data


def R(theta, adjust=False):
    """
    Computes the resultant length R for a set of angles.
    R = |sum(exp(i*theta))| / N
    """
    n = theta.shape[0]
    R = np.abs(np.sum(np.exp(1j * theta))) / n
    if adjust:
        # Adjust R to be in the range [0, 1]
        R = R - (1 - R**2) / (2 * n * R)
    return R


def cVAR(theta, adjust=False):
    """
    Computes the circular dispersion for a set of angles.
    Dispersion = 1 - R
    """
    R_value = R(theta, adjust=adjust)
    return 1 - R_value


def cSTD(theta, adjust=False):
    """
    Computes the circular standard deviation for a set of angles.
    STD = sqrt(-2 * log(R))
    """
    R_value = R(theta, adjust=adjust)
    if R_value == 0:
        return np.inf  # Handle case where R is zero
    return np.sqrt(-2 * np.log(R_value))


def cstd2R(cstd):
    """return mean resultant length R given circular standard deviation in radians"""
    R = np.exp(-(cstd**2) / 2)
    return R

def R2cstd(R):
    """return circular standard deviation in radians given mean resultant length R"""
    cstd = np.sqrt(-2 * np.log(R))
    return cstd


def compute_posterior_statistics(l_xc):
    """
    Compute posterior circular statistics.
    To be sure that l_xc is a discrete pdf it re normalizes it.
    """
    l_xc = l_xc / l_xc.sum(axis=0)
    post_mean_c = np.apply_along_axis(circ_mean_P, 0, l_xc)
    post_var_c = np.apply_along_axis(circ_var_P, 0, l_xc)
    post_std_c = np.apply_along_axis(circ_std_P, 0, l_xc)
    return post_mean_c, post_var_c, post_std_c


def compute_posterior_mean(l_xc):
    """
    Compute posterior circular statistics.
    To be sure that l_xc is a discrete pdf it re normalizes it.
    """
    l_xc = l_xc / l_xc.sum(axis=0)
    post_mean_c = np.apply_along_axis(circ_mean_P, 0, l_xc)
    return post_mean_c


def compute_posterior_mode(l_xc):
    """
    This function is used to find the MODE of the distribution.
    Very often distributions are bimodal, and gradient descent fails
    to find the correct solution.
    """
    # for every cell get the argmax of the likelihood
    phi_x = np.linspace(0, 2 * np.pi, l_xc.shape[0] + 1)[:-1]

    phi_max_ind = np.argmax(l_xc, axis=0)
    # phi_max_ind = list(phi_max_ind)
    phi_mode = phi_x[phi_max_ind]
    return phi_mode


# fucntions that used to get the moments of the numerical approximation of the
# posterior distribution of the phase
def circ_mean_P(P):

    phis = np.linspace(0, 2 * np.pi, P.shape[0] + 1)[:-1]
    # take complex arg of sum
    mu = np.angle(np.sum(np.exp(1j * phis) * P))
    return mu % (2 * np.pi)


def circ_var_P(P):

    phis = np.linspace(0, 2 * np.pi, P.shape[0] + 1)[:-1]
    # take complex arg of sum
    var = 1 - np.abs(np.sum(np.exp(1j * phis) * P))
    return var


def circ_std_P(P):

    phis = np.linspace(0, 2 * np.pi, P.shape[0] + 1)[:-1]
    std = np.sqrt(-2 * np.log(np.abs(np.sum(np.exp(1j * phis) * P))))
    return std
