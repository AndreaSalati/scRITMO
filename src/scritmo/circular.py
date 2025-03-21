import numpy as np
from scipy.stats import circmean, circstd, circvar


# metrics to evaluate the performance of the model
def circular_deviation(x, y, period=2 * np.pi):
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


def circular_median_absolute_deviation(x, y, period=24.0):
    """
    It computes the circular mean absolute deviation between two vectors x and y
    PASS SQUEEZED ARRAYS
    """
    # throw error if x and y are not squeezed

    v = circular_deviation(x, y, period)
    return np.median(v)


def circular_mean_absolute_deviation(x, y, period=24):
    """
    It computes the circular mean absolute deviation between two vectors x and y
    PASS SQUEEZED ARRAYS
    """
    v = circular_deviation(x, y, period)
    return np.mean(v)


def circular_sqrt_mean_squared_error(x, y, period=24):
    """
    It computes the srt of sum of squared errors absolute deviation between two vectors x and y
    PASS SQUEEZED ARRAYS
    """
    v = circular_deviation(x, y, period)
    return np.sqrt(np.mean(v**2))


def optimal_shift(p, p0, n_s=200):
    """
    Aligns two sequences defined on the unit circle, taking care of the periodicity
    and the flipping symmetry of the circle.
    It uses the median absolute deviation (MAD) as a measure of the distance between the two sequences.

    Parameters:
    p: phase array to adjust
    p0: phase array (reference)
    n_s: number of shifts to consider

    Returns:
    phi_aligned: the aligned phase array
    best_mad: the MAD of the best alignment
    """

    def circular_deviation2(x, y, period=2 * np.pi):
        """
        Function called by optimal_shift
        Inputs:
        x: phase array
        y: phase array
        period: period of the circular variable
        """
        x, y = x % period, y % period
        v1 = np.abs(x - y)
        v2 = (period - v1) % period
        return np.minimum(v1, v2)

    Nc = p.shape[0]
    shifts = np.linspace(0, 2 * np.pi, n_s)
    # creating a matrix of all possible shifts
    theta_cs = (p.reshape(Nc, 1) - shifts.reshape(1, n_s)) % (2 * np.pi)
    theta_cs_neg = (-p.reshape(Nc, 1) - shifts.reshape(1, n_s)) % (2 * np.pi)

    # for each shift, computing the circular deviation, using apply_along_axis
    delta_cs = circular_deviation2(p0[:, None], theta_cs)
    delta_cs_neg = circular_deviation2(p0[:, None], theta_cs_neg)

    # computing the median absolute deviation for all shifts
    v = np.median(delta_cs, axis=0)
    v_neg = np.median(delta_cs_neg, axis=0)
    # selecting the best shift
    best_shift_ind = np.argmin(v)
    best_shift_ind_neg = np.argmin(v_neg)
    mad, mad_neg = v[best_shift_ind], v_neg[best_shift_ind_neg]

    # selecting which direction is the best
    if mad < mad_neg:
        phi_aligned = theta_cs[:, best_shift_ind]
        best_mad = mad
    else:
        phi_aligned = theta_cs_neg[:, best_shift_ind_neg]
        best_mad = mad_neg

    return phi_aligned, best_mad


# used for Differential Expression part
def angle_deviation(angle, c_mean):
    """
    This function computes the deviation of an angle from a circular mean.
    Taking into account the periodicity of the circle.
    It returns the deviation in the range [-pi, pi]
    Input:
    angle: vector of angles in radians
    c_mean: circular mean in radians

    Output:
    adjusted_diff: deviation of the angle from the circular mean
    """
    # Direct difference
    direct_diff = angle - c_mean

    # Adjust for circular nature
    adjusted_diff = np.where(
        # condition
        direct_diff > np.pi,
        # if true substitute with
        direct_diff - 2 * np.pi,
        # if false substitute with
        # this checks the other condition on the left
        np.where(direct_diff <= -np.pi, direct_diff + 2 * np.pi, direct_diff),
    )

    return adjusted_diff


def circular_dispersion_samples(phi, samples):
    """
    This function computes some statistics after the inference
    It takes the cricular mean of all cells beloning to the same
    sample and computes the circular mean and deviation of the sample mean
    """
    samples_u = np.unique(samples)
    circ_mean = np.zeros(samples_u.shape)
    mad = np.zeros(samples_u.shape)
    mead = np.zeros(samples_u.shape)

    for i, s in enumerate(samples_u):
        idx = samples == s
        circ_mean[i] = circmean(phi[idx])
        mead[i] = circular_mean_absolute_deviation(
            phi[idx], circ_mean[i], period=2 * np.pi
        )
        mad[i] = circular_median_absolute_deviation(
            phi[idx], circ_mean[i], period=2 * np.pi
        )

    return circ_mean, mad, mead
