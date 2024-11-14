import numpy as np
from scipy.stats import circmean, circstd, circvar


def circular_mean(phis, P=None):
    """
    It outputs the circular mean of vector phis, given a PDF P
    This is the same implementation as scipy
    """

    if P is None:
        P = np.ones(len(phis)) / len(phis)
    # take complex arg of sum
    mu = np.angle(np.sum(np.exp(1j * phis) * P))

    return mu % (2 * np.pi)


def circ_mean_P(P):
    """
    It outputs the circular mean of PDF P
    """

    phis = np.linspace(0, 2 * np.pi, P.shape[0])
    # take complex arg of sum
    mu = np.angle(np.sum(np.exp(1j * phis) * P))
    return mu % (2 * np.pi)


def circ_var_P(P):
    """
    It outputs the circular variance of PDF P
    """

    phis = np.linspace(0, 2 * np.pi, P.shape[0])
    # take complex arg of sum
    var = 1 - np.abs(np.sum(np.exp(1j * phis) * P))
    return var


def circ_std_P(P):
    """
    It outputs the circular std of PDF P
    """

    phis = np.linspace(0, 2 * np.pi, P.shape[0])
    std = np.sqrt(-2 * np.log(np.abs(np.sum(np.exp(1j * phis) * P))))

    return std


# used for DE
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


def optimal_shift(p, p0, n_s=200, return_mad=True):
    """
    Aligns two sequences defined on the unit circle, taking care of the periodicity
    and the flipping symmetry of the circle.
    It uses the median absolute deviation (MAD) as a measure of the distance between the two sequences.
    """
    n_c = p.shape[0]
    shifts = np.linspace(0, 2 * np.pi, n_s)
    # creating a matrix of all possible shifts
    theta_cs = (p.reshape(n_c, 1) - shifts.reshape(1, n_s)) % (2 * np.pi)
    theta_cs_neg = (-p.reshape(n_c, 1) - shifts.reshape(1, n_s)) % (2 * np.pi)
    delta_cs = np.abs(theta_cs - p0.reshape(n_c, 1)) % (2 * np.pi)
    delta_cs_neg = np.abs(theta_cs_neg - p0.reshape(n_c, 1)) % (2 * np.pi)
    # computing the median absolute deviation for all shifts
    v = np.median(delta_cs, axis=0)
    v_neg = np.median(delta_cs_neg, axis=0)
    # selecting the best shift
    best_shift_ind = np.argmin(v)
    best_shift_ind_neg = np.argmin(v_neg)
    mad, mad_neg = v[best_shift_ind], v_neg[best_shift_ind_neg]
    # selecting which direction is the best
    best_ind = best_shift_ind if mad < mad_neg else best_shift_ind_neg

    if return_mad:
        return theta_cs[:, best_ind], mad  # , shifts[best_ind]
    else:
        return theta_cs[:, best_ind]


# def optimal_shift(p, p0, n_s=200, return_mad=True):
#     """
#     Aligns two sequences defined on the unit circle, taking care of the periodicity
#     and the flipping symmetry of the circle.
#     It uses the median absolute deviation (MAD) as a measure of the distance between the two sequences.
#     """
#     n_c = p.shape[0]
#     shifts = np.linspace(0, 2 * np.pi, n_s)
#     theta_cs = (p.reshape(n_c, 1) - shifts.reshape(1, n_s)) % (2 * np.pi)
#     theta_cs_neg = (-p.reshape(n_c, 1) - shifts.reshape(1, n_s)) % (2 * np.pi)
#     delta_cs = np.abs(theta_cs - p0.reshape(n_c, 1)) % (2 * np.pi)
#     delta_cs_neg = np.abs(theta_cs_neg - p0.reshape(n_c, 1)) % (2 * np.pi)
#     v = np.median(delta_cs, axis=0)
#     v_neg = np.median(delta_cs_neg, axis=0)
#     best_shift_ind = np.argmin(v)
#     best_shift_ind_neg = np.argmin(v_neg)
#     mad, mad_neg = v[best_shift_ind], v_neg[best_shift_ind_neg]
#     if v[best_shift_ind] < v_neg[best_shift_ind_neg]:
#         if return_mad:
#             return theta_cs[:, best_shift_ind], mad
#         else:
#             return theta_cs[:, best_shift_ind]

#     else:
#         if return_mad:
#             return theta_cs_neg[:, best_shift_ind_neg], mad_neg
#         else:
#             return theta_cs_neg[:, best_shift_ind_neg]


def phi2category(phi, n_tmp=4):
    """
    Function that maps a phase phi to a category in the range [0, n_tmp-1]
    It is used to get an classification accuracy measure, comparable with
    logistic regression.
    Input:
    phi: phase in radians, between 0 and 2*pi
    n_tmp: number of categories, they are hardcoded
    """
    if n_tmp == 4:
        z = np.pi / 4
        if (phi >= 7 * z) or (phi <= z):
            return 0
        elif z < phi <= 3 * z:
            return 6
        elif 3 * z < phi <= 5 * z:
            return 12
        elif 5 * z < phi <= 7 * z:
            return 18
        else:
            print(phi)
            raise ValueError("phi not in the range")

    elif n_tmp == 6:
        z = np.pi / 6
        if 0 < phi <= 2 * z:
            return 2
        elif 2 * z < phi <= 4 * z:
            return 6
        elif 4 * z < phi <= 6 * z:
            return 10
        elif 6 * z < phi <= 8 * z:
            return 14
        elif 8 * z < phi <= 10 * z:
            return 18
        elif 10 * z < phi <= 12 * z:
            return 22
        else:
            print(phi)
            raise ValueError("phi not in the range")


# metrics to evaluate the performance of the model
def circular_deviation(x, y, period=24):
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
    v1 = np.abs(x.squeeze() - y.squeeze()) % period
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


def circular_correlation(x, y):
    """
    It computes the circular correlation between two vectors x and y
    DOUBLE CHECK THIS FUNCTION
    """
    # n = len(x)


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
