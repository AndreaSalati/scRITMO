import numpy as np
from scipy.stats import circmean, circstd, circvar
import pandas as pd


def optimal_shift(
    p,
    p0,
    n_s=200,
    return_shift=False,
    return_shift_only=False,
    allow_flip=True,
    verbose=True,
):
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
    best_shift: the shift required to best align
    """

    def circular_deviation2(x, y, period=2 * np.pi):
        """
        gives difference in phases
        """
        x, y = x % period, y % period
        v1 = np.abs(x - y)
        v2 = (period - v1) % period
        return np.minimum(v1, v2)

    Nc = p.shape[0]
    shifts = np.linspace(0, 2 * np.pi, n_s + 1)[:-1]
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
    if allow_flip:
        if mad < mad_neg:
            phi_aligned = theta_cs[:, best_shift_ind]
            best_mad = mad
            best_shift = shifts[best_shift_ind]
        else:
            if verbose:
                print("Flipping occurred")
            phi_aligned = theta_cs_neg[:, best_shift_ind_neg]
            best_mad = mad_neg
            best_shift = shifts[best_shift_ind_neg]
    else:
        # Force non-flipped version regardless of MAD comparison
        phi_aligned = theta_cs[:, best_shift_ind]
        best_mad = mad
        best_shift = shifts[best_shift_ind]

    if return_shift_only:
        return best_shift
    elif return_shift:
        return phi_aligned, best_mad, best_shift
    else:
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


# def circular_dispersion_samples(phi, samples):
#     """
#     This function computes some statistics after the inference
#     It takes the cricular mean of all cells beloning to the same
#     sample and computes the circular mean and deviation of the sample mean
#     """
#     samples_u = np.unique(samples)
#     circ_mean = np.zeros(samples_u.shape)
#     mad = np.zeros(samples_u.shape)
#     mead = np.zeros(samples_u.shape)

#     for i, s in enumerate(samples_u):
#         idx = samples == s
#         circ_mean[i] = circmean(phi[idx])
#         mead[i] = circular_mean_absolute_deviation(
#             phi[idx], circ_mean[i], period=2 * np.pi
#         )
#         mad[i] = circular_median_absolute_deviation(
#             phi[idx], circ_mean[i], period=2 * np.pi
#         )

#     return circ_mean, mad, mead


def get_shift_y(ext_time, ph, context):
    """
    This function aligns all cells beloging to the
    same context class to the ext_time vector.
    """
    context_u = np.unique(context)
    shift_y = pd.Series(index=context_u)

    for i, ct in enumerate(context_u):
        mask = context == ct
        tp_ct = ext_time[mask]
        delta = optimal_shift(ph[mask], tp_ct, return_shift_only=True)
        if delta > np.pi:
            delta -= 2 * np.pi

        shift_y[ct] = delta

    return shift_y
