import numpy as np
import pandas as pd


# error metrics for circular data
def circular_deviation_(x, y, period):
    """
    It computes the circular absolute deviation between two vectors x and y
    Inputs:
    x: phase array
    y: phase array
    period: period of the circular variable
    """
    x, y = x % period, y % period
    v1 = np.abs(x - y)
    v2 = period - v1

    return np.minimum(v1, v2)


def circmedian(x, n_grid=1000, period=2 * np.pi):
    """
    It computes the circular median of a vector x
    Inputs:
    x: phase array
    n_grid: number of grid points to evaluate the median
    period: period of the circular variable
    """
    x = np.asarray(x)
    x = x % period
    grid = np.linspace(0, period, n_grid)
    abs_dev_ij = circular_deviation_(x[:, None], grid[None, :], period)
    sum_abs_dev_j = np.nansum(abs_dev_ij, axis=0)
    median_idx = np.argmin(sum_abs_dev_j)
    circ_median = grid[median_idx]
    return circ_median


def median_dispersion(x, med=None, period=2 * np.pi):
    """
    It computes the circular median absolute deviation of a vector x
    Inputs:
    x: phase array
    med: precomputed circular median of x (optional)
    period: period of the circular variable
    """
    x = x % period
    if med is None:
        med = circmedian(x, period=period)
    v = circular_deviation_(x, med, period)
    return np.median(v)
