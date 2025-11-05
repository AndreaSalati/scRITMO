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


    
