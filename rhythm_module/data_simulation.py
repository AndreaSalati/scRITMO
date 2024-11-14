import numpy as np
import numpyro
from numpyro import distributions as dist

from jax import random
import jax.numpy as jnp


def simulate_timepoint_counts_G(
    t_true,
    omega,
    Ng,
    Nc,
    A_g,
    P_g,
    M_g,
    std_angle=np.pi / 4,
    std_gauss=0.01,
    key=random.PRNGKey(0),
):
    """
    this generates the expression levels of circardian genes, for single cells
    It does so following a particular model where there is NO NOISE and
    a spread of the phases around the true phase

    Inputs:
    t_true: the true phase of the circadian genes
    omega: the frequency of the circadian genes
    Ng: the number of genes
    Nc: the number of cells
    A_g: the amplitude of the genes
    P_g: the phase of the genes
    M_g: the mean of the genes
    std_angle: the standard deviation of the phase
    std_gauss: the standard deviation of the gaussian noise

    """
    kappa = 1 / std_angle

    alphas = numpyro.distributions.VonMises(t_true, kappa).sample(key, (Nc,))
    # creating the matrix of expression levels
    X = (
        A_g[None, :] * np.cos(omega * alphas[:, None] - P_g[None, :])
        + M_g[None, :]
        + std_gauss * np.random.randn(Nc, Ng)
    )
    # than maybe implement NB to give integer counts
    return X, alphas % (2 * np.pi)


# data-generating functions that create integer data. So far we do this at the price of
# addign extra noise, not sure it is the best way to do it
# here we have a unique function, where for the complete phase coherence
# you just need to set the dispersion to 0


def simulate_cell_noiseless_integer(phi_c, Ng, a_g, b_g, m_g, count=1):
    """
    This function simulates the expression levels of ONE cell with no noise.
    Hence it outputs a vector of expression levels for each gene.
    """
    # creating the matrix of expression levels
    y = np.exp(a_g * np.cos(phi_c) + b_g * np.sin(phi_c) + m_g) * count
    # round the vector to the nearest integer
    y = np.round(y)
    return y


# def simulate_timepoint_counts_NB(
#     t_true, omega, Ng, Nc, A_g, P_g, M_g, std_angle, disp_NB, key=random.PRNGKey(0)
# ):
#     """
#     this generates the expression levels of circardian genes, for single cells
#     It does so following a particular model where there is NO NOISE and
#     a spread of the phases around the true phase
#     """
#     kappa = 1 / std_angle
#     conc = 1 / disp_NB

#     alphas = dist.VonMises(0.0, kappa).sample(key, (Nc,))
#     true_phases = t_true + alphas
#     # creating the matrix of expression levels
#     X = (
#         A_g[None, :] * np.cos(omega * t_true + alphas[:, None] - P_g[None, :])
#         + M_g[None, :]
#     )
#     # adding noise, Negative Binomial
#     Y = dist.NegativeBinomial2(jnp.exp(X), conc).sample(key)
#     # than maybe implement NB to give integer counts
#     return Y, true_phases % (2 * np.pi)
