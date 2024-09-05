import numpy as np
import numpyro
import jax
import jax.numpy as jnp

# helper functions for posterior distributions


def entropy(l_x):
    """
    entropy of discrete distribution l_x
    """
    return -np.sum(l_x * np.log(l_x))


def delta_entropy(l_x):
    """
    Compute the entropy of the distribution l_x and subtract the maximum entropy
    0 is maximum entropy
    negative value is less than maximum entropy
    """
    max_entropy = np.log(l_x.shape[0])
    entropy = -np.sum(l_x * np.log(l_x))
    return entropy - max_entropy


def frac_entropy(l_x):
    """
    Compute the entropy of the distribution l_x and divide by the maximum entropy
    1 is maximum entropy
    0 is less than maximum entropy
    """
    max_entropy = np.log(l_x.shape[0])
    entropy = -np.sum(l_x * np.log(l_x))
    return entropy / max_entropy


# functions that normalize distributions
def normalize_l(v):
    """
    This function takes a vector of likelihoods and returns a vector of probabilities
    """
    Z = np.sum(v) * (2 * np.pi / v.shape[0])
    return v / Z


def normalize_l_xc(v):
    """
    This function takes a vector of likelihoods and returns a vector of probabilities
    """
    Z = np.sum(v, axis=0) * (2 * np.pi / v.shape[0])
    return v / Z


# same functions as above but for log likelihoods
def normalize_ll(v):
    """
    This function takes a vector of log likelihoods and returns a vector of probabilities
    """
    vm = v.min()
    v = v - vm
    v = np.exp(v)
    Z = np.sum(v) * (2 * np.pi / v.shape[0])
    return v / Z


def normalize_ll_xc(ll_xc):
    """
    Takes a matrix of log likelihoods and returns a matrix of probabilities
    it is normalized across dim=0, the pdf is normalized over the 0, 2pi range
    Later in the code this is changed to 0,24 hours
    Input:
    ll_xc: Nx x Nc, where Nx is the number of points and Nc is the number of cells/samples

    """
    Nx, Nc = ll_xc.shape
    mins = np.min(ll_xc, axis=0)
    a = ll_xc - mins
    a = np.exp(a)
    area = a.sum(axis=0) * (2 * np.pi / Nx)
    return a / area


# numpyro functions
def get_trace(model, y, svi_g, svi_c, **kwargs):
    """
    This function ouputs the trace of the model given the data and the trained parameters.
    The traace contains a lot of information about the model, including the likelihood fn itself.
    Args:
    model: the model function
    y: the data Nc x Ng
    svi_g: the SVI object containing the trained gene coefficients
    svi_c: the SVI object containing the trained phases

    Kwargs:
    counts: the counts Nc x 1, use it for NB models, don't for gaussian models
    """
    Nc, Ng = y.shape
    counts = kwargs.get("counts", None)
    # preparing an array with size Nx x Nc x 1

    model_sub = numpyro.handlers.substitute(model, svi_g)
    model_sub = numpyro.handlers.substitute(model_sub, svi_c)

    trace = numpyro.handlers.trace(model_sub).get_trace(y, counts=counts)
    return trace


def cells_posteriors(model, y, svi_g, svi_c, Nx=1000, **kwargs):
    """
    Compute the posterior of the cells phases. It akes the fitted model, fixes gene parameters
    and computes the log likelihood for every combination of cell phases phi_xc
    Args:
    model: the model function
    y: the data Nc x Ng
    svi_g: the SVI object containing the trained gene coefficients
    svi_c: the SVI object containing the trained phases
    Nx: the number of points to evaluate the posterior
    Kwargs:
    counts: the counts Nc x 1, use it for NB models, don't for gaussian models
    """
    Nc, Ng = y.shape
    counts = kwargs.get("counts", None)
    # preparing an array with size Nx x Nc x 1
    phi_xc = np.linspace(0, 2 * np.pi, Nx)
    # repeat phi_range Nc times
    phi_xc = np.tile(phi_xc, (Nc, 1)).T
    phi_xc = phi_xc.reshape(Nx, Nc, 1)

    model_sub = numpyro.handlers.substitute(model, svi_g)
    model_sub = numpyro.handlers.substitute(model_sub, svi_c)

    model_sub = numpyro.handlers.substitute(model_sub, {"phi_c": phi_xc})
    trace = numpyro.handlers.trace(model_sub).get_trace(y, counts=counts)
    ll_xcg = trace["obs"]["fn"].log_prob(y)
    return ll_xcg


def fix_phases(l_xc, phi_x):
    """
    This function is used to fix the phases of the cells.
    Very often distributions are bimodal, and gradient descent fails
    to find the correct solution.
    """
    # for every cell get the argmax of the likelihood
    phi_max_ind = np.argmax(l_xc, axis=0)
    # phi_max_ind = list(phi_max_ind)
    phi_NB2 = phi_x[phi_max_ind]
    return phi_NB2
