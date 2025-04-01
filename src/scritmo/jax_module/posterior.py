import numpy as np
import numpyro
import jax
import jax.numpy as jnp
from scipy.stats import chi2

# helper functions for posterior distributions


def entropy(l_x):
    """
    entropy of discrete distribution l_x
    """
    return -np.sum(l_x * np.log(l_x))


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


# # numpyro functions
# def get_trace(model, y, svi_g, svi_c, **kwargs):
#     """
#     This function ouputs the trace of the model given the data and the trained parameters.
#     The traace contains a lot of information about the model, including the likelihood fn itself.
#     Args:
#     model: the model function
#     y: the data Nc x Ng
#     svi_g: the SVI object containing the trained gene coefficients
#     svi_c: the SVI object containing the trained phases

#     Kwargs:
#     counts: the counts Nc x 1, use it for NB models, don't for gaussian models
#     """
#     Nc, Ng = y.shape
#     counts = kwargs.get("counts", None)
#     # preparing an array with size Nx x Nc x 1

#     model_sub = numpyro.handlers.substitute(model, svi_g)
#     model_sub = numpyro.handlers.substitute(model_sub, svi_c)

#     trace = numpyro.handlers.trace(model_sub).get_trace(y, counts=counts)
#     return trace


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


def fix_phases(ll_xc, phi_x):
    """
    This function is used to find the MODE of the distribution.
    Very often distributions are bimodal, and gradient descent fails
    to find the correct solution.
    """
    # for every cell get the argmax of the likelihood
    phi_max_ind = np.argmax(ll_xc, axis=0)
    phi_mode = phi_x[phi_max_ind]
    ll_mle = ll_xc[phi_max_ind, np.arange(ll_xc.shape[1])]
    return phi_mode, ll_mle


def get_likelihood(model, params, data, jax_device="cpu", **kwargs):
    """
    Takes as input a numpyro model and a dict
    of parameters and returns the likelihood function
    This is useful for both sampling or evaluation of the likelihood
    Parameters:
    model: the model function
    params: the dictionary of parameters, keys need to match the model
    data: the data, needs to be passed to the model, even if it is not used
    kwargs: additional arguments to pass to the model
    """
    jax.config.update("jax_platform_name", jax_device)
    model = numpyro.handlers.seed(model, 0)
    model = numpyro.handlers.substitute(model, params)

    # pass the kwargs to the model
    tr = numpyro.handlers.trace(model).get_trace(data, **kwargs)
    like = tr["obs"]["fn"]
    return like


def get_ll(model, params, data, **kwargs):
    """
    Takes as input a numpyro model and a dict
    of parameters and returns the log likelihood
    """
    like = get_likelihood(model, params, data, **kwargs)
    return like.log_prob(data)


###########################
# tests
###########################


def AIC_test(ll_full, ll_null, k_full, k_null, n_samples, correction=True):
    """
    AIC calculation

    ll_full: log likelihood of the full model
    ll_null: log likelihood of the null model
    k_full: number of parameters in the full model
    k_null: number of parameters in the null model
    """

    delta_k = k_full - k_null
    delta_ll = ll_full - ll_null
    AIC = 2 * delta_k - 2 * delta_ll

    if correction:
        AIC += 2 * delta_k * (delta_k + 1) / (n_samples - delta_k - 1)
    return AIC


def BIC_test(ll_full, ll_null, k_full, k_null, n_samples):
    """
    BIC calculation

    ll_full: log likelihood of the full model
    ll_null: log likelihood of the null model
    k_full: number of parameters in the full model
    k_null: number of parameters in the null model
    """

    delta_k = k_full - k_null
    delta_ll = ll_full - ll_null
    return delta_k * np.log(n_samples) - 2 * delta_ll


def BH_correction(pvals):
    """
    Benjamini-Hochberg correction
    returns the corrected p-values
    """
    m = len(pvals)
    rank = np.argsort(np.argsort(pvals)) + 1

    pvals_ = pvals * m / rank
    return pvals_


def pvalue(ll_full, ll_null, k_full, k_null, BH=False):
    """
    Basic implementation of the p-value calculation for the likelihood ratio test

    ll_full: log likelihood of the full model
    ll_null: log likelihood of the null model
    k_full: number of parameters in the full model
    k_null: number of parameters in the null model
    """
    delta_k = k_full - k_null
    delta_ll = ll_full - ll_null
    pval = chi2.sf(2 * delta_ll, delta_k)
    if BH:
        return BH_correction(pval)
    return pval
