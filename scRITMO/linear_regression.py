import numpy as np
from .basics import BIC
import scipy.stats as stats
import pandas as pd


def harmonic_regression(t, y, omega=1):
    """
    t: time
    y: expression of a gene
    This function clearly uses guassian noise onlys
    """
    X = np.zeros((len(t), 3))
    X[:, 0] = np.cos(omega * t)
    X[:, 1] = np.sin(omega * t)
    X[:, -1] = 1
    # fit linear regression
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta


def polar_parameters(beta, rel_ampl=False):
    """
    takes the parameters of the linear regression and returns the
    amplitude and phase, and the radial amplitude if rel_ampl is True
    """
    amp = np.sqrt(beta[0] ** 2 + beta[1] ** 2)
    # here the arguments of arctan2 should be y,x and they are as the previous functions switches already
    phase = np.arctan2(beta[1], beta[0])
    if rel_ampl:
        amp = amp / beta[-1]
    return amp, phase, beta[-1]


def genes_polar_parameters(res, rel_ampl=False):
    """
    Converts all genes in the res array to polar coordinates
    Input:
    res: np.array of shape (N_ct, N_g, 3) in cartesian coordinates
    rel_ampl: bool, if True, the amplitude is divided by the offset
    Output:
    res_pol: np.array of shape (N_ct, N_g, 3) in polar coordinates
    where the first coordinate is the amplitude, the second the phase and the third the offset
    """
    N_ct, N_g, _ = res.shape
    res_pol = res.copy()
    for i in range(N_ct):
        for j in range(N_g):
            amp, phase, mu = polar_parameters(res[i, j, :], rel_ampl=rel_ampl)
            res_pol[i, j, 0] = amp
            res_pol[i, j, 1] = phase
            res_pol[i, j, 2] = mu
    return res_pol


# Inverse of the previous function
def cartesian_parameters(par, rel_ampl=False):
    """
    transforms the polar parameters into cartesian
    """
    mean = par[-1]
    if rel_ampl:
        a = par[0] * mean * np.cos(par[1])
        b = par[0] * mean * np.sin(par[1])
        return a, b, mean
    else:
        a = par[0] * np.cos(par[1])
        b = par[0] * np.sin(par[1])
        return a, b, mean


def harmonic_function_exp(phi, beta, omega=1):
    """
    takes the parameters of the linear regression and returns the
    the function evaluated at times t, exponential version

    """
    a_g, b_g, m_g = beta
    E = a_g * np.cos(omega * phi) + b_g * np.sin(omega * phi) + m_g
    return np.exp(E)


def harmonic_function_exp2(phi, beta, omega=1):
    """
    takes the parameters of the linear regression and returns the
    the function evaluated at times t, exponential version

    """
    a_g, b_g, m_g = beta
    E = a_g * np.cos(omega * phi) + b_g * np.sin(omega * phi) + m_g
    return np.exp2(E)


def harmonic_function(t, beta, omega=1, basis="cartesian", rel_ampl=False):
    """
    takes the parameters of the linear regression and returns the
    the function evaluated at times t:
    beta: parameters of the linear regression
    omega: frequency
    basis: 'cartesian' or 'polar'
    rel_ampl: if the amplitude is relative to the mean
    """
    if basis == "cartesian":
        return beta[0] * np.cos(omega * t) + beta[1] * np.sin(omega * t) + beta[-1]
    elif basis == "polar":
        if rel_ampl:
            out = beta[0] * beta[-1] * np.cos(omega * t - beta[1]) + beta[-1]
        else:
            out = beta[0] * np.cos(omega * t - beta[1]) + beta[-1]
        return out


def harmonic_regression_bic(t, y, omega=1):
    """
    this function peforms linear harmonic regression but also
    compares the fit to a flat model and returns which model is better
    minimizing the BIC. It can be used to test if a gene is rhythmic

    t: time
    y: expression of a gene
    omega: frequency

    returns:
    beta: parameters of the harmonic regression
    delta_bic: bic of the harmonic regression minus bic of the flat
        it quantifies the evidence in favor of the harmonic regression
    """
    flats = y.mean() * np.ones_like(y)
    beta = harmonic_regression(t, y, omega=omega)
    # calculate bic of the harmonic regression
    bic_h = BIC(y, harmonic_function(t, beta, omega), 3)
    bic_f = BIC(y, flats, 1)

    delta_bic = bic_h - bic_f
    is_h = bic_h < bic_f

    if is_h:
        return beta, delta_bic
    else:
        return np.array([0.0, 0.0, y.mean()]), delta_bic


def harmonic_regression_pvalue(t, y, omega=1):
    """
    Performs harmonic regression, compares it against a model with no harmonic terms (intercept only),
    using a chi-square test for the comparison, and returns the coefficients and chi-square p-value.


    Parameters:
    - t: numpy array of time points
    - y: numpy array of gene expression levels at the corresponding time points
    - omega: frequency of the harmonic component

    Returns:
    - beta: Coefficients of the harmonic regression
    - chi_square_p_value: P-value from the chi-square test of the model comparison
    """
    # Harmonic model
    X_harmonic = np.zeros((len(t), 3))
    X_harmonic[:, 0] = np.cos(omega * t)
    X_harmonic[:, 1] = np.sin(omega * t)
    X_harmonic[:, -1] = 1  # Intercept
    beta, residuals, rank, s = np.linalg.lstsq(X_harmonic, y, rcond=None)

    # Intercept-only model
    X_intercept = np.ones((len(t), 1))  # Intercept
    _, residuals_intercept, _, _ = np.linalg.lstsq(X_intercept, y, rcond=None)

    # Calculate the chi-square statistic
    # Here, we assume the residual sum of squares effectively approximates twice the difference in log-likelihoods
    chi_square_stat = (residuals_intercept - residuals) / np.var(y) * len(y)

    # Degrees of freedom: number of parameters in harmonic model - number of parameters in intercept-only model
    df = 2  # cos and sin terms are the extra parameters

    # Compute the chi-square p-value
    chi_square_p_value = stats.chi2.sf(chi_square_stat, df)

    return beta, chi_square_p_value


def harmonic_regression_loop(E_ygt, t_u, omega=1, eval_fit="bic", return_bic=False):
    """
    It takes the output of pseudobulk_loop and returns the parameters of the harmonic regression
    for each gene and celltype. It use harmonic_regression_bic to fit the data
    Input:
    - E_ygt: 3D array with dimensions celltype, gene, timepoint
    - t_u: timepoints
    - omega: frequency
    - eval_fit: 'bic' or 'pvalue'
    - return_bic: if True, the function returns the BIC/Pvalue of the harmonic regression
    """
    N_y, N_g, N_t = E_ygt.shape
    res = np.zeros((N_y, N_g, 3))
    vals = np.zeros((N_y, N_g))

    for i in range(N_y):
        for j in range(N_g):
            y = E_ygt[i, j, :].squeeze()

            if eval_fit == "pvalue":
                beta, v = harmonic_regression_pvalue(t_u, y, omega=omega)
            elif eval_fit == "bic":
                beta, v = harmonic_regression_bic(t_u, y, omega=omega)
            # if beta is not None:
            # amp, phase, mu = radial_parameters(beta)
            res[i, j, 0] = beta[0]
            res[i, j, 1] = beta[1]
            res[i, j, 2] = beta[2]
            vals[i, j] = v
    if return_bic:
        return res, vals
    else:
        return res


def cSVD(res, center_around="mean", return_explained=False):
    """
    This function performs cSVD to find the common phase and amplitude for all genes
    across different cell types, and the common phase shift for all cell types

    Input:
    - res: np.array of shape (n_celltypes/conditions , n_genes, 3 (a,b,mu))
    - center_around: str, 'mean' or 'strongest', if 'mean' we center the data around the mean
    of the phase, if 'strongest' we center the data around the phase of the 'strongest' sample
    - return_explained: bool, if True, the function returns the explained variance of SVD
    Output:
    - U_: np.array of shape (n_genes, n_genes)
    - V_: np.array of shape (n_celltypes, n_celltypes)
    - S_norm: np.array of shape (n_genes, ), explained variance of SVD
    """

    # passing to complex polar form
    C = np.zeros((res.shape[0], res.shape[1]), dtype=complex)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            C[i, j] = res[i, j, 0] + 1j * res[i, j, 1]

    # SVD
    U, S, Vh = np.linalg.svd(C.T, full_matrices=False)
    V = Vh.T

    # normalizing S
    S_norm = S / np.sum(S)

    U_ = np.zeros((U.shape[0], U.shape[1]), dtype=complex)
    V_ = np.zeros((Vh.shape[0], Vh.shape[1]), dtype=complex)
    # U is gene
    # V is celltype

    # if we want to find the common shift
    if center_around == "mean":
        for i in range(len(S)):
            v = V[:, i].sum()
            # rotation
            rot = np.conj(v) / np.abs(v)
            max_s = np.abs(V[:, i]).max()

            U_[:, i] = U[:, i] * np.conj(rot) * S[i] * max_s
            V_[:, i] = V[:, i] * rot / max_s
    elif center_around == "strongest":
        for i in range(len(S)):

            # getting the index max entry of the i-th column, based on the absolute value
            max_sample = np.argmax(np.abs(V[:, i]))
            rot = V[max_sample, i]
            # rotation, we will define a complex number on the uit circle
            rot = np.conj(rot) / np.abs(rot)
            max_s = np.abs(V[:, i]).max()
            U_[:, i] = U[:, i] * np.conj(rot) * S[i] * max_s
            # since we took the conj earlier, here we just multiply by rot
            V_[:, i] = V[:, i] * rot / max_s

    if return_explained:
        return U_, V_, S_norm
    else:
        return U_, V_
