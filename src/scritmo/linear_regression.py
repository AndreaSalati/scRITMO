import numpy as np
from .basics import BIC
import scipy.stats as stats
import pandas as pd
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from tqdm import tqdm
import statsmodels.api as sm
from patsy import dmatrix, build_design_matrices


def create_harmonic_design_matrix(phases, n_harmonics=1, add_intercept=True):
    """
    Constructs a design matrix for harmonic regression with multiple harmonics.

    Parameters:
    -----------
    phases : array-like
        Vector of phases in radians (0 to 2π)
    n_harmonics : int, default=1
        Number of harmonics to include in the model
    add_intercept : bool, default=True
        Whether to add an intercept (constant) term to the design matrix

    Returns:
    --------
    X : numpy.ndarray or pandas.DataFrame
        Design matrix with columns for each harmonic (cos1, sin1, cos2, sin2, etc.)
        If add_intercept is True, the last column will be the intercept term.
    """
    phases = np.asarray(phases)
    n_samples = len(phases)

    # Initialize matrix with 2 columns per harmonic
    X = np.zeros((n_samples, 2 * n_harmonics))

    # Populate matrix with cos and sin terms for each harmonic
    for h in range(1, n_harmonics + 1):
        X[:, 2 * (h - 1)] = np.cos(h * phases)  # Cosine terms
        X[:, 2 * (h - 1) + 1] = np.sin(h * phases)  # Sine terms

    # Add column names
    column_names = []
    for h in range(1, n_harmonics + 1):
        column_names.extend([f"cos{h}", f"sin{h}"])

    # Convert to DataFrame with named columns
    X_df = pd.DataFrame(X, columns=column_names)

    if add_intercept:
        X_df = add_constant(X_df, prepend=True)

    return X_df


def evaluate_harmonic_fn(x, beta, n_harmonics=1):
    """
    Evaluates the harmonic function at given x-values using the coefficients beta.

    Parameters:
    -----------
    x : array-like
        Input values where the harmonic function will be evaluated.
    beta : array-like
        Coefficients of the harmonic regression model.
    n_harmonics : int, default=1
        Number of harmonics included in the model.

    Returns:
    --------
    y : numpy.ndarray
        Evaluated values of the harmonic function at x.
    """
    X = create_harmonic_design_matrix(x, n_harmonics=n_harmonics, add_intercept=True)
    y = X.values @ beta  # Matrix multiplication to get the predicted values
    return y


def fit_periodic_spline(x, y, df, period=2 * np.pi):
    """
    Fits a periodic spline to x and y data and returns a prediction function.

    This function will prompt the user to enter the number of degrees of
    freedom for the spline.

    Args:
        x (np.array): The independent variable data.
        y (np.array): The dependent variable data.

    Returns:
        function: A callable function that takes a new array of x-values
                  and returns the corresponding predicted y-values.
    """

    # 2. Create the design matrix for the training data
    # The 'cc()' function creates a cyclic cubic (periodic) spline basis.
    # formula = f"cc(x, df={df})"
    formula = f"cc(x, df={df}, lower_bound=0, upper_bound={period})"
    design_matrix_train = dmatrix(formula, {"x": x}, return_type="dataframe")

    # Capture the design information to use for prediction later
    design_info = design_matrix_train.design_info

    # 3. Fit the Generalized Linear Model
    model = sm.GLM(y, design_matrix_train).fit()

    # 4. Define and return the prediction function (a closure)
    def predict_function(new_x):
        """
        Predicts y-values for new x-values using the fitted spline model.
        """
        # Build a new design matrix for the new_x data using the original's info
        new_design_matrix = build_design_matrices([design_info], {"x": new_x})[0]

        # Return the model's prediction for this new matrix
        return model.predict(new_design_matrix)

    return predict_function


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


def polar_genes_pandas(res, rel_ampl=False):
    """
    This function takes a panda df of size Ngx4
    where teh first column is the gene name, and
    teh following 3 are a,b,m
    """
    N_g, _ = res.shape
    res_pol = res.copy()

    res_pol.columns = ["amp", "phase", "mean"] + list(res_pol.columns[3:])

    for j in range(N_g):
        amp, phase, mu = polar_parameters(res.iloc[j, 0:3].values, rel_ampl=rel_ampl)
        res_pol.iloc[j, 0] = amp
        res_pol.iloc[j, 1] = phase
        res_pol.iloc[j, 2] = mu

    return res_pol


def cartesian_genes_pandas(res, rel_ampl=False):
    """
    This function takes a panda df of size Ngx4
    where teh first column is the gene name, and
    teh following 3 are a,b,m
    """
    N_g, _ = res.shape
    res_pol = res.copy()

    res_pol.columns = ["a_g", "b_g", "m_g", "disp"]

    for j in range(N_g):
        amp, phase, mu = cartesian_parameters(
            res.iloc[j, 0:3].values, rel_ampl=rel_ampl
        )
        res_pol.iloc[j, 0] = amp
        res_pol.iloc[j, 1] = phase
        res_pol.iloc[j, 2] = mu

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
        return beta[1] * np.cos(omega * t) + beta[2] * np.sin(omega * t) + beta[0]
    elif basis == "polar":
        if rel_ampl:
            out = beta[0] * beta[-1] * np.cos(omega * t - beta[1]) + beta[-1]
        else:
            out = beta[0] * np.cos(omega * t - beta[1]) + beta[-1]
        return out


# def harmonic_regression_bic(t, y, omega=1):
#     """
#     this function peforms linear harmonic regression but also
#     compares the fit to a flat model and returns which model is better
#     minimizing the BIC. It can be used to test if a gene is rhythmic

#     t: time
#     y: expression of a gene
#     omega: frequency

#     returns:
#     beta: parameters of the harmonic regression
#     delta_bic: bic of the harmonic regression minus bic of the flat
#         it quantifies the evidence in favor of the harmonic regression
#     """
#     flats = y.mean() * np.ones_like(y)
#     beta = harmonic_regression(t, y, omega=omega)
#     # calculate bic of the harmonic regression
#     bic_h = BIC(y, harmonic_function(t, beta, omega), 3)
#     bic_f = BIC(y, flats, 1)

#     delta_bic = bic_h - bic_f
#     is_h = bic_h < bic_f

#     if is_h:
#         return beta, delta_bic
#     else:
#         return np.array([0.0, 0.0, y.mean()]), delta_bic


# def harmonic_regression_pvalue(t, y, omega=1):
#     """
#     Performs harmonic regression, compares it against a model with no harmonic terms (intercept only),
#     using a chi-square test for the comparison, and returns the coefficients and chi-square p-value.


#     Parameters:
#     - t: numpy array of time points
#     - y: numpy array of gene expression levels at the corresponding time points
#     - omega: frequency of the harmonic component

#     Returns:
#     - beta: Coefficients of the harmonic regression
#     - chi_square_p_value: P-value from the chi-square test of the model comparison
#     """
#     # Harmonic model
#     X_harmonic = np.zeros((len(t), 3))
#     X_harmonic[:, 0] = np.cos(omega * t)
#     X_harmonic[:, 1] = np.sin(omega * t)
#     X_harmonic[:, -1] = 1  # Intercept
#     beta, residuals, rank, s = np.linalg.lstsq(X_harmonic, y, rcond=None)

#     # Intercept-only model
#     X_intercept = np.ones((len(t), 1))  # Intercept
#     _, residuals_intercept, _, _ = np.linalg.lstsq(X_intercept, y, rcond=None)

#     # Calculate the chi-square statistic
#     # Here, we assume the residual sum of squares effectively approximates twice the difference in log-likelihoods
#     chi_square_stat = (residuals_intercept - residuals) / np.var(y) * len(y)

#     # Degrees of freedom: number of parameters in harmonic model - number of parameters in intercept-only model
#     df = 2  # cos and sin terms are the extra parameters

#     # Compute the chi-square p-value
#     chi_square_p_value = stats.chi2.sf(chi_square_stat, df)
#     # adjusted_p_value = benjamini_hochberg_correction([chi_square_p_value])[0]

#     return beta, chi_square_p_value


def benjamini_hochberg_correction(p_values):

    p_values = np.array(p_values)
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    corrected_p_values = np.empty(n)

    # Apply the BH formula
    for i, p in enumerate(sorted_p_values):
        corrected_p_values[sorted_indices[i]] = min(p * n / (i + 1), 1.0)

    return corrected_p_values


def cSVD(res, center_around="mean", return_explained=False):
    """
    This function performs cSVD to find the common phase and amplitude for all genes
    across different cell types, and the common phase shift for all cell types.
    Defined for ONLY one harmonic!

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
    U, S, Vh = np.linalg.svd(C.T, full_matrices=True)
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
            U_[:, i] = U[:, i] * rot * S[i] * max_s
            # since we took the conj earlier, here we just multiply by rot
            V_[:, i] = V[:, i] * rot / max_s

    if return_explained:
        return U_, V_, S_norm
    else:
        return U_, V_


# def harmonic_regression_adata(
#     adata,
#     genes,
#     phases,
#     layer=None,
#     n_harmonics=1,
# ):
#     """
#     Performs harmonic regression using OLS on genes in an AnnData object and computes
#     statistics comparing against a flat model.

#     Parameters:
#     -----------
#     adata : AnnData
#         AnnData object containing log-transformed gene expression data
#     genes : list
#         List of gene names to analyze
#     phases : array-like
#         Vector of phases in radians (0 to 2π) for each cell/sample
#     layer : str, optional
#         Layer in AnnData to use. If None, uses .X
#     n_harmonics : int, default=1
#         Number of harmonics to include in the model
#     return_full_results : bool, default=False
#         Whether to return full regression results for each gene

#     Returns:
#     --------
#     results_df : pandas.DataFrame
#         DataFrame containing regression results with the following columns:
#         - gene: Gene name
#         - a_g, b_g, m_g: Coefficients for cos, sin, and intercept
#         - pvalue: P-value from likelihood ratio test vs. flat model
#         - padj: Adjusted p-value (Benjamini-Hochberg)
#         - BIC: BIC difference between harmonic and flat models
#         - amplitude: Amplitude of the first harmonic
#         - phase: Phase of the first harmonic
#         - rsquared: R-squared value for the model

#     full_results : dict, optional
#         Dictionary with gene names as keys and full regression result objects as values
#         (only returned if return_full_results=True)
#     """
#     # Create design matrices
#     X_harmonic = create_harmonic_design_matrix(phases, n_harmonics)
#     X_flat = create_harmonic_design_matrix(phases, 0)  # Just the intercept

#     # Filter genes to ensure they exist in the dataset
#     genes = [g for g in genes if g in adata.var_names]

#     results_list = []

#     # tqdm is used to show progress
#     for gene in tqdm(genes, desc="Fitting genes", unit="gene"):
#         # Extract gene expression
#         if layer is None:
#             y = (
#                 adata[:, gene].X.toarray().flatten()
#                 if hasattr(adata[:, gene].X, "toarray")
#                 else adata[:, gene].X
#             )
#         else:
#             y = (
#                 adata[:, gene].layers[layer].toarray().flatten()
#                 if hasattr(adata[:, gene].layers[layer], "toarray")
#                 else adata[:, gene].layers[layer]
#             )

#         # Fit harmonic model using OLS
#         harmonic_model = sm.OLS(y, X_harmonic)
#         flat_model = sm.OLS(y, X_flat)

#         try:
#             harmonic_fit = harmonic_model.fit()
#             flat_fit = flat_model.fit()

#             # Likelihood ratio test for p-value
#             # For OLS, we can use F-test or directly compare RSS
#             n = len(y)
#             df_harmonic = X_harmonic.shape[1]
#             df_flat = X_flat.shape[1]
#             df_diff = df_harmonic - df_flat

#             rss_harmonic = harmonic_fit.ssr
#             rss_flat = flat_fit.ssr

#             f_stat = ((rss_flat - rss_harmonic) / df_diff) / (
#                 rss_harmonic / (n - df_harmonic)
#             )
#             pvalue = 1 - stats.f.cdf(f_stat, df_diff, n - df_harmonic)

#             # Calculate BIC difference
#             bic_diff = (
#                 harmonic_fit.bic - flat_fit.bic
#             )  # Positive value means harmonic model is better

#             # Extract parameters for first harmonic
#             params = harmonic_fit.params

#             # For the standard form (first harmonic only): y ~ a_g * cos(phi) + b_g * sin(phi) + m_g
#             if n_harmonics >= 1:
#                 a_g = params[X_harmonic.columns.get_loc("cos1")]
#                 b_g = params[X_harmonic.columns.get_loc("sin1")]

#                 amplitude = np.sqrt(a_g**2 + b_g**2)
#                 phase = np.arctan2(b_g, a_g) % (2 * np.pi)
#             else:
#                 a_g = 0
#                 b_g = 0
#                 amplitude = 0
#                 phase = 0

#             # Get intercept
#             if "const" in X_harmonic.columns:
#                 m_g = params[X_harmonic.columns.get_loc("const")]
#             else:
#                 m_g = y.mean()

#             # Compile results
#             result = {
#                 "gene": gene,
#                 "a_g": a_g,
#                 "b_g": b_g,
#                 "m_g": m_g,
#                 "pvalue": pvalue,
#                 "BIC": bic_diff,
#                 "amp": amplitude,
#                 "phase": phase,
#                 "R2": harmonic_fit.rsquared,
#             }

#             results_list.append(result)

#         except Exception as e:
#             print(f"Error fitting model for gene {gene}: {str(e)}")

#     # Create results DataFrame
#     results_df = pd.DataFrame(results_list)
#     # columns gene as index
#     results_df.set_index("gene", inplace=True)

#     # Adjust p-values for multiple testing (Benjamini-Hochberg)
#     if len(results_list) > 0:
#         results_df["pvalue_correctedBH"] = benjamini_hochberg_correction(
#             results_df["pvalue"].values
#         )

#     return results_df

# def harmonic_regression_loop(E_ygt, t_u, omega=1, eval_fit="bic", return_eval=True):
#     """
#     It takes the output of pseudobulk_loop and returns the parameters of the harmonic regression
#     for each gene and celltype. It use harmonic_regression_bic to fit the data
#     Input:
#     - E_ygt: 3D array with dimensions celltype, gene, timepoint
#     - t_u: timepoints
#     - omega: frequency
#     - eval_fit: 'bic' or 'pvalue'
#     - return_eval: if True, the function returns the BIC/Pvalue of the harmonic regression

#     Output:
#     - res: np.array of shape (N_ct, N_g, 3) with the parameters of the harmonic regression
#         where the last dimension is a, b, mu
#     - vals: np.array of shape (N_ct, N_g) with the BIC/Pvalue of the harmonic
#     """
#     N_y, N_g, N_t = E_ygt.shape
#     res = np.zeros((N_y, N_g, 3))
#     vals = np.zeros((N_y, N_g))

#     for i in range(N_y):
#         for j in range(N_g):
#             y = E_ygt[i, j, :].squeeze()

#             if eval_fit == "pvalue":
#                 beta, v = harmonic_regression_pvalue(t_u, y, omega=omega)
#             elif eval_fit == "bic":
#                 beta, v = harmonic_regression_bic(t_u, y, omega=omega)
#             # if beta is not None:
#             # amp, phase, mu = radial_parameters(beta)
#             res[i, j, 0] = beta[0]
#             res[i, j, 1] = beta[1]
#             res[i, j, 2] = beta[2]
#             vals[i, j] = v
#     if return_eval:
#         return res, vals
#     else:
#         return res
