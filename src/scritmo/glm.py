import numpy as np
import scanpy as sc
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from scipy.stats import chi2
from tqdm import tqdm
from statsmodels.tools import add_constant
from scritmo import create_harmonic_design_matrix, benjamini_hochberg_correction


def glm_gene_fit(
    data,
    phases,
    genes=None,
    counts=None,
    fixed_disp=0.1,
    fit_disp=False,
    layer=None,
    n_harmonics=1,
):
    """
    Fits gene expression data to a harmonic model using statsmodels.

    This function is a standalone version of the fit_genes_fast method from RITMO class,
    without dependencies on JAX or the class structure.

    Parameters:
    -----------
    data : numpy.ndarray, pandas.DataFrame, or AnnData
        Gene expression matrix of shape (cells/samples, genes)
        If DataFrame, genes should be in columns
        If AnnData, genes should be in var_names
    phases : numpy.ndarray
        Phases in radians for each cell/sample (shape: n_samples)
    genes : list, optional
        List of gene names to analyze.
        Required if data is numpy array, optional for DataFrame and AnnData
    counts : numpy.ndarray, optional
        Total counts for each cell (e.g., library size). If None, no normalization is applied
        For AnnData, if None and counts are needed, they will be computed from the data
    fixed_disp : float, default=0.1
        Fixed dispersion parameter when fit_disp=False
    fit_disp : bool, default=False
        Whether to fit the dispersion parameter, not really well supported yet
    layer : str, optional
        For AnnData input, specifies which layer to use (e.g., 'counts', 'spliced').
        If None, uses adata.X
    n_harmonics : int, default=1
        Number of harmonics to fit. 1 for cosine and sine, 2 for second harmonic, etc.

    Returns:
    --------
    params_g : pandas.DataFrame
        DataFrame with fitted parameters for each gene:
        - a_i: coefficient for cos(phase)
        - b_i: coefficient for sin(phase)
        - a_0: intercept
        - disp: dispersion parameter (if fit_disp=True)
        - pvalue: significance of rhythmicity
        - BIC: Bayesian Information Criterion for model selection
        - amp: amplitude of the fitted curve
        - phase: phase of the fitted curve
    """

    # Get gene list
    if genes is None:
        genes = data.var_names.tolist()
    else:
        genes = [gene for gene in genes if gene in data.var_names]
        if len(genes) == 0:
            raise ValueError("None of the specified genes were found in AnnData object")

    # Get expression data from the specified layer
    if layer is None:
        data_c = data[:, genes].X
    else:
        if layer not in data.layers:
            raise ValueError(f"Layer '{layer}' not found in AnnData object")
        data_c = data[:, genes].layers[layer]

    # Try to convert to dense array if it's sparse
    try:
        data_c = data_c.toarray()
    except AttributeError:
        # Already dense
        pass

    # Compute counts if not provided
    if counts is None and (fit_disp or True):  # Always need counts for NB model
        if layer is None:
            total_counts = data.X.sum(axis=1)
        else:
            total_counts = data.layers[layer].sum(axis=1)

        # Convert to dense if needed
        try:
            counts = total_counts.A1  # For CSR matrix
        except AttributeError:
            counts = total_counts

    # Create design matrix
    X = create_harmonic_design_matrix(phases.squeeze(), n_harmonics=n_harmonics)

    # Design matrix for null model (intercept only)
    X_null = create_harmonic_design_matrix(phases.squeeze(), 0)

    # Fit models for each gene
    results_list = []

    # loop with tqdm for progress bar
    for gene_index, gene_name in tqdm(
        enumerate(genes), total=len(genes), desc="Fitting genes"
    ):
        # In case I am using bulk data, which has a gene dependent offset
        if type(counts) == pd.DataFrame:
            counts = counts[gene_name].values

        # Extract the gene counts
        if isinstance(data, pd.DataFrame):
            gene_counts = data[gene_name].values
        else:
            gene_counts = data_c[:, gene_index]

        # Fit the models using statsmodels
        if fit_disp:
            model = NegativeBinomial(gene_counts, X, offset=np.log(counts))
            model_null = NegativeBinomial(gene_counts, X_null, offset=np.log(counts))
        else:
            model = sm.GLM(
                gene_counts,
                X,
                family=sm.families.NegativeBinomial(alpha=fixed_disp),
                offset=np.log(counts),
            )
            model_null = sm.GLM(
                gene_counts,
                X_null,
                family=sm.families.NegativeBinomial(alpha=fixed_disp),
                offset=np.log(counts),
            )

        # Fit models
        try:
            result = model.fit(disp=False)
            result_null = model_null.fit(disp=False)

            # Calculate p-value for rhythmicity using likelihood ratio test
            llr = 2 * (result.llf - result_null.llf)
            pval = 1 - chi2.cdf(llr, 2)

            # Calculate BIC for model selection
            bic = -2 * result.llf + np.log(len(gene_counts)) * (len(result.params) + 1)
            bic_null = -2 * result_null.llf + np.log(len(gene_counts)) * (
                len(result_null.params) + 1
            )
            delta_bic = bic - bic_null

            # Store the results dynamically based on number of harmonics
            result_dict = {"gene": gene_name}

            # Extract all parameters except dispersion (if fitted) and name them
            param_values = result.params[:-1] if fit_disp else result.params
            a_0 = param_values.iloc[-1]  # Intercept is the last before dispersion
            result_dict["a_0"] = a_0

            # Loop through harmonics to assign a_k and b_k dynamically
            for h in range(1, n_harmonics + 1):
                a_index = (h - 1) * 2
                b_index = a_index + 1
                if a_index < len(param_values):
                    result_dict[f"a_{h}"] = param_values.iloc[a_index]
                if b_index < len(param_values):
                    result_dict[f"b_{h}"] = param_values.iloc[b_index]

            if fit_disp:
                # fix this line!
                result_dict["disp"] = result.params.iloc[3]
            else:
                result_dict["disp"] = fixed_disp

            result_dict["BIC"] = delta_bic
            result_dict["pvalue"] = pval
            results_list.append(result_dict)

        except Exception as e:
            print(f"Warning: Could not fit model for gene {gene_name}: {str(e)}")
            # Add a placeholder with NaN values for each harmonic parameter
            result_dict = {"gene": gene_name}
            for h in range(1, n_harmonics + 1):
                result_dict[f"a_{h}"] = np.nan
                result_dict[f"b_{h}"] = np.nan
            result_dict["a_0"] = np.nan
            result_dict["disp"] = np.nan if fit_disp else fixed_disp
            result_dict["pvalue"] = np.nan
            result_dict["BIC"] = np.nan
            results_list.append(result_dict)
            continue

    # Convert results to DataFrame
    params_g = pd.DataFrame(results_list)
    params_g = params_g.set_index("gene")

    # adjust p-values for multiple testing

    if n_harmonics == 1:
        params_g["amp"] = np.sqrt(params_g["a_1"] ** 2 + params_g["b_1"] ** 2)
        params_g["phase"] = np.arctan2(params_g["b_1"], params_g["a_1"]) % (2 * np.pi)

    if n_harmonics > 1:
        # Calculate amplitude and phase numerically for multi-harmonic
        thetas = np.linspace(0, 2 * np.pi, 100)
        amps = []
        phases = []
        for idx, row in params_g.iterrows():
            profile = np.full_like(thetas, row["a_0"], dtype=float)
            for h in range(1, n_harmonics + 1):
                a = row.get(f"a_{h}", 0)
                b = row.get(f"b_{h}", 0)
                profile += a * np.cos(h * thetas) + b * np.sin(h * thetas)
            amplitude = (profile.max() - profile.min()) / 2
            phase = thetas[np.argmax(profile)]
            amps.append(amplitude)
            phases.append(phase)
        params_g["amp"] = amps
        params_g["phase"] = phases

    params_g["pvalue_correctedBH"] = benjamini_hochberg_correction(
        params_g["pvalue"].values
    )

    return params_g


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
        X_df = add_constant(X_df, prepend=False)

    return X_df


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
