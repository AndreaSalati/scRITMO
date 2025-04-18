import numpy as np
import scanpy as sc
from scritmo import polar_genes_pandas
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from scipy.stats import chi2


def harmonic_fit_genes(
    data, phases, genes=None, counts=None, fixed_disp=0.1, fit_disp=False, layer=None
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
        Whether to fit the dispersion parameter (slower but more accurate)
    layer : str, optional
        For AnnData input, specifies which layer to use (e.g., 'counts', 'spliced').
        If None, uses adata.X

    Returns:
    --------
    params_g : pandas.DataFrame
        DataFrame with fitted parameters for each gene:
        - a_g: coefficient for cos(phase)
        - b_g: coefficient for sin(phase)
        - m_g: intercept
        - disp: dispersion parameter (if fit_disp=True)
        - pvalue: significance of rhythmicity
    params_g_pol : pandas.DataFrame
        DataFrame with polar parameters for each gene:
        - amp: amplitude
        - phase: phase
        - mean: mean expression level
        - disp: dispersion parameter
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
    X = np.ones((data_c.shape[0], 3))
    X = pd.DataFrame(X, columns=["cos", "sin", "intercept"])
    X.cos = np.cos(phases)
    X.sin = np.sin(phases)

    # Design matrix for null model (intercept only)
    X_null = np.ones((phases.shape[0], 1))

    # Fit models for each gene
    results_list = []
    pvals = []

    for gene_index, gene_name in enumerate(genes):
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

            # Store the results
            result_dict = {
                "gene": gene_name,
                "a_g": result.params.iloc[0],
                "b_g": result.params.iloc[1],
                "m_g": result.params.iloc[2],
            }

            if fit_disp:
                result_dict["disp"] = result.params.iloc[3]
            else:
                result_dict["disp"] = fixed_disp

            results_list.append(result_dict)
            pvals.append(pval)

        except Exception as e:
            print(f"Warning: Could not fit model for gene {gene_name}: {str(e)}")
            # Add a placeholder with NaN values
            result_dict = {
                "gene": gene_name,
                "a_g": np.nan,
                "b_g": np.nan,
                "m_g": np.nan,
                "disp": np.nan if fit_disp else fixed_disp,
            }
            results_list.append(result_dict)
            pvals.append(np.nan)

    # Convert results to DataFrame
    params_g = pd.DataFrame(results_list)
    params_g = params_g.set_index("gene")

    # Add p-values
    params_g["pvalue"] = pvals

    params_g["amp"] = np.sqrt(params_g["a_g"] ** 2 + params_g["b_g"] ** 2)
    params_g["phase"] = np.arctan2(params_g["b_g"], params_g["a_g"]) % (2 * np.pi)

    return params_g
