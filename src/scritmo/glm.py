import numpy as np
import scanpy as sc
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from scipy.stats import chi2
from tqdm import tqdm
from statsmodels.tools import add_constant
from .beta import Beta


def glm_gene_fit(
    data,
    phases,
    genes=None,
    counts=None,
    fixed_disp=0.1,
    fit_disp=False,
    layer=None,
    n_harmonics=1,
    outlier_treshold=100.0,
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

        threshold = np.percentile(gene_counts, outlier_treshold)
        mask = gene_counts <= threshold

        if mask.sum() == 0:
            print(f"Skipping gene {gene_name} due to empty mask")
            continue

        gene_counts = gene_counts[mask]
        counts_ = counts[mask]

        # Fit the models using statsmodels
        if fit_disp:
            model = NegativeBinomial(gene_counts, X[mask], offset=np.log(counts_))
            model_null = NegativeBinomial(
                gene_counts, X_null[mask], offset=np.log(counts_)
            )
        else:
            model = sm.GLM(
                gene_counts,
                X[mask],
                family=sm.families.NegativeBinomial(alpha=fixed_disp),
                offset=np.log(counts_),
            )
            model_null = sm.GLM(
                gene_counts,
                X_null[mask],
                family=sm.families.NegativeBinomial(alpha=fixed_disp),
                offset=np.log(counts_),
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
            param_values = param_values.to_dict()
            for k, v in param_values.items():
                result_dict[k] = v

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
    params_g["pvalue_correctedBH"] = benjamini_hochberg_correction(
        params_g["pvalue"].values
    )

    params_g = Beta(params_g)
    params_g.get_amp(inplace=True)

    return params_g


def lm_gene_fit(
    data,
    phases,
    genes=None,
    layer=None,
    n_harmonics=1,
):
    """
    Fits log-transformed gene expression data to a harmonic model using OLS.

    Parameters:
    ----------
    data : numpy.ndarray, pandas.DataFrame, or AnnData
        Log-transformed expression matrix (samples × genes). If DataFrame,
        genes should be columns. If AnnData, genes in var_names.
    phases : array-like
        Phases in radians for each sample (length = n_samples).
    genes : list of str, optional
        Subset of genes to fit. If None, use all in data.
    layer : str, optional
        If AnnData, which .layers[layer] to use; else adata.X.
    n_harmonics : int, default=1
        Number of harmonics to include (cos1/sin1, cos2/sin2, …).

    Returns:
    -------
    params_g : pandas.DataFrame
        Indexed by gene, with columns:
        - a_0       intercept
        - a_1, b_1  cos/sin coefficients (and a_2, b_2, … if n_harmonics>1)
        - BIC       ΔBIC = BIC_full − BIC_null
        - pvalue    from nested F-test
        - amp       amplitude = √(a_1² + b_1²) (or max-minus-min/2 for ≥2)
        - phase     phase in [0,2π)
        - pvalue_correctedBH
    """

    # --- select gene list ---
    if hasattr(data, "var_names"):
        all_genes = list(data.var_names)
    elif isinstance(data, pd.DataFrame):
        all_genes = data.columns.tolist()
    else:
        raise ValueError("For numpy input you must pass genes=list_of_names")

    if genes is None:
        genes = all_genes
    else:
        genes = [g for g in genes if g in all_genes]
        if not genes:
            raise ValueError("None of the specified genes found in data")

    # --- extract expression matrix ---
    if hasattr(data, "layers"):
        mat = data[:, genes].layers[layer] if layer else data[:, genes].X
    elif isinstance(data, pd.DataFrame):
        mat = data[genes].values
    else:  # numpy array
        mat = data[:, [all_genes.index(g) for g in genes]]

    # if sparse
    try:
        mat = mat.toarray()
    except AttributeError:
        pass

    # --- build design matrices ---
    X = create_harmonic_design_matrix(phases, n_harmonics)
    X_null = create_harmonic_design_matrix(phases, 0)  # intercept only

    results = []
    for i, gene in enumerate(genes):
        y = mat[:, i]
        # full model
        mod = sm.OLS(y, X)
        res = mod.fit()
        # null model
        mod0 = sm.OLS(y, X_null)
        res0 = mod0.fit()
        # nested F-test
        f_stat, p_val, _ = res.compare_f_test(res0)
        # ΔBIC
        delta_bic = res.bic - res0.bic
        # collect params
        d = {"gene": gene}
        param_values = res.params.to_dict()
        for k, v in param_values.items():
            d[k] = v

        d["BIC"] = delta_bic
        d["pvalue"] = p_val

        results.append(d)

    df = pd.DataFrame(results).set_index("gene")
    df = Beta(df)
    df.get_amp(inplace=True)

    # BH correction
    df["pvalue_correctedBH"] = benjamini_hochberg_correction(df["pvalue"].values)
    return Beta(df)


def create_harmonic_design_matrix(phases, n_harmonics=1, add_intercept=True):
    """
    Constructs a design matrix for harmonic regression with multiple harmonics,
    with columns ordered as: intercept (a_0), a_1, b_1, a_2, b_2, ..., a_n, b_n.

    Parameters:
    -----------
    phases : array-like
        Vector of phases in radians (length = n_samples).
    n_harmonics : int, default=1
        Number of harmonics to include (cos1/sin1 through cos_n/sin_n).
    add_intercept : bool, default=True
        If True, adds an intercept column named 'a_0'.

    Returns:
    --------
    X_df : pandas.DataFrame
        Design matrix of shape (n_samples, 1 + 2*n_harmonics) if add_intercept,
        otherwise (n_samples, 2*n_harmonics). Columns are:
        ['a_0', 'a_1', 'b_1', ..., 'a_n', 'b_n'].
    """
    phases = np.asarray(phases).squeeze()
    n_samples = len(phases)
    cols = {}
    if add_intercept:
        cols["a_0"] = np.ones(n_samples, dtype=float)
    for h in range(1, n_harmonics + 1):
        cols[f"a_{h}"] = np.cos(h * phases)
        cols[f"b_{h}"] = np.sin(h * phases)

    X_df = pd.DataFrame(cols)
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
