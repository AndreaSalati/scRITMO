import numpy as np
import scanpy as sc
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from scipy.stats import chi2
from tqdm import tqdm
from statsmodels.tools import add_constant
from .beta import Beta
from .pseudobulk import pseudobulk
from .basics import w, rh, check_library_size_fallback

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse
from joblib import Parallel, delayed
from functools import partial
import warnings


def _fit_single_gene_glm(
    gene_name,
    gene_counts,
    X,
    X_null,
    counts_,
    outlier_threshold,
    noise_model="nb",
    fixed_disp=None,
    fit_disp=True,
    n_harmonics=None,  # Parameter kept in signature for compatibility
    show_warnings=False,  # New parameter for the switch
):
    """
    Fits a model for a single gene with selectable noise distributions.

    Supports Negative Binomial ('nb'), Poisson ('poisson'), and
    Gaussian ('gaussian') noise models.
    """
    try:
        # --- 1. Data Filtering ---
        threshold = np.percentile(gene_counts, outlier_threshold)
        mask = gene_counts <= threshold

        if mask.sum() == 0:
            # Return None if no data remains after filtering
            return None

        gene_counts = gene_counts[mask]
        counts_ = counts_[mask]
        X = X[mask]
        X_null = X_null[mask]

        if isinstance(counts_, pd.DataFrame):
            counts_ = counts_[gene_name].values

        # --- 2. Model Selection and Initialization ---
        model = None
        model_null = None
        pseudocount = 1e-8  # Add a pseudocount to prevent log(0) errors

        if noise_model == "nb":
            offset = np.log(counts_ + pseudocount)
            if fit_disp:
                # Alpha (dispersion) is estimated automatically by the fit method
                family = sm.families.NegativeBinomial()
            else:
                if fixed_disp is None:
                    raise ValueError(
                        "Argument 'fixed_disp' must be provided for Negative Binomial model when 'fit_disp' is False."
                    )
                # Use a fixed alpha
                family = sm.families.NegativeBinomial(alpha=fixed_disp)

            model = sm.GLM(gene_counts, X, family=family, offset=offset)
            model_null = sm.GLM(gene_counts, X_null, family=family, offset=offset)

        elif noise_model == "poisson":
            offset = np.log(counts_ + pseudocount)
            family = sm.families.Poisson()
            model = sm.GLM(gene_counts, X, family=family, offset=offset)
            model_null = sm.GLM(gene_counts, X_null, family=family, offset=offset)

        elif noise_model == "gaussian":
            # For a Gaussian model, we use Ordinary Least Squares (OLS).
            # OLS does not use an offset term for exposure.
            model = sm.OLS(gene_counts, X)
            model_null = sm.OLS(gene_counts, X_null)

        else:
            raise ValueError(
                f"Unknown noise_model: '{noise_model}'. Choose from 'nb', 'poisson', or 'gaussian'."
            )

        # Conditional block for warnings
        if show_warnings:
            result = model.fit(disp=False)
            result_null = model_null.fit(disp=False)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered in divide",
                    category=RuntimeWarning,
                )
                result = model.fit(disp=False)
                result_null = model_null.fit(disp=False)

        bic = -2 * result.llf + np.log(len(gene_counts)) * (len(result.params) + 1)
        bic_null = -2 * result_null.llf + np.log(len(gene_counts)) * (
            len(result_null.params) + 1
        )
        delta_bic = bic - bic_null

        if noise_model == "gaussian":
            # For OLS, use the standard R-squared based on variance
            r2 = result.rsquared
            # pval from nested F-test
            pval = result.f_pvalue
        else:
            # For GLMs (NB, Poisson), use McFadden's pseudo R-squared
            r2 = 1 - (result.llf / result_null.llf)
            # pval
            df_diff = result.df_model - result_null.df_model
            llr = 2 * (result.llf - result_null.llf)
            pval = 1 - chi2.cdf(llr, df_diff)

        result_dict = {"gene": gene_name}
        param_values = result.params.to_dict()
        result_dict.update(param_values)

        if fit_disp:
            result_dict["disp"] = result.params.iloc[-1]
        else:
            result_dict["disp"] = fixed_disp

        result_dict["BIC"] = delta_bic
        result_dict["pvalue"] = pval
        result_dict["r2"] = r2

        return result_dict

    except Exception as e:
        # Build a dictionary with NaNs for failed fits
        # print(f"Warning: Could not fit model for gene {gene_name}: {str(e)}")
        result_dict = {"gene": gene_name}
        for h in range(1, n_harmonics + 1):
            result_dict[f"a_{h}"] = np.nan
            result_dict[f"b_{h}"] = np.nan
        result_dict["a_0"] = np.nan
        result_dict["disp"] = np.nan if fit_disp else fixed_disp
        result_dict["pvalue"] = np.nan
        result_dict["BIC"] = np.nan
        result_dict["r2"] = np.nan
        return result_dict


def _disp_mom(params_g, X, data_c, genes, counts):
    """
    Method-of-moments negative-binomial dispersion, per gene, from the
    residuals of that gene's own fitted rhythm.

    mu_cg   = counts_cg * exp(sum_k X_ck beta_kg)      (all fitted coefficients)
    alpha_g = sum_c[(y_cg - mu_cg)^2 - mu_cg] / sum_c[mu_cg^2],  clipped at 1e-3

    Uses ALL cells: unlike the GLM fit, no outlier trimming is applied here.
    Genes with sum_c[mu_cg^2] == 0 get NaN.
    """
    coef_cols = [c for c in X.columns if c in params_g.columns]
    pos = {}
    for i, g in enumerate(genes):
        pos.setdefault(g, i)
    cols = [pos[g] for g in params_g.index]

    Y = np.asarray(data_c[:, cols], dtype=float)
    B = params_g[coef_cols].to_numpy(dtype=float)
    # (n_cells, n_coef) @ (n_coef, n_genes) -> (n_cells, n_genes)
    mu = np.exp(X[coef_cols].to_numpy(dtype=float) @ B.T)
    if isinstance(counts, pd.DataFrame):
        mu *= counts[list(params_g.index)].to_numpy(dtype=float)
    else:
        mu *= np.asarray(counts, dtype=float).reshape(-1, 1)

    num = ((Y - mu) ** 2 - mu).sum(0)
    den = (mu**2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = num / den
    alpha = np.where(den == 0, np.nan, alpha)
    alpha = np.clip(alpha, 1e-3, None)
    return pd.Series(alpha, index=params_g.index, name="disp_MoM")


def glm_gene_fit(
    data,
    phases,
    genes=None,
    counts=None,
    fixed_disp=0.1,
    fit_disp=False,
    layer="spliced",
    n_harmonics=1,
    outlier_threshold=98.0,
    use_mi=None,
    n_jobs=-1,
    pseudobulk_by=None,
    pb_replicates=1,
    noise_model="nb",
    show_warnings=False,  # New parameter for the switch
    add_slope=False,
    disp_mom=True,
):
    """
    Fits gene expression data to a harmonic model using statsmodels.
    Runs in parallel if n_jobs != 1.

    Parameters:
    ----------
    data : AnnData
        AnnData object containing the expression data.
    phases : array-like
        Array of phases in radians for each cell/sample.
    genes : list of str, optional
        List of gene names to fit. If None, all genes in data.var_names are used.
    counts : array-like, optional
        Total counts per cell/sample for offset. If None, computed from data.
        In case of bulk, can be a DataFrame with genes as columns.
    fixed_disp : float or array-like or pandas.Series, default=0.1
        Fixed dispersion value if fit_disp is False. Either a scalar used for
        every gene, or one value per gene: a numpy array / list aligned
        positionally with the final `genes` list, or a pandas Series indexed by
        gene name (aligned by name; genes missing from the Series fall back to
        0.1, and the number of fallbacks is printed).
    fit_disp : bool, default=False
        If True, fits the dispersion parameter for each gene.
    layer : str, default="spliced"
        Layer in AnnData to use for expression data.
    n_harmonics : int, default=1
        Number of harmonics to include in the model.
    outlier_threshold : float, default=98.0
        Percentile threshold to exclude outliers in gene expression.
    use_mi : str or None, default=None
        If 'classif', computes mutual information with a discrete label in data.obs.
        If 'reg', computes mutual information with the continuous phase.
        If None, MI is not computed.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available cores. 1 runs in serial.
    pseudobulk_by : list of str or None, default=None
        If provided, pseudobulks the data by these obs keys before fitting.
    disp_mom : bool, default=True
        If True, adds a 'disp_MoM' column: the method-of-moments
        negative-binomial dispersion of each gene, computed from the residuals
        of its own fitted rhythm over ALL cells (no outlier trimming, unlike
        the fit itself). It builds a (n_cells, n_genes) mu array, which roughly
        doubles the peak memory of the call - that is why it can be switched
        off.
    """
    if pseudobulk_by:
        data = pseudobulk(
            data,
            groupby_obs_list=pseudobulk_by,
            pseudobulk_layer=layer,
            n_replicates=pb_replicates,
        )

        data.layers[layer] = data.layers["sum"]
        # data_c = data[:, genes].layers[layer]
        phases = data.obs["ZTmod"].astype(float).values * w
        outlier_threshold = 100.0  # no outlier removal in pseudobulk

    if genes is None:
        genes = data.var_names.tolist()
    else:
        genes = [gene for gene in genes if gene in data.var_names]
        if not genes:
            raise ValueError("None of the specified genes were found in AnnData object")

    if layer is None:
        data_c = data[:, genes].X
    else:
        data_c = data[:, genes].layers[layer]

    try:
        data_c = data_c.toarray()
    except AttributeError:
        pass

    if counts is None:
        counts = check_library_size_fallback(data, layer=layer, context="glm_gene_fit")

    X = create_harmonic_design_matrix(
        phases.squeeze(), n_harmonics=n_harmonics, add_slope=add_slope
    )
    X_null = create_harmonic_design_matrix(phases.squeeze(), 0, add_slope=add_slope)

    # --- fixed_disp: one scalar for the call, or one value per gene ---
    disp_per_gene = None
    if isinstance(fixed_disp, pd.Series):
        aligned = fixed_disp.reindex(genes).astype(float)
        n_missing = int(aligned.isna().sum())
        if n_missing:
            print(
                f"fixed_disp: {n_missing} of {len(genes)} genes are missing from the "
                "Series, falling back to the default 0.1 for them."
            )
        disp_per_gene = aligned.fillna(0.1).values
    elif isinstance(fixed_disp, (list, tuple, np.ndarray)):
        disp_per_gene = np.asarray(fixed_disp, dtype=float)
        if disp_per_gene.ndim == 0:  # 0-d array, i.e. a scalar
            fixed_disp = float(disp_per_gene)
            disp_per_gene = None
        elif len(disp_per_gene) != len(genes):
            raise ValueError(
                f"'fixed_disp' has {len(disp_per_gene)} values but there are "
                f"{len(genes)} genes to fit."
            )

    # --- Create the "slim" partial function ---
    # Pre-fill all arguments that are the same for every gene
    fit_function = partial(
        _fit_single_gene_glm,
        X=X,
        X_null=X_null,
        # counts_=counts,
        fit_disp=fit_disp,
        outlier_threshold=outlier_threshold,
        n_harmonics=n_harmonics,
        noise_model=noise_model,
        show_warnings=show_warnings,
    )
    if disp_per_gene is None:
        # really constant across genes, so it belongs in the partial
        fit_function = partial(fit_function, fixed_disp=fixed_disp)

    def _disp_kw(i):
        # empty when fixed_disp is already baked into the partial
        return {} if disp_per_gene is None else {"fixed_disp": float(disp_per_gene[i])}

    # --- Dispatch to serial or parallel execution ---
    if n_jobs == 1:
        print("Running in serial mode.")

        results_list = [
            fit_function(
                gene_name=genes[i],
                gene_counts=data_c[:, i],
                counts_=counts,
                **_disp_kw(i),
            )
            for i in tqdm(range(len(genes)), desc="Fitting genes (serial)")
        ]

    else:
        print(f"Running in parallel on {n_jobs} jobs.")

        results_list = Parallel(n_jobs=n_jobs)(
            delayed(fit_function)(
                gene_name=genes[i],
                gene_counts=data_c[:, i],
                counts_=counts,
                **_disp_kw(i),
            )
            for i in tqdm(range(len(genes)), desc="Fitting genes (parallel)")
        )

    # --- The final cleanup and DataFrame creation remains the same ---
    results_list = [r for r in results_list if r is not None]
    if not results_list:
        print("Warning: All gene fits failed.")
        return pd.DataFrame()

    params_g = pd.DataFrame(results_list).set_index("gene")
    # remove all columns with NaN a_0
    params_g = params_g.dropna(subset=["a_0"])

    if disp_mom:
        disp_mom_g = _disp_mom(params_g, X, data_c, genes, counts)
        cols = list(params_g.columns)
        cols.insert(cols.index("disp") + 1, "disp_MoM")
        params_g["disp_MoM"] = disp_mom_g
        params_g = params_g[cols]

    params_g["pvalue_correctedBH"] = benjamini_hochberg_correction(
        params_g["pvalue"].values
    )
    params_g = Beta(params_g)
    params_g.get_amp(inplace=True)

    if use_mi:
        print("computing Mutual Information...")
        # Note: You might need to adjust the gene list for MI if some fits failed
        valid_genes = params_g.index.tolist()
        mi = compute_mutual_information_classif(data, phases, valid_genes)
        params_g["mi"] = mi

    return params_g


def create_harmonic_design_matrix(
    phases, n_harmonics=1, add_intercept=True, add_slope=False
):
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

    if add_slope:
        cols["slope"] = phases

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


def compute_mutual_information_classif(adata, label, genes, layer=None, n_jobs=None):
    """
    Computes the mutual information between a discrete label (given as a key
    in `adata.obs` or directly as a pandas Series) and the log-normalized
    expression of selected genes from an AnnData object.

    Parameters:
    ----------
    adata : AnnData
        Annotated data matrix.
    label : str or array-like
        If str: key in `adata.obs` corresponding to a discrete categorical variable.
        If array-like: a 1D array or pandas Series with labels for each cell.
    genes : list of str
        List of gene names to include.
    layer : str or None
        If specified, the name of the layer in `adata.layers` to extract expression from.
        Otherwise, uses `adata.X`.

    Returns:
    -------
    pandas.Series
        Mutual information scores for each gene, indexed by gene name.
    """
    # Get the counts matrix
    X = adata[:, genes].layers[layer] if layer else adata[:, genes].X
    if issparse(X):
        X = X.toarray()

    # Get discrete labels
    if isinstance(label, str):
        y = adata.obs[label].astype("category").cat.codes.values
    else:
        label = pd.Series(label, index=adata.obs_names)  # to ensure same indexing
        y = label.astype("category").cat.codes.values

    # Compute MI
    mi_scores = mutual_info_classif(
        X, y, discrete_features=False, random_state=0, n_jobs=n_jobs
    )
    return pd.Series(mi_scores, index=genes, name="mutual_information")


def compute_mutual_information_reg(adata, phase, genes, layer=None, n_jobs=None):
    """
    Computes the mutual information between a circular phase
    and the log-normalized expression of selected genes from
    an AnnData object.

    Parameters:
    ----------
    adata : AnnData
        Annotated data matrix.
    phase : array-like
       array of phases, in radians
    genes : list of str
        List of gene names to include.
    layer : str or None
        If specified, the name of the layer in `adata.layers` to extract expression from.
        Otherwise, uses `adata.X`.

    Returns:
    -------
    pandas.DataFrame
        Mutual information scores for each gene, indexed by gene name.
    """
    # Get the counts matrix
    X = adata[:, genes].layers[layer] if layer else adata[:, genes].X
    if issparse(X):
        X = X.toarray()

    y_cos = np.cos(phase)
    y_sin = np.sin(phase)

    X = StandardScaler().fit_transform(X)
    y_cos = StandardScaler().fit_transform(y_cos.reshape(-1, 1)).ravel()
    y_sin = StandardScaler().fit_transform(y_sin.reshape(-1, 1)).ravel()

    mi_cos = mutual_info_regression(X, y_cos, n_jobs=n_jobs)
    mi_sin = mutual_info_regression(X, y_sin, n_jobs=n_jobs)

    # 2. Build the DataFrame
    df = pd.DataFrame(
        {
            "cos": mi_cos,
            "sin": mi_sin,
        }
    )
    # 3. Combine into scalar summaries
    df["norm"] = np.sqrt(df["cos"] ** 2 + df["sin"] ** 2)
    df["max"] = df[["cos", "sin"]].max(axis=1)
    df["sum"] = df["cos"] + df["sin"]

    # 4. Label rows by gene name
    df.index = genes
    df.index.name = "gene"

    return df
