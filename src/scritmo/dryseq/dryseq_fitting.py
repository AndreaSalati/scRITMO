import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydeseq2.dds import DeseqDataSet
from joblib import Parallel, delayed


def estimate_dispersions(counts_df, metadata, design_formula="group"):
    """
    Step 1: Use PyDESeq2 to estimate size factors and dispersions.

    Args:
        counts_df: (Samples x Genes) DataFrame
        metadata: (Samples x Metadata) DataFrame
    """
    # 1. Setup PyDESeq2 Object
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design_factors=design_formula,
        refit_cooks=True,
        quiet=True,
    )

    # 2. Run minimal pipeline to get dispersions
    dds.deseq2()

    # 3. Extract parameters
    # Dispersions: stored in dds.varm['dispersions'] -> Shape (n_genes,)
    dispersions = dds.var["dispersions"]

    # Size Factors: stored in dds.obsm['size_factors'] -> Shape (n_samples,)
    if "size_factors" in dds.obsm:
        size_factors = dds.obsm["size_factors"]
        offsets = np.log(size_factors)
    else:
        offsets = np.zeros(counts_df.shape[0])

    return dds, dispersions, offsets


def fit_single_gene(gene_counts_vector, design_matrices, dispersion, offset, gene_name):
    """
    Fits all models for a single gene vector.
    """
    deviances = []
    params = []

    # If dispersion is invalid, skip
    if np.isnan(dispersion) or dispersion <= 0:
        return None

    # Statsmodels NB alpha is the same as DESeq2 dispersion
    family = sm.families.NegativeBinomial(alpha=dispersion)

    for design_matrix in design_matrices:
        try:
            # GLM expects endog (y) to be (n_samples,)
            model = sm.GLM(
                endog=gene_counts_vector,
                exog=design_matrix,
                family=family,
                offset=offset,
            )
            res = model.fit()

            deviances.append(res.deviance)
            params.append(res.params)

        except Exception:
            # Convergence failure or rank issue
            deviances.append(np.nan)
            params.append(None)

    return {"gene": gene_name, "deviances": deviances, "params": params}


def run_iterative_fitting(counts_df, design_matrices, dispersions, offsets, n_jobs=4):
    """
    Parallel loop over all genes.

    Args:
        counts_df: (Samples x Genes) DataFrame
        dispersions: Series indexed by gene name
    """
    genes = counts_df.columns  # Iterating over columns (Genes)

    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_single_gene)(
            counts_df[gene].values,  # Extract column as vector
            design_matrices,
            dispersions.loc[gene],  # Get specific dispersion
            offsets,
            gene,
        )
        for gene in genes
    )

    return [r for r in results if r is not None]
