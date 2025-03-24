import numpy as np
import pandas as pd

from scritmo.basics import df2dict, dict2df, ccg
from scritmo.circular import *
from scipy.stats import circmean, circvar, circstd

from scritmo.jax_module.numpyro_models import (
    model_MLE_NB,
    guide_MLE,
    model_null,
)

from scritmo.jax_module.posterior import *


def check_shapes(
    self,
    genes_sc,
    params_g,
):
    """
    This function checks wether the list of genes and the
    params_g dict/df have the same number of genes.
    It avoid shape mismatch errors later on.
    """
    if type(params_g) is dict:
        if params_g["a_g"].shape[1] != genes_sc.shape[0]:
            raise ValueError(
                "The number of genes in the gene coefficients does not match the number of genes in the data"
            )

    elif type(params_g) is pd.DataFrame:
        genes_sc = np.intersect1d(genes_sc, params_g.index)
        params_g = params_g.loc[genes_sc]
        # maybe not necessary afterall
        params_g = df2dict(params_g)

    return genes_sc, params_g


def _eval_performance(self, pred_ph, true_ph, period=2 * np.pi):
    """
    This method evaluates the performance of the model by comparing the predicted
    phases with the true phases. Remember to sepcify what is the period
    of the circular variable (2pi or 24h). The output is in hours
    """
    median_abs_error = circular_median_absolute_deviation(pred_ph, true_ph, period)
    mean_abs_error = circular_mean_absolute_deviation(pred_ph, true_ph, period)
    root_mean_sq_error = circular_sqrt_mean_squared_error(pred_ph, true_ph, period)

    if period != 24:
        median_abs_error *= self.rh
        mean_abs_error *= self.rh
        root_mean_sq_error *= self.rh

    return median_abs_error, mean_abs_error, root_mean_sq_error


def _filter_genes_and_prepare_data(self, genes_sc, counts, params_g, use_PB=False):
    """
    This function filters the genes that are not-expressed in the subset of
    the adata object that will be used for the inference.
    Also, it prepared data_c and counts for the inference.
    CAREFUL: counts should be the total counts for each cell, including
        ALL genes, and not only the subset of genes used for the inference.
    """

    genes_sc = np.intersect1d(genes_sc, self.adata.var_names)
    if use_PB:
        circ = self.PB[:, genes_sc]
    else:
        circ = self.adata[:, genes_sc]

    # getting rid of genes with no expression
    mask_g = np.array(circ.layers["spliced"].sum(0) != 0).squeeze()
    genes_sc = genes_sc[mask_g]

    if params_g is not None:
        for key in params_g.keys():
            # checking that I am filtering only the gene related parameteres
            if params_g[key].shape[1] == mask_g.shape[0]:
                params_g[key] = params_g[key][:, mask_g]
    circ = circ[:, genes_sc]
    data_c = circ.layers["spliced"].toarray()

    if counts is None:
        if use_PB:
            counts = self.PB[:, :].layers["spliced"].toarray().sum(1).reshape(-1, 1)
        else:
            counts = self.adata[:, :].layers["spliced"].toarray().sum(1).reshape(-1, 1)
    self.counts = counts
    self.genes_sc = genes_sc

    return genes_sc, data_c, counts, params_g


########################################################
# sc_inference method helper functions
########################################################


def eval_fit_genes(self, svi_g, svi_null, data_c, ph):
    """
    This function computes the log likelihoods for the genes and the cells.
    Than proceedes to compute the loglikelihood for null flat model.
    Than it computes pvalues, BIC and AIC for the model.
    Parameters:
    - svi_g: the SVI object containing the gene coefficients
    - svi_null: the SVI object containing the null model
    - data_c: the data matrix Nc x Ng
    - ph: the external time values
    """
    # add ph to the dictionary of parameters
    params = svi_g.params
    params["phi_c"] = ph

    ll_cg = get_ll(
        model=model_MLE_NB,
        params=svi_g.params,
        data=data_c,
        counts=self.counts,
    )

    ll_null_cg = get_ll(
        model=model_null,
        params=svi_null.params,
        data=data_c,
        counts=self.counts,
    )

    ll_g = ll_cg.sum(0)
    ll_null_g = ll_null_cg.sum(0)

    NS = data_c.shape[0]
    # print(f"NS = {NS}")
    pvals = pvalue(ll_g, ll_null_g, 3, 1, BH=True)
    bics = BIC_test(ll_g, ll_null_g, 3, 1, NS)
    aics = AIC_test(ll_g, ll_null_g, 3, 1, NS, correction=True)
    # data frame with genes on index and pvalues, bics, aics as columns
    model_stats = pd.DataFrame(
        np.array([pvals, bics, aics]).T,
        index=self.genes_sc,
        columns=["pvalue", "BIC", "AIC"],
    )

    return model_stats


########################################################
# sc_posterior method helper functions
########################################################


def _configure_jax_device(self, jax_device):
    """Configure JAX device."""
    if jax_device == "cpu":
        jax.config.update("jax_platform_name", "cpu")


def _compute_posterior_statistics(self, l_xc, Nx):
    """Compute posterior statistics."""
    delta_phi = 2 * np.pi / Nx
    # deltaH_c = np.apply_along_axis(delta_entropy, axis=0, arr=l_xc * delta_phi)
    post_mean_c = np.apply_along_axis(circmean, 0, l_xc * delta_phi)
    post_var_c = np.apply_along_axis(circvar, 0, l_xc * delta_phi)
    post_std_c = np.apply_along_axis(circstd, 0, l_xc * delta_phi)
    print("output of this is wrong! fix it!")
    return post_mean_c, post_var_c, post_std_c
