import numpy as np
import torch
from torch import tensor as tt
from torch import nn
from scritmo import Beta, optimal_shift
import pandas as pd
from .utils import harmonic_dm_torch


def prepare_extra_genes(self, mp):
    """
    Prepare the model to handle extra genes that are not pan-rhythmic, but
    celltype specific. This function adds the extra parameters to the model.
    It prepares the more complex design matrix and the me_g and abe_g parameters.

    Args:
        mp (dict): A dictionary containing the model parameters.
    """
    dm = (
        self.design_matrix(
            mp["params_e_g"].context.values, all_categories=self.context_u
        )
        .to(self.dev)
        .bool()
    )

    self.register_buffer("dm_e_gy", dm)

    # model parameters
    self.m_e_g = nn.Parameter(
        torch.tensor(mp["params_e_g"].a_0.values, dtype=torch.float32)
    )

    acrophase_e_tensor = tt(mp["params_e_g"]["phase"].values, dtype=torch.float32)

    safe_amp = torch.clamp(
        tt(mp["params_e_g"]["amp"].values, dtype=torch.float32),
        min=1e-2,
        max=self.max_amp - 1e-2,
    )
    log_amp_e = torch.logit(safe_amp / self.max_amp)
    self.log_amp_e = nn.Parameter(log_amp_e)

    if self.fix_phase:
        self.register_buffer("acrophase_e", acrophase_e_tensor)
    else:
        self.acrophase_e = nn.Parameter(acrophase_e_tensor)


def nb_dist_extra(self, ct_index, indices=slice(None)):
    """
    Computes the likelihood distribution (Negative Binomial or Poisson)
    for the given parameters.
    Called by several methods
    """
    if indices is None:
        indices = slice(None)

    # --- Common computations for the expected mean (rate) ---
    mask_c = self.dm[indices, ct_index].bool()
    mask_g = self.dm_e_gy[:, ct_index].bool()

    counts = self.counts[indices, :][mask_c, :]
    X = self.X[:, indices, :][:, mask_c, :]
    intercept_g = self.m_e_g[mask_g]
    ab_e = self._get_ab_extra()
    ab_pg = ab_e[:, mask_g]
    disp = torch.exp(self.log_disp)
    if self.fix_disp_val == "context":
        disp = disp[ct_index]

    # E_xcg is the expected mean of the distribution
    E_xcg = (X @ ab_pg) + intercept_g

    # --- Select distribution based on the noise model ---
    if self.noise_model == "nb":
        E_xcg = torch.exp(E_xcg) * counts
        # Negative Binomial distribution
        r = 1 / disp
        eps = 1e-6
        p = disp * E_xcg / (1 + disp * E_xcg)
        p = p.clamp(min=eps, max=1 - eps)

        return torch.distributions.NegativeBinomial(total_count=r, probs=p)

    elif self.noise_model == "poisson":
        # Poisson distribution, where the rate is the expected mean
        E_xcg = torch.exp(E_xcg) * counts
        return torch.distributions.Poisson(rate=E_xcg)

    elif self.noise_model == "gaussian":
        # Gaussian distribution with mean E_xcg and fixed std dev
        std_dev = 1.0
        return torch.distributions.Normal(loc=E_xcg, scale=std_dev)

    else:
        raise NotImplementedError(
            f"Noise model '{self.noise_model}' is not implemented."
        )


def nb_dist_extra_posterior(
    self, ct_index, indices=slice(None), counts=None, n_theta=None
):
    """
    Computes the likelihood distribution (Negative Binomial or Poisson)
    for the given parameters.
    Called by several methods
    """
    if indices is None:
        indices = slice(None)

    if counts is None:
        counts = self.counts[indices]

    if n_theta is not None:

        phi_x_new = torch.linspace(
            0, 2 * torch.pi, n_theta + 1, dtype=torch.float32, device=self.dev
        )[:-1]

        X_new = harmonic_dm_torch(phi_x_new, self.nh, False)
        X = X_new.unsqueeze(1).expand(n_theta, self.Nc, self.nh * 2)
        X = X[:, indices, :]
    else:
        X = self.X[:, indices, :]

    # --- Common computations for the expected mean (rate) ---
    mask_c = self.dm[indices, ct_index].bool()
    mask_g = self.dm_e_gy[:, ct_index].bool()

    counts = self.counts[indices, :][mask_c, :]
    X = X[:, mask_c, :]
    intercept_g = self.m_e_g[mask_g]
    ab_e = self._get_ab_extra()
    ab_pg = ab_e[:, mask_g]
    disp = torch.exp(self.log_disp)
    if self.fix_disp_val == "context":
        disp = disp[ct_index]

    # E_xcg is the expected mean of the distribution
    E_xcg = (X @ ab_pg) + intercept_g

    # --- Select distribution based on the noise model ---
    if self.noise_model == "nb":
        E_xcg = torch.exp(E_xcg) * counts
        # Negative Binomial distribution
        r = 1 / disp
        eps = 1e-6
        p = disp * E_xcg / (1 + disp * E_xcg)
        p = p.clamp(min=eps, max=1 - eps)

        return torch.distributions.NegativeBinomial(total_count=r, probs=p)

    elif self.noise_model == "poisson":
        # Poisson distribution, where the rate is the expected mean
        E_xcg = torch.exp(E_xcg) * counts
        return torch.distributions.Poisson(rate=E_xcg)

    elif self.noise_model == "gaussian":
        # Gaussian distribution with mean E_xcg and fixed std dev
        std_dev = 1.0
        return torch.distributions.Normal(loc=E_xcg, scale=std_dev)

    else:
        raise NotImplementedError(
            f"Noise model '{self.noise_model}' is not implemented."
        )


def loss_extra_xc(
    self,
    indices=slice(None),
    y_e=None,
    n_theta=None,
):
    """
    Computes the log likelihood of the extra genes given the context.
    It loops one celltype at a time, and subsets the data accordingly.
    """
    Nx, Nc, _ = y_e.shape
    ll_e_xc = torch.zeros((Nx, Nc), dtype=torch.float32, device=self.dev)
    # ll_e_xc = torch.zeros((self.Nc, self.Nc), dtype=torch.float32, device=self.dev)[
    #     :, indices
    # ]
    for i, ct in enumerate(self.context_u):
        mask_c = self.dm[indices, i].squeeze().bool()
        if n_theta is None:
            dist_e = self.nb_dist_extra(
                i,
                indices=indices,
            )
        else:
            dist_e = self.nb_dist_extra_posterior(
                i,
                indices=indices,
                n_theta=n_theta,
                counts=self.counts[indices],
            )
        # to keep flat last dimension
        y_ct = y_e[:, mask_c, :][:, :, self.dm_e_gy[:, i]]
        ll_e_xcg = dist_e.log_prob(y_ct)
        ll_e_xc_ = ll_e_xcg.sum(dim=2)
        ll_e_xc[:, mask_c] = ll_e_xc_

    return ll_e_xc


def prepare_params_df(pars_dict, params_pan=None, ct_for_pan=None):
    """
    Prepares two dataframes from a dictionary of dataframes.

    This function first identifies the set of common genes (indices) shared across all input dataframes. It then returns two separate dataframes: one containing only the common genes from a specified or default cell type, and another containing all of the non-common genes from all cell types, along with their original 'context'.

    Parameters
    ----------
    pars_dict : dict
        A dictionary where keys represent cell types and values are pandas DataFrames.
        The index of each DataFrame is expected to be gene names.
    ct_for_pan : str, optional
        The key from `pars_dict` to use for the `params_pan` DataFrame. If `None`
        (default), the first key in the dictionary is used to create this DataFrame.

    Returns
    -------
    params_pan : pandas.DataFrame
        A DataFrame containing only the genes that are common to all input dataframes.
        The data is taken from the DataFrame specified by `ct_for_pan`.
    par_extra : pandas.DataFrame
        A DataFrame containing all the genes that are not common across all
        input dataframes. It includes a new 'context' column indicating the
        original cell type for each gene.
    """
    ctu = list(pars_dict.keys())
    if params_pan is None:
        # find common indices between all dataframes
        common_genes = list(
            set.intersection(*[set(df.index) for df in pars_dict.values()])
        )

        if ct_for_pan is None:
            params_pan = pars_dict[ctu[0]].loc[common_genes].copy()
        else:
            params_pan = pars_dict[ct_for_pan].loc[common_genes].copy()
    else:
        common_genes = list(set(params_pan.index))

    # now for each dataset take the remaining gens

    for i, ct in enumerate(ctu):
        par = pars_dict[ct]
        par["context"] = ct
        diff_genes = list(set(par.index) - set(common_genes))
        par = par.loc[diff_genes]
        if i == 0:
            par_extra = par.copy()
        else:
            par_extra = pd.concat([par_extra, par], axis=0)

    return Beta(params_pan), Beta(par_extra)

def _get_ab_extra(self):

    amp = torch.sigmoid(self.log_amp_e) * self.max_amp
    cos = amp * torch.cos(self.acrophase_e).unsqueeze(0)
    sin = amp * torch.sin(self.acrophase_e).unsqueeze(0)
    return torch.cat([cos, sin], dim=0)