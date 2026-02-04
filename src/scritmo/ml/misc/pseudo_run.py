import numpy as np
import scanpy as sc
import seaborn as sns
from matplotlib import pyplot as plt

from tqdm import tqdm
from scipy.stats import circvar
import pandas as pd
import torch

import scritmo as sr

# here w converts h in radians, rh the inverse
from scritmo import w, rh, ccg, pseudobulk
from ..context_model import ContextModel
from ..utils import assemble_mp
from ..discrete_MI import MI
# TODO: train_deterministic_tempo function not found in the codebase, may need to be implemented or imported from elsewhere


def run_pseudobulk(
    adata_full,
    params_g,
    by=["sample_name", "celltype"],
    n_theta=100,
    n_steps=5000,
    context_col="celltype",
    context_mode="full",
):
    """ """
    # batch vector to pass to the model
    batch_c = (
        adata_full.obs[by[0]].astype(str) + "_" + adata_full.obs[by[1]].astype(str)
    )
    adata = pseudobulk(adata_full, by)
    adata.layers["spliced"] = adata.layers["sum"].copy()

    counts = adata.obs.n_counts.values[:, None]

    # Get the list of genes for inference directly from the params index
    genes = np.intersect1d(params_g.index, adata.var_names)
    ext_phase = adata.obs.ZTmod * w

    glm_means = sr.glm_gene_fit(
        adata,
        phases=ext_phase,
        genes=genes,
        n_harmonics=0,
    )

    glm_means = glm_means.a_0
    params_g.m_g = glm_means

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_c, mp = assemble_mp(
        adata=adata,
        params_g=params_g,
        counts=counts,
        labels=adata.obs[context_col].values,
        layer="spliced",
        device=device,
        n_theta=n_theta,
    )

    cmodel = ContextModel(mp, data_c, context_mode=context_mode, method="simpson")
    cmodel.to(device)
    batch_size = adata.shape[0]

    _ = train_deterministic_tempo(
        cmodel, data_c, mp, n_steps=n_steps, batch_size=batch_size
    )

    posterior_xc = cmodel.get_phase_posteriors(data_c, method="sum")
    post_mean_c, post_var_c, post_std_c = sr.compute_posterior_statistics(
        posterior_xc,
    )

    obs = adata.obs.copy()
    obs["post_mean_c"] = post_mean_c
    obs["post_std_c"] = post_std_c
    obs["batch"] = obs[by[0]].astype(str) + "_" + obs[by[1]].astype(str)
    sorted_obs = obs.sort_values(by="batch")

    return sorted_obs, batch_c
