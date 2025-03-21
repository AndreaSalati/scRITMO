import numpy as np
import scanpy as sc
import anndata
import jax.numpy as jnp
from jax.random import PRNGKey
import pandas as pd

from scritmo.jax_module.posterior import get_likelihood
from scritmo.jax_module.numpyro_models import model_MLE_NB
from scritmo.basics import df2dict, dict2df


# preparing for the sampling


def simulate_dataset(ph, params_g, seq_depth=2e4, ref_dataset=None, n_samples=1):
    """
    Simulate a dataset from the model. The model is defined by the parameters in params_g
    and the phase ph. The data is simulated at a given sequencing depth.

    Parameters:
    ph: array-like
        The phase of the data to be simulated. Shape (Nc, 1)
    params_g: The parameters of the model.
    seq_depth: The sequencing depth of the simulated data.
    ref_dataset: The reference dataset to be used for the simulation. Needs
        to be anndata object. It basically copies the metadata from the reference.
        Notice that the n_samples parameter is harcdoced to 1 when ref_dataset is not None.
    n_samples: The number of samples to be simulated. It's differents "version"
        of the same cell.
    """

    Nc = ph.shape[0]

    if type(params_g) is pd.DataFrame:
        params_model = df2dict(params_g)

    params_model["phi_c"] = ph
    counts = np.ones((Nc, 1)) * seq_depth

    # sampling from the model
    like = get_likelihood(
        model=model_MLE_NB, params=params_model, data=None, counts=counts
    )
    sim_data = like.sample(key=PRNGKey(0), sample_shape=(n_samples,))

    # making sim_data compatible with RITMO class
    if ref_dataset is not None:
        sim_data = sim_data.squeeze()
        sim_data = np.array(sim_data)
        adata_sim = anndata.AnnData(X=sim_data, var=params_g.index.values)
        adata_sim.var_names = params_g.index.values
        adata_sim.layers["spliced"] = adata_sim.X
        adata_sim.obs[["ZTmod", "sample_name", "celltype"]] = ref_dataset.obs[
            ["ZTmod", "sample_name", "celltype"]
        ].values
        return adata_sim
    else:
        return sim_data
