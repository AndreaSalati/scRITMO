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


def simulate_dataset(
    ph,
    params_g,
    seq_depth=2e4,
    ref_dataset=None,
    n_samples=1,
    random_seed=0,
    md_columns=["ZTmod", "sample_name", "celltype"],
):
    """
    Simulate a dataset from the model. The model is defined by the parameters in params_g
    and the phase ph. The data is simulated at a given sequencing depth.

    Parameters:
    ph: array-like
        The phase of the data to be simulated. Shape (Nc, 1)
    params_g: The gene parameters of the model (amplitudes and phases)
    seq_depth: The sequencing depth of the simulated data.
    ref_dataset: The reference dataset to be used for the simulation. Needs
        to be anndata object. It basically copies the metadata from the reference.
        Notice that the n_samples parameter is harcdoced to 1 when ref_dataset is not None.
    n_samples: The number of samples to be simulated. It's differents "version"
        of the same cell. Passing mutiple times the same phase is sort of similar
        to pass n_sample > 1
    random_seed: The random seed to be used for the simulation.
    md_columns: The metadata columns to be used for the simulation. It should
        be the same as the ones used in the reference dataset. If ref_dataset is None,
        this parameter is ignored.
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
    sim_data = like.sample(key=PRNGKey(random_seed), sample_shape=(n_samples,))

    # making sim_data compatible with RITMO class
    if ref_dataset is not None:
        sim_data = sim_data.squeeze()
        sim_data = np.array(sim_data)
        adata_sim = anndata.AnnData(X=sim_data, var=params_g.index.values)
        adata_sim.var_names = params_g.index.values
        adata_sim.layers["spliced"] = adata_sim.X
        adata_sim.obs[md_columns] = ref_dataset.obs[md_columns].values
        adata_sim.obs["true_phase"] = ph
        return adata_sim
    else:
        return sim_data


def simulate_dataset2(
    ph,
    params_g,
    seq_depth=2e4,
    return_adata=True,
    n_samples=1,
    random_seed=0,
):
    """
    Simulate a dataset from the model. The model is defined by the parameters in params_g
    and the phase ph. The data is simulated at a given sequencing depth.

    Parameters:
    ph: array-like
        The phase of the data to be simulated. Shape (Nc, 1)
    params_g: The gene parameters of the model (amplitudes and phases)
    seq_depth: The sequencing depth of the simulated data.
    return_adata: If True, the function returns an anndata object. If False,
        it returns a numpy array.
    n_samples: The number of samples to be simulated. It's differents "version"
        of the same cell. Passing mutiple times the same phase is sort of similar
        to pass n_sample > 1
    random_seed: The random seed to be used for the simulation.
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
    sim_data = like.sample(key=PRNGKey(random_seed), sample_shape=(n_samples,))
    sim_data = sim_data.squeeze()
    sim_data = np.array(sim_data)

    # making sim_data compatible with RITMO class
    if return_adata:
        adata_sim = anndata.AnnData(X=sim_data, var=params_g.index.values)
        adata_sim.var_names = params_g.index.values
        adata_sim.layers["spliced"] = adata_sim.X
        adata_sim.obs["true_phase"] = ph
        return adata_sim
    else:
        return sim_data
