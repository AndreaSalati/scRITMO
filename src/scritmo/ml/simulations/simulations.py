import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from ..misc.power_spherical.torch_distribution import PowerSpherical
from ..utils import df2dict, nmp
import seaborn as sns
import anndata
import scanpy as sc
from scipy.stats import circstd, circmean

from scipy.sparse import csr_matrix

import scritmo as sr
from scritmo import w, rh, ccg
import seaborn as sns
from context_repo import context_model
from context_repo import trainer

import contextlib
import io
import warnings
from tqdm import tqdm

import numpy as np
from torch import tensor as tt
import matplotlib.pyplot as plt

from .utils import (
    circular_std,
    get_df_clock,
    get_ext_time,
    results_to_df,
    assemble_mp_simulation,
    get_df_var_inference,
)


# essential simulation function used everywhere
def simulate_data(
    n_cells=10000,
    seq_depth=10**3,
    genes_parameters="clock_ccg_params_g.csv",
    device="cuda",
    n_theta=24,  # model param
    dispersion=0.1,
    population_kappa=None,
    n_pop=None,
    context_mus=[],
    context_lambdas=[],
    noise_model="nb",
    phi_sim=None,
):
    """
    Simulates single-cell gene expression data with optional population structure and context-specific parameters.
    Parameters
    ----------
    n_cells : int, optional
        Number of cells to simulate. Default is 10,000.
    seq_depth : int, optional
        Sequencing depth per cell. Default is 1,000.
    genes_parameters : str, optional
        Path to CSV file containing gene parameters. Default is "clock_ccg_params_g.csv".
    device : str, optional
        Device to place the output tensor ("cuda" or "cpu"). Default is "cuda".
    n_theta : int, optional
        Number of theta values for model expansion. Default is 24.
    dispersion : float, optional
        Dispersion parameter for the noise model. Default is 0.1.
    population_kappa : float or None, optional
        Concentration parameter for Von Mises distribution to simulate population structure. If None, no population structure is used.
    n_pop : int or None, optional
        Number of populations to simulate. Required if population_kappa is set.
    context_mus : list or np.ndarray, optional
        Context-specific mean parameters for each population and gene. Shape should be (n_celltypes, n_genes).
    context_lambdas : list or np.ndarray, optional
        Context-specific lambda parameters for each cell type. Length should be n_celltypes.
    noise_model : str, optional
        Noise model to use ("nb" for negative binomial). Default is "nb".
    Returns
    -------
    mp : object
        Simulation metadata or assembled simulation object.
    data_c : torch.Tensor
        Simulated gene expression data tensor of shape (n_theta, n_cells, n_genes).
    phi_sim : torch.Tensor
        Simulated phase values for each cell.
    population : list or None
        List of population labels for each cell, or None if no population structure.
    """

    if len(context_lambdas) and len(context_lambdas) != context_mus.shape[0]:
        raise ValueError(
            f"Size context_lambdas and number of cell types should be identical: {len(context_lambdas)}, {context_mus.shape[0]}"
        )

    if phi_sim is None and population_kappa is not None:
        if n_pop is None:
            raise ValueError("n_pop must be specified when population_kappa is set.")
        population_mean = torch.linspace(0, 2 * np.pi, n_pop + 1)[:-1]
        phi_sim = torch.Tensor([])
        population = []
        n_cells_in_pop = n_cells // n_pop
        for i, mean in enumerate(population_mean):
            phi_sim = torch.cat(
                [
                    phi_sim,
                    torch.distributions.VonMises(mean, population_kappa).sample(
                        (n_cells // n_pop,)
                    ),
                ]
            )
            phi_sim = phi_sim % (2 * np.pi)
            population.extend([str(i)] * (n_cells // n_pop))
    elif phi_sim is None and population_kappa is None:
        phi_sim = torch.linspace(0, 2 * np.pi, n_cells)
        population = None
    else:
        population = None
        phi_sim = torch.tensor(phi_sim, dtype=torch.float32)

    params_g_init = sr.Beta(genes_parameters)

    n_celltypes = context_mus.shape[0] if len(context_mus) else 1
    if len(context_mus) and context_mus.shape != (n_celltypes, params_g_init.shape[0]):
        raise ValueError(
            f"Sizes context_mus and (n_celltypes, n_genes) should be identical: {context_mus.shape}, {(n_celltypes, params_g_init.shape[0])}"
        )

    # Assign cells to cell types randomly, ensuring even distribution across populations
    celltypes = np.random.choice(
        [f"Type_{i}" for i in range(n_celltypes)], size=n_cells
    )

    dm = design_matrix(celltypes)

    if not len(context_mus):
        m_yg = np.zeros((n_celltypes, params_g_init.shape[0]))
    else:
        m_yg = context_mus

    if not len(context_lambdas):
        kappa = np.ones(n_celltypes)
    else:
        kappa = context_lambdas

    data = generate_nb_data(
        params_g_init,
        phi_c=phi_sim,
        counts=seq_depth,
        dm=dm,
        m_yg=m_yg,
        lambdaa=kappa,
        dispersion=dispersion,
        noise_model=noise_model,
    ).numpy()

    counts = torch.ones_like(phi_sim) * seq_depth
    mp = assemble_mp_simulation(
        data, counts, celltypes, params_g_init, use_data_mean=True
    )

    data_c = torch.tensor(data, dtype=torch.float32, device=device)
    data_c = data_c.unsqueeze(0).expand(n_theta, n_cells, params_g_init.shape[0])
    if n_celltypes == 1:
        return mp, data_c, phi_sim, population
    else:
        return mp, data_c, phi_sim, population, celltypes


# extract info from posteriors
def get_results_model(cmodel, data_c, phi_sim, realign=True):

    lambda_y = cmodel.log_lambda_y.exp().cpu().detach().numpy()
    m_yg = cmodel.m_yg.cpu().detach().numpy()
    # posterior_xc = cmodel.get_phase_posteriors(data_c, method="simpson")
    # post_mean_c, post_var_c, post_std_c = sr.compute_posterior_statistics(
    #     posterior_xc,
    # )

    post_mean_c = cmodel.get_inferred_phases(data_c)
    post_std_c = cmodel.post_std_c
    post_mode_c = cmodel.post_mode_c

    if realign:
        _, _, shift = sr.optimal_shift(post_mean_c, phi_sim, return_shift=True)
        post_mean_c = (post_mean_c - shift) % (2 * np.pi)

    cad = sr.circular_deviation(phi_sim, post_mean_c, period=2 * np.pi)
    cad = cad * rh
    if isinstance(cad, torch.Tensor):
        cad = cad.numpy()
    mad = np.median(cad)
    post_std_c = post_std_c * rh
    std = post_std_c.mean()
    return {
        "cad": cad,
        "mad": mad,
        "post_std_c": post_std_c,
        "std": std,
        "post_mean_c": post_mean_c,
        "post_mode_c": post_mode_c,
        "lambda_y": lambda_y,
        "delta_mean": m_yg,
        "model_params": cmodel.get_parameter_dataframe_context(),
    }


def get_simulation_results(
    n_cells=10000,
    seq_depth=10**3,
    genes_parameters="clock_ccg_params_g.csv",
    device="cuda",
    n_theta=24,  # model param
    realign=True,
    dispersion=0.1,
    population_kappa=None,
    n_pop=None,
    context_lambdas=[],
    context_mus=[],
    context_mode="none",
    n_epochs=10,
):
    if n_pop is not None and n_cells % n_pop != 0:
        n_cells = (n_cells // n_pop) * n_pop
    mp, data_c, phi_sim, population = simulate_data(
        n_cells=n_cells,
        seq_depth=seq_depth,
        genes_parameters=genes_parameters,
        device=device,
        n_theta=n_theta,
        dispersion=dispersion,
        population_kappa=population_kappa,
        n_pop=n_pop,
        context_lambdas=context_lambdas,
        context_mus=np.array(context_mus),
    )
    circadian_counts = data_c[0].sum(axis=1).cpu().numpy()

    cmodel = context_model.ContextModel(mp, data_c, context_mode=context_mode)
    cmodel.to(device)
    with torch.no_grad():
        cmodel.log_disp.copy_(np.log(dispersion))

    res_not_trained = get_results_model(cmodel, data_c, phi_sim, realign=realign)
    if n_epochs == 0:
        res_trained = None
    else:
        losses = trainer.train_ritmo(
            cmodel, data_c, mp, n_epochs=n_epochs, batch_size=1024
        )

        res_trained = get_results_model(cmodel, data_c, phi_sim, realign=realign)

    return {
        "res_not_trained": res_not_trained,
        "res_trained": res_trained,
        "circadian_counts": circadian_counts,
        "phi_sim": phi_sim.cpu().numpy(),
        "seq_depth": seq_depth,
        "population": population,
        "model_params": cmodel.get_parameter_dataframe_context(),
    }


def simulate_data_no_context(
    phases,
    seq_depths,
    fourier_coefficients,
    noise_model,
    context_label="A",
    dispersion=0.1,
):
    assert len(phases) == len(seq_depths)
    n_cells = len(phases)
    n_genes = fourier_coefficients.shape[0]
    # set-up
    context = np.array([context_label] * n_cells)
    dm = design_matrix(context)
    m_yg = np.zeros((1, n_genes))
    kappa = np.ones(1)

    data = generate_nb_data(
        fourier_coefficients,
        phi_c=phases,
        counts=seq_depths,
        dm=dm,
        m_yg=m_yg,
        lambdaa=kappa,
        dispersion=dispersion,
        noise_model=noise_model,
    ).numpy()
    return data


def get_simulation_results_indep_pop(
    n_cells=10000,
    seq_depth=10**3,
    genes_parameters="clock_ccg_params_g.csv",
    device="cuda",
    n_theta=24,  # model param
    realign=True,
    dispersion=0.1,
    population_kappa=1,
    n_pop=4,
    context_lambdas=[],
    context_mus=[],
    n_epochs=100,
    context_mode="none",
):
    mp, data_c, phi_sim, population = simulate_data(
        n_cells=n_cells,
        seq_depth=seq_depth,
        genes_parameters=genes_parameters,
        device=device,
        n_theta=n_theta,
        dispersion=dispersion,
        population_kappa=population_kappa,
        n_pop=n_pop,
        context_lambdas=context_lambdas,
        context_mus=np.array(context_mus),
    )

    result_cells = {
        "phi_sim": [],
        "post_mean_c": [],
        "population": [],
    }
    param_models = {}
    for y in np.unique(mp["context"]):
        i = mp["context"] == y
        mp_tmp = mp.copy()
        mp_tmp["context"] = mp["context"][i]
        mp_tmp["counts"] = mp["counts"][i]
        phi_sim_tmp = phi_sim[i]
        data_c_tmp = data_c[:, i, :]
        cmodel = context_model.ContextModel(
            mp_tmp, data_c_tmp, context_mode=context_mode
        )
        cmodel.to(device)
        with torch.no_grad():
            cmodel.log_disp.copy_(np.log(dispersion))
        cmodel.forward(data_c_tmp)
        print(np.exp(cmodel.log_disp.cpu().detach().numpy()))
        losses = trainer.train_ritmo(
            cmodel, data_c_tmp, n_epochs=n_epochs, batch_size=1024
        )
        print(np.exp(cmodel.log_disp.cpu().detach().numpy()))
        res_trained = get_results_model(
            cmodel, data_c_tmp, phi_sim_tmp, realign=realign
        )
        result_cells["phi_sim"].extend(phi_sim_tmp.cpu().numpy())
        result_cells["post_mean_c"].extend(res_trained["post_mean_c"])
        result_cells["population"].extend(mp_tmp["context"])
        param_models[str(y)] = res_trained["model_params"]
    return result_cells, param_models


def get_simulation_results_seq_depths(
    seq_depths=[3000, 10000, 30000],
    genes_parameters="clock_ccg_params_g.csv",
    population_kappa=None,
    n_pop=None,
    n_epochs=10,
    n_theta=24,
):
    """
    Wrapper to run simulations at different sequencing depths"""
    results = []
    for depth in seq_depths:

        with contextlib.redirect_stdout(io.StringIO()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress all warnings
                res = get_simulation_results(
                    seq_depth=depth,
                    population_kappa=population_kappa,
                    n_pop=n_pop,
                    genes_parameters=genes_parameters,
                    n_epochs=n_epochs,
                    n_theta=n_theta,
                )
        results.append(res)
    return results


def generate_nb_data(
    params_g,
    phi_c,
    counts,
    dm,
    m_yg,
    lambdaa,
    noise_model,
    dispersion=0.1,
    seed=None,
):
    """
    Generate synthetic data based on the negative binomial model with context effects.

    Parameters:s
    -----------
    params_g : pandas.DataFrame
        DataFrame containing gene parameters with columns 'a_g', 'b_g', 'm_g'
    phi_c : array-like
        Phases for each cell (in radians)
    counts : array-like
        Total RNA counts for each cell
    dm : torch.Tensor or numpy.ndarray
        Design matrix for cell categories (one-hot encoded)
    m_yg : torch.Tensor or numpy.ndarray
        Correction to m_g for each cell category and gene
    kappa : torch.Tensor or numpy.ndarray
        Amplitude scaling for each cell category
    dispersion : float
        Dispersion parameter for negative binomial
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    y : torch.Tensor
        Generated count data of shape [num_cells, num_genes]
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Convert inputs to torch tensors if they aren't already
    if not isinstance(dm, torch.Tensor):
        dm = torch.tensor(dm, dtype=torch.float32)
    if not isinstance(m_yg, torch.Tensor):
        m_yg = torch.tensor(m_yg, dtype=torch.float32)
    if not isinstance(lambdaa, torch.Tensor):
        lambdaa = torch.tensor(lambdaa, dtype=torch.float32)
    if not isinstance(counts, torch.Tensor):
        counts = torch.tensor(counts, dtype=torch.float32)
    if not isinstance(phi_c, torch.Tensor):
        phi_c = torch.tensor(phi_c, dtype=torch.float32)

    # Extract parameters from dataframe
    a_g = torch.tensor(params_g["a_1"].values, dtype=torch.float32)
    b_g = torch.tensor(params_g["b_1"].values, dtype=torch.float32)
    m_g = torch.tensor(params_g["a_0"].values, dtype=torch.float32)

    # Reshape as needed
    num_cells = len(phi_c)
    num_genes = len(a_g)

    # Reshape tensors for proper broadcasting
    phi_c = phi_c.reshape(-1, 1)  # [num_cells, 1]
    counts = counts.reshape(-1, 1)  # [num_cells, 1]

    a_g = a_g.reshape(1, -1)  # [1, num_genes]
    b_g = b_g.reshape(1, -1)  # [1, num_genes]
    m_g = m_g.reshape(1, -1)  # [1, num_genes]

    # Calculate category-specific adjustments
    category_intercept = torch.matmul(dm, m_yg)  # [num_cells, num_genes]
    category_lambdaa = torch.matmul(dm, lambdaa).unsqueeze(1)  # [num_cells, 1]

    # Calculate expected values
    E_cg = (
        category_lambdaa * a_g * torch.cos(phi_c)
        + category_lambdaa * b_g * torch.sin(phi_c)
        + m_g
        + category_intercept
    )  # [num_cells, num_genes]

    # Apply exponentiation and multiply by counts
    E_cg = torch.exp(E_cg) * counts  # [num_cells, num_genes]

    if noise_model == "poisson":
        dist = torch.distributions.Poisson(rate=E_cg)
        return dist.sample()

    elif noise_model == "nb":
        # Calculate negative binomial parameters
        disp = torch.tensor(dispersion, dtype=torch.float32)
        r = 1 / disp
        p = disp * E_cg / (1 + disp * E_cg)

        # Create negative binomial distribution and sample
        nb_dist = torch.distributions.NegativeBinomial(r, p)
        return nb_dist.sample()


def design_matrix(vec):
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_matrix = encoder.fit_transform(vec.reshape(-1, 1))
    return torch.tensor(one_hot_matrix, dtype=torch.float32)


def power_noise_generator(n_samples, kappa):
    """
    It creates n_samples angles centered around 0, with
    concentration parameter kappa
    """

    # mean of distribution is zero
    mu = torch.tensor([1, 0], dtype=torch.float32)
    dist = PowerSpherical(mu, kappa)
    samples = dist.rsample((n_samples,))
    angles = torch.arctan2(samples[:, 1], samples[:, 0]) % (2 * np.pi)
    return angles


def generate_phases(n_cells: int, n_timepoints: int, kappa=0.0) -> torch.Tensor:
    """
    Generates phases for different replicates aroud the clock.
    When kappa > 0. it uses power sperical noise around the
    time points values
    """
    if n_cells % n_timepoints != 0:
        raise ValueError(
            f"n_cells={n_cells} must be divisible by n_timepoints={n_timepoints}."
        )

    # 1) Precompute the 4 “equally spaced” angles around the unit circle:
    #    0, π/2, π, 3π/2
    base_angles = torch.linspace(0, 2 * np.pi, n_timepoints + 1, dtype=torch.float32)[
        :-1
    ]
    cells_per_timepoint = n_cells // n_timepoints

    # 2) For each timepoint t, pick angle = base_angles[t % 4]
    #    and repeat it cells_per_timepoint times.
    phases = torch.empty((n_timepoints, cells_per_timepoint), dtype=torch.float32)
    for t in range(n_timepoints):
        angle_idx = t % 4
        phases[t, :] = base_angles[angle_idx]

    phases = phases.flatten()

    if kappa > 0:
        kappa = torch.tensor(kappa)
        eps_phi = power_noise_generator(n_samples=n_cells, kappa=kappa)
        # return the no noise (labels) and denoised phis
        return phases, phases + eps_phi

    # only labels
    return phases


def data_c_to_adata(data_c, sd, context=None, gene_names=None):

    adata = sc.AnnData(data_c[0, :, :].cpu().numpy())

    if gene_names is not None:
        adata.var_names = gene_names

    adata.layers["norm"] = adata.X / sd
    adata.layers["log1p"] = np.log1p(adata.layers["norm"] * 1e4)
    adata.obs["counts"] = sd
    if context is not None:
        adata.obs["context"] = context

    circadian_counts = adata.X.sum(axis=1)
    adata.obs["circadian_counts"] = circadian_counts
    return adata
