import numpy as np
from tqdm import trange  # Progress bar
from numpy.linalg import det
import pandas as pd

from .helper_fn import (
    ccg,
    process_expression_data,
)

from .stat_phys import (
    J_tilde,
    phase_initialization_mf,
)

from .em import (
    EM_initialization,
    EM_step,
    update_EM_parameters,
)


def CHIRAL(
    E,
    genes,
    layer=None,
    iter_em=500,
    iter_mf=1000,
    add_noise_mf_phase=False,
    tau2=None,
    u=None,
    sigma2=None,
    TSM=True,
    standardize=True,
    q=0.1,
    update_q=False,
    phi_start=None,
):
    """
    Infer sample's phases per periodic gene expression

    Parameters:
        - E (numpy.ndarray): Matrix of gene expression. Samples should be on columns, genes on rows
        - genes (list): Set of clock genes (subset of .var_names), default is None, which uses core clock genes.
        - layer (string): Layer of the data to use, default is None, which useses adata.X
        - iter_mf (int): Number of iterations for the mean field approximation, default is 1000.
        - iter_em (int): Number of maximum iterations. Default is 500.
        - add_noise_mf_phase (bool): Whether to add noise to the initial phase from mf, default is False.
        - tau2 (float): Tau parameter for the prior on gene coefficient, default is None.
        - u (float): u parameter for the prior on gene means, default is None.
        - sigma2 (float): Standard deviation of data points for prediction, default is None.
        - TSM (bool): Switches two-state model in EM, default is True.
        - mean_centre_E (bool): Whether to center data around the empirical mean, default is True.
        - q (float): Probability weight for EM procedure.
        - update_q (bool): Whether to update q during EM, default is False.
        - phi_start (numpy.ndarray): Initial guess for phases, default is None.
        - standardize (bool): Whether to standardize the matrix for inference, default is False.
        - q (float): Probability weight for EM procedure, default is 0.1.
        - update_q (bool): Whether to update q during EM, default is False.
        - phi_start (numpy.ndarray): Initial guess for phases, default is None.

    Returns:
        dict: Inferred phases, sigma, params_g, weights, iteration number, and other metrics.
    """

    E, E_full, Nc, Ng = process_expression_data(E, genes, ccg, layer, standardize)
    phi = phi_start

    # Initializing phase using spin glass model (or any other method)
    if phi_start is None:
        # Use the spin glass initialization (for now, we skip implementation of J.tilde and Zeta.mf.ordered)
        beta = 1000
        J = J_tilde(E)
        Theta = phase_initialization_mf(J, beta, E.shape[0], iter_mf=iter_mf)
        phi = Theta
        if add_noise_mf_phase:
            phi += np.random.uniform(-0.5, 0.5, size=Nc)
            phi % (2 * np.pi)

    sigma2, u, tau2, T, S, W = EM_initialization(E, sigma2, u, tau2)
    dTinv = 1 / det(T)
    Q_hist = pd.DataFrame(columns=["cos", "sin", "Q", "Q_old", "iteration", "sample"])

    # Start EM iterations
    for i in trange(iter_em, desc="Progress", bar_format="{percentage:3.0f}%"):
        phi, phi_old, Q_hist, W, alpha, Nn, X, X_old = EM_step(
            phi, T, E, sigma2, Nc, Ng, i, TSM, dTinv, q, W, Q_hist
        )

        # Exit the loop if convergence is reached
        if np.max(np.abs(phi - phi_old)) < 0.001:
            par_df = pd.DataFrame(alpha, index=genes, columns=["a_0", "a_1", "b_1"])
            print("Algorithm has converged")
            return {
                "phi": phi,
                "sigma": sigma2,
                "params_g": par_df,
                "weights": W,
                "iteration": i + 1,
                "E": E,
                "Q_hist": Q_hist,
                "genes": genes,
            }

        cos_phi, sin_phi, X, Mold, Moldinv, sigma2_m1, sigma2_m0, sigma2, q = (
            update_EM_parameters(phi, Nn, X, X_old, S, E, T, sigma2, W, q, update_q)
        )

    print("EM algorithm finished after maximum iterations")

    par_df = pd.DataFrame(alpha, index=genes, columns=["a_0", "a_1", "b_1"])

    return {
        "phi": phi,
        "Q_hist": Q_hist,
        "sigma": sigma2,
        "params_g": par_df,
        "weights": W,
        "iteration": iter_em,
        "sigma.m1": sigma2_m1,
        "genes": genes,
        "E": E,
    }
