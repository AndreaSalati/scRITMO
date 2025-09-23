import numpy as np
from scipy.special import iv  # Modified Bessel function of the first kind
from tqdm import tqdm


def J_tilde(E, n_genes=0, n_samples=0):
    """
    Calculates the interaction matrix for the spin glass model.

    Parameters:
        E (numpy.ndarray): Expression matrix with genes on rows and samples on columns.
        n_genes (int): Number of genes (rows). If 0, use all genes.
        n_samples (int): Number of samples (columns). If 0, use all samples.

    Returns:
        numpy.ndarray: Interaction matrix (n_samples x n_samples).
    """

    # Default to using all genes and all samples if values are not provided
    if n_genes == 0:
        n_genes = E.shape[1]
    if n_samples == 0:
        n_samples = E.shape[0]

    # Initialize the interaction matrix with zeros
    Jtilde = np.zeros((n_samples, n_samples))

    # Calculate the interaction matrix, vecotrized
    Jtilde = E @ E.T / (n_samples * n_genes)

    # Set the diagonal elements to zero
    np.fill_diagonal(Jtilde, 0)

    return Jtilde


def phase_initialization_mf(J, beta, n_samples, A_0=0.1, iter_mf=1000):
    """
    Spin glass approximation to initialize the EM model.

    Parameters:
        J (numpy.ndarray): Interaction matrix.
        beta (float): Temperature-like parameter controlling phase interactions.
        n_samples (int): Number of samples (columns of expression matrix).
        A_0 (float): Initial condition for the order parameter A (amplitude).
        iterations (int): Maximum number of iterations.

    Returns:
        numpy.ndarray: Matrix containing amplitudes (A) and phases (Theta).
    """
    A = np.full(n_samples, A_0)
    Theta = np.random.uniform(0, 2 * np.pi, n_samples)

    # use trange
    # for _ in tqdm(range(iterations), desc="Finding an initial guess for phases..."):
    for _ in range(iter_mf):
        A_cos = A * np.cos(Theta)
        A_sin = A * np.sin(Theta)

        u = beta * (J @ A_cos)
        v = beta * (J @ A_sin)
        mod = np.sqrt(u**2 + v**2)

        # Avoid division by zero
        Zeta = np.vstack([u / mod, v / mod]).T
        Zeta[mod == 0] = [1, 0]  # Handle case where mod == 0

        # Update amplitudes and phases
        Theta = np.arctan2(Zeta[:, 1], Zeta[:, 0])
        A = np.where(mod <= 20, iv(1, mod) / iv(0, mod), 1)

    return Theta
