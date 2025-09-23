import numpy as np
from tqdm import tqdm  # Progress bar
from numpy.linalg import inv, det
from scipy.linalg import solve
import pandas as pd
from scipy.special import iv  # Modified Bessel function of the first kind

# This file contains the functions for the EM algorithm
# in the loop over iterations, the functions are called in
# same order as they are defined in this file


def EM_step(phi, T, E, sigma2, Nc, Ng, i, TSM, dTinv, q, W, Q_hist):
    phi_old, alpha, M, M_inv, XTX, X, X_old = update_matrices(phi, T, E, sigma2, Nc)

    # If two-state model is off, reset weights to initial values
    if not TSM:
        W = np.ones(Ng)

    Tot = np.sum(W)  # Total sum of weights

    K, Om, A, B, C, D = solve_lagrange(alpha, M_inv, E, sigma2, W, Tot, i)
    # Ensure no NaN values in Om
    if np.any(np.isnan(Om)):
        return {"alpha": alpha, "weights": W, "iteration": i}

    # Apply find_roots function for each column of Om
    rooted = np.apply_along_axis(find_roots, 0, Om, A, B, C, D)

    # Test roots to find the minimum (corresponds to the minimum of the Q function)
    # Vectorized version of the function
    possible_solutions = evaluate_possible_solutions(
        rooted, K, Om, alpha, E, W, sigma2, M_inv, Tot, phi_old, Ng
    )
    # Extract Q values (the third column)
    Q_values = possible_solutions[:, :, 2]  # Shape: [Nroots, Nc]

    # Find the indices of the minimum Q value for each observation
    min_indices = np.argmin(Q_values, axis=0)  # Shape: [Nc]

    # Extract the minimum Q values
    min_Q_values = Q_values[min_indices, np.arange(Q_values.shape[1])]  # Shape: [Nc]

    # Check for invalid solutions
    if np.any(min_Q_values == 1e5):
        raise ValueError("No solution found on the circle for some observations")

    # Extract the corresponding solutions # Shape: [Nc, 4]
    ze = possible_solutions[min_indices, np.arange(Q_values.shape[1]), :]
    # Update phi using the best root solutions
    ze = np.array(ze).T
    phi = np.arctan2(ze[1, :], ze[0, :]) % (2 * np.pi)

    Q_hist = update_Q_hist(Q_hist, ze, i, Nc)
    W = update_weights(E, X, M_inv, sigma2, dTinv, M, q)
    # T is fixed
    return phi, phi_old, Q_hist, W, alpha, XTX, X, X_old


def EM_initialization(E, sigma2=None, u=None, tau2=None):
    """
    Initializes parameters for a probabilistic model.

    Parameters:
    - E: The expression data matrix (2D NumPy array).
    - sigma2: Initial value for sigma squared. If None, calculated from E.
    - u: Initial value for u. If None, set to 0.2.
    - tau2: Initial value for tau squared. If None, calculated based on E.
    - pbar: Boolean indicating whether to show a progress bar.
    - iterations: Number of iterations for the progress bar.

    Returns:
    - sigma2: Initialized sigma squared.
    - T: Diagonal matrix of variances.
    - S: Precomputed outer products of columns of E.
    - W: Array of ones with length equal to the number of genes (Ng).
    """

    # Initialize parameters for probabilistic model
    if sigma2 is None:
        sigma2 = np.mean(np.var(E, axis=0))
    if u is None:
        u = 0.2
    if tau2 is None:
        tau2 = 4 / (24 + E.shape[0])

    T = np.diag([u**2, tau2, tau2])

    # Precompute some variables used in the EM loop
    Ng = E.shape[1]
    S = np.einsum("il, jl -> lij", E, E)
    W = np.ones(Ng)

    return sigma2, u, tau2, T, S, W


def update_matrices(phi, T, E, sigma2, N):
    """
    Update matrices for the EM algorithm.
    """
    # Trigonometric components based on the current phi values
    phi_old = phi.copy()
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Design matrix X with columns: [1, cos(phi), sin(phi)]
    X = np.column_stack((np.ones(N), cos_phi, sin_phi))
    X_old = X.copy()

    # Matrix multiplication and inversion for parameter updates
    XTX = X.T @ X  # This is t(X) %*% X in R

    T_inv = inv(T)  # Inverse of the T matrix (covariance of gene's coefficients)
    M = XTX + sigma2 * T_inv
    M_inv = inv(M)  # Inverse of M

    # Update the alpha parameters
    alpha = M_inv @ X.T @ E
    alpha = alpha.T  # Transpose back to match R's convention

    return phi_old, alpha, M, M_inv, XTX, X, X_old


def solve_lagrange(alpha, M_inv, E, sigma2, W, Tot, i):
    """
    First step to get the 4 zeros of teh derivative of the Q function.
    Find some matrix elements and construct the matrix K.
    """

    # Calculate intermediate variables for matrix K
    A = np.sum(W * (alpha[:, 1] ** 2 / sigma2 + M_inv[1, 1])) / Tot
    B = np.sum(W * (alpha[:, 1] * alpha[:, 2] / sigma2 + M_inv[1, 2])) / Tot
    C = np.sum(W * (alpha[:, 2] * alpha[:, 1] / sigma2 + M_inv[2, 1])) / Tot
    D = np.sum(W * (alpha[:, 2] ** 2 / sigma2 + M_inv[2, 2])) / Tot

    # Construct matrix K
    K = np.array([[A, B], [C, D]])

    # Return early if any elements of K are NaN
    if np.any(np.isnan(K)):
        return {"alpha": alpha, "weights": W, "iteration": i}

    # Calculate al and be, which depend on the current weight values and alpha
    al = np.apply_along_axis(
        lambda x: np.sum(W * (alpha[:, 1] * (x - alpha[:, 0]) / sigma2 - M_inv[0, 1]))
        / Tot,
        1,
        E,
    )
    be = np.apply_along_axis(
        lambda x: np.sum(W * (alpha[:, 2] * (x - alpha[:, 0]) / sigma2 - M_inv[0, 2]))
        / Tot,
        1,
        E,
    )

    # Create matrix O
    O = np.vstack([al, be])

    return K, O, A, B, C, D


def find_roots(x, A, B, C, D):
    """
    Computes the roots of a polynomial based on the input parameters.

    Parameters:
    - A, B, C, D: Coefficients of the polynomial.
    - x: A list or array containing the values for x[0] and x[1].

    Returns:
    - roots: Roots of the polynomial.
    """
    zero = (
        B**2 * C**2
        + A**2 * D**2
        - x[0] ** 2 * (D**2 + C**2)
        - x[1] ** 2 * (A**2 + B**2)
        + 2 * x[0] * x[1] * (A * B + C * D)
        - 2 * A * B * C * D
    )
    one = 2 * (
        (A + D) * (A * D - B * C)
        - x[0] ** 2 * D
        - x[1] ** 2 * A
        + x[0] * x[1] * (B + C)
    )
    two = A**2 + D**2 + 4 * A * D - x[0] ** 2 - x[1] ** 2 - 2 * B * C
    three = 2 * (A + D)
    four = 1

    # Roots of the polynomial (use numpy roots for equivalent of polyroot)
    return np.roots([four, three, two, one, zero])


def evaluate_possible_solutions(
    rooted, K, O, alpha, E, W, sigma2, M_inv, Tot, phi_old, Ng
):
    """
    Fully vectorized version of the function to evaluate possible solutions for the roots of the polynomial.
    Processes all Nc observations at once.
    """
    Nroots, Nc = rooted.shape

    # Extract real roots and create a mask for valid roots
    y = rooted
    mask = np.abs(np.imag(y)) <= 1e-8  # Shape: [Nroots, Nc]

    # Initialize possible_solutions with default values
    possible_solutions = np.zeros((Nroots, Nc, 4))
    possible_solutions[:, :, :2] = 0
    possible_solutions[:, :, 2:] = 100000  # Default Q values

    # Process valid roots
    valid_indices = np.where(mask)
    if valid_indices[0].size == 0:
        # No valid roots found
        return possible_solutions

    # Extract valid real parts of roots
    real_y_valid = y.real[valid_indices]  # Shape: [N_valid]

    # Compute K_lambda for valid roots
    K_lambda = K + np.einsum(
        "n,ij->nij", real_y_valid, np.eye(2)
    )  # Shape: [N_valid, 2, 2]

    # Extract corresponding O_j for valid indices
    O_j = O[:, valid_indices[1]].T  # Shape: [N_valid, 2]

    # Solve K_lambda * zet = O_j
    zet = np.linalg.solve(K_lambda, O_j[..., None])  # Shape: [N_valid, 2]
    zet = zet.squeeze()  # Remove the last dimension

    # Compute phit from zet
    phit = np.arctan2(zet[:, 1], zet[:, 0]) % (2 * np.pi)  # Shape: [N_valid]

    # Compute Xt and Mt for valid roots
    Xt = np.column_stack(
        (np.ones_like(phit), np.cos(phit), np.sin(phit))
    )  # Shape: [N_valid, 3]
    Mt = Xt[:, :, None] * Xt[:, None, :]  # Shape: [N_valid, 3, 3]

    # Compute Xt_old and Mt_old based on phi_old
    phi_old_j = phi_old[valid_indices[1]]  # Shape: [N_valid]
    Xt_old = np.column_stack(
        (np.ones_like(phi_old_j), np.cos(phi_old_j), np.sin(phi_old_j))
    )  # Shape: [N_valid, 3]
    Mt_old = Xt_old[:, :, None] * Xt_old[:, None, :]  # Shape: [N_valid, 3, 3]

    # Prepare alpha and E_j for vectorized computations
    alpha = np.array(alpha)  # Shape: [Ng, 3]
    E_j = E[valid_indices[1], :]  # Shape: [N_valid, Ng]

    # Compute Qs for the new values
    Qs_term1 = np.einsum("ki,nij,kj->nk", alpha, Mt, alpha)  # Shape: [N_valid, Ng]
    temp = np.dot(Xt, alpha.T)  # Shape: [N_valid, Ng]
    Term2 = -2 * temp * E_j  # Shape: [N_valid, Ng]
    Term3 = sigma2 * np.einsum("ij,nji->n", M_inv, Mt)  # Shape: [N_valid]
    Qs = Qs_term1 + Term2 + Term3[:, None]  # Shape: [N_valid, Ng]

    # Compute Qs for the old values
    Qs_term1_old = np.einsum(
        "ki,nij,kj->nk", alpha, Mt_old, alpha
    )  # Shape: [N_valid, Ng]
    temp_old = np.dot(Xt_old, alpha.T)  # Shape: [N_valid, Ng]
    Term2_old = -2 * temp_old * E_j  # Shape: [N_valid, Ng]
    Term3_old = sigma2 * np.einsum("ij,nji->n", M_inv, Mt_old)  # Shape: [N_valid]
    Qs_old = Qs_term1_old + Term2_old + Term3_old[:, None]  # Shape: [N_valid, Ng]

    # Compute the Q function for new and old values
    W_sigma2 = W / sigma2  # Shape: [Ng]
    Q = np.sum(Qs * W_sigma2[None, :], axis=1) / Tot  # Shape: [N_valid]
    Q_old = np.sum(Qs_old * W_sigma2[None, :], axis=1) / Tot  # Shape: [N_valid]

    # Store the solutions in possible_solutions
    possible_solutions[valid_indices[0], valid_indices[1], 0:2] = (
        zet  # zet: [N_valid, 2]
    )
    possible_solutions[valid_indices[0], valid_indices[1], 2] = Q  # Q: [N_valid]
    possible_solutions[valid_indices[0], valid_indices[1], 3] = (
        Q_old  # Q_old: [N_valid]
    )

    return possible_solutions


def update_Q_hist(Q_hist, ze, i, Nc):
    """
    Updates the Q_hist DataFrame with the current iteration results.
    """
    # add ze.T  to values of first 4 columns
    # Create a new DataFrame for the current block of 30 rows
    Q_temp = pd.DataFrame(ze.T, columns=["cos", "sin", "Q", "Q_old"])
    # Set 'iteration' and 'sample' columns for the new block
    Q_temp["iteration"] = i + 1  # Assign iteration number
    Q_temp["sample"] = np.arange(Nc)  # Assign sample number
    Q_hist = pd.concat([Q_hist, Q_temp])  # Append the new iteration results
    return Q_hist


def update_weights(E, X, M_inv, sigma2, dTinv, M, q):
    Nc, Ng = E.shape

    # Compute the quadratic terms for all p at once
    E_M = E.T @ X @ M_inv @ X.T @ E  # Shape: (Ng, Ng)

    # Diagonal elements represent E[:, p] @ ... @ E[:, p].T for all p
    quad_terms = np.diag(E_M) / (2 * sigma2)  # Shape: (Ng,)

    # Compute P1_0 for all p
    P1_0 = np.exp(quad_terms) * np.sqrt(dTinv * sigma2**3 / np.linalg.det(M))

    # Compute W for all p
    W = q * P1_0 / (1 - q + q * P1_0)

    # Handle NaNs
    W[np.isnan(W)] = 1

    return W


def update_EM_parameters(phi, Nn, X, X_old, S, E, T, sigma2, W, q, update_q):
    """
    Updates the parameters for the next EM step.
    """
    Nc, Ng = E.shape
    # Update parameters for the next EM step
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    X = np.column_stack(
        (np.ones(len(phi)), cos_phi, sin_phi)
    )  # Equivalent of cbind(1, cos(phi), sin(phi))

    M_old = Nn + sigma2 * inv(T)
    M_old_inv = inv(M_old)  # Inverse of M_old

    # Update sigma2.m1
    sigma2_m1 = np.trace(S - S @ X_old @ M_old_inv @ X.T, axis1=1, axis2=2) / Nc + 0.01

    sigma2_m0 = np.var(E, axis=0)
    sigma2 = np.mean(sigma2_m1 * W + sigma2_m0 * (1 - W))
    sigma2_m1 = np.sum(sigma2_m1 * W) / np.sum(W)

    # Update q if needed
    if update_q:
        q = np.mean(W)
        q = max(0.05, min(q, 0.3))

    return cos_phi, sin_phi, X, M_old, M_old_inv, sigma2_m1, sigma2_m0, sigma2, q
