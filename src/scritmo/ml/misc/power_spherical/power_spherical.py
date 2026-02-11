import numpy as np
import torch
import scipy
from scipy.special import i0, i1


@torch.jit.script
def log_von_mises_jit(x: torch.Tensor, mu: torch.Tensor, kappa: float) -> torch.Tensor:
    """
    JIT-compiled log Von Mises distribution.
    
    Args:
        x: Tensor of angles (any shape)
        mu: Tensor of mean angles (broadcastable to x's shape)
        kappa: scalar concentration parameter
    
    Returns:
        Tensor of log-densities
    """
    # Compute log-normalization constant: log(2*pi) + log(I_0(kappa))
    log_2pi = 1.8378770664093453  # log(2*pi) precomputed
    kappa_t = torch.tensor(kappa, dtype=x.dtype, device=x.device)
    log_I0 = torch.log(torch.i0(kappa_t))
    logN = log_2pi + log_I0
    
    # Compute log-unnormalized term: kappa * cos(x - mu)
    angle_diff = x - mu
    log_unnorm = kappa_t * torch.cos(angle_diff)
    
    return log_unnorm - logN


def powershperical2beta(k, d):
    alpha = (d - 1) / 2 + k
    beta = (d - 1) / 2
    return alpha, beta


def power_spherical(x: torch.Tensor, mu: torch.Tensor, k: float) -> torch.Tensor:
    """
    Torch‐version of the 2D power‐spherical distribution
      x: any shape tensor of angles (radians)
      mu: same shape as x (or broadcastable) of mean angles
      k: scalar concentration parameter

    Returns:
      f(x) = (1 + cos(x - mu))^k  /  N(k, d=2)
    """

    # get scalar Beta params
    alpha, beta = powershperical2beta(k, d=2)

    # cast to tensor on the same device/dtype as x
    alpha_t = torch.as_tensor(alpha, dtype=x.dtype, device=x.device)
    beta_t = torch.as_tensor(beta, dtype=x.dtype, device=x.device)

    # normalization constant:
    #   N = 2^(α+β) * π^β * Γ(α) / Γ(α+β)
    N = (
        2.0 ** (alpha + beta)
        * (torch.pi**beta_t)
        * torch.exp(torch.lgamma(alpha_t))
        / torch.exp(torch.lgamma(alpha_t + beta_t))
    )

    # un‐normalized density
    num = (1.0 + torch.cos(x - mu)) ** k

    return num / N


def log_power_spherical(x: torch.Tensor, mu: torch.Tensor, k: float) -> torch.Tensor:
    """
    Compute log f(x) for the 2D power‐spherical distribution
      f(x) ∝ (1 + cos(x - mu))^k

    Log‐density:
      log f(x) = k * log(1 + cos(x - mu))  -  log N(k)

    where
      log N(k) = (α+β)·ln 2  +  β·ln π  +  ln Γ(α)  –  ln Γ(α+β)
      α = k + 1,   β = d/2 = 1

    Args:
      x   : Tensor of angles (any shape)
      mu  : Tensor of mean angles (broadcastable to x’s shape)
      k   : scalar concentration parameter

    Returns:
      Tensor of same shape as x with log‐densities.
    """
    alpha, beta = powershperical2beta(k, d=2)

    # promote to torch tensors on the correct device/dtype
    device, dtype = x.device, x.dtype
    alpha_t = torch.as_tensor(alpha, dtype=dtype, device=device)
    beta_t = torch.as_tensor(beta, dtype=dtype, device=device)

    # compute log‐normalization constant log N
    #   (α+β)*ln2 + β*lnπ + lgamma(α) – lgamma(α+β)
    logN = (
        (alpha_t + beta_t) * torch.log(torch.tensor(2.0, device=device, dtype=dtype))
        + beta_t * torch.log(torch.tensor(torch.pi, device=device, dtype=dtype))
        + torch.lgamma(alpha_t)
        - torch.lgamma(alpha_t + beta_t)
    )

    # compute the log‐unnormalized term: k * ln(1 + cos(x - mu))
    # use log1p for stability when cos is near zero
    angle_diff = x - mu
    log_unnorm = k * torch.log1p(torch.cos(angle_diff))

    # return log‐density
    return log_unnorm - logN


def log_power_spherical_unnorm(
    x: torch.Tensor, mu: torch.Tensor, k: float
) -> torch.Tensor:
    """
    Un-normalized log pdf(x)

    Args:
      x   : Tensor of angles (any shape)
      mu  : Tensor of mean angles (broadcastable to x’s shape)
      k   : scalar concentration parameter

    Returns:
      Tensor of same shape as x with log‐densities.
    """
    # compute the log‐unnormalized term: k * ln(1 + cos(x - mu))
    # use log1p for stability when cos is near zero
    angle_diff = x - mu
    log_unnorm = k * torch.log1p(torch.cos(angle_diff) + 1e-6)

    # return log‐density
    return log_unnorm


def ps_cSTD(kappa):
    R = kappa / (kappa + 1)
    cstd = np.sqrt(-2 * np.log(R))
    return cstd


def log_von_mises(x: torch.Tensor, mu: torch.Tensor, kappa: float) -> torch.Tensor:
    """
    Compute log f(x) for the Von Mises distribution
      f(x) = exp(kappa * cos(x - mu)) / (2 * pi * I_0(kappa))

    Log-density:
      log f(x) = kappa * cos(x - mu) - log(2 * pi * I_0(kappa))
               = kappa * cos(x - mu) - (log(2 * pi) + log(I_0(kappa)))

    where
      I_0 is the modified Bessel function of the first kind, order 0.

    Args:
      x     : Tensor of angles (any shape)
      mu    : Tensor of mean angles (broadcastable to x’s shape)
      kappa : scalar concentration parameter (kappa >= 0)

    Returns:
      Tensor of same shape as x with log-densities.
    """
    device, dtype = x.device, x.dtype

    # promote kappa to a tensor on the correct device/dtype
    kappa_t = torch.as_tensor(kappa, dtype=dtype, device=device)

    # compute log-normalization constant log N
    # log N = log(2 * pi) + log(I_0(kappa))
    log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=device, dtype=dtype))

    # Use torch.i0 and take its log
    log_I0 = torch.log(torch.i0(kappa_t))
    logN = log_2pi + log_I0

    # compute the log-unnormalized term: kappa * cos(x - mu)
    angle_diff = x - mu
    log_unnorm = kappa_t * torch.cos(angle_diff)

    # return log-density
    return log_unnorm - logN


def vm_cSTD(kappa):
    R = i1(kappa) / i0(kappa)
    cstd = np.sqrt(-2 * np.log(R))
    return cstd
