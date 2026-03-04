import torch
from .misc.power_spherical.power_spherical import log_power_spherical_unnorm, log_von_mises, log_von_mises_jit
import torch
import matplotlib.pyplot as plt
from .utils import nmp


@torch.jit.script
def vectorized_simpson_jit(y_values: torch.Tensor, h: float) -> torch.Tensor:
    """
    JIT-compiled vectorized Periodic Simpson's rule.
    
    Args:
        y_values: Tensor of function values, shape (N, B)
        h: Step size (uniform grid)
    
    Returns:
        Integrals, shape (B,)
    """
    n_points = y_values.shape[0]
    
    if n_points < 2:
        return torch.zeros(y_values.shape[1], dtype=y_values.dtype, device=y_values.device)
    
    # Create weights: [2, 4, 2, 4, ...] for periodic Simpson's rule
    weights = torch.ones(n_points, dtype=y_values.dtype, device=y_values.device)
    weights[1::2] = 4.0  # Odd indices
    weights[0::2] = 2.0  # Even indices
    
    # Apply weights and sum
    weighted_y = y_values * weights.unsqueeze(1)
    integrals = (h / 3.0) * torch.sum(weighted_y, dim=0)
    
    return integrals


@torch.jit.script
def marginalize_theta_core(
    ll_xc: torch.Tensor,
    log_prior: torch.Tensor,
    ll_e_xc: torch.Tensor,
    phi_x: torch.Tensor,
    method: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    JIT-compiled core marginalization computation.
    
    Returns:
        l_c: Marginalized likelihood
        max_c: Max values for log-sum-exp
        l_xc: Exponentiated likelihoods
    """
    # Add prior and extra term
    ll_xc_combined = ll_xc + log_prior + ll_e_xc
    
    # Log-sum-exp trick
    max_c = torch.max(ll_xc_combined, dim=0, keepdim=True).values
    ll_xc_stable = ll_xc_combined - max_c
    l_xc = torch.exp(ll_xc_stable)
    
    # Integration
    if method == "simpson":
        h = 2.0 * 3.141592653589793 / float(ll_xc.shape[0])  # 2*pi/Nx
        l_c = vectorized_simpson_jit(l_xc, h)
    else:  # sum
        l_c = torch.sum(l_xc, dim=0) * (2.0 * 3.141592653589793 / float(ll_xc.shape[0]))
    
    return l_c, max_c, l_xc


@torch.jit.script
def normalize_log_dist_jit(ll_xc: torch.Tensor, method: str, Nx: int) -> torch.Tensor:
    """
    JIT-compiled log distribution normalization.
    
    Args:
        ll_xc: Log likelihood tensor, shape (Nx, ...)
        method: "simpson" or "sum"
        Nx: Number of grid points
    
    Returns:
        Normalized distribution
    """
    max_c = torch.max(ll_xc, dim=0, keepdim=True).values
    ll_xc_stable = ll_xc - max_c
    l_xc = torch.exp(ll_xc_stable)
    
    if method == "simpson":
        h = 2.0 * 3.141592653589793 / float(Nx)
        l_c = vectorized_simpson_jit(l_xc, h)
    else:  # sum
        l_c = torch.sum(l_xc, dim=0) * (2.0 * 3.141592653589793 / float(Nx))
    
    return l_xc / l_c

class MarginalizationMixin:
    @staticmethod
    def vectorized_simpson(y_values, x_values):
        """
        Vectorized implementation of the Periodic Simpson's rule for PyTorch tensors.

        Integrates along dimension 0 for functions on a periodic domain (e.g., a circle).
        It assumes the input grid `x_values` covers one full period without
        repeating the endpoint.

        Args:
            y_values (torch.Tensor): Tensor of function values, shape (N, B).
                                    N is the number of points, B is the batch size.
            x_values (torch.Tensor): Tensor of sample points, shape (N,).

        Returns:
            torch.Tensor: A tensor of integrals, shape (B,).
        """
        n_points = y_values.shape[0]

        # 1. Enforce even number of points
        if n_points % 2 != 0:
            raise ValueError(
                "Periodic Simpson's rule requires an even number of sample points."
            )

        if n_points < 2:
            # For a batch, returning zeros might be preferable to raising an error
            return torch.zeros(
                y_values.shape[1], device=y_values.device, dtype=y_values.dtype
            )

        # 2. Calculate step size (assuming uniform grid)
        h = x_values[1] - x_values[0]

        # Use JIT-compiled version for the core computation
        return vectorized_simpson_jit(y_values, float(h))


    def marginalize_theta(
        self, ll_xc_, log_prior, ll_e_xc, method="simpson", return_integrand=False
    ):
        r"""
        This function gives the log pf the marginal distribution P(D, beta)
        by integrating over the theta, given a prior P(theta)
        \int P(D, beta, theta) P(theta) d theta = P(D, beta)
        The integration is done using Simpson's rule.
        Returns the log of the marginal distribution and the m
        """

        # log (L(D|theta, beta) * P(theta))
        # ADD HERE ll_e_xc already summed over genes
        ll_xc = ll_xc_ + log_prior + ll_e_xc
        # simpson integration + logsumexp trick
        max_c = torch.max(ll_xc, dim=0, keepdim=True).values
        ll_xc = ll_xc - max_c
        l_xc = torch.exp(ll_xc)

        if method == "simpson":
            l_c = self.vectorized_simpson(l_xc, self.phi_x)
        elif method == "sum":
            l_c = torch.sum(l_xc, dim=0) * (2 * torch.pi / self.Nx)

        if return_integrand:
            return l_c, max_c, l_xc
        else:
            return l_c, max_c


    def cell_prior(self, indices=None, n_theta=None):
        """
        This method calculates the batch prior that will be used by
        marginalize_theta.
        """
        if indices is None:
            indices = slice(None)

        if self.batch_mode:
            dm = self.dm_batch[indices, :]
            kappa_b = torch.exp(self.kappa_b)
            # Use JIT-compiled version when kappa_b is a scalar float
            if isinstance(kappa_b, torch.Tensor) and kappa_b.numel() == 1:
                kappa_val = float(kappa_b.item())
                prior_xb = log_von_mises_jit(self.phi_x.unsqueeze(1), self.phi_b, kappa_val)
            else:
                prior_xb = log_von_mises(self.phi_x.unsqueeze(1), self.phi_b, kappa_b)
            # than expand by mutiplying by the dm prior_xb
            prior_xc = prior_xb @ dm.T

        else:
            prior_xc = torch.log(torch.tensor(1 / (2 * torch.pi), dtype=torch.float32))

        return prior_xc

    @staticmethod
    def log_like_loss(l_c, max_c):
        # log and add back the max
        ll_c = torch.log(l_c) + max_c
        return ll_c.sum()


    def marginalize_theta_svi(self, ll_xcg, method="simpson", return_integrand=False):
        r"""
        This function gives the log pf the marginal distribution P(D, beta)
        by integrating over the theta, given a  flat prior P(theta) = 1 / (2 * pi)
        \int P(D, beta, theta) P(theta) d theta = P(D, beta)
        The integration is done using Simpson's rule.
        Returns the log of the marginal distribution and the m
        """

        # Sum over genes dimension
        ll_xc = ll_xcg.sum(dim=2)
        N_grid = ll_xc.shape[0]
        phi_range = torch.linspace(0, 2 * torch.pi, N_grid)

        # log prior is log(1 / 2*pi)
        log_prior = torch.log(torch.tensor(1 / (2 * torch.pi), dtype=torch.float32))
        # log (L(D|theta, beta) * P(theta))
        ll_xc = ll_xc + log_prior
        # simpson integration + logsumexp trick
        max_c = torch.max(ll_xc, dim=0, keepdim=True).values
        ll_xc = ll_xc - max_c
        l_xc = torch.exp(ll_xc)

        if method == "simpson":
            l_c = self.vectorized_simpson(l_xc, phi_range)
        elif method == "sum":
            l_c = torch.sum(l_xc, dim=0) * (2 * torch.pi / N_grid)

        if return_integrand:
            return l_c, max_c, l_xc
        else:
            return l_c, max_c
