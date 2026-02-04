import torch
from .misc.power_spherical.power_spherical import log_power_spherical_unnorm, log_von_mises
import torch
import matplotlib.pyplot as plt
from .utils import nmp

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
        # This is more direct than calculating from the endpoints for a periodic grid.
        h = x_values[1] - x_values[0]

        # 3. Create weights for Periodic Simpson's rule: [2, 4, 2, 4, ...]
        weights = torch.ones(n_points, device=y_values.device, dtype=y_values.dtype)
        weights[1::2] = 4.0  # All odd indices are 4
        weights[0::2] = 2.0  # All even indices are 2

        # Apply weights to y values. The unsqueeze adapts the weights for broadcasting.
        # y_values shape: (n_points, B)
        # weights shape: (n_points,) -> unsqueeze(1) -> (n_points, 1)
        weighted_y = y_values * weights.unsqueeze(1)

        # Calculate the integral by summing weighted values and scaling
        integrals = (h / 3.0) * torch.sum(weighted_y, dim=0)

        return integrals


    def marginalize_theta(
        self, ll_xc_, log_prior, ll_e_xc, method="simpson", return_integrand=False
    ):
        """
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

        # if n_theta is not None:
        #     dm = self.design_matrix(nmp.linspace(0, 2 * torch.pi, n_theta)).to(

        if self.batch_mode:
            dm = self.dm_batch[indices, :]
            kappa_b = torch.exp(self.kappa_b)
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
        """
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
