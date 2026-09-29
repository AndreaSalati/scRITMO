"""
Backward-compatible home of the marginalization helpers.

The maths now lives in :mod:`scritmo.ml.model.likelihood` and the methods on
``Scritmo`` itself; this module keeps the old names importable.
"""

import torch

from .model.likelihood import vectorized_simpson  # noqa: F401  (old import path)
from .misc.power_spherical.power_spherical import log_von_mises  # noqa: F401


class MarginalizationMixin:
    """
    DEPRECATED. The marginalization methods are defined on ``Scritmo`` directly;
    this class keeps them available to old code that mixed it in.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        from .context_model import Scritmo

        for name in (
            "vectorized_simpson",
            "marginalize_theta",
            "_phase_width",
            "cell_prior",
            "log_like_loss",
        ):
            if name not in cls.__dict__:
                setattr(cls, name, Scritmo.__dict__[name])

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
