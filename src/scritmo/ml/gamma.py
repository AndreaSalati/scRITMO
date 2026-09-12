"""
GammaAmplitudeMixin: per-cell amplitude scaling, marginalized like the phase.

Motivation
----------
The standard scRITMO model writes each gene as

    log(X_cg) = m_g + A_g * cos(theta_c - phi_g)

so every cell shares the same rhythm amplitude ``A_g``. Experimentally this is
too rigid: after a phase shift (jet lag) the population rhythm is not only
shifted, its amplitude collapses during the transient (Yamaguchi et al. 2013).
A model with a single fixed amplitude can only absorb that by broadening the
phase posterior, which is a poor description of what happened.

This mixin adds one scalar per cell (or per sample, for bulk),

    log(X_cg) = m_g + gamma_c * A_g * cos(theta_c - phi_g)

``gamma_c`` rescales *all* genes of that cell by the same factor: gamma = 1 is
the reference population amplitude, gamma = 0 is an arrhythmic cell.

Implementation
--------------
``gamma_c`` is a latent variable treated exactly like ``theta_c``: it is never a
point parameter, it is integrated out on a grid. The trick that keeps the code
(and the memory layout) unchanged is that the mean is *linear* in the harmonic
design row, so a point ``(theta_i, gamma_j)`` of the 2-D grid is just the
harmonic row at ``theta_i`` scaled by ``gamma_j``:

    X[(i, j), :] = gamma_j * [cos(theta_i), ..., sin(theta_i), ...]

The 2-D grid is therefore *flattened* into the existing leading "phase-grid"
axis, with size ``Nx = n_theta * n_gamma`` and layout ``x = i * n_gamma + j``
(so ``reshape(n_theta, n_gamma)`` recovers the two axes). Every downstream step
(``log_prob``, sum over genes, log-sum-exp marginalization, posterior
normalization) is untouched; only the integration measure, the prior and the
final posterior reshaping know about gamma.

Two notes on the parameterization:

* The grid is restricted to ``gamma >= 0``. Negative gamma is not a new model,
  it is ``theta_c + pi``, so it would only duplicate the posterior.
* The global scale is degenerate: ``(gamma_c, A_g) -> (c * gamma_c, A_g / c)``
  leaves the likelihood unchanged. The grid bounds alone already break it
  weakly; a prior on gamma (``gamma_prior``) pins it properly. Only the
  *relative* gamma between cells is identified without a prior.
"""

import numpy as np
import torch

from .utils import nmp


def _normal_log_prob(x, mu, sigma):
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)


class GammaAmplitudeMixin:
    """
    Per-cell amplitude scaling ``gamma_c``, marginalized on a grid.

    Disabled by default: with ``n_gamma=None`` every method here is a no-op and
    the model is bit-for-bit the original one (``gamma == 1``).
    """

    # ------------------------------------------------------------------ setup

    def _setup_gamma(self, n_gamma=None, gamma_range=(0.0, 2.0), gamma_prior=None):
        """
        Configure the gamma axis. Called once from ``__init__``, before
        :meth:`X_matrix`.

        Args:
            n_gamma: Number of grid points on the gamma axis. ``None`` (default)
                or 1 disables the feature entirely.
            gamma_range: ``(lo, hi)`` bounds of the gamma grid, inclusive. Must
                satisfy ``0 <= lo < hi``.
            gamma_prior: Prior on gamma. ``None``/"uniform" is flat on the grid;
                a float is a Gaussian of that standard deviation centered on 1;
                a ``(mu, sigma)`` tuple or ``{"mu":, "sigma":}`` dict is an
                explicit Gaussian; a callable gets the gamma grid tensor and
                must return log-densities of the same shape.
        """
        self.gamma_mode = n_gamma is not None and int(n_gamma) > 1
        self.n_gamma = int(n_gamma) if self.gamma_mode else 1
        self.gamma_range = tuple(float(v) for v in gamma_range)
        self.gamma_prior = gamma_prior

        if not self.gamma_mode:
            return

        lo, hi = self.gamma_range
        if lo < 0 or hi <= lo:
            raise ValueError(
                f"gamma_range must satisfy 0 <= lo < hi, got {self.gamma_range}. "
                "Negative gamma is redundant with a phase shift of pi."
            )
        if getattr(self, "unspliced_mode", False):
            raise NotImplementedError(
                "gamma amplitude scaling is not implemented for unspliced_mode."
            )

        self.register_buffer(
            "gamma_values", torch.linspace(lo, hi, self.n_gamma, dtype=torch.float32)
        )

    # ------------------------------------------------------------------- grid

    def grid_size(self, n_theta=None):
        """Number of points on the flattened (theta, gamma) grid."""
        if n_theta is None:
            return self.Nx
        return n_theta * self.n_gamma

    def grid_axes(self, n_theta):
        """
        The flattened grid coordinates.

        Returns:
            ``(phi_x, gamma_x)``, both of length ``n_theta * n_gamma``, with
            ``gamma_x = None`` when gamma is disabled. Layout is
            ``x = i_theta * n_gamma + j_gamma``.
        """
        theta = torch.linspace(0, 2 * torch.pi, n_theta + 1, dtype=torch.float32)[:-1]

        if not self.gamma_mode:
            return theta, None

        gamma_values = self.gamma_values.to(theta.device)
        phi_x = theta.repeat_interleave(self.n_gamma)
        gamma_x = gamma_values.repeat(n_theta)
        return phi_x, gamma_x

    @staticmethod
    def scale_design(X, gamma_x):
        """Scale each harmonic design row by its gamma (no-op if ``gamma_x`` is None)."""
        if gamma_x is None:
            return X
        return X * gamma_x.unsqueeze(-1)

    def expand_to_grid(self, y, n_grid, indices=slice(None)):
        """
        Slice cells out of the data tensor and broadcast it along the grid axis.

        The data does not depend on the grid, so the leading axis is an
        ``expand`` (a view): enlarging the grid costs no extra memory for ``y``.
        """
        if y.shape[0] == n_grid:
            return y[:, indices, :]
        return y[0, indices, :].unsqueeze(0).expand(n_grid, -1, -1)

    def integration_method(self, method):
        """
        Force plain summation on the flattened grid.

        Simpson's rule assumes a contiguous 1-D periodic axis, which the
        flattened (theta, gamma) grid is not. With a uniform grid the summation
        weight is a constant factor, so the normalized posterior is unaffected.
        """
        return "sum" if self.gamma_mode else method

    # ------------------------------------------------------------------ prior

    def log_gamma_prior(self, gamma_x=None):
        """
        Log prior density on gamma, evaluated on the flattened grid.

        Returns a ``[Nx]`` tensor, or a scalar zero when gamma is disabled or
        the prior is flat.
        """
        if not self.gamma_mode:
            return torch.zeros((), dtype=torch.float32, device=self.dev)

        if gamma_x is None:
            gamma_x = self.gamma_x

        prior = self.gamma_prior
        if prior is None or prior == "uniform":
            return torch.zeros_like(gamma_x)
        if callable(prior):
            return prior(gamma_x)
        if isinstance(prior, dict):
            return _normal_log_prob(gamma_x, prior.get("mu", 1.0), prior["sigma"])
        if isinstance(prior, (tuple, list)):
            mu, sigma = prior
            return _normal_log_prob(gamma_x, mu, sigma)
        if isinstance(prior, (int, float)):
            return _normal_log_prob(gamma_x, 1.0, float(prior))

        raise ValueError(f"Unrecognized gamma_prior: {prior!r}")

    # -------------------------------------------------------------- posterior

    def split_gamma_grid(self, P_xc):
        """
        Reshape a flattened-grid posterior ``(Nx, Nc)`` into ``(n_theta, n_gamma, Nc)``.

        Accepts numpy arrays or torch tensors.
        """
        n_theta = P_xc.shape[0] // self.n_gamma
        return P_xc.reshape(n_theta, self.n_gamma, P_xc.shape[1])

    def gamma_posterior_statistics(self, P_tgc):
        """
        Summaries of the gamma marginal.

        Args:
            P_tgc: joint posterior, shape ``(n_theta, n_gamma, Nc)`` (numpy).

        Returns:
            ``(p_gc, mean_c, std_c, mode_c)`` where ``p_gc`` is the normalized
            gamma marginal, shape ``(n_gamma, Nc)``.
        """
        gamma = nmp(self.gamma_values)
        p_gc = P_tgc.sum(axis=0)
        p_gc = p_gc / p_gc.sum(axis=0, keepdims=True)

        mean_c = (gamma[:, None] * p_gc).sum(axis=0)
        var_c = (gamma[:, None] ** 2 * p_gc).sum(axis=0) - mean_c**2
        std_c = np.sqrt(np.clip(var_c, 0.0, None))
        mode_c = gamma[np.argmax(p_gc, axis=0)]
        return p_gc, mean_c, std_c, mode_c

    def gamma_boundary_mass(self, p_gc=None, n_edge=1):
        """
        Fraction of the gamma posterior sitting on the grid edges, per cell.

        A diagnostic for whether the bounds are binding: values close to 1 mean
        the grid is too narrow (or the cell is genuinely arrhythmic and pinned
        at ``gamma = lo``).

        Args:
            p_gc: gamma marginal ``(n_gamma, Nc)``. Defaults to the stored
                ``gamma_post_gc`` from the last inference.
            n_edge: how many grid points at each end count as "boundary".
        """
        if p_gc is None:
            p_gc = self.gamma_post_gc
        return p_gc[:n_edge].sum(axis=0), p_gc[-n_edge:].sum(axis=0)

    # --------------------------------------------------------------- plotting

    def plot_gamma_posterior(
        self, cell=0, P_tgc=None, axes=None, cmap="viridis", period=24.0, truth=None
    ):
        """
        The joint ``(theta, gamma)`` posterior of one cell/sample, as a heatmap
        with both marginals drawn alongside.

        Use it to check the posterior is well behaved: mass in the interior
        rather than pinned at a gamma bound, and no ridge running along a
        ``gamma(theta)`` manifold, which would say the two are not separately
        identified for that cell.

        Args:
            cell: index into the cell axis of ``P_tgc``.
            P_tgc: joint posterior ``(n_theta, n_gamma, Nc)``. Defaults to
                ``self.posterior_tgc`` from the last inference.
            axes: a 2x2 array of axes, as returned by a previous call.
            period: phase axis units; 24 plots hours, ``2 * np.pi`` radians.
            truth: optional ``(theta, gamma)`` ground truth to mark.
        """
        import matplotlib.pyplot as plt

        if P_tgc is None:
            P_tgc = self.posterior_tgc
        P = P_tgc[:, :, cell]
        P = P / P.sum()

        theta = np.linspace(0, period, P.shape[0] + 1)[:-1]
        gamma = nmp(self.gamma_values)

        if axes is None:
            _, axes = plt.subplots(
                2,
                2,
                figsize=(5, 4.5),
                sharex="col",
                sharey="row",
                gridspec_kw={"width_ratios": [3, 1], "height_ratios": [1, 3]},
            )
        ax_top, ax_none, ax, ax_right = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
        ax_none.axis("off")

        ax.pcolormesh(theta, gamma, P.T, shading="auto", cmap=cmap)
        ax.set_xlabel(r"$\theta$" + (" [h]" if period == 24.0 else " [rad]"))
        ax.set_ylabel(r"$\gamma$")

        ax_top.plot(theta, P.sum(axis=1), color="k", lw=1)
        ax_top.set_ylabel(r"$P(\theta)$")
        ax_right.plot(P.sum(axis=0), gamma, color="k", lw=1)
        ax_right.set_xlabel(r"$P(\gamma)$")

        if truth is not None:
            theta_t, gamma_t = truth
            ax.plot(theta_t, gamma_t, "r+", ms=10, mew=2)

        ax_top.set_title(f"cell {cell}", fontsize=9)
        return axes
