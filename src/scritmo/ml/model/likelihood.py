"""
Pure tensor functions behind the Scritmo likelihood.

Nothing here holds state: the model class passes in its grid, parameters and
flags. Keeping the maths out of the class means each piece can be tested on its
own and the phase grid / design matrix are built in exactly one place.

- Phase grid and harmonic design matrix: :func:`phase_grid`, :func:`grid_design`.
- Observation model: :func:`compute_nb_params`, :func:`compute_poisson_rate`,
  :func:`count_distribution`.
- Integration over the phase grid: :func:`vectorized_simpson`,
  :func:`periodic_simpson`, :func:`marginalize`, :func:`normalize_log_dist`,
  :func:`log_like_loss`.
- Priors: :func:`flat_log_prior`, :func:`batch_log_prior`.
"""

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder

from ..utils import harmonic_dm_torch
from ..misc.power_spherical.power_spherical import log_von_mises


# --------------------------------------------------------------------------
# phase grid and design matrices
# --------------------------------------------------------------------------


def phase_grid(n_theta, phase_range=None, device=None):
    """
    The phase grid the likelihood is evaluated on.

    On the full circle (``phase_range=None``): ``n_theta`` even points on
    [0, 2π), endpoint excluded. On an arc ``(lo, hi)``: the ``n_theta``
    midpoints of equal bins of [lo, hi], so a plain sum times
    ``(hi - lo) / n_theta`` is the midpoint rule.

    Returns:
        1D float32 tensor of phases in radians, shape [n_theta].
    """
    if phase_range is None:
        return torch.linspace(
            0, 2 * torch.pi, n_theta + 1, dtype=torch.float32, device=device
        )[:-1]
    lo, hi = phase_range
    step = (hi - lo) / n_theta
    return lo + step * (
        torch.arange(n_theta, dtype=torch.float32, device=device) + 0.5
    )


def grid_design(phi_x, n_harmonics, n_cells):
    """
    Harmonic design matrix on a phase grid, broadcast over cells.

    Args:
        phi_x: Grid phases, shape [Nx].
        n_harmonics: Number of harmonics.
        n_cells: Number of cells to expand over.

    Returns:
        Tensor [Nx, Nc, 2 * n_harmonics] (an expanded view, no copy).
    """
    X = harmonic_dm_torch(phi_x, n_harmonics, False)
    return X.unsqueeze(1).expand(phi_x.shape[0], n_cells, n_harmonics * 2)


def one_hot(vec, n_rows, device, all_categories=None):
    """
    One-hot design matrix of a vector of category labels.

    ``vec=None`` gives a single all-ones column of ``n_rows`` rows (one global
    context). ``all_categories`` fixes the column set, so the shape does not
    depend on which labels happen to be present.
    """
    if vec is None:
        return torch.ones((n_rows, 1), dtype=torch.float32, device=device)

    data = np.array(vec).reshape(-1, 1)
    if all_categories is not None:
        encoder = OneHotEncoder(
            categories=[all_categories],
            sparse_output=False,
            handle_unknown="ignore",
        )
    else:
        encoder = OneHotEncoder(sparse_output=False)
    return torch.tensor(encoder.fit_transform(data), dtype=torch.float32, device=device)


# --------------------------------------------------------------------------
# observation model
# --------------------------------------------------------------------------


def compute_nb_params(
    E_xcg: torch.Tensor, disp: torch.Tensor, counts: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Negative Binomial parameter computation (r, p) from the log-mean.

    Args:
        E_xcg: Expected mean values (before exp transform)
        disp: Dispersion parameter
        counts: Library size counts
        eps: Epsilon for numerical stability

    Returns:
        r: Total count parameter
        p: Success probability parameter (clamped)
    """
    E_xcg_exp = torch.exp(E_xcg) * counts
    r = 1.0 / disp
    p = disp * E_xcg_exp / (1.0 + disp * E_xcg_exp)
    p = p.clamp(min=eps, max=1.0 - eps)
    return r, p


def compute_poisson_rate(E_xcg: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    return torch.exp(E_xcg) * counts


def count_distribution(noise_model, E_xcg, disp, counts):
    """
    The count distribution for a log-mean ``E_xcg``.

    Args:
        noise_model: "nb", "poisson" or "gaussian" (unit-variance Normal on
            ``E_xcg`` itself; no library-size scaling).
    """
    if noise_model == "nb":
        r, p = compute_nb_params(E_xcg, disp, counts)
        return torch.distributions.NegativeBinomial(total_count=r, probs=p)
    elif noise_model == "poisson":
        return torch.distributions.Poisson(rate=compute_poisson_rate(E_xcg, counts))
    elif noise_model == "gaussian":
        return torch.distributions.Normal(loc=E_xcg, scale=1.0)
    raise NotImplementedError(f"Noise model '{noise_model}' is not implemented.")


# --------------------------------------------------------------------------
# integration over the phase grid
# --------------------------------------------------------------------------


def vectorized_simpson(y_values: torch.Tensor, h: float) -> torch.Tensor:
    """
    Periodic Simpson's rule along dim 0 with step ``h``.

    Args:
        y_values: Tensor of function values, shape (N, B)
        h: Step size (uniform grid)

    Returns:
        Integrals, shape (B,)
    """
    n_points = y_values.shape[0]

    if n_points < 2:
        return torch.zeros(
            y_values.shape[1], dtype=y_values.dtype, device=y_values.device
        )

    # weights [2, 4, 2, 4, ...] for the periodic rule
    weights = torch.ones(n_points, dtype=y_values.dtype, device=y_values.device)
    weights[1::2] = 4.0
    weights[0::2] = 2.0

    weighted_y = y_values * weights.unsqueeze(1)
    return (h / 3.0) * torch.sum(weighted_y, dim=0)


def periodic_simpson(y_values, x_values):
    """
    Periodic Simpson's rule on a uniform grid ``x_values`` covering one period
    without repeating the endpoint. Integrates along dim 0.

    Args:
        y_values: shape (N, B); N must be even.
        x_values: shape (N,).

    Returns:
        Integrals, shape (B,).
    """
    n_points = y_values.shape[0]
    if n_points % 2 != 0:
        raise ValueError(
            "Periodic Simpson's rule requires an even number of sample points."
        )
    if n_points < 2:
        return torch.zeros(
            y_values.shape[1], device=y_values.device, dtype=y_values.dtype
        )
    h = x_values[1] - x_values[0]
    return vectorized_simpson(y_values, float(h))


def integrate_grid(l_xc, method, phi_x, phase_width, Nx):
    """Integral over the phase axis (dim 0): periodic Simpson or a plain sum."""
    if method == "simpson":
        return periodic_simpson(l_xc, phi_x)
    elif method == "sum":
        # phase_width is 2π on the full circle, hi - lo on a phase_range arc
        return torch.sum(l_xc, dim=0) * (phase_width / Nx)
    raise ValueError(f"Unknown integration method '{method}'. Use 'simpson' or 'sum'.")


def marginalize(ll_xc, log_prior, method, phi_x, phase_width, Nx):
    r"""
    Marginal likelihood per cell, :math:`\int P(D|\theta) P(\theta) d\theta`.

    Log-sum-exp stabilized: returns ``(l_c, max_c, l_xc)`` with
    ``log P(D) = log(l_c) + max_c`` and ``l_xc`` the shifted integrand.
    """
    ll_xc = ll_xc + log_prior
    max_c = torch.max(ll_xc, dim=0, keepdim=True).values
    l_xc = torch.exp(ll_xc - max_c)
    l_c = integrate_grid(l_xc, method, phi_x, phase_width, Nx)
    return l_c, max_c, l_xc


def normalize_log_dist(ll_xc, method, phi_x, phase_width, Nx, on_arc=False):
    """
    Normalize a log density over the phase axis (dim 0) into a density.

    The periodic Simpson rule is wrong on an arc, so ``on_arc`` forces "sum".
    """
    max_c = torch.max(ll_xc, dim=0, keepdim=True).values
    l_xc = torch.exp(ll_xc - max_c)
    if method == "simpson" and on_arc:
        method = "sum"
    l_c = integrate_grid(l_xc, method, phi_x, phase_width, Nx)
    return l_xc / l_c


def log_like_loss(l_c, max_c):
    """Summed log marginal likelihood from the output of :func:`marginalize`."""
    ll_c = torch.log(l_c) + max_c
    return ll_c.sum()


# --------------------------------------------------------------------------
# priors
# --------------------------------------------------------------------------


def flat_log_prior(phase_width):
    """Log of the uniform phase density on the support (circle or arc)."""
    return torch.log(torch.tensor(1 / phase_width, dtype=torch.float32))


def batch_log_prior(phi_x, phi_b, log_kappa_b, dm_batch):
    """
    Per-cell Von-Mises phase prior centred on each cell's batch phase.

    Args:
        phi_x: Grid phases [Nx].
        phi_b: Batch mean phases [Nb].
        log_kappa_b: Log concentration, scalar or [Nb].
        dm_batch: One-hot batch design [Nc, Nb].

    Returns:
        [Nx, Nc] log prior.
    """
    prior_xb = log_von_mises(phi_x.unsqueeze(1), phi_b, torch.exp(log_kappa_b))
    return prior_xb @ dm_batch.T
