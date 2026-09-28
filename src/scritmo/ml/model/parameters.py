"""
Gene-parameter transforms and the ``Beta`` tables built from a fitted model.

The model stores amplitudes in an unconstrained form (``log_amp``) and phases as
acrophases; these helpers turn them into the Cartesian harmonic coefficients the
likelihood uses and into ``Beta`` DataFrames for users.
"""

import numpy as np
import pandas as pd
import torch

from scritmo import Beta
from ..utils import nmp

# largest amplitude the "logit" parametrization can reach: a log2 fold change of 8
MAX_AMP = 8 / (2 * np.log2(np.e))


def amplitude(log_amp, log_amp_fn, max_amp=MAX_AMP):
    """Amplitude from its unconstrained parameter ("logit" in (0, max_amp), or "log")."""
    if log_amp_fn == "logit":
        return torch.sigmoid(log_amp) * max_amp
    elif log_amp_fn == "log":
        return torch.exp(log_amp)
    raise ValueError(f"Unknown log_amp_fn '{log_amp_fn}'. Use 'logit' or 'log'.")


def raw_amplitude(amp, log_amp_fn, max_amp=MAX_AMP):
    """Inverse of :func:`amplitude`; "logit" clamps into (0.01, max_amp - 0.01) first."""
    if log_amp_fn == "logit":
        safe_amp = torch.clamp(amp, min=1e-2, max=max_amp - 1e-2)
        return torch.logit(safe_amp / max_amp)
    elif log_amp_fn == "log":
        return torch.log(amp)
    raise ValueError(f"Unknown log_amp_fn '{log_amp_fn}'. Use 'logit' or 'log'.")


def harmonic_coefficients(amp, acrophase):
    """Cartesian coefficients ``[a; b]`` (shape [2, Ng]) from amplitude and acrophase."""
    cos = amp * torch.cos(acrophase).unsqueeze(0)
    sin = amp * torch.sin(acrophase).unsqueeze(0)
    return torch.cat([cos, sin], dim=0)


def parameter_table(m_g, ab, genes, log_disp):
    """
    ``Beta`` table with columns a_0, a_1, b_1, ..., amp, phase (per harmonic)
    and disp, one row per gene.
    """
    a_0_np = nmp(m_g).squeeze()
    ab_np = nmp(ab).T
    n_harmonics = ab_np.shape[1] // 2
    col_names = ["a_0"] + [
        f"{c}_{i+1}" for i in range(n_harmonics) for c in ("a", "b")
    ]
    full_array = np.concatenate([a_0_np[:, None], ab_np], axis=1)
    params_g = Beta(pd.DataFrame(full_array, columns=col_names, index=genes))
    params_g.get_amp(inplace=True)
    params_g["disp"] = nmp(log_disp.exp()).squeeze()
    return params_g


def context_parameter_tables(params_g, contexts, m_yg, log_lambda_y):
    """
    LEGACY. One ``Beta`` table per context label, folding in the per-context
    intercept ``m_yg[i]`` and amplitude scale ``exp(log_lambda_y[i])``.
    """
    params_y = {}
    for i, ct in enumerate(contexts):
        par = params_g.copy()
        par.a_0 += m_yg[i, :].cpu().detach().numpy()
        col_names = par.get_ab_column_names(keep_a_0=False)
        lambda_y = nmp(log_lambda_y[i].exp().squeeze())
        par[col_names] = par[col_names] * lambda_y
        par["amp"] = par["amp"] * lambda_y
        par.get_cartesian(inplace=True)
        params_y[ct] = par
    return params_y
