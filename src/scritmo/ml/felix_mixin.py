"""Mixin that adds Felix's marginal-likelihood desynchrony estimation to
ContextModel.

Usage
-----
After training a ContextModel::

    sigma_result = model.estimate_sigma(
        Y=counts,            # (N, G) — model's genes, same order as model.genes
        ZT=zt_hours,         # (N,)  — external time per cell
        L=lib_sizes,         # (N,)  — total UMIs per cell (recommended)
    )
    print(f"Desynchrony: {sigma_result['sigma_phi_hr']:.2f}h")

Two modes:
  - ``mode="sigma_only"`` (default): uses scRITMO's fitted gene parameters,
    optimises only the per-timepoint σ. Fast, Level A integration.
  - ``mode="full"``: runs the independent two-stage MLE (Felix's method).
    Slower but does not depend on scRITMO's phase inference.
"""

from __future__ import annotations

import numpy as np

from .sigma_marginal import (
    fit_sigma_only,
    fit_felix,
)


class FelixDesynchronyMixin:
    """Mixin providing :meth:`estimate_sigma` to ContextModel.

    Relies on the parent having:
      - ``Ng``           : int — number of genes
      - ``genes``        : list-like of gene names
      - ``log_disp``     : nn.Parameter — log of NB dispersion (always exists)
      - ``get_parameter_dataframe_context(gene_names)`` — returns dict of
        DataFrames keyed by context label, each with ``a_0``, ``a_1``, ``b_1``
    """

    def estimate_sigma(
        self,
        Y: np.ndarray,
        ZT: np.ndarray,
        L: np.ndarray | None = None,
        context_key: str | None = None,
        mode: str = "sigma_only",
        n_gh: int = 15,
        maxiter: int = 200,
        verbose: bool = False,
    ) -> dict:
        """Estimate desynchrony σ from count data using Felix's marginal
        likelihood.

        Parameters
        ----------
        Y : ndarray, shape (N, G)
            Integer counts for the **exact same genes** this model was trained
            on (``self.genes``, same order).  Shape is (N, self.Ng).
        ZT : ndarray, shape (N,)
            External time in **hours** for every cell (e.g. ZTmod values).
        L : ndarray, shape (N,), optional
            Library sizes — **total UMI count per cell** across all genes, not
            just the model's gene subset.  This is crucial for correct size-
            factor normalisation.  If ``None``, falls back to row sums of
            ``Y`` (a rough approximation that may bias σ downward).
        context_key : str, optional
            Which fitted context's parameters to use.  Required when the model
            has more than one context.  If ``None`` and exactly one context
            exists, it is selected automatically.
        mode : {"sigma_only", "full"}
            - ``"sigma_only"`` (default): fix gene parameters to scRITMO's
              fitted values, optimise only σ.  The post-hoc Level A estimator.
            - ``"full"``: run the independent two-stage MLE, fitting all gene
              parameters and σ jointly.  Does **not** use scRITMO's fitted
              params.
        n_gh : int
            Number of Gauss-Hermite quadrature nodes (default 15).
        maxiter : int
            Maximum iterations for the L-BFGS-B optimiser.
        verbose : bool
            Print convergence info.

        Returns
        -------
        dict
            Result with keys ``sigma_phi_hr`` (scalar median, hours),
            ``sigma_hr`` (per-timepoint), ``sigma_phi_rad``, ``sigma_rad``,
            and mode-specific keys.
        """
        # ── validation ──────────────────────────────────────────────────
        Y = np.asarray(Y, dtype=np.int64)
        ZT = np.asarray(ZT, dtype=np.float64).ravel()

        if Y.ndim != 2 or Y.shape[1] != self.Ng:
            raise ValueError(
                f"Y shape {Y.shape} does not match model's gene dimension "
                f"(self.Ng = {self.Ng}).  Expected (N, {self.Ng})."
            )
        if ZT.shape[0] != Y.shape[0]:
            raise ValueError(
                f"ZT length {ZT.shape[0]} != Y rows {Y.shape[0]}"
            )

        # Build time-index mapping (cells → timepoint index)
        timepoints = np.sort(np.unique(ZT))
        tp_to_idx = {float(t): k for k, t in enumerate(timepoints)}
        time_index = np.array(
            [tp_to_idx[float(t)] for t in ZT], dtype=np.int64
        )

        # Library sizes — default to row sums with a warning
        if L is None:
            L = Y.sum(axis=1).astype(np.float64)
            L = np.maximum(L, 1.0)
            import warnings
            warnings.warn(
                "L not provided; using row sums of Y as library sizes. "
                "For correct normalisation, pass total-UMI library sizes.",
                UserWarning,
                stacklevel=2,
            )
        else:
            L = np.asarray(L, dtype=np.float64).ravel()
            if L.shape[0] != Y.shape[0]:
                raise ValueError(
                    f"L length {L.shape[0]} != Y rows {Y.shape[0]}"
                )
            L = np.maximum(L, 1.0)

        # ── dispatch ────────────────────────────────────────────────────
        if mode == "sigma_only":
            fixed = self._extract_felix_params(context_key)
            result = fit_sigma_only(
                Y, L, time_index, timepoints, fixed,
                n_gh=n_gh, maxiter=maxiter, verbose=verbose,
            )

        elif mode == "full":
            result = fit_felix(
                Y, L, time_index, timepoints,
                n_gh=n_gh, maxiter=maxiter, verbose=verbose,
            )

        else:
            raise ValueError(
                f"Unknown mode '{mode}'.  Choose 'sigma_only' or 'full'."
            )

        result["n_genes"] = self.Ng
        result["n_timepoints"] = len(timepoints)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_felix_params(
        self, context_key: str | None = None
    ) -> dict:
        """Convert scRITMO's fitted parameters to the dict Felix expects.

        Uses ``self.log_disp`` (always a Parameter) rather than ``self.disp``
        (only set after ``forward()``)."""
        param_df = self.get_parameter_dataframe_context(
            np.arange(self.Ng)
        )

        if context_key is None:
            keys = list(param_df.keys())
            if len(keys) == 1:
                context_key = keys[0]
            else:
                raise ValueError(
                    f"Model has {len(keys)} contexts ({keys}); "
                    f"specify which one via context_key=."
                )

        gp = param_df[context_key]
        alpha = gp["a_0"].values.astype(float)
        beta_cos = gp["a_1"].values.astype(float)
        beta_sin = gp["b_1"].values.astype(float)

        # Dispersion: read log_disp (always exists as a Parameter) →
        # exp() to get linear disp.
        import torch
        disp_t = torch.exp(self.log_disp).detach().cpu()
        disp_arr = disp_t.numpy().ravel()
        if len(disp_arr) == 1:
            disp = np.full(self.Ng, float(disp_arr[0]))
        elif len(disp_arr) == self.Ng:
            disp = disp_arr.astype(float)
        else:
            raise ValueError(
                f"disp length ({len(disp_arr)}) doesn't match Ng ({self.Ng})"
            )

        return dict(
            alpha=alpha,
            beta_cos=beta_cos,
            beta_sin=beta_sin,
            disp=disp,
        )
