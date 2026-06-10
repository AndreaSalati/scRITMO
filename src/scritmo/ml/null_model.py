import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom as scipy_nbinom


class NullModelMixin:
    """
    Mixin for ContextModel: fits a null (flat-amplitude) NB model per gene
    for model comparison via BIC.

    The null model has no amplitude and no phase:
        Y_cg ~ NB(exp(a0_g) * counts_c, disp_g)
    with 2 free parameters per gene (a0, disp), fitted via MLE.

    The full model likelihood is evaluated at each cell's MAP phase
    (posterior mode), giving 4 parameters per gene (a0, amp, phase, disp).
    """

    def fit_null_model(self, adata, layer=None, counts=None, phase_estimator="mode", mask=None):
        """
        Fit per-gene intercept-only NB null model and compute per-gene BIC
        comparison against the fitted rhythmic model.

        Must be called after training + get_inferred_phases, so that
        self.post_mode_c, self.disp, and self.m_g are available.

        Args:
            adata  : AnnData object whose var_names cover self.genes.
            layer  : Layer to use for counts (None → adata.X).
            counts : (Nc,) library-size vector; computed from data if None.
            mask   : Boolean or integer index array selecting a subset of cells
                     from the full training set (indexes into self.post_mode_c).
                     Use this when adata/counts are already a subset so that
                     the phase vector is sliced to match.

        Returns:
            pd.DataFrame indexed by gene with columns:
                ll_null    – total log-likelihood under null model
                ll_full    – total log-likelihood under full model at post_mode_c
                delta_ll   – ll_full - ll_null  (> 0 → full model fits better)
                bic_null   – BIC for null model  (k=2)
                bic_full   – BIC for full model  (k=4)
                delta_bic  – bic_null - bic_full (> 0 → full model preferred)
                a0_null    – fitted mesor under null
                disp_null  – fitted dispersion under null
        """
        from .utils import nmp

        # ---- data -------------------------------------------------------
        genes = self.genes
        if layer is None:
            Y = adata[:, genes].X
        else:
            Y = adata[:, genes].layers[layer]
        try:
            Y = Y.toarray()
        except AttributeError:
            Y = np.asarray(Y)
        Y = Y.astype(float)  # (Nc, Ng)
        Nc, Ng = Y.shape

        if counts is None:
            raw = adata.X if layer is None else adata.layers[layer]
            counts = np.asarray(raw.sum(1)).squeeze().astype(float)
        counts = np.asarray(counts).squeeze().astype(float)

        # ---- initialisation from trained model --------------------------
        a0_init = nmp(self.m_g).squeeze()  # (Ng,)
        if hasattr(self, "disp"):
            disp_init = np.asarray(self.disp).squeeze()
        else:
            disp_init = nmp(self.log_disp.exp()).squeeze()

        # ---- fit null model per gene ------------------------------------
        a0_null = np.empty(Ng)
        disp_null = np.empty(Ng)
        ll_null = np.empty(Ng)

        print(f"Fitting null model for {Ng} genes...")
        for g in range(Ng):
            y_g = Y[:, g]
            a0_opt, disp_opt = _fit_null_gene(y_g, counts, a0_init[g], disp_init[g])
            a0_null[g] = a0_opt
            disp_null[g] = disp_opt
            ll_null[g] = _nb_loglik_gene(y_g, a0_opt, disp_opt, counts)

        # ---- full model LL at MAP (posterior mode) phase per gene -------
        if phase_estimator == "mode":
            phases_c = self.post_mode_c  # (Nc,) radians
        else:
            phases_c = self.post_mean_c  # (Nc,) radians
        if mask is not None:
            phases_c = phases_c[mask]
        params_inf = self.get_parameter_dataframe()

        a0_f = params_inf["a_0"].values  # (Ng,)
        amp_f = params_inf["amp"].values  # (Ng,)
        phase_f = params_inf["phase"].values  # (Ng,)
        disp_f = disp_init  # (Ng,)

        ll_full = np.empty(Ng)
        for g in range(Ng):
            log_mu_c = a0_f[g] + amp_f[g] * np.cos(phases_c - phase_f[g])
            mu_c = np.exp(log_mu_c) * counts
            ll_full[g] = _nb_loglik_mu(Y[:, g], mu_c, disp_f[g])

        # ---- BIC  (N = Nc per gene) -------------------------------------
        # null : a0 + disp             → 2 params
        # full : a0 + amp + phase + disp → 4 params
        bic_null = -2 * ll_null + 2 * np.log(Nc)
        bic_full = -2 * ll_full + 4 * np.log(Nc)

        # ---- store and return -------------------------------------------
        self.null_a0 = a0_null
        self.null_disp = disp_null

        return pd.DataFrame(
            {
                "ll_null": ll_null,
                "ll_full": ll_full,
                "delta_ll": ll_full - ll_null,
                "bic_null": bic_null,
                "bic_full": bic_full,
                "delta_bic": bic_null - bic_full,
                "a0_null": a0_null,
                "disp_null": disp_null,
            },
            index=genes,
        )


# ------------------------------------------------------------------ helpers


def _fit_null_gene(y_g, counts, a0_init, disp_init):
    """MLE for one gene under the NB intercept-only model (Nelder-Mead)."""

    def neg_ll(params):
        a0, log_disp = params
        return -_nb_loglik_gene(y_g, a0, np.exp(log_disp), counts)

    x0 = [float(a0_init), np.log(max(float(disp_init), 1e-6))]
    result = minimize(
        neg_ll,
        x0=x0,
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-5},
    )
    a0_opt, log_disp_opt = result.x
    return a0_opt, np.exp(log_disp_opt)


def _nb_loglik_gene(y_g, a0, disp, counts):
    """NB log-likelihood: Y ~ NB(exp(a0)*counts, disp)."""
    mu = np.exp(float(a0)) * counts
    return _nb_loglik_mu(y_g, mu, disp)


def _nb_loglik_mu(y_g, mu, disp):
    """NB log-likelihood given per-cell mean vector mu and scalar disp."""
    r = 1.0 / float(disp)
    # scipy nbinom(n=r, p): P(Y=k) ∝ p^r*(1-p)^k, mean = r*(1-p)/p → p = r/(r+mu)
    p = r / (r + mu)
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    return scipy_nbinom.logpmf(y_g.astype(int), r, p).sum()
