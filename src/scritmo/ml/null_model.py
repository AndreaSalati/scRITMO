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

    def rhythmic_evidence_per_cell(
        self,
        adata,
        layer=None,
        counts=None,
        phase_estimator="mode",
        mask=None,
        mode="marginal",
        n_theta=100,
    ):
        r"""Per-cell rhythmic-vs-flat log-likelihood ratio Δ_c (Reviewer 1, idea i).

        The per-cell transpose of :meth:`fit_null_model`. Where ``fit_null_model``
        sums the NB log-likelihood over *cells* to score each *gene* (``delta_ll``
        per gene), this sums over *genes* to score each *cell*:

            Δ_c = log L_c(β*) − log L_c(β_flat)

        ``β_flat`` reuses the same fitted mesor ``a0_g`` with all amplitudes
        ``A_g = 0`` (shared-a0 null). The full (rhythmic) term log L_c(β*) is offered
        in two flavors (``mode``):

          - ``"marginal"`` (default, recommended): the cell's likelihood under the
            fitted gene params with the latent phase **integrated out** over a uniform
            grid, ``log[(1/Nx) Σ_x exp Σ_g log NB(y_cg | μ_g(θ_x))]``. This is the
            faithful per-cell model evidence: an arrhythmic cell gets Δ_c ≤ 0 (the
            oscillating template can only *hurt* a flat cell on average), so the flat
            floor really is ~0 — no phase is cherry-picked.
          - ``"map"``: the full likelihood at the cell's MAP phase (``post_mode_c``).
            Faster, but because the MAP phase is *selected* to maximize the fit, even a
            flat cell scores a small positive Δ_c (a depth-dependent overfitting floor),
            so the flat group does NOT sit at 0. Use only if a profile statistic is
            specifically wanted; prefer ``"marginal"`` for arrhythmic classification.

        Δ_c is large/positive for cells with detectable rhythmic structure and ≈ 0
        (marginal) for arrhythmic cells *independently of where the cell sits in the
        cycle* — the companion statistic to σ_u (``post_std_c``) the reviewer asked for.

        Must be called after training + ``get_inferred_phases`` (needs
        ``post_mode_c``/``post_mean_c``, ``disp`` and ``m_g``). Genes are summed
        unweighted (matching ``fit_null_model``; assumes the default unit
        ``weights_g``). Pure forward evaluation: no retraining, only ``self.delta_c``
        is cached.

        Args:
            adata  : AnnData whose var_names cover ``self.genes``.
            layer  : Layer to use for counts (None → adata.X). Same convention as
                     :meth:`fit_null_model`.
            counts : (Nc,) library-size vector; summed from the data if None.
            phase_estimator : "mode" (MAP, default) or "mean" posterior phase (only
                     used for the ``"map"`` full term).
            mask   : Boolean/integer index selecting a subset of cells from the full
                     training set, so the stored ``post_mode_c``/``post_std_c`` are
                     sliced to match a subset ``adata``/``counts`` (same semantics as
                     :meth:`fit_null_model`).
            mode   : "marginal" (default) or "map" — which full term feeds ``delta_c``.
            n_theta: phase-grid resolution for the marginal integral (default 100).

        Returns:
            pd.DataFrame indexed by ``adata.obs_names`` with columns:
                delta_c            – the selected ``mode``'s Δ_c, summed over genes
                delta_c_per_gene   – delta_c / Ng (depth/panel-size comparable)
                delta_c_marginal   – marginal-evidence Δ_c (phase integrated out)
                delta_c_map        – MAP-profile Δ_c (phase at post_mode_c)
                ll_full_marginal_c – marginal full-model log-lik, summed over genes
                ll_full_map_c      – full-model log-lik at MAP phase, summed over genes
                ll_flat_c          – flat-model (A_g=0) log-lik, summed over genes
            and, when their length matches Nc, ``post_std_c`` and ``post_mode_c``
            carried through for convenient joint reporting with σ_u.
        """
        from .utils import nmp
        from scipy.special import logsumexp

        # ---- data (mirror fit_null_model) -------------------------------
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

        # ---- fitted template + per-cell MAP phase -----------------------
        params_inf = self.get_parameter_dataframe()
        a0_f = params_inf["a_0"].values  # (Ng,)
        amp_f = params_inf["amp"].values  # (Ng,)
        phase_f = params_inf["phase"].values  # (Ng,)
        if hasattr(self, "disp"):
            disp_f = np.asarray(self.disp).squeeze()
        else:
            disp_f = nmp(self.log_disp.exp()).squeeze()
        disp_f = np.broadcast_to(disp_f, (Ng,))  # scalar or per-gene → (Ng,)

        if phase_estimator == "mode":
            phases_c = self.post_mode_c
        else:
            phases_c = self.post_mean_c
        if mask is not None:
            phases_c = phases_c[mask]
        phases_c = np.asarray(phases_c, dtype=float).reshape(-1)  # (Nc,)
        if phases_c.shape[0] != Nc:
            raise ValueError(
                f"phase vector length {phases_c.shape[0]} != n_cells {Nc}; "
                "pass `mask` if adata/counts are a subset of the training set."
            )

        # ---- flat null + full-at-MAP (shared a0), summed over genes -----
        log_mu_map = a0_f[None, :] + amp_f[None, :] * np.cos(
            phases_c[:, None] - phase_f[None, :]
        )  # (Nc, Ng)
        mu_map = np.exp(log_mu_map) * counts[:, None]
        mu_flat = np.exp(a0_f)[None, :] * counts[:, None]  # A_g=0

        ll_map = np.empty((Nc, Ng))
        ll_flat = np.empty((Nc, Ng))
        for g in range(Ng):
            ll_map[:, g] = _nb_logpmf_mu(Y[:, g], mu_map[:, g], disp_f[g])
            ll_flat[:, g] = _nb_logpmf_mu(Y[:, g], mu_flat[:, g], disp_f[g])
        ll_full_map_c = ll_map.sum(axis=1)
        ll_flat_c = ll_flat.sum(axis=1)

        # ---- marginal full-model evidence: integrate phase over a uniform grid ----
        # log L_c = log( (1/Nx) Σ_x exp Σ_g log NB(y_cg | μ_g(θ_x)·counts_c) )
        theta_x = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)  # (Nx,)
        ll_xc = np.zeros((n_theta, Nc))
        for g in range(Ng):
            log_mu_xg = a0_f[g] + amp_f[g] * np.cos(theta_x - phase_f[g])  # (Nx,)
            rate_xc = np.exp(log_mu_xg)[:, None] * counts[None, :]  # (Nx, Nc)
            ll_xc += _nb_logpmf_mu(Y[:, g][None, :], rate_xc, disp_f[g])
        ll_full_marg_c = logsumexp(ll_xc, axis=0) - np.log(n_theta)

        delta_c_map = ll_full_map_c - ll_flat_c
        delta_c_marg = ll_full_marg_c - ll_flat_c
        delta_c = delta_c_marg if mode == "marginal" else delta_c_map

        self.delta_c = delta_c

        out = pd.DataFrame(
            {
                "delta_c": delta_c,
                "delta_c_per_gene": delta_c / Ng,
                "delta_c_marginal": delta_c_marg,
                "delta_c_map": delta_c_map,
                "ll_full_marginal_c": ll_full_marg_c,
                "ll_full_map_c": ll_full_map_c,
                "ll_flat_c": ll_flat_c,
            },
            index=np.asarray(adata.obs_names),
        )
        # carry σ_u / MAP phase through when they line up with this cell set
        if getattr(self, "post_std_c", None) is not None:
            ps = np.asarray(self.post_std_c).reshape(-1)
            pm = np.asarray(self.post_mode_c).reshape(-1)
            if mask is not None:
                ps, pm = ps[mask], pm[mask]
            if ps.shape[0] == Nc:
                out["post_std_c"] = ps
                out["post_mode_c"] = pm
        return out


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


def _nb_logpmf_mu(y_g, mu, disp):
    """Per-observation NB log-pmf given per-cell mean vector mu and scalar disp.

    Returns the un-summed log-pmf array (same shape as ``y_g``). Both
    :func:`_nb_loglik_mu` (sum over cells, per gene) and
    :meth:`NullModelMixin.rhythmic_evidence_per_cell` (sum over genes, per cell)
    are reductions of this same quantity along different axes.
    """
    r = 1.0 / float(disp)
    # scipy nbinom(n=r, p): P(Y=k) ∝ p^r*(1-p)^k, mean = r*(1-p)/p → p = r/(r+mu)
    p = r / (r + mu)
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    return scipy_nbinom.logpmf(np.asarray(y_g).astype(int), r, p)


def _nb_loglik_mu(y_g, mu, disp):
    """NB log-likelihood given per-cell mean vector mu and scalar disp."""
    return _nb_logpmf_mu(y_g, mu, disp).sum()
