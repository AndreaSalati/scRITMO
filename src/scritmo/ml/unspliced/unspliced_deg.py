import numpy as np
import torch
from torch import tensor as tt
from torch import nn
import torch.nn.functional as F
from scipy.stats import chi2
from scritmo import Beta, optimal_shift, w, rh
import pandas as pd
from ..utils import harmonic_dm_torch, nmp
import scritmo as sr


class UnsplicedMixin:
    """
    Joint spliced/unspliced modeling.

    Kinetics: du/dt = alpha(t) - beta u,  ds/dt = beta u - gamma(t) s. Solving the
    second equation for u gives the exact identity

        u(theta) = s(theta) / beta * (gamma(theta) + d ln s / dt)

    With a single-harmonic spliced curve and gamma(theta) = gamma_mean + G . x(theta),
    the ratio curve h = u / s is a first harmonic in LINEAR space,

        h(theta) = m + r . x(theta),   m = gamma_mean / beta,   r = (D + G) / beta,

    where D = omega * (b, -a) is fixed by the spliced fit. The unspliced likelihood
    sees the parameters only through (m, r): 3 numbers per gene. Two parametrizations
    are available (``mp["unspliced_param"]``):

    - ``"ratio"`` (default): fit h directly in identifiable coordinates,

          h(theta) = exp(lambda) * (1 + rho * cos(theta - psi)),
          psi = phi - pi/2 + delta,

      with lambda free (log level of u/s; absorbs beta and the intronic capture
      ratio), rho = sigmoid(.) in (0, 1) (positivity of u is automatic) and delta the
      phase mismatch of u/s against the constant-degradation prediction.
      ``rhythmic_degradation=False`` fixes delta = 0 (null), True frees it. The null
      and the alternative differ by exactly one identifiable parameter. Kinetic rates
      are DERIVED quantities, see :meth:`get_kinetic_parameters`.

    - ``"kinetic"`` (legacy): fit (beta, gamma_mean, A_gamma, phi_gamma). With rhythmic
      degradation this has a flat likelihood direction (the beta-gamma_mean ridge);
      kept so old pickles and scripts keep reproducing.
    """

    def _unspliced_param(self):
        # models pickled before the ratio parametrization existed are kinetic
        return getattr(self, "unspliced_param", "kinetic")

    def prepare_unspliced_genes(self, mp):
        """
        Initialize the unspliced parameters.

        Args:
            mp: Model parameters dictionary. Reads:
                - 'unspliced_param': "ratio" (default) or "kinetic" (legacy).
                - 'rhythmic_degradation': bool (default True). False is the null model
                  (constant degradation): delta = 0 ("ratio") or A_gamma = 0 ("kinetic").
                - 'u_level_init': optional [Ng] initial log(u/s) level ("ratio" only).
                - 'counts_u': unspliced library sizes, or None to reuse the spliced ones.
        """
        self.rhythmic_degradation = mp.get("rhythmic_degradation", True)
        self.unspliced_param = mp.get("unspliced_param", "ratio")
        if self.nh != 1:
            raise ValueError(
                "The unspliced model assumes a single-harmonic spliced curve (nh=1); "
                f"got nh={self.nh}."
            )

        self.register_buffer(
            "omega", tt(2.0 * torch.pi / 24.0, device=self.dev, dtype=torch.float32)
        )

        if self.unspliced_param == "ratio":
            self._prepare_ratio_params(mp)
        elif self.unspliced_param == "kinetic":
            self._prepare_kinetic_params()
        else:
            raise ValueError(
                f"Unknown unspliced_param '{self.unspliced_param}'. Use 'ratio' or 'kinetic'."
            )

        if mp.get("counts_u") is None:
            self.register_buffer("counts_u", self.counts)
            print("\nUsing SPLICED library size for unspliced counts\n")
        else:
            self.register_buffer("counts_u", mp.get("counts_u"))
            print("\nUsing provided UNSPLICED library size\n")

        # --- Dispersion ---
        if self.fix_disp_val == "gene":
            self.log_disp_u = nn.Parameter(-torch.ones(self.Ng))
        elif self.fix_disp_val is None:
            self.log_disp_u = nn.Parameter(tt(-1.0))
        elif self.fix_disp_val == "context":
            self.log_disp_u = nn.Parameter(-torch.ones(self.Ny, 1))
        else:
            self.log_disp_u = nn.Parameter(tt(np.log(self.fix_disp_val)))
            self.log_disp_u.requires_grad = (
                self.fix_disp_val is not None
            ) and not isinstance(self.fix_disp_val, (int, float))

    def _prepare_ratio_params(self, mp):
        """Ratio parametrization: lambda (log level), rho (depth), delta (phase mismatch)."""
        ones = torch.ones(self.Ng, device=self.dev, dtype=torch.float32)

        level_init = mp.get("u_level_init")
        if level_init is None:
            level_init = -1.0 * ones
        else:
            level_init = torch.as_tensor(
                level_init, dtype=torch.float32, device=self.dev
            ).reshape(self.Ng)
        self.u_log_level = nn.Parameter(level_init.clone())

        # rho = sigmoid(u_logit_rho) in (0, 1); start at rho = 0.3
        self.u_logit_rho = nn.Parameter(float(np.log(0.3 / 0.7)) * ones)

        # delta = 0 is the constant-degradation prediction; the null keeps it there
        if self.rhythmic_degradation:
            self.u_delta = nn.Parameter(torch.zeros_like(ones))
        else:
            self.register_buffer("u_delta", torch.zeros_like(ones))

    def _prepare_kinetic_params(self):
        """Legacy kinetic parametrization (beta, gamma_mean, A_gamma, phi_gamma)."""
        # --- 1. Splicing Rate (Beta) ---
        init_log_k_splice = 0.8 * torch.ones(self.Ng, device=self.dev, dtype=torch.float32)
        self.log_k_splice_g = nn.Parameter(init_log_k_splice)

        # --- 2. Degradation Amplitude (Epsilon / A_gamma) ---
        if self.rhythmic_degradation:
            # Full model: degradation is a learnable parameter
            self.raw_epsilon_gamma = nn.Parameter(
                -2.0 * torch.ones(self.Ng, device=self.dev, dtype=torch.float32)
            )
            # Phase of Degradation Rhythm (phi_gamma)
            self.phi_gamma_g = nn.Parameter(
                torch.zeros(self.Ng, device=self.dev, dtype=torch.float32)
            )
        else:
            # Null model: degradation amplitude fixed to 0
            self.register_buffer(
                "raw_epsilon_gamma",
                torch.zeros(self.Ng, device=self.dev, dtype=torch.float32)
            )
            self.register_buffer(
                "phi_gamma_g",
                torch.zeros(self.Ng, device=self.dev, dtype=torch.float32)
            )

        # --- 3. Gamma Mean Excess ---
        # gamma_mean = A_R + softplus(excess)
        init_excess_gamma = -0.5 * torch.ones(self.Ng, device=self.dev, dtype=torch.float32)
        self.param_excess_gamma = nn.Parameter(init_excess_gamma)

    ##################
    # Internal Calculation (Tensors with Gradients)
    ##################

    def _get_ratio_tensors(self):
        """
        Ratio-curve coefficients: h(theta) = level + r_cos cos(theta) + r_sin sin(theta),
        with level = exp(lambda), |r| = level * rho and angle(r) = psi = phi - pi/2 + delta.
        """
        level = torch.exp(self.u_log_level)
        rho = torch.sigmoid(self.u_logit_rho)
        delta = self.u_delta
        psi = self.acrophase - torch.pi / 2 + delta
        return {
            "level": level,
            "rho": rho,
            "delta": delta,
            "psi": psi,
            "r_cos": level * rho * torch.cos(psi),
            "r_sin": level * rho * torch.sin(psi),
        }

    def _get_gamma_kinetics_tensors(self):
        """
        Legacy kinetic parametrization. Calculates the kinetic parameters ensuring
        the positivity constraint:
        1. Vector D = Derivative oscillation
        2. Vector G = Gamma oscillation (or 0 if rhythmic_degradation=False)
        3. Vector R = D + G (Resultant)
        4. A_R = length(R)
        5. gamma_mean = A_R + softplus(excess)
        """
        # d(ln s)/dt = omega * (b_s * cos - a_s * sin)
        ab_s = self._get_ab()
        a_s = ab_s[0, :]
        b_s = ab_s[1, :]

        D_cos = self.omega * b_s
        D_sin = -self.omega * a_s

        if self.rhythmic_degradation:
            # Full model: A_gamma is learnable (constrained to 0-1 via sigmoid)
            A_gamma = torch.sigmoid(self.raw_epsilon_gamma)
            phi_gamma = self.phi_gamma_g
        else:
            # Null model: A_gamma = 0, phi_gamma doesn't matter
            A_gamma = torch.zeros_like(self.raw_epsilon_gamma)
            phi_gamma = self.phi_gamma_g

        G_cos = A_gamma * torch.cos(phi_gamma)
        G_sin = A_gamma * torch.sin(phi_gamma)

        R_cos = D_cos + G_cos
        R_sin = D_sin + G_sin
        A_R = torch.sqrt(R_cos**2 + R_sin**2)

        # gamma_mean > A_R ensures gamma_mean + R(t) > 0 everywhere.
        excess = F.softplus(self.param_excess_gamma)
        gamma_mean = A_R + excess + 1e-6

        return {
            "gamma_mean": gamma_mean,
            "A_gamma": A_gamma,
            "phi_gamma": phi_gamma,
            "R_cos": R_cos,
            "R_sin": R_sin,
            "k_splice": torch.exp(self.log_k_splice_g)
        }

    def _unspliced_ratio_curve(self, X):
        """h(theta) = u / s on the design X (either parametrization), shape of X[..., 0]."""
        cos_basis, sin_basis = X[..., 0], X[..., 1]
        if self._unspliced_param() == "ratio":
            k = self._get_ratio_tensors()
            return k["level"] + k["r_cos"] * cos_basis + k["r_sin"] * sin_basis

        k = self._get_gamma_kinetics_tensors()
        factor = k["gamma_mean"] + k["R_cos"] * cos_basis + k["R_sin"] * sin_basis
        # numerical safety only: gamma_mean > |R| makes it positive analytically
        factor = factor.clamp(min=1e-8)
        return factor / k["k_splice"]

    def _unspliced_formula(self, X, indices=slice(None), counts=None, n_theta=None):
        """
        Expected unspliced rate per unit library: u = s * h(theta).
        """
        spliced_rate_log, _, _ = self.model_formula(indices, counts, n_theta)
        spliced_rate = torch.exp(spliced_rate_log)
        h = self._unspliced_ratio_curve(X.unsqueeze(-2))
        return spliced_rate * h

    ##################
    # Post-Training / External Methods
    ##################

    def get_kinetic_parameters(self):
        """
        Per-gene unspliced parameters as a DataFrame.

        "ratio" parametrization. Fitted (identifiable) columns:
          - u_level: lambda = log of the mean u/s level (includes the capture ratio q)
          - rho: relative modulation depth of u/s, in (0, 1)
          - delta, delta_h: u/s peak phase minus the constant-degradation prediction
            (phi - pi/2), in rad and h, wrapped to [-pi, pi). 0 in the null.
          - psi_h: u/s peak phase [h]
        Derived columns (omega fixed gives absolute rates):
          - A_gamma_min = omega A |sin delta| [1/h]: smallest degradation rhythm
            compatible with the data. Identified; 0 in the null.
          - gamma_mean = omega A cos(delta) / rho [1/h]. EXACT in the null. With free
            delta it is a CONVENTION: the point of the beta-gamma ridge with the
            smallest degradation rhythm. NaN when not ``feasible``.
          - half_life_h = ln 2 / gamma_mean
          - beta_over_q = gamma_mean * exp(-lambda): splicing rate / capture ratio
          - feasible: cos(delta) > 0 and |tan delta| <= 1 / rho, i.e. the minimal-rhythm
            point has beta > 0 and gamma(theta) >= 0 everywhere.

        "kinetic" parametrization (legacy): gamma_mean, amp_gamma, phase_gamma, k_splice.
        """
        if self._unspliced_param() == "kinetic":
            k = self._get_gamma_kinetics_tensors()
            df = pd.DataFrame({
                "gamma_mean": nmp(k["gamma_mean"]),
                "amp_gamma": nmp(k["A_gamma"]),
                "phase_gamma": nmp(k["phi_gamma"]),
                "k_splice": nmp(k["k_splice"])
            }, index=self.genes)
            df["log2fc_gamma"] = np.log2((1 + df["amp_gamma"]) / (1 - df["amp_gamma"]))
            return df

        k = self._get_ratio_tensors()
        lam = nmp(self.u_log_level).astype(float)
        rho = nmp(k["rho"]).astype(float)
        delta = (nmp(k["delta"]).astype(float) + np.pi) % (2 * np.pi) - np.pi
        psi = nmp(k["psi"]).astype(float) % (2 * np.pi)
        omega_A = float(nmp(self.omega)) * nmp(self._amp_s()).astype(float)

        cos_d, sin_d = np.cos(delta), np.sin(delta)
        feasible = (cos_d > 0) & (np.abs(sin_d) * rho <= cos_d)
        gamma_mean = np.where(feasible, omega_A * cos_d / rho, np.nan)

        return pd.DataFrame({
            "u_level": lam,
            "rho": rho,
            "delta": delta,
            "delta_h": delta * rh,
            "psi_h": psi * rh,
            "A_gamma_min": omega_A * np.abs(sin_d),
            "gamma_mean": gamma_mean,
            "half_life_h": np.log(2) / gamma_mean,
            "beta_over_q": gamma_mean * np.exp(-lam),
            "feasible": feasible,
        }, index=self.genes)

    def analyze_rhythmic_dominance(self):
        """
        LEGACY ("kinetic" only). Analyzes whether rhythmic transcription or rhythmic
        degradation dominates the unspliced dynamics for each gene. For the "ratio"
        parametrization use :meth:`get_kinetic_parameters` (delta, A_gamma_min).

        Returns:
            pd.DataFrame with columns:
            - amp_s: Spliced (transcription) amplitude [log2FC]
            - phi_s: Spliced phase [rad]
            - A_D: Derivative amplitude [1/h] = omega * amp_s
            - phi_D: Derivative phase [rad] = phi_s + pi/2
            - A_gamma: Degradation amplitude [1/h]
            - phi_gamma: Degradation phase [rad]
            - A_R: Resultant amplitude [1/h]
            - rel_amp_ratio: A_gamma / A_D (relative amplitude ratio)
            - phase_diff_h: phi_gamma - phi_D [hours] (0-24h range)
            - deg_contrib_frac: Fraction of resultant from degradation (0-1)
            - dominance: Category: "transcription", "mixed", or "degradation"
        """
        if self._unspliced_param() != "kinetic":
            raise NotImplementedError(
                "analyze_rhythmic_dominance is for the legacy 'kinetic' parametrization; "
                "use get_kinetic_parameters() (delta, A_gamma_min) instead."
            )
        # --- 1. Get Spliced Parameters ---
        params_s = self.get_parameter_dataframe()
        amp_s_log2 = params_s["amp"].values  # log2 fold change
        phi_s = params_s["phase"].values  # radians

        # Convert amp_s to natural log space for derivative calculation
        amp_ln = amp_s_log2 * np.log(2)

        # --- 2. Derivative from Transcription (dE/dt) ---
        omega = nmp(self.omega)
        A_D = omega * amp_ln  # amplitude of derivative oscillation [1/h]
        phi_D = phi_s + np.pi / 2  # derivative leads by 90 degrees

        # --- 3. Degradation Parameters ---
        k = self._get_gamma_kinetics_tensors()
        A_gamma = nmp(k["A_gamma"]).squeeze()  # [1/h]
        phi_gamma = nmp(k["phi_gamma"]).squeeze()  # [rad]
        A_R = np.sqrt(nmp(k["R_cos"]).squeeze()**2 + nmp(k["R_sin"]).squeeze()**2)

        # --- 4. Relative Amplitude Ratio ---
        rel_amp_ratio = np.where(A_D > 1e-6, A_gamma / A_D, np.inf)

        # --- 5. Phase Difference ---
        phase_diff_rad = (phi_gamma - phi_D + np.pi) % (2 * np.pi) - np.pi
        phase_diff_h = phase_diff_rad * rh
        phase_diff_h = (phase_diff_h + 24) % 24  # Ensure 0-24 range

        # --- 6. Degradation Contribution Fraction ---
        D_cos = A_D * np.cos(phi_D)
        D_sin = A_D * np.sin(phi_D)
        G_cos = A_gamma * np.cos(phi_gamma)
        G_sin = A_gamma * np.sin(phi_gamma)
        R_cos = D_cos + G_cos
        R_sin = D_sin + G_sin

        dot_GR = G_cos * R_cos + G_sin * R_sin
        A_R_sq = np.maximum(R_cos**2 + R_sin**2, 1e-12)
        deg_contrib_frac = dot_GR / A_R_sq
        deg_contrib_frac = np.clip(deg_contrib_frac, 0, 1)

        # --- 7. Dominance Classification ---
        dominance = np.empty(len(self.genes), dtype=object)
        dominance[rel_amp_ratio < 0.5] = "transcription"
        dominance[rel_amp_ratio > 2.0] = "degradation"
        dominance[(rel_amp_ratio >= 0.5) & (rel_amp_ratio <= 2.0)] = "mixed"

        # --- 8. Create DataFrame ---
        df = pd.DataFrame(
            {
                "amp_s_log2fc": amp_s_log2,
                "phi_s_rad": phi_s,
                "phi_s_h": phi_s * rh,
                "A_D_1perh": A_D,
                "phi_D_h": (phi_D * rh) % 24,
                "A_gamma_1perh": A_gamma,
                "phi_gamma_h": (phi_gamma * rh) % 24,
                "A_R_1perh": A_R,
                "rel_amp_ratio": rel_amp_ratio,
                "phase_diff_h": phase_diff_h,
                "deg_contrib_frac": deg_contrib_frac,
                "dominance": dominance,
            },
            index=self.genes,
        )

        return df

    def extract_params_u(self, n_theta=24):
        """
        Same job as get_parameter_dataframe but for unspliced data.
        It needs the extras step of computing unspliced rates first,
        and a posteriori find the amplitude/phase from there.
        """

        phi_x = self.posterior_grid(n_theta)
        if phi_x is None:
            phi_x = np.linspace(0, 2 * np.pi, n_theta + 1)[:-1]
        X = self.X_matrix(fixed_cell_mode=False, n_theta=n_theta)
        u_xcg = self._unspliced_formula(X=X)
        u_xg = nmp(u_xcg[:,0,:])
        log_u_xg = np.log(u_xg)


        mean = log_u_xg.mean(axis=0)
        amp = (log_u_xg.max(axis=0) - log_u_xg.min(axis=0)) / 2
        peak_idx = log_u_xg.argmax(axis=0)
        phase = phi_x[peak_idx]
        # create df
        df = pd.DataFrame(
            {
                "a_0": mean,
                "amp": amp,
                "phase": phase,
            },
            index=self.genes,
        )
        par = sr.Beta(df)
        par.get_cartesian(inplace=True)
        par.get_amp(inplace=True)
        disp_u = nmp(self.log_disp_u.exp())
        par["disp"] = disp_u

        return par

    ##################
    # Model Selection / Comparison (Per Gene)
    ##################

    def compute_gene_log_likelihoods(self, data, data_u, indices=None):
        """
        Computes per-gene log-likelihoods for both spliced and unspliced data.

        Only a proper likelihood in fixed-cell mode (one known phase per cell). On a
        phase grid this sums log p(y | theta_x) over ALL grid points with equal
        weight, which is not a likelihood; genes are coupled through the unknown
        phase there, so a per-gene likelihood does not exist.

        Args:
            data: Spliced data tensor [Nx, Nc, Ng]
            data_u: Unspliced data tensor [Nx, Nc, Ng]
            indices: Optional cell indices to evaluate on subset

        Returns:
            dict with per-gene log-likelihoods:
                - total: [Ng] total LL per gene
                - spliced: [Ng] spliced LL per gene
                - unspliced: [Ng] unspliced LL per gene
        """
        if indices is None:
            indices = slice(None)
        with torch.no_grad():
            dist_s = self.nb_dist(indices=indices)
            ll_spliced = dist_s.log_prob(data)

            dist_u = self.nb_dist_unspliced(indices=indices)
            ll_unspliced = dist_u.log_prob(data_u)

            # Genes are always the last dimension
            ll_spliced_g = ll_spliced.sum(dim=tuple(range(ll_spliced.dim() - 1)))
            ll_unspliced_g = ll_unspliced.sum(dim=tuple(range(ll_unspliced.dim() - 1)))

            ll_total_g = ll_spliced_g + ll_unspliced_g

            return {
                "total": ll_total_g.cpu().numpy(),
                "spliced": ll_spliced_g.cpu().numpy(),
                "unspliced": ll_unspliced_g.cpu().numpy(),
            }

    def n_unspliced_params_per_gene(self):
        """
        Identifiable unspliced parameters per gene (excluding dispersion): 2 for the
        null, 3 with rhythmic degradation. In the legacy "kinetic" parametrization
        the rhythmic model has 4 raw parameters, but only one of (A_gamma, phi_gamma)
        is identifiable (beta-gamma ridge), so it counts 3 too.
        """
        return 2 + (1 if self.rhythmic_degradation else 0)

    def compute_bic_per_gene(self, data, data_u, indices=None):
        """
        Computes BIC per gene for model comparison.

        BIC_g = -2 * LL_g + k_g * log(n_g)

        Only valid in fixed-cell mode (see :meth:`compute_gene_log_likelihoods`). For
        null vs rhythmic degradation prefer :func:`unspliced_lrt`, which uses the
        exact 1-degree-of-freedom difference.

        Args:
            data: Spliced data tensor [Nx, Nc, Ng]
            data_u: Unspliced data tensor [Nx, Nc, Ng]
            indices: Optional cell indices

        Returns:
            pd.DataFrame with per-gene BIC results.
        """
        if not self.fixed_cell_mode:
            print(
                "Warning: compute_bic_per_gene on a phase grid is not a likelihood; "
                "use fixed-cell mode (fixed_cell_phases=...)."
            )
        ll_dict = self.compute_gene_log_likelihoods(data, data_u, indices=indices)

        # For simplicity, we attribute an equal share of shared parameters to each gene
        Ny = self.Ny

        n_params_spliced_per_gene = (
            1 +      # m_g (a_0)
            1 +      # log_amp
            1 +      # acrophase
            (1 if self.fix_disp_val == "gene" else 1/self.Ng)  # share of log_disp
        )

        # Context parameters (attributed per gene)
        n_params_context_per_gene = (
            Ny +     # m_yg per context
            (Ny if self.context_mode == "full_lambda" else 1)  # log_lambda_y
        )

        n_params_unspliced_per_gene = (
            self.n_unspliced_params_per_gene()
            + (1 if self.fix_disp_val == "gene" else 1/self.Ng)  # share of log_disp_u
        )

        n_params_per_gene = (
            n_params_spliced_per_gene +
            n_params_context_per_gene +
            n_params_unspliced_per_gene
        )

        # Observations per gene: Nx * Nc * 2 (spliced + unspliced)
        n_obs_per_gene = data.shape[0] * data.shape[1] + data_u.shape[0] * data_u.shape[1]

        bic_per_gene = -2 * ll_dict["total"] + n_params_per_gene * np.log(n_obs_per_gene)

        bic_per_gene = np.atleast_1d(bic_per_gene).flatten()
        ll_total = np.atleast_1d(ll_dict["total"]).flatten()
        ll_spliced = np.atleast_1d(ll_dict["spliced"]).flatten()
        ll_unspliced = np.atleast_1d(ll_dict["unspliced"]).flatten()
        n_params_per_gene = np.atleast_1d(n_params_per_gene).flatten()

        df = pd.DataFrame({
            "bic": bic_per_gene,
            "log_likelihood_total": ll_total,
            "log_likelihood_spliced": ll_spliced,
            "log_likelihood_unspliced": ll_unspliced,
            "n_params": n_params_per_gene,
            "n_obs": n_obs_per_gene,
            "model_type": "alternative" if self.rhythmic_degradation else "null",
        }, index=self.genes)

        return df

    ##################
    # Distribution
    ##################

    def nb_dist_unspliced(self, indices=slice(None), counts=None, n_theta=None):
        if counts is None:
            counts = self.counts_u[indices]

        if n_theta is not None:
            phi_x_new = self.phase_grid(n_theta, device=self.dev)
            X_new = harmonic_dm_torch(phi_x_new, self.nh, False)
            X = X_new.unsqueeze(1).expand(n_theta, self.Nc, self.nh * 2)
            X = X[:, indices, :]
        else:
            X = self.X[:, indices, :]

        disp = torch.exp(self.log_disp_u)
        E_xcg = self._unspliced_formula(X, indices, counts, n_theta)

        if self.noise_model == "nb":
            E_xcg = E_xcg * counts
            r = 1 / disp
            eps = 1e-6
            p = disp * E_xcg / (1 + disp * E_xcg)
            p = p.clamp(min=eps, max=1 - eps)
            return torch.distributions.NegativeBinomial(total_count=r, probs=p)

        elif self.noise_model == "poisson":
            E_xcg = E_xcg * counts
            return torch.distributions.Poisson(rate=E_xcg)

        else:
            raise NotImplementedError(f"Noise model '{self.noise_model}' is not implemented.")


def refine_mle(model, data, data_u=None, max_iter=500, tol=1e-9):
    """
    Polish a fixed-cell-mode fit to the exact maximum likelihood with full-batch L-BFGS.

    Minibatch Adam leaves the log-likelihood jittering by O(10) units, the same size
    as a likelihood-ratio signal, so run this on both models before
    :func:`unspliced_lrt`. Trains every parameter with ``requires_grad``, in place.

    Returns:
        The final loss (negative log-likelihood plus any prior terms).
    """
    if not model.fixed_cell_mode:
        raise ValueError("refine_mle is for fixed-cell mode (one known phase per cell).")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.LBFGS(
        params, lr=1.0, max_iter=max_iter, tolerance_grad=tol,
        tolerance_change=tol, history_size=50, line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = model(data, y_u=data_u, indices=slice(None))
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return model(data, y_u=data_u, indices=slice(None)).item()


def unspliced_lrt(null_model, alt_model, data, data_u):
    """
    Per-gene likelihood-ratio test of constant vs rhythmic degradation.

    Both models must be fitted in fixed-cell mode on the same cells (then genes are
    independent and the per-gene likelihood is exact) and polished with
    :func:`refine_mle`, with ``rhythmic_degradation``
    False for ``null_model`` and True for ``alt_model``. In the "ratio"
    parametrization they differ by the single identifiable parameter delta, and
    delta = 0 is an interior point of a regular model, so the statistic is
    asymptotically chi^2 with 1 degree of freedom.

    Returns:
        pd.DataFrame indexed by gene: ll_null, ll_alt, lrt (2 * gain, clipped at 0),
        pval, delta_bic (> 0 favours rhythmic degradation; 1-parameter penalty).
    """
    for mdl in (null_model, alt_model):
        if not mdl.fixed_cell_mode:
            raise ValueError("unspliced_lrt needs models fitted in fixed-cell mode.")
    if null_model.rhythmic_degradation or not alt_model.rhythmic_degradation:
        raise ValueError("Pass (null: rhythmic_degradation=False, alt: True).")

    ll0 = null_model.compute_gene_log_likelihoods(data, data_u)["total"]
    ll1 = alt_model.compute_gene_log_likelihoods(data, data_u)["total"]
    lrt = np.clip(2 * (ll1 - ll0), 0, None)
    n_obs = data.shape[0] * data.shape[1] + data_u.shape[0] * data_u.shape[1]
    return pd.DataFrame({
        "ll_null": ll0,
        "ll_alt": ll1,
        "lrt": lrt,
        "pval": chi2.sf(lrt, df=1),
        "delta_bic": 2 * (ll1 - ll0) - np.log(n_obs),
    }, index=alt_model.genes)


def min_gamma(log2fc):
    amp = log2fc / (np.log2(np.e)*2)
    gamma_min = amp * w
    return float(gamma_min)

def max_half_life(log2fc):
    gamma_min = min_gamma(log2fc)
    half_life_max = np.log(2) / gamma_min
    return float(half_life_max)
