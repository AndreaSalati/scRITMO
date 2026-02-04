import numpy as np
import torch
from torch import tensor as tt
from torch import nn
import torch.nn.functional as F
from scritmo import Beta, optimal_shift, w, rh
import pandas as pd
from ..utils import harmonic_dm_torch, nmp
import scritmo as sr


class UnsplicedMixin:
    def prepare_unspliced_genes(self, mp):
        """
        Initializes parameters for the unspliced model with rhythmic degradation.
        Equation: u = (s / beta) * (gamma(t) + d(ln s)/dt)
        
        Logic:
        1. We have an oscillation from degradation: Gamma_osc(t)
        2. We have an oscillation from splicing derivative: dE/dt(t)
        3. The sum is a Resultant Oscillation: R(t) with amplitude A_R
        4. To ensure positivity, gamma_mean must be > A_R.
        
        Args:
            mp: Model parameters dictionary. Can include:
                - 'rhythmic_degradation': bool (default True)
                  If False, degradation amplitude is fixed to 0 (null model)
        """
        
        # --- 0. Toggle for Model Selection ---
        self.rhythmic_degradation = mp.get("rhythmic_degradation", True)
        
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
        # We start with a healthy excess
        init_excess_gamma = -0.5 * torch.ones(self.Ng, device=self.dev, dtype=torch.float32)
        self.param_excess_gamma = nn.Parameter(init_excess_gamma)

        # --- 4. Constants and Buffers ---
        self.register_buffer(
            "omega", tt(2.0 * torch.pi / 24.0, device=self.dev, dtype=torch.float32)
        )

        if mp["counts_u"] is None:
            self.register_buffer("counts_u", self.counts)
            print("\nUsing SPLICED library size for unspliced counts\n")
        else:
            self.register_buffer("counts_u", mp.get("counts_u"))
            print("\nUsing provided UNSPLICED library size\n")

        # --- 5. Dispersion ---
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

    ##################
    # Internal Calculation (Tensors with Gradients)
    ##################

    def _get_gamma_kinetics_tensors(self):
        """
        Calculates the kinetic parameters ensuring the positivity constraint.
        Analytical steps:
        1. Vector D = Derivative oscillation
        2. Vector G = Gamma oscillation (or 0 if rhythmic_degradation=False)
        3. Vector R = D + G (Resultant)
        4. A_R = length(R)
        5. gamma_mean = A_R + softplus(excess)
        """
        
        # --- A. Splicing Derivative Vector (D) ---
        # s(t) ~ exp(a_s * cos + b_s * sin)
        # d(ln s)/dt = omega * (b_s * cos - a_s * sin)
        # In terms of cosine/sine coefficients:
        # D_cos = omega * b_s
        # D_sin = -omega * a_s
        
        ab_s = self._get_ab() 
        a_s = ab_s[0, :] 
        b_s = ab_s[1, :] 
        
        D_cos = self.omega * b_s
        D_sin = -self.omega * a_s

        # --- B. Gamma Vector (G) ---
        # gamma_osc(t) = A_gamma * cos(wt - phi)
        #              = (A_gamma cos_phi) * cos + (A_gamma sin_phi) * sin
        
        if self.rhythmic_degradation:
            # Full model: A_gamma is learnable (constrained to 0-1 via sigmoid)
            A_gamma = torch.sigmoid(self.raw_epsilon_gamma)
            phi_gamma = self.phi_gamma_g
        else:
            # Null model: A_gamma = 0, phi_gamma doesn't matter
            A_gamma = torch.zeros_like(self.raw_epsilon_gamma)
            phi_gamma = self.phi_gamma_g  # Still need for return, but won't affect calc
        
        G_cos = A_gamma * torch.cos(phi_gamma)
        G_sin = A_gamma * torch.sin(phi_gamma)
        
        # --- C. Resultant Vector (R) ---
        # Sum of coefficients
        R_cos = D_cos + G_cos
        R_sin = D_sin + G_sin
        
        # Resultant Amplitude A_R
        A_R = torch.sqrt(R_cos**2 + R_sin**2)
        
        # --- D. Gamma Mean Constraint ---
        # gamma_mean must be strictly greater than A_R to ensure
        # gamma_mean + R(t) > 0 everywhere.
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

    def _unspliced_formula(self, X, indices=slice(None), counts=None, n_theta=None):
        """
        Calculates the expected unspliced rate.
        u = (s/beta) * (gamma_mean + gamma_osc(t) + dE/dt)
        u = (s/beta) * (gamma_mean + R_cos * cos + R_sin * sin)
        """
        # 1. Spliced Dynamics
        spliced_rate_log, _, _ = self.model_formula(indices, counts, n_theta)
        spliced_rate = torch.exp(spliced_rate_log)

        # 2. Basis
        cos_basis, sin_basis = X.chunk(2, dim=-1) 
        
        # 3. Kinetics
        k = self._get_gamma_kinetics_tensors()
        
        # 4. Resultant Oscillation (gamma_osc + dE/dt)
        # We calculated the summed vector coefficients R_cos/R_sin analytically above
        resultant_osc = k["R_cos"] * cos_basis + k["R_sin"] * sin_basis
        
        # 5. Factor & Rate
        # gamma_mean is guaranteed > amplitude of resultant_osc
        factor = k["gamma_mean"] + resultant_osc
        
        # Additional clamp just for numerical safety (though analytically positive)
        factor = factor.clamp(min=1e-8)
        
        unspliced_rate = (spliced_rate / k["k_splice"]) * factor

        return unspliced_rate

    ##################
    # Post-Training / External Methods
    ##################

    def get_kinetic_parameters(self):
        """
        Returns the learned kinetic parameters as a nice DataFrame.
        Includes:
        - gamma_mean: The average degradation rate
        - amp_gamma: The amplitude of the degradation rhythm
        - phase_gamma: The phase of the degradation rhythm
        - k_splice: The splicing rate constant
        """
        k = self._get_gamma_kinetics_tensors()
        
        df = pd.DataFrame({
            "gamma_mean": nmp(k["gamma_mean"]),
            "amp_gamma": nmp(k["A_gamma"]),
            "phase_gamma": nmp(k["phi_gamma"]),
            "k_splice": nmp(k["k_splice"])
        }, index=self.genes)
        df["log2fc_gamma"] = np.log2((1 + df["amp_gamma"]) / (1 - df["amp_gamma"]))
        
        return df

    def analyze_rhythmic_dominance(self):
        """
        Analyzes whether rhythmic transcription or rhythmic degradation dominates
        the unspliced dynamics for each gene.

        The model equation is:
            u = (s / beta) * (gamma_mean + gamma_osc(t) + d(ln s)/dt)

        The oscillatory part of the unspliced comes from two sources:
        1. Transcription derivative: d(ln s)/dt with amplitude A_D = omega * amp_s
        2. Degradation rhythm: gamma_osc(t) with amplitude A_gamma

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
        # --- 1. Get Spliced Parameters ---
        # amp_s is in log2FC space, need to convert to linear for derivative calc
        params_s = self.get_parameter_dataframe()
        amp_s_log2 = params_s["amp"].values  # log2 fold change
        phi_s = params_s["phase"].values  # radians

        # Convert amp_s to natural log space for derivative calculation
        # log2FC -> lnFC: lnFC = log2FC * ln(2)
        amp_ln = amp_s_log2 * np.log(2)

        # --- 2. Derivative from Transcription (dE/dt) ---
        # d(ln s)/dt has amplitude: omega * amp_ln (since s ~ exp(a*cos + b*sin))
        # But in the code, the derivative amplitude is computed as:
        # A_D = omega * sqrt(a_s^2 + b_s^2) = omega * amp_ln
        omega = nmp(self.omega)
        A_D = omega * amp_ln  # amplitude of derivative oscillation [1/h]
        phi_D = phi_s + np.pi / 2  # derivative leads by 90 degrees

        # --- 3. Degradation Parameters ---
        k = self._get_gamma_kinetics_tensors()
        A_gamma = nmp(k["A_gamma"]).squeeze()  # [1/h]
        phi_gamma = nmp(k["phi_gamma"]).squeeze()  # [rad]
        A_R = np.sqrt(nmp(k["R_cos"]).squeeze()**2 + nmp(k["R_sin"]).squeeze()**2)

        # --- 4. Relative Amplitude Ratio ---
        # Use safe division to handle near-zero transcription amplitudes
        rel_amp_ratio = np.where(A_D > 1e-6, A_gamma / A_D, np.inf)

        # --- 5. Phase Difference ---
        # Phase difference in radians, wrapped to [-pi, pi]
        phase_diff_rad = (phi_gamma - phi_D + np.pi) % (2 * np.pi) - np.pi
        # Convert to hours (0-24 range, where 0 = in-phase, 12 = anti-phase)
        phase_diff_h = phase_diff_rad * rh
        phase_diff_h = (phase_diff_h + 24) % 24  # Ensure 0-24 range

        # --- 6. Degradation Contribution Fraction ---
        # Compute what fraction of the resultant comes from degradation vs derivative
        # In vector terms: R = D + G
        # The contribution can be quantified via the law of cosines:
        # A_R^2 = A_D^2 + A_gamma^2 + 2*A_D*A_gamma*cos(phase_diff)
        # We compute the projection of G onto R as a proxy for contribution

        # Vector components
        D_cos = A_D * np.cos(phi_D)
        D_sin = A_D * np.sin(phi_D)
        G_cos = A_gamma * np.cos(phi_gamma)
        G_sin = A_gamma * np.sin(phi_gamma)
        R_cos = D_cos + G_cos
        R_sin = D_sin + G_sin

        # Projection of G onto R: (G · R) / |R|^2 * |R| = (G · R) / |R|
        # Contribution fraction: |projection| / |R| = (G · R) / |R|^2
        dot_GR = G_cos * R_cos + G_sin * R_sin
        A_R_sq = np.maximum(R_cos**2 + R_sin**2, 1e-12)
        deg_contrib_frac = dot_GR / A_R_sq

        # Clamp to [0, 1] for interpretability (projection can be negative if anti-phase)
        deg_contrib_frac = np.clip(deg_contrib_frac, 0, 1)

        # --- 7. Dominance Classification ---
        # Based on relative amplitude ratio and phase alignment
        dominance = np.empty(len(self.genes), dtype=object)

        # Transcription-dominated: small degradation amplitude
        dominance[rel_amp_ratio < 0.5] = "transcription"

        # Degradation-dominated: large degradation amplitude
        dominance[rel_amp_ratio > 2.0] = "degradation"

        # Mixed: intermediate ratio
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
        with torch.no_grad():
            # Spliced likelihood - shape may vary due to broadcasting
            dist_s = self.nb_dist(indices=indices)
            ll_spliced = dist_s.log_prob(data)
            
            # Unspliced likelihood
            dist_u = self.nb_dist_unspliced(indices=indices)
            ll_unspliced = dist_u.log_prob(data_u)
            
            # Sum over all dimensions except the last (genes dimension)
            # Genes are always the last dimension
            ll_spliced_g = ll_spliced.sum(dim=tuple(range(ll_spliced.dim() - 1)))
            ll_unspliced_g = ll_unspliced.sum(dim=tuple(range(ll_unspliced.dim() - 1)))
            
            ll_total_g = ll_spliced_g + ll_unspliced_g
            
            return {
                "total": ll_total_g.cpu().numpy(),
                "spliced": ll_spliced_g.cpu().numpy(),
                "unspliced": ll_unspliced_g.cpu().numpy(),
            }

    def compute_bic_per_gene(self, data, data_u, indices=None):
        """
        Computes BIC per gene for model comparison.
        
        Each gene has its own set of parameters, so BIC should be computed
        gene-by-gene to determine which model fits each gene better.
        
        BIC_g = -2 * LL_g + k_g * log(n_g)
        
        where:
        - LL_g = sum over all cells of log P(y_cg, y_u_cg | params_g)
        - k_g = number of parameters for gene g
        - n_g = number of observations for gene g (N_cells * 2 for spliced + unspliced)
        
        Args:
            data: Spliced data tensor [Nx, Nc, Ng]
            data_u: Unspliced data tensor [Nx, Nc, Ng]
            indices: Optional cell indices
            
        Returns:
            pd.DataFrame with per-gene BIC results:
                - gene: gene name
                - bic: BIC value
                - log_likelihood_total: total LL
                - log_likelihood_spliced: spliced LL
                - log_likelihood_unspliced: unspliced LL
                - n_params: number of parameters for this gene
                - n_obs: number of observations
        """
        # Get per-gene likelihoods
        ll_dict = self.compute_gene_log_likelihoods(data, data_u, indices=indices)
        
        # Per-gene parameter counts
        # Spliced params per gene: m_g (1), log_amp (1), acrophase (1), log_disp (1/Ng or 1)
        # + context params: m_yg (Ny), log_lambda_y (Ny or 1)
        
        # For simplicity, we attribute an equal share of shared parameters to each gene
        Ny = self.Ny  # Number of contexts
        
        # Spliced parameters per gene
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
        
        # Unspliced parameters per gene
        n_params_unspliced_per_gene = (
            1 +      # log_k_splice
            1 +      # param_excess_gamma
            (1 if self.fix_disp_val == "gene" else 1/self.Ng)  # share of log_disp_u
        )
        
        if self.rhythmic_degradation:
            n_params_unspliced_per_gene += 2  # raw_epsilon_gamma + phi_gamma_g
        
        n_params_per_gene = (
            n_params_spliced_per_gene + 
            n_params_context_per_gene + 
            n_params_unspliced_per_gene
        )
        
        # Observations per gene: Nx * Nc * 2 (spliced + unspliced)
        n_obs_per_gene = data.shape[0] * data.shape[1] + data_u.shape[0] * data_u.shape[1]
        
        # Compute BIC per gene
        bic_per_gene = -2 * ll_dict["total"] + n_params_per_gene * np.log(n_obs_per_gene)
        
        # Ensure all arrays are 1D
        bic_per_gene = np.atleast_1d(bic_per_gene).flatten()
        ll_total = np.atleast_1d(ll_dict["total"]).flatten()
        ll_spliced = np.atleast_1d(ll_dict["spliced"]).flatten()
        ll_unspliced = np.atleast_1d(ll_dict["unspliced"]).flatten()
        n_params_per_gene = np.atleast_1d(n_params_per_gene).flatten()
        
        # Create DataFrame
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
            phi_x_new = torch.linspace(
                0, 2 * torch.pi, n_theta + 1, dtype=torch.float32, device=self.dev
            )[:-1]
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

def min_gamma(log2fc):
    amp = log2fc / (np.log2(np.e)*2)
    gamma_min = amp * w
    return float(gamma_min)

def max_half_life(log2fc):
    gamma_min = min_gamma(log2fc)
    half_life_max = np.log(2) / gamma_min
    return float(half_life_max)