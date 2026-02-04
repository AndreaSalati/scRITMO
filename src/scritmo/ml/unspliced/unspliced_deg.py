import numpy as np
import torch
from torch import tensor as tt
from torch import nn
import torch.nn.functional as F
from scritmo import Beta, optimal_shift, w
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
        """
        
        # --- 1. Splicing Rate (Beta) ---
        init_log_k_splice = 0.8 * torch.ones(self.Ng, device=self.dev, dtype=torch.float32)
        self.log_k_splice_g = nn.Parameter(init_log_k_splice)

        # --- 2. Degradation Amplitude (Epsilon / A_gamma) ---
        # User requested relative amplitude epsilon between 0 and 1.
        # Since degradation rates are usually < 1.0 (e.g. 0.17 for 4h half-life),
        # 0-1 is a safe range for the amplitude A_gamma.
        self.raw_epsilon_gamma = nn.Parameter(
            -2.0 * torch.ones(self.Ng, device=self.dev, dtype=torch.float32)
        )
        
        # Phase of Degradation Rhythm (phi_gamma)
        self.phi_gamma_g = nn.Parameter(
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
        2. Vector G = Gamma oscillation
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
        
        # A_gamma restricted to (0, 1) using sigmoid
        A_gamma = torch.sigmoid(self.raw_epsilon_gamma)
        phi_gamma = self.phi_gamma_g
        
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