import numpy as np
import torch
from torch import tensor as tt
from torch import nn
from scritmo import Beta, optimal_shift, w, rh
import pandas as pd
from ..utils import harmonic_dm_torch, nmp
import scritmo as sr



class UnsplicedMixin:

    ##################
    # methods called by context_model
    ##################

    def prepare_unspliced_genes(self, mp):
        # New kinetic parameters (per-gene)
        # Initial log(gamma) ~ log(0.17) -> T1/2 ~ 4h
        # Initial log(k_splice) ~ log(2.3) -> T1/2 ~ 18min
        init_log_k_splice = 0.8 * torch.ones(self.Ng, device=self.dev, dtype=torch.float32)
        self.log_k_splice_g = nn.Parameter(init_log_k_splice)

        init_excess_gamma = -1.0 * torch.ones(self.Ng, device=self.dev, dtype=torch.float32)
        self.param_excess_gamma = nn.Parameter(init_excess_gamma)

        # Omega constant (2pi / 24h)
        self.register_buffer(
            "omega", tt(2.0 * torch.pi / 24.0, device=self.dev, dtype=torch.float32)
        )

        # Buffer for counts (per cell) for unspliced
        if mp["counts_u"] is None:
            self.register_buffer("counts_u", self.counts)
            print("\nUsing SPLICED library size for unspliced counts\n")
        else:
            self.register_buffer("counts_u", mp.get("counts_u"))
            print("\nUsing provided UNSPLICED library size\n")

        # Independent dispersion for unspliced data (logic remains the same)
        if self.fix_disp_val == "gene":
            self.log_disp_u = nn.Parameter(-torch.ones(self.Ng))
        elif self.fix_disp_val is None:
            self.log_disp_u = nn.Parameter(tt(-1.0))
        elif self.fix_disp_val == "context":
            # Assuming context-disp for unspliced follows same logic
            self.log_disp_u = nn.Parameter(-torch.ones(self.Ny, 1))
        else:
            self.log_disp_u = nn.Parameter(tt(np.log(self.fix_disp_val)))
            # Make sure requires_grad is set correctly if fix_disp_val is a number
            self.log_disp_u.requires_grad = (
                self.fix_disp_val is not None
            ) and not isinstance(self.fix_disp_val, (int, float))

    def nb_dist_unspliced(self, indices=slice(None), counts=None, n_theta=None):
        """
        Computes the likelihood distribution (Negative Binomial or Poisson)
        for the given data.

        Called by several methods.
        """

        # --- Common computations for the expected mean (rate) ---
        if counts is None:
            counts = self.counts_u[indices]
        else:
            print("not implemented")

        if n_theta is not None:

            phi_x_new = torch.linspace(
                0, 2 * torch.pi, n_theta + 1, dtype=torch.float32, device=self.dev
            )[:-1]
            X_new = harmonic_dm_torch(phi_x_new, self.nh, False)
            X = X_new.unsqueeze(1).expand(n_theta, self.Nc, self.nh * 2)
            X = X[:, indices, :]
        else:
            X = self.X[:, indices, :]

        dm = self.dm[indices, :]
        disp = torch.exp(self.log_disp_u)

        # E_xcg is the expected mean of the distribution
        E_xcg = self.unspliced_formula(X, indices, counts, n_theta)

        # --- Select distribution based on the noise model ---
        if self.noise_model == "nb":
            E_xcg = E_xcg * counts
            # Negative Binomial distribution
            r = 1 / disp
            eps = 1e-6
            p = disp * E_xcg / (1 + disp * E_xcg)
            p = p.clamp(min=eps, max=1 - eps)

            return torch.distributions.NegativeBinomial(total_count=r, probs=p)

        elif self.noise_model == "poisson":
            E_xcg = E_xcg * counts
            # Poisson distribution, where the rate is the expected mean
            return torch.distributions.Poisson(rate=E_xcg)

        elif self.noise_model == "gaussian":
            # return not implemented
            raise NotImplementedError("Gaussian noise model is not implemented yet.")

        else:
            raise NotImplementedError(
                f"Noise model '{self.noise_model}' is not implemented."
            )
        
    ################
    # methods used internally
    ################

    def gamma_rate(self, return_ab=False):
        """
        Returns the gamma splicing rates for all genes.
        This method is both used by unspliced_model for evaualtion
        and also post trianing to get gamma_values
        
        """
        ab = self._get_ab() # Shape (2, Ng) -> a_1=row0, b_1=row1
        a_1 = ab[0, :].unsqueeze(0).unsqueeze(0) # (1, 1, Ng)
        b_1 = ab[1, :].unsqueeze(0).unsqueeze(0) # (1, 1, Ng)

        # 4. Enforce Positivity Constraint (gamma > max(dE/dt))
        # For a single harmonic, max(dE/dt) = amp * omega
        amp = torch.sqrt(a_1**2 + b_1**2)
        threshold = amp * self.omega

        # 6. Final Formula: u = (s / beta) * (gamma - dE/dt)
        # Note: (gamma - dE/dt) is guaranteed positive now because:
        # gamma > amp * omega >= max(dE/dt)
        epsilon = 0
        gamma_g = threshold + torch.nn.functional.softplus(self.param_excess_gamma) + epsilon
        if return_ab:    
            return gamma_g, a_1, b_1
        else:
            return nmp(gamma_g).squeeze()

        
    def _unspliced_formula(self, X, indices=slice(None), counts=None, n_theta=None):
        # 1. Get Spliced Dynamics
        spliced_rate_log, _, _ = self.model_formula(indices, counts, n_theta)
        spliced_rate = torch.exp(spliced_rate_log)
        
        # However, using your existing X structure (assuming order is cos, sin):
        cos_basis, sin_basis = X.chunk(2, dim=-1) 
        
        # We need to reconstruct the derivative of the exponent.
        # The exponent is: a_1 * cos_basis + b_1 * sin_basis + a_0
        # The derivative is: a_1 * (-w * sin_basis) + b_1 * (w * cos_basis)
        
        # Let's get alpha and beta coefficients from the model
        gamma_g, a_1, b_1 = self.gamma_rate(return_ab=True)

        # Derivative of exponent dE/dt
        dE_dt = self.omega * (-a_1 * sin_basis + b_1 * cos_basis)

        # 5. Splicing Rate (beta parameter)
        k_splice = torch.exp(self.log_k_splice_g)
 
        unspliced_rate = (spliced_rate / k_splice) * (gamma_g + dE_dt)

        return unspliced_rate
        
    def extract_params_u(self):
        """
        Same job as get_parameter_dataframe but for unspliced data.
        It needs the extras step of computing unspliced rates first,
        and a posteriori find the amplitude/phase from there.
        """

        phi_x = np.linspace(0, 2 * np.pi, self.Nx + 1)[:-1]
        u_xcg = self._unspliced_formula(X=self.X)
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
