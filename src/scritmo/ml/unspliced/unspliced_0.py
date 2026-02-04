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
        # Simple harmonic parameterization for unspliced counts
        # m_u_g: baseline (a_0) for unspliced per gene
        # log_u_amp: parameterization of amplitude (same encoding as spliced)
        # acrophase_u: acrophase for unspliced

        # Initialize baseline from provided params_g a_0 (fallback)
        self.m_u_g = nn.Parameter(tt(mp["a_0_u"], dtype=torch.float32))

        # Initialize amplitude using same values as spliced if available
        amp_values = tt(mp.get("params_g")["amp"].values, dtype=torch.float32)
        if self.log_amp_fn == "logit":
            safe_amp = torch.clamp(amp_values, min=1e-2, max=self.max_amp - 1e-2)
            log_u_amp = torch.logit(safe_amp / self.max_amp)
            self.log_u_amp = nn.Parameter(log_u_amp)
        else:
            log_u_amp = torch.log(amp_values)
            self.log_u_amp = nn.Parameter(log_u_amp)

        # Initialize acrophase for unspliced to be one hour earlier than spliced
        phase_values = mp.get("params_g")["phase"].values
        # Subtract one hour (assuming phase is in hours, and 24-hour cycle)
        acrophase_tensor = tt((phase_values - 1*w) % (2*np.pi), dtype=torch.float32)
        if self.fix_phase:
            self.register_buffer("acrophase_u", acrophase_tensor)
        else:
            self.acrophase_u = nn.Parameter(acrophase_tensor)

        # Buffer for counts (per cell) for unspliced
        if mp["counts_u"] is None:
            self.register_buffer("counts_u", self.counts)
            print("\nUsing SPLICED library size for unspliced counts\n")
        else:
            self.register_buffer("counts_u", mp.get("counts_u"))
            print("\nUsing provided UNSPLICED library size\n")

        # Dispersion for unspliced
        if self.fix_disp_val == "gene":
            self.log_disp_u = nn.Parameter(-torch.ones(self.Ng))
        elif self.fix_disp_val is None:
            self.log_disp_u = nn.Parameter(tt(-1.0))
        elif self.fix_disp_val == "context":
            self.log_disp_u = nn.Parameter(-torch.ones(self.Ny, 1))
        else:
            self.log_disp_u = nn.Parameter(tt(np.log(self.fix_disp_val)))
            self.log_disp_u.requires_grad = False

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

        # Build expected mean using dedicated unspliced harmonic parameters
        ab = self._get_ab_u()

        # E_xcg is the expected mean of the distribution
        E_xcg = (X @ ab) + self.m_u_g

        # --- Select distribution based on the noise model ---
        if self.noise_model == "nb":
            E_xcg = torch.exp(E_xcg) * counts
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
        # Deprecated in simpler unspliced model: placeholder to avoid API breakage
        raise NotImplementedError("gamma_rate is not available in the simplified unspliced mixin")

        
    def unspliced_formula(self, X, indices=slice(None), counts=None, n_theta=None):
        # Deprecated in simplified model; keep for compatibility
        raise NotImplementedError("unspliced_formula is not available in the simplified unspliced mixin")

    def _amp_u(self):
        if self.log_amp_fn == "logit":
            amp = torch.sigmoid(self.log_u_amp) * self.max_amp
            return amp
        elif self.log_amp_fn == "log":
            amp = torch.exp(self.log_u_amp)
            return amp
        
    def _get_ab_u(self):
        amp = self._amp_u()
        cos = amp * torch.cos(self.acrophase_u).unsqueeze(0)
        sin = amp * torch.sin(self.acrophase_u).unsqueeze(0)
        ab = torch.cat([cos, sin], dim=0)
        return ab

    def extract_params_u(self):
        return self.get_parameter_dataframe(unspliced=True)