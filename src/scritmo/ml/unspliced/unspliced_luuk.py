import numpy as np
import torch
from torch import tensor as tt
from torch import nn
from scritmo import Beta, optimal_shift, w
import pandas as pd
from ..utils import harmonic_dm_torch

class UnsplicedLuukMixin:
    def prepare_unspliced_genes(self, mp):
        # New kinetic parameters (per-gene)
        # Initial log(gamma) ~ log(0.17) -> T1/2 ~ 4h
        init_log_gamma = -1.8 * torch.ones(self.Ng, device=self.dev, dtype=torch.float32)
        # Initial log(k_splice) ~ log(2.3) -> T1/2 ~ 18min
        init_log_k_splice = 0.8 * torch.ones(self.Ng, device=self.dev, dtype=torch.float32)

        self.log_gamma_g = nn.Parameter(init_log_gamma)
        self.log_k_splice_g = nn.Parameter(init_log_k_splice)

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


    def _compute_derived_unspliced_params(self):
        """
        Computes derived unspliced parameters (mean, amplitude, acrophase)
        from spliced parameters and kinetic rates (gamma, k_splice).
        """
        # Get base spliced parameters
        if self.log_amp_fn == "logit":
            amp_s_g = torch.sigmoid(self.log_amp) * self.max_amp
        elif self.log_amp_fn == "log":
            amp_s_g = torch.exp(self.log_amp)
        m_s_g = self.m_g
        acrophase_s_g = self.acrophase

        # Get kinetic parameters (ensure positivity)
        gamma_g = torch.exp(self.log_gamma_g)
        k_splice_g = torch.exp(self.log_k_splice_g)

        # Apply linking equations
        # 1. Mean
        m_u_g = (gamma_g * m_s_g) / k_splice_g

        # 2. Amplitude
        amp_scaling_factor = torch.sqrt(gamma_g**2 + self.omega**2) / k_splice_g
        amp_u_g = amp_s_g * amp_scaling_factor

        # 3. Acrophase
        shift_u = torch.arctan(self.omega / gamma_g)
        acrophase_u = acrophase_s_g - shift_u

        return m_u_g, amp_u_g, acrophase_u


    def _get_ab_u(self):
        # Get derived amplitude and acrophase
        _, amp_u_g, acrophase_u = self._compute_derived_unspliced_params()

        # Convert to Euclidean coordinates (a, b)
        cos_u = amp_u_g * torch.cos(acrophase_u).unsqueeze(0)
        sin_u = amp_u_g * torch.sin(acrophase_u).unsqueeze(0)
        return torch.cat([cos_u, sin_u], dim=0)


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
        m_u_g, _, _ = self._compute_derived_unspliced_params()

        intercept_cg = torch.matmul(dm, self.m_yg) + m_u_g
        lambda_cg = torch.matmul(dm, self.log_lambda_y.exp())
        disp = torch.exp(self.log_disp_u)
        if self.fix_disp_val == "context":
            disp = torch.matmul(dm, disp)
        ab = self._get_ab_u()

        # E_xcg is the expected mean of the distribution
        E_xcg = (X @ ab) * lambda_cg + intercept_cg

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
            E_xcg = torch.exp(E_xcg) * counts
            # Poisson distribution, where the rate is the expected mean
            return torch.distributions.Poisson(rate=E_xcg)

        elif self.noise_model == "gaussian":
            # Gaussian distribution with mean E_xcg and fixed std dev
            std_dev = 1.0
            return torch.distributions.Normal(loc=E_xcg, scale=std_dev)

        else:
            raise NotImplementedError(
                f"Noise model '{self.noise_model}' is not implemented."
            )