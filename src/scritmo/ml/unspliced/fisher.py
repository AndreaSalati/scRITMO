import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.autograd.functional import hessian, jacobian
from ..utils import nmp

class FisherUncertaintyMixin:
    """
    Mixin to compute parameter uncertainty (standard deviations) using the 
    Fisher Information Matrix (inverse Hessian of NLL) and the Delta Method.
    """

    def compute_fisher_uncertainty(
        self, 
        y_u,
        gene_indices=None, 
        batch_size=10, 
        param_subset=None,
    ):
        """
        Computes the standard deviation for biophysical parameters.

        Args:
            gene_indices (list or slice, optional): Subset of genes to process. 
                                                    Defaults to all genes.
            batch_size (int): Not used in this specific implementation but kept for API consistency.
                              (This implementation loops gene-by-gene for memory safety).
            param_subset (list, optional): List of physical parameter names to return.
                                           Options: ['k_splice', 'gamma_mean', 'A_gamma', 'phi_gamma']
                                           If None, returns all.

        Returns:
            pd.DataFrame: DataFrame containing the standard deviations of the requested parameters.
        """
        if not self.unspliced_mode:
            raise ValueError("Fisher uncertainty currently only implemented for unspliced kinetics.")

        if gene_indices is None:
            gene_indices = range(self.Ng)
        elif isinstance(gene_indices, slice):
            gene_indices = range(*gene_indices.indices(self.Ng))
        
        # Mapping for output storage
        results = {
            "gene": [],
            "k_splice": [],
            "gamma_mean": [],
            "amp_gamma": [],
            "phi_gamma": [],
            "correlation_beta_gamma": [] # To check the correlation you mentioned
        }

        # Prepare static tensors needed for functional evaluation (Spliced params are fixed constants here)
        # We perform this outside the loop to avoid re-overhead
        with torch.no_grad():
            ab_s_all = self._get_ab() # [2, Ng]
            # Spliced Rate Log is treated as a constant input signal for the unspliced component
            # to isolate kinetic uncertainty.
            spliced_rate_log_all, _, _ = self.model_formula() 
            spliced_rate_all = torch.exp(spliced_rate_log_all)

        print(f"Computing Fisher Information for {len(gene_indices)} genes...")

        counts_s = self.counts
        counts_u = self.counts_u

        for g_idx in gene_indices:
            # 1. Extract Current Parameter Values (Points of Tangency)
            # These are the unconstrained parameters
            p_log_beta = self.log_k_splice_g[g_idx].detach().clone()
            p_raw_eps = self.raw_epsilon_gamma[g_idx].detach().clone()
            p_phi = self.phi_gamma_g[g_idx].detach().clone()
            p_excess = self.param_excess_gamma[g_idx].detach().clone()
            
            # Enable grad for Hessian computation
            params_vec = [p_log_beta, p_raw_eps, p_phi, p_excess]
            for p in params_vec:
                p.requires_grad = True

            # Data for this gene
            data_u = y_u[:, :, g_idx].unsqueeze(-1) # Using counts as provided in buffer

            # Spliced constants for this gene
            a_s_g = ab_s_all[0, g_idx]
            b_s_g = ab_s_all[1, g_idx]
            spliced_rate_g = spliced_rate_all[:, :, g_idx]

            # 2. Define Functional Loss (NLL) for this single gene
            # This function takes the *parameters* as arguments
            def local_nll(log_beta, raw_eps, phi, excess):
                # Recalculate physical params from inputs
                k_splice = torch.exp(log_beta)
                A_gamma = torch.sigmoid(raw_eps)
                
                # Recalculate Resultant Oscillation (R)
                # D (derivative) components
                D_cos = self.omega * b_s_g
                D_sin = -self.omega * a_s_g
                
                # G (gamma) components
                G_cos = A_gamma * torch.cos(phi)
                G_sin = A_gamma * torch.sin(phi)
                
                # R components
                R_cos = D_cos + G_cos
                R_sin = D_sin + G_sin
                
                A_R = torch.sqrt(R_cos**2 + R_sin**2 + 1e-8)
                
                # Gamma Mean
                gamma_mean = A_R + torch.nn.functional.softplus(excess) + 1e-6
                
                X_g = self.X # Global X
                cos_basis = X_g[:, :, 0] # First harmonic cos
                sin_basis = X_g[:, :, 1] # First harmonic sin
                
                resultant_osc = R_cos * cos_basis + R_sin * sin_basis
                factor = gamma_mean + resultant_osc
                factor = factor.clamp(min=1e-8)
                
                E_xcg = (spliced_rate_g / k_splice) * factor
                
                # Likelihood (Negative Binomial)
                disp = torch.exp(self.log_disp_u) if self.log_disp_u.ndim == 0 else torch.exp(self.log_disp_u[g_idx])
                
                mu = E_xcg.unsqueeze(-1) * self.counts_u
                
                r = 1.0 / disp
                p = disp * mu / (1.0 + disp * mu)
                p = p.clamp(min=1e-6, max=1.0 - 1e-6)
                
                # Log Probability
                dist = torch.distributions.NegativeBinomial(total_count=r, probs=p)
                log_prob = dist.log_prob(data_u)
                
                # Marginalization or Sum
                # We replicate the 'forward' logic.
                ll_xc = log_prob # [Nx, Nc]
                
                if hasattr(self, 'probabilistic_phase') and not self.probabilistic_phase:
                     # Sum over cells
                     loss = -ll_xc.sum()
                else:
                    # Marginalize over theta (sum over Nx, sum over Nc)
                    # Use existing marginalize method? Hard to invoke cleanly in functional wrapper.
                    # We implement simplified Sum integration here for the gradient.
                    l_xc = torch.exp(ll_xc - ll_xc.max())
                    l_c = l_xc.sum(dim=0)
                    loss = -torch.log(l_c).sum()
                
                return loss

            # 3. Define Transformation to Physical Space
            def physical_transform(log_beta, raw_eps, phi, excess):
                k_splice = torch.exp(log_beta)
                A_gamma = torch.sigmoid(raw_eps)
                
                D_cos = self.omega * b_s_g
                D_sin = -self.omega * a_s_g
                G_cos = A_gamma * torch.cos(phi)
                G_sin = A_gamma * torch.sin(phi)
                
                R_cos = D_cos + G_cos
                R_sin = D_sin + G_sin
                A_R = torch.sqrt(R_cos**2 + R_sin**2 + 1e-8)
                
                gamma_mean = A_R + torch.nn.functional.softplus(excess) + 1e-6
                
                # Return vector of physical params
                return torch.stack([k_splice, gamma_mean, A_gamma, phi])

            # 4. Compute Matrices
            inputs = tuple(params_vec)
            
            # A. Hessian of NLL w.r.t Unconstrained Params
            try:
                H = hessian(local_nll, inputs)
                # H is a tuple of tuples. We need to assemble the 4x4 matrix.
                # H[i][j] is the block for param i and param j. Since params are scalars, it's 1x1.
                H_mat = torch.zeros(4, 4, device=self.dev)
                for i in range(4):
                    for j in range(4):
                        H_mat[i, j] = H[i][j].item()
                
                # B. Invert Hessian -> Covariance (Unconstrained)
                # Add jitter for stability
                H_mat = H_mat + 1e-4 * torch.eye(4, device=self.dev)
                Cov_unc = torch.inverse(H_mat)
                
                # C. Jacobian of Transformation w.r.t Unconstrained Params
                J = jacobian(physical_transform, inputs)
                # J is tuple of 4 tensors (one per input). Stack them.
                # Output of transform is size 4. J[i] is size 4 (deriv of output wrt input i).
                J_mat = torch.stack([j_tens for j_tens in J], dim=1).squeeze() # [4, 4]
                
                # D. Delta Method: Cov_phys = J @ Cov_unc @ J.T
                Cov_phys = J_mat @ Cov_unc @ J_mat.T
                
                # Extract variances (diagonal)
                variances = torch.diag(Cov_phys)
                stds = torch.sqrt(variances.clamp(min=1e-9))
                
                # Extract Correlation between k_splice (idx 0) and gamma_mean (idx 1)
                # Corr = Cov_01 / (std_0 * std_1)
                cov_01 = Cov_phys[0, 1]
                corr_01 = cov_01 / (stds[0] * stds[1] + 1e-9)

                results["gene"].append(self.genes[g_idx])
                results["k_splice"].append(stds[0].item())
                results["gamma_mean"].append(stds[1].item())
                results["amp_gamma"].append(stds[2].item())
                results["phi_gamma"].append(stds[3].item())
                results["correlation_beta_gamma"].append(corr_01.item())

            except RuntimeError as e:
                # Handle cases where Hessian is singular or calculation fails
                print(f"Warning: Hessian failed for gene {self.genes[g_idx]}: {e}")
                results["gene"].append(self.genes[g_idx])
                results["k_splice"].append(np.nan)
                results["gamma_mean"].append(np.nan)
                results["amp_gamma"].append(np.nan)
                results["phi_gamma"].append(np.nan)
                results["correlation_beta_gamma"].append(np.nan)

        df = pd.DataFrame(results)
        df.set_index("gene", inplace=True)
        
        if param_subset:
            cols = [f"{p}_std" for p in param_subset]
            if "correlation_beta_gamma" in results:
                cols.append("correlation_beta_gamma")
            return df[cols]
            
        return df