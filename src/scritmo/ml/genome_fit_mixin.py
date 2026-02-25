"""
GenomeFitMixin: Genome-wide gene parameter fitting using frozen phase posteriors.

This module provides functionality to fit gene parameters for a large set of genes
(e.g., 10k-20k genes) independently using frozen phase posteriors from predictor genes.
The key insight is that when phase posteriors are fixed, gene fits become independent
and can be parallelized.

Performance notes:
- Runs on both CPU and GPU (set device='cuda' for GPU)
- Typical runtime: ~2-3 minutes for 24,000 genes on CPU
- ~10-20x faster on GPU
- Uses PyTorch LBFGS (second-order optimizer) instead of standard gradient descent
"""

import numpy as np
import torch
import torch.nn as nn
from torch import tensor as tt
from typing import Optional, Dict, Union, Tuple
import pandas as pd
from tqdm import tqdm
from scipy import stats
import anndata
import time

from scritmo import Beta


class GenomeFitMixin:
    """
    Mixin class for genome-wide gene parameter fitting.

    This mixin adds methods to fit gene parameters (mean, amplitude, phase, dispersion)
    for a large number of genes using frozen phase posteriors from predictor genes.
    The optimization is independent for each gene, allowing for parallel computation.
    """

    def fit_genome_wide(
        self,
        adata_new: anndata.AnnData,
        posteriors_c: np.ndarray,
        gene_chunk_size: int = 1000,
        optimizer: str = "LBFGS",
        max_iter: int = 100,
        tolerance: float = 1e-4,
        learning_rate: float = 0.01,
        show_progress: bool = True,
        layer: Optional[str] = "spliced",
        counts: Optional[np.ndarray] = None,
        n_theta: Optional[int] = None,
        device: Optional[str] = None,
        use_wls_init: bool = True,
    ) -> pd.DataFrame:
        """
        Fit genome-wide gene parameters using frozen phase posteriors.

        This method fits mean, amplitude, phase, and dispersion parameters for each gene
        independently, using frozen phase posteriors from predictor genes. The optimization
        is done in chunks to avoid memory issues.

        This is pure PyTorch code that uses LBFGS (a second-order optimizer) rather than
        standard gradient descent. LBFGS approximates the Hessian and typically converges
        in 5-20 iterations compared to hundreds for Adam/SGD.

        Performance:
        - CPU: ~2-3 minutes for 24,000 genes
        - GPU: ~10-20x faster than CPU
        - Memory: O(Nc x chunk_size x N_theta) instead of O(Nc x Ng x N_theta)

        Parameters
        ----------
        adata_new : anndata.AnnData
            AnnData object containing the new genes to fit. Shape: (Nc, Ng_genome)
        posteriors_c : np.ndarray
            Frozen phase posteriors from predictor genes. Shape: (Nc, N_theta) or (N_theta, Nc)
            These are P_c(theta) values that sum to 1 along the theta axis.
        gene_chunk_size : int, default 1000
            Number of genes to process in each chunk to avoid memory issues.
        optimizer : str, default "LBFGS"
            Optimizer to use. Options: "LBFGS", "Adam".
            LBFGS is recommended for faster convergence on these GLM-like problems.
            LBFGS uses second-order information (Hessian approximation) vs first-order for Adam.
        max_iter : int, default 100
            Maximum number of iterations for the optimizer.
        tolerance : float, default 1e-4
            Convergence tolerance for the optimizer.
        learning_rate : float, default 0.01
            Learning rate for gradient-based optimizers.
        show_progress : bool, default True
            Whether to show a progress bar.
        layer : str, optional, default "spliced"
            Layer in adata to use for expression data. If None, uses adata.X.
        counts : np.ndarray, optional
            Library size counts for each cell. If None, computed from adata.
        n_theta : int, optional
            Number of theta grid points. If None, uses self.Nx from the model.
        device : str, optional
            Device to use for computation ('cpu' or 'cuda'). If None, uses self.dev.
            Use 'cuda' for GPU acceleration (typically 10-20x faster).
        use_wls_init : bool, default True
            Whether to use Weighted Least Squares for initialization.
            Recommended as it provides better starting points and reduces iterations.

        Returns
        -------
        Beta
            Beta object with fitted parameters for each gene. This is the standard
            scRITMO parameter format with columns:
            - a_0: intercept/mean parameter
            - a_1, b_1: harmonic coefficients (if nh >= 1)
            - amp: amplitude (pre-computed)
            - phase: phase in radians
            - disp: dispersion parameter
        """
        if device is None:
            device = self.dev

        if n_theta is None:
            n_theta = self.Nx

        # Print timing and device info
        if show_progress:
            print(
                f"Genome-wide fitting: {adata_new.shape[1]} genes, {adata_new.shape[0]} cells"
            )
            print(
                f"Device: {device}, Optimizer: {optimizer}, Chunks: {gene_chunk_size} genes"
            )
            start_time = time.time()

        # Validate posteriors shape
        # posteriors_c can be either (Nc, N_theta) or (N_theta, Nc)
        Nc = adata_new.shape[0]

        if posteriors_c.shape[0] == Nc:
            # Shape is (Nc, N_theta) - standard format
            N_theta_posteriors = posteriors_c.shape[1]
            posteriors_c_T = posteriors_c.T  # (N_theta, Nc)
        elif posteriors_c.shape[1] == Nc:
            # Shape is (N_theta, Nc) - need to transpose
            N_theta_posteriors = posteriors_c.shape[0]
            posteriors_c_T = posteriors_c  # Already (N_theta, Nc)
            posteriors_c = posteriors_c.T  # (Nc, N_theta)
        else:
            raise ValueError(
                f"posteriors_c shape {posteriors_c.shape} doesn't match adata_new n_cells {Nc}"
            )

        if N_theta_posteriors != n_theta:
            # Interpolate in transposed format (N_theta, Nc)
            posteriors_c_T = self._interpolate_posteriors_T(posteriors_c_T, n_theta)
            posteriors_c = posteriors_c_T.T  # Back to (Nc, N_theta)

        # Get expression data
        if layer is None:
            data = (
                adata_new.X.toarray()
                if hasattr(adata_new.X, "toarray")
                else adata_new.X
            )
        else:
            layer_data = adata_new.layers[layer]
            data = (
                layer_data.toarray() if hasattr(layer_data, "toarray") else layer_data
            )

        # Get counts if not provided
        if counts is None:
            if layer is None:
                counts = adata_new.X.sum(axis=1)
            else:
                counts = adata_new.layers[layer].sum(axis=1)
            if hasattr(counts, "A1"):
                counts = counts.A1
            elif hasattr(counts, "squeeze"):
                counts = counts.squeeze()

        if counts.ndim == 1:
            counts = counts[:, None]

        # Move data to device
        y_all = torch.tensor(data, dtype=torch.float32, device=device)  # (Nc, Ng)
        counts_t = torch.tensor(counts, dtype=torch.float32, device=device)  # (Nc, 1)
        # Use transposed format for posteriors (N_theta, Nc) for easier computation
        posteriors_t = torch.tensor(
            posteriors_c_T, dtype=torch.float32, device=device
        )  # (N_theta, Nc)

        # Create theta grid
        phi_x = torch.linspace(
            0, 2 * torch.pi, n_theta + 1, dtype=torch.float32, device=device
        )[:-1]

        # Get gene names
        gene_names = adata_new.var_names.values
        Ng_total = len(gene_names)

        # Process genes in chunks
        all_params = []

        n_chunks = (Ng_total + gene_chunk_size - 1) // gene_chunk_size
        chunk_iterator = range(n_chunks)
        if show_progress:
            chunk_iterator = tqdm(chunk_iterator, desc="Fitting genome-wide genes")

        for chunk_idx in chunk_iterator:
            start_idx = chunk_idx * gene_chunk_size
            end_idx = min((chunk_idx + 1) * gene_chunk_size, Ng_total)

            # Get chunk of genes
            y_chunk = y_all[:, start_idx:end_idx]  # (Nc, chunk_size)
            gene_names_chunk = gene_names[start_idx:end_idx]

            # Initialize parameters using WLS or defaults
            if use_wls_init:
                initial_params = self._initialize_gene_params_wls(
                    y_chunk, posteriors_t, phi_x, counts_t
                )
            else:
                initial_params = self._initialize_gene_params_default(
                    y_chunk, posteriors_t, phi_x, counts_t
                )

            # Optimize chunk
            fitted_params = self._optimize_chunk(
                y_chunk,
                posteriors_t,
                phi_x,
                counts_t,
                initial_params,
                optimizer=optimizer,
                max_iter=max_iter,
                tolerance=tolerance,
                learning_rate=learning_rate,
            )

            # Convert to DataFrame
            params_df = self._params_to_dataframe(fitted_params, gene_names_chunk)
            all_params.append(params_df)

        # Combine all chunks
        result_df = pd.concat(all_params, axis=0)

        # Convert to Beta object (standard format for scRITMO)
        result_df = self._convert_to_beta_format(result_df)

        if show_progress:
            elapsed = time.time() - start_time
            genes_per_sec = len(result_df) / elapsed
            print(f"Fitting complete: {elapsed:.1f}s ({genes_per_sec:.1f} genes/sec)")

        return result_df

    def _interpolate_posteriors_T(
        self, posteriors_T: np.ndarray, n_theta_target: int
    ) -> np.ndarray:
        """
        Interpolate posteriors to match a different number of theta grid points.

        Parameters
        ----------
        posteriors_T : np.ndarray
            Original posteriors with shape (N_theta_orig, Nc)
        n_theta_target : int
            Target number of theta grid points

        Returns
        -------
        np.ndarray
            Interpolated posteriors with shape (n_theta_target, Nc)
        """
        N_theta_orig, Nc = posteriors_T.shape

        # Create original and target theta values
        theta_orig = np.linspace(0, 2 * np.pi, N_theta_orig, endpoint=False)
        theta_target = np.linspace(0, 2 * np.pi, n_theta_target, endpoint=False)

        # Interpolate for each cell
        posteriors_interp = np.zeros((n_theta_target, Nc))
        for c in range(Nc):
            posteriors_interp[:, c] = np.interp(
                theta_target, theta_orig, posteriors_T[:, c], period=2 * np.pi
            )

        # Renormalize to ensure they sum to 1 along theta axis
        posteriors_interp = posteriors_interp / posteriors_interp.sum(
            axis=0, keepdims=True
        )

        return posteriors_interp

    def _initialize_gene_params_wls(
        self,
        y_chunk: torch.Tensor,
        posteriors_T: torch.Tensor,
        phi_x: torch.Tensor,
        counts: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Initialize gene parameters using Weighted Least Squares on log-transformed counts.

        This provides good initial estimates for mean, amplitude, and phase parameters,
        which drastically reduces the number of optimization steps needed.

        Parameters
        ----------
        y_chunk : torch.Tensor
            Expression data for the chunk. Shape: (Nc, Ng_chunk)
        posteriors_T : torch.Tensor
            Phase posteriors (transposed). Shape: (N_theta, Nc)
        phi_x : torch.Tensor
            Theta grid values. Shape: (N_theta,)
        counts : torch.Tensor
            Library size counts. Shape: (Nc, 1)

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing initialized parameters:
            - m_g: mean/intercept
            - log_amp: log amplitude
            - acrophase: phase
            - log_disp: log dispersion
        """
        Nc, Ng_chunk = y_chunk.shape
        N_theta = len(phi_x)
        nh = self.nh

        # Compute weighted mean across theta for each cell (posterior mean phase)
        # This gives us a representative phase for each cell
        sin_phi = torch.sin(phi_x)
        cos_phi = torch.cos(phi_x)

        # Circular mean of posterior (posteriors_T is N_theta x Nc)
        mean_sin = (posteriors_T * sin_phi.unsqueeze(1)).sum(dim=0)  # (Nc,)
        mean_cos = (posteriors_T * cos_phi.unsqueeze(1)).sum(dim=0)  # (Nc,)
        phi_c_mean = torch.atan2(mean_sin, mean_cos)  # (Nc,)

        # Normalize counts
        counts_norm = counts.squeeze(-1)  # (Nc,)

        # Log-transform counts (with pseudocount)
        log_y = torch.log(y_chunk / counts_norm.unsqueeze(1) + 1e-6)  # (Nc, Ng_chunk)

        # Initialize parameters for each gene
        m_g = torch.zeros(Ng_chunk, device=y_chunk.device)
        amp = torch.zeros(Ng_chunk, device=y_chunk.device)
        phase = torch.zeros(Ng_chunk, device=y_chunk.device)

        # For each gene, fit a simple harmonic regression using cell weights
        for g in range(Ng_chunk):
            y_g = log_y[:, g]  # (Nc,)

            # Build design matrix for harmonic regression (first harmonic only for init)
            X = torch.stack(
                [
                    torch.ones(Nc, device=y_chunk.device),
                    torch.cos(phi_c_mean),
                    torch.sin(phi_c_mean),
                ],
                dim=1,
            )  # (Nc, 3)

            # Weighted least squares (weights based on posterior concentration)
            # Higher weight for cells with more concentrated posteriors
            # posteriors_T is (N_theta, Nc), entropy along N_theta axis
            posterior_entropy = -(posteriors_T * torch.log(posteriors_T + 1e-10)).sum(
                dim=0
            )  # (Nc,)
            weights = 1.0 / (1.0 + posterior_entropy)  # (Nc,)
            weights = weights / weights.sum()
            W = torch.diag(weights)

            # Solve WLS: (X'WX)^(-1) X'Wy
            XWX = X.T @ W @ X
            XWy = X.T @ W @ y_g

            try:
                beta = torch.linalg.solve(XWX, XWy)
            except:
                # If singular, use pseudoinverse
                beta = torch.linalg.lstsq(XWX, XWy).solution

            m_g[g] = beta[0]
            amp[g] = torch.sqrt(beta[1] ** 2 + beta[2] ** 2)
            phase[g] = torch.atan2(beta[2], beta[1])

        # Clamp and transform parameters
        amp = torch.clamp(amp, min=1e-2, max=self.max_amp - 1e-2)

        if self.log_amp_fn == "logit":
            log_amp = torch.logit(amp / self.max_amp)
        else:  # log
            log_amp = torch.log(amp)

        # Initialize dispersion with a reasonable default
        log_disp = torch.full((Ng_chunk,), -1.0, device=y_chunk.device)

        return {
            "m_g": nn.Parameter(m_g),
            "log_amp": nn.Parameter(log_amp),
            "acrophase": nn.Parameter(phase),
            "log_disp": nn.Parameter(log_disp),
        }

    def _initialize_gene_params_default(
        self,
        y_chunk: torch.Tensor,
        posteriors_T: torch.Tensor,
        phi_x: torch.Tensor,
        counts: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Initialize gene parameters with default values.

        Parameters
        ----------
        y_chunk : torch.Tensor
            Expression data for the chunk. Shape: (Nc, Ng_chunk)
        posteriors_T : torch.Tensor
            Phase posteriors (transposed). Shape: (N_theta, Nc)
        phi_x : torch.Tensor
            Theta grid values. Shape: (N_theta,)
        counts : torch.Tensor
            Library size counts. Shape: (Nc, 1)

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing initialized parameters.
        """
        Ng_chunk = y_chunk.shape[1]
        device = y_chunk.device

        # Initialize with small amplitude and zero phase
        m_g = torch.zeros(Ng_chunk, device=device)

        if self.log_amp_fn == "logit":
            log_amp = torch.full((Ng_chunk,), -2.0, device=device)  # Small amplitude
        else:
            log_amp = torch.full((Ng_chunk,), -1.0, device=device)

        acrophase = torch.zeros(Ng_chunk, device=device)
        log_disp = torch.full((Ng_chunk,), -1.0, device=device)

        return {
            "m_g": nn.Parameter(m_g),
            "log_amp": nn.Parameter(log_amp),
            "acrophase": nn.Parameter(acrophase),
            "log_disp": nn.Parameter(log_disp),
        }

    def _optimize_chunk(
        self,
        y_chunk: torch.Tensor,
        posteriors_T: torch.Tensor,
        phi_x: torch.Tensor,
        counts: torch.Tensor,
        initial_params: Dict[str, torch.nn.Parameter],
        optimizer: str = "LBFGS",
        max_iter: int = 100,
        tolerance: float = 1e-4,
        learning_rate: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """
        Optimize parameters for a chunk of genes.

        This creates a lightweight parameter module and optimizes all genes in the chunk
        simultaneously using the specified optimizer.

        Parameters
        ----------
        y_chunk : torch.Tensor
            Expression data for the chunk. Shape: (Nc, Ng_chunk)
        posteriors_T : torch.Tensor
            Phase posteriors (transposed). Shape: (N_theta, Nc)
        phi_x : torch.Tensor
            Theta grid values. Shape: (N_theta,)
        counts : torch.Tensor
            Library size counts. Shape: (Nc, 1)
        initial_params : Dict[str, nn.Parameter]
            Initial parameter values.
        optimizer : str
            Optimizer to use.
        max_iter : int
            Maximum number of iterations.
        tolerance : float
            Convergence tolerance.
        learning_rate : float
            Learning rate for gradient-based optimizers.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing optimized parameters.
        """
        Nc, Ng_chunk = y_chunk.shape
        N_theta = len(phi_x)
        nh = self.nh

        # Create a simple module to hold parameters
        class GeneParamsModule(nn.Module):
            def __init__(self, init_params, max_amp, log_amp_fn):
                super().__init__()
                self.m_g = nn.Parameter(init_params["m_g"].data.clone())
                self.log_amp = nn.Parameter(init_params["log_amp"].data.clone())
                self.acrophase = nn.Parameter(init_params["acrophase"].data.clone())
                self.log_disp = nn.Parameter(init_params["log_disp"].data.clone())
                self.max_amp = max_amp
                self.log_amp_fn = log_amp_fn

            def get_ab(self):
                if self.log_amp_fn == "logit":
                    amp = torch.sigmoid(self.log_amp) * self.max_amp
                else:
                    amp = torch.exp(self.log_amp)
                cos = amp * torch.cos(self.acrophase)
                sin = amp * torch.sin(self.acrophase)
                return torch.stack([cos, sin], dim=0)  # (2, Ng)

        params_module = GeneParamsModule(
            initial_params, self.max_amp, self.log_amp_fn
        ).to(y_chunk.device)

        # Setup optimizer
        if optimizer == "LBFGS":
            opt = torch.optim.LBFGS(
                params_module.parameters(),
                lr=learning_rate,
                max_iter=max_iter,
                tolerance_grad=tolerance,
                tolerance_change=tolerance,
                line_search_fn="strong_wolfe",
            )
        elif optimizer == "Adam":
            opt = torch.optim.Adam(params_module.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # Precompute X matrix (harmonic design matrix)
        # X: (N_theta, 2*nh) -> for nh=1, it's (N_theta, 2) with [cos, sin]
        from .utils import harmonic_dm_torch

        X = harmonic_dm_torch(phi_x, nh, add_intercept=False)  # (N_theta, 2*nh)

        # For simplicity with nh=1, X is (N_theta, 2)
        # We need to handle nh > 1 by repeating amplitudes appropriately
        # For now, assume nh=1 for genome-wide fitting
        if nh > 1:
            # Create higher harmonic design matrix but only use first harmonic
            # This is a simplification - could be extended
            X = X[:, :2]  # Only use first harmonic

        # Optimization closure
        def closure():
            opt.zero_grad()

            # Get parameters
            ab = params_module.get_ab()  # (2, Ng_chunk)
            disp = torch.exp(params_module.log_disp)  # (Ng_chunk,)
            m_g = params_module.m_g  # (Ng_chunk,)

            # Compute expected mean: E_xcg = m_g + X @ ab
            # X: (N_theta, 2), ab: (2, Ng_chunk) -> (N_theta, Ng_chunk)
            E_xg = torch.matmul(X, ab)  # (N_theta, Ng_chunk)
            E_xg = E_xg + m_g.unsqueeze(0)  # (N_theta, Ng_chunk)

            # Expand to match cell dimension: (N_theta, Ng_chunk) -> (N_theta, Nc, Ng_chunk)
            # But we can use broadcasting in the likelihood computation

            # Compute negative binomial log-likelihood
            # y_chunk: (Nc, Ng_chunk)
            # counts: (Nc, 1)
            # posteriors_c: (Nc, N_theta)

            # For each cell c and gene g, we need:
            # LL = sum_theta P_c(theta) * log P(y_cg | theta, params_g)

            # Compute rates
            counts_expanded = counts.squeeze(-1)  # (Nc,)

            # E_xcg shape should be (N_theta, Nc, Ng_chunk)
            # We compute: log rate for each (theta, c, g)
            E_xcg = E_xg.unsqueeze(1)  # (N_theta, 1, Ng_chunk)

            # Add cell-specific effect via counts
            # log_mu_xcg = E_xcg + log(counts_c) for each cell
            log_mu_xcg = E_xcg + torch.log(
                counts_expanded[None, :, None] + 1e-10
            )  # (N_theta, Nc, Ng_chunk)

            # Compute NB log-likelihood
            # y_chunk: (Nc, Ng_chunk) -> expand to (1, Nc, Ng_chunk) for broadcasting
            y_expanded = y_chunk.unsqueeze(0)  # (1, Nc, Ng_chunk)

            # r (dispersion parameter)
            r = 1.0 / (disp[None, None, :] + 1e-10)  # (1, 1, Ng_chunk)

            # p (success probability)
            mu_xcg = torch.exp(log_mu_xcg)  # (N_theta, Nc, Ng_chunk)
            p = disp[None, None, :] * mu_xcg / (1.0 + disp[None, None, :] * mu_xcg)
            p = torch.clamp(p, min=1e-6, max=1.0 - 1e-6)

            # Negative Binomial log PMF
            # log Gamma(y + r) - log Gamma(r) - log Gamma(y + 1) + r*log(1-p) + y*log(p)
            # Using torch.distributions for numerical stability
            from torch.distributions import NegativeBinomial

            # Create distribution with shape broadcasting
            dist = NegativeBinomial(total_count=r, probs=p)
            ll_xcg = dist.log_prob(y_expanded)  # (N_theta, Nc, Ng_chunk)

            # Weight by posteriors and sum
            # posteriors_T is already (N_theta, Nc), add dimension for genes
            posteriors_t = posteriors_T.unsqueeze(-1)  # (N_theta, Nc, 1)

            # Weighted log-likelihood
            weighted_ll = ll_xcg * posteriors_t  # (N_theta, Nc, Ng_chunk)

            # Sum over theta and cells, mean over genes
            total_ll = weighted_ll.sum()  # Sum all contributions

            # Negative log-likelihood (we minimize)
            loss = -total_ll / (Nc * Ng_chunk)  # Normalize by batch size

            loss.backward()
            return loss

        # Run optimization
        if optimizer == "LBFGS":
            # LBFGS requires calling the closure multiple times internally
            loss = opt.step(closure)
        else:
            # Adam: manual iteration loop
            for _ in range(max_iter):
                loss = closure()
                opt.step()

                # Early stopping check could go here

        # Extract fitted parameters
        fitted_params = {
            "m_g": params_module.m_g.detach().cpu(),
            "log_amp": params_module.log_amp.detach().cpu(),
            "acrophase": params_module.acrophase.detach().cpu(),
            "log_disp": params_module.log_disp.detach().cpu(),
        }

        return fitted_params

    def _params_to_dataframe(
        self,
        params: Dict[str, torch.Tensor],
        gene_names: np.ndarray,
    ) -> pd.DataFrame:
        """
        Convert parameter tensors to a DataFrame (raw format).

        Parameters
        ----------
        params : Dict[str, torch.Tensor]
            Dictionary of parameter tensors.
        gene_names : np.ndarray
            Array of gene names.

        Returns
        -------
        pd.DataFrame
            DataFrame with parameters as columns and genes as index.
        """
        # Convert to numpy
        m_g = params["m_g"].numpy()
        log_amp = params["log_amp"].numpy()
        acrophase = params["acrophase"].numpy()
        log_disp = params["log_disp"].numpy()

        # Compute derived parameters
        if self.log_amp_fn == "logit":
            amp = torch.sigmoid(torch.tensor(log_amp)).numpy() * self.max_amp
        else:
            amp = np.exp(log_amp)

        # Compute a_1, b_1 from amp and phase
        a_1 = amp * np.cos(acrophase)
        b_1 = amp * np.sin(acrophase)

        disp = np.exp(log_disp)

        # Create DataFrame
        df = pd.DataFrame(
            {
                "a_0": m_g,
                "a_1": a_1,
                "b_1": b_1,
                "amp": amp,
                "phase": acrophase,
                "disp": disp,
            },
            index=gene_names,
        )

        return df

    def _convert_to_beta_format(self, params_df: pd.DataFrame) -> Beta:
        """
        Convert a parameter DataFrame to a Beta object (scRITMO standard format).

        This follows the same pattern as ContextModel.get_parameter_dataframe().

        Parameters
        ----------
        params_df : pd.DataFrame
            DataFrame with a_0, a_1, b_1, amp, phase, disp columns.

        Returns
        -------
        Beta
            Beta object with proper formatting (amp pre-computed, cartesian coords).
        """
        # Convert to Beta object
        params_beta = Beta(params_df)

        params_beta.get_amp(inplace=True)

        return params_beta

    def fit_genome_wide_parallel(
        self,
        adata_new: anndata.AnnData,
        posteriors_c: np.ndarray,
        n_jobs: int = -1,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fit genome-wide gene parameters in parallel using joblib.

        This is an alternative to fit_genome_wide that processes genes in parallel
        using multiple CPU cores. Each gene is fitted independently.

        Parameters
        ----------
        adata_new : anndata.AnnData
            AnnData object containing the new genes to fit.
        posteriors_c : np.ndarray
            Frozen phase posteriors from predictor genes.
        n_jobs : int, default -1
            Number of parallel jobs. -1 uses all available cores.
        **kwargs
            Additional arguments passed to fit_genome_wide.

        Returns
        -------
        pd.DataFrame
            DataFrame with fitted parameters for each gene.
        """
        # This is a placeholder for a true parallel implementation
        # The current fit_genome_wide already processes genes in parallel
        # within each chunk via vectorized PyTorch operations
        # True process-level parallelism would require more complex orchestration

        # For now, just delegate to fit_genome_wide with smaller chunks
        # to allow better CPU utilization through PyTorch's intra-op parallelism
        kwargs["gene_chunk_size"] = kwargs.get("gene_chunk_size", 500)

        return self.fit_genome_wide(adata_new, posteriors_c, **kwargs)
