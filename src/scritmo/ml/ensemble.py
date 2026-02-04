import torch
import numpy as np
from .utils import nmp


class EnsembleMixin:
    """
    Mixin class for ContextModel to handle feature bagging (ensemble) 
    phase estimation.
    """

    def _sample_gene_mask(self, m, base_genes_indices=None):
        """
        Generates a boolean mask for a random subset of m genes.
        Always includes base_genes if provided.
        """
        if m > self.Ng:
            raise ValueError(f"m ({m}) cannot be larger than total genes ({self.Ng})")

        # Start with all False
        mask = torch.zeros(self.Ng, dtype=torch.bool, device=self.dev)
        
        # Handle mandatory base genes
        if base_genes_indices is not None:
            mask[base_genes_indices] = True
            n_fixed = len(base_genes_indices)
        else:
            n_fixed = 0

        # Calculate how many random genes we still need
        n_random = m - n_fixed
        if n_random < 0:
            raise ValueError(f"Number of base genes ({n_fixed}) exceeds m ({m})")

        if n_random > 0:
            # Get indices of genes NOT in base set
            available_indices = torch.where(~mask)[0]
            # Randomly sample from available
            perm = torch.randperm(len(available_indices), device=self.dev)
            selected_indices = available_indices[perm[:n_random]]
            mask[selected_indices] = True

        return mask

    def compute_ensemble_phases(
        self, 
        y, 
        n_bags=50, 
        m=None, 
        base_genes=None, 
        counts=None, 
        n_theta=None,
        return_samples=False
    ):
        """
        Performs feature bagging to estimate phase uncertainty.
        
        Args:
            y: Data tensor
            n_bags: Number of bagging iterations
            m: Number of genes per bag. Defaults to Ng // 3.
            base_genes: List of gene names (str) to always include.
            counts: Raw counts (required if not using cached model counts)
            n_theta: Resolution for phase grid.
        
        Returns:
            post_mean_ensemble: Circular mean of inferred phases (Nc,)
            R_ensemble: Resultant vector length (measure of confidence) (Nc,)
            samples (optional): The individual predictions (n_bags, Nc)
        """
        
        # 1. Defaults and Setup
        if m is None:
            m = max(1, self.Ng // 3)
            
        if counts is None:
            # Fallback to model counts if compatible, else raise error
            if y.shape[1] == self.Nc:
                counts = self.counts
            else:
                raise ValueError("Counts must be provided for new data.")

        # Resolve n_theta and grid
        if n_theta is None:
            n_theta = self.Nx
            phi_grid = self.phi_x
        else:
            phi_grid = torch.linspace(
                0, 2 * torch.pi, n_theta + 1, dtype=torch.float32, device=self.dev
            )[:-1]

        # 2. Compute or Retrieve Full Likelihood (Cache check)
        # Check if cache exists and dimensions match current request
        cache_valid = False
        if hasattr(self, 'll_xcg') and self.ll_xcg is not None:
            if self.ll_xcg.shape[0] == n_theta and self.ll_xcg.shape[1] == y.shape[1]:
                cache_valid = True

        if not cache_valid:
            with torch.no_grad():
                # We use existing nb_dist logic but must ensure proper indices/counts
                # Note: nb_dist expects indices or counts. 
                dist = self.nb_dist(counts=counts, n_theta=n_theta)
                # Compute log_prob and cache it
                self.ll_xcg = dist.log_prob(y)
        
        # 3. Prepare Base Genes Indices
        base_indices = None
        if base_genes is not None:
            # Convert string names to indices
            # self.genes is a numpy array from params_g.index
            base_indices = [np.where(self.genes == g)[0][0] for g in base_genes if g in self.genes]
            base_indices = torch.tensor(base_indices, device=self.dev, dtype=torch.long)

        # 4. Bagging Loop
        phi_samples = torch.zeros((n_bags, y.shape[1]), dtype=torch.float32, device=self.dev)
        
        with torch.no_grad():
            for i in range(n_bags):
                # Get mask for this iteration
                mask = self._sample_gene_mask(m, base_indices)
                
                # Slice cached likelihood: (Nx, Nc, Ng_subset)
                ll_subset = self.ll_xcg[:, :, mask]
                
                # Sum over genes (dim 2) -> (Nx, Nc)
                ll_xc = (ll_subset * self.weights_g[mask]).sum(2)
                
                # MLE: Argmax over phases (dim 0) -> (Nc,)
                max_indices = ll_xc.argmax(dim=0)
                
                # Map indices to actual phase values
                phi_samples[i, :] = phi_grid[max_indices]

        # 5. Compute Circular Statistics across bags (dim 0)
        # Resultant Vector R and Mean Phase
        sin_sum = torch.sin(phi_samples).sum(dim=0)
        cos_sum = torch.cos(phi_samples).sum(dim=0)
        
        # R = sqrt(C^2 + S^2) / N
        R_ensemble = torch.sqrt(sin_sum**2 + cos_sum**2) / n_bags
        
        # Mean = atan2(S, C) (adjusts to -pi, pi, map to 0, 2pi)
        post_mean_ensemble = torch.atan2(sin_sum, cos_sum)
        post_mean_ensemble = post_mean_ensemble % (2 * torch.pi)

        if return_samples:
            return post_mean_ensemble, R_ensemble, phi_samples
        
        return nmp(post_mean_ensemble), nmp(R_ensemble)