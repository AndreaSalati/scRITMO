"""Backward-compatibility mixins delegating to :mod:`scritmo.ml.tools`.

These mixin classes keep the old ``NullModelMixin`` / ``GenomeFitMixin``
inheritance interface working: each method is a one-line delegate to the
corresponding module-level function in :mod:`scritmo.ml.tools.null_model`
and :mod:`scritmo.ml.tools.genome_fit`.
"""

import numpy as np
import torch
import pandas as pd
import anndata
from typing import Optional, Dict

from scritmo import Beta

from . import null_model as _null
from . import genome_fit as _genome


class NullModelMixin:
    """
    Mixin for ContextModel: fits a null (flat-amplitude) NB model per gene
    for model comparison via BIC.

    The null model has no amplitude and no phase:
        Y_cg ~ NB(exp(a0_g) * counts_c, disp_g)
    with 2 free parameters per gene (a0, disp), fitted via MLE.

    The full model likelihood is evaluated at each cell's MAP phase
    (posterior mode), giving 4 parameters per gene (a0, amp, phase, disp).

    Methods delegate to the plain functions in :mod:`scritmo.ml.tools.null_model`.
    """

    def fit_null_model(self, adata, layer = None, counts = None, phase_estimator = 'mode', mask = None):
        """See :func:`scritmo.ml.tools.null_model.fit_null_model`."""
        return _null.fit_null_model(self, adata=adata, layer=layer, counts=counts, phase_estimator=phase_estimator, mask=mask)

    def rhythmic_evidence_per_cell(self, adata, layer = None, counts = None, phase_estimator = 'mode', mask = None, mode = 'marginal', n_theta = 100):
        """See :func:`scritmo.ml.tools.null_model.rhythmic_evidence_per_cell`."""
        return _null.rhythmic_evidence_per_cell(self, adata=adata, layer=layer, counts=counts, phase_estimator=phase_estimator, mask=mask, mode=mode, n_theta=n_theta)

    def per_gene_residuals(self, adata, layer = None, counts = None, phase_estimator = 'mode', mask = None, pseudocount = 1.0):
        """See :func:`scritmo.ml.tools.null_model.per_gene_residuals`."""
        return _null.per_gene_residuals(self, adata=adata, layer=layer, counts=counts, phase_estimator=phase_estimator, mask=mask, pseudocount=pseudocount)


class GenomeFitMixin:
    """
    Mixin class for genome-wide gene parameter fitting.

    This mixin adds methods to fit gene parameters (mean, amplitude, phase, dispersion)
    for a large number of genes using frozen phase posteriors from predictor genes.
    The optimization is independent for each gene, allowing for parallel computation.

    Methods delegate to the plain functions in :mod:`scritmo.ml.tools.genome_fit`.
    """

    def fit_genome_wide(self, adata_new: anndata.AnnData, posteriors_c: Optional[np.ndarray] = None, adata_predictors: Optional[anndata.AnnData] = None, gene_chunk_size: int = 1000, optimizer: str = 'LBFGS', max_iter: int = 100, tolerance: float = 0.0001, learning_rate: float = 0.01, show_progress: bool = True, layer: Optional[str] = 'spliced', counts: Optional[np.ndarray] = None, n_theta: Optional[int] = None, device: Optional[str] = None, use_wls_init: bool = True) -> Beta:
        """See :func:`scritmo.ml.tools.genome_fit.fit_genome_wide`."""
        return _genome.fit_genome_wide(self, adata_new=adata_new, posteriors_c=posteriors_c, adata_predictors=adata_predictors, gene_chunk_size=gene_chunk_size, optimizer=optimizer, max_iter=max_iter, tolerance=tolerance, learning_rate=learning_rate, show_progress=show_progress, layer=layer, counts=counts, n_theta=n_theta, device=device, use_wls_init=use_wls_init)

    def _interpolate_posteriors_T(self, posteriors_T: np.ndarray, n_theta_target: int) -> np.ndarray:
        """See :func:`scritmo.ml.tools.genome_fit._interpolate_posteriors_T`."""
        return _genome._interpolate_posteriors_T(self, posteriors_T=posteriors_T, n_theta_target=n_theta_target)

    def _initialize_gene_params_wls(self, y_chunk: torch.Tensor, posteriors_T: torch.Tensor, phi_x: torch.Tensor, counts: torch.Tensor) -> Dict[str, torch.Tensor]:
        """See :func:`scritmo.ml.tools.genome_fit._initialize_gene_params_wls`."""
        return _genome._initialize_gene_params_wls(self, y_chunk=y_chunk, posteriors_T=posteriors_T, phi_x=phi_x, counts=counts)

    def _initialize_gene_params_default(self, y_chunk: torch.Tensor, posteriors_T: torch.Tensor, phi_x: torch.Tensor, counts: torch.Tensor) -> Dict[str, torch.Tensor]:
        """See :func:`scritmo.ml.tools.genome_fit._initialize_gene_params_default`."""
        return _genome._initialize_gene_params_default(self, y_chunk=y_chunk, posteriors_T=posteriors_T, phi_x=phi_x, counts=counts)

    def _optimize_chunk(self, y_chunk: torch.Tensor, posteriors_T: torch.Tensor, phi_x: torch.Tensor, counts: torch.Tensor, initial_params: Dict[str, torch.nn.Parameter], optimizer: str = 'LBFGS', max_iter: int = 100, tolerance: float = 0.0001, learning_rate: float = 0.01) -> Dict[str, torch.Tensor]:
        """See :func:`scritmo.ml.tools.genome_fit._optimize_chunk`."""
        return _genome._optimize_chunk(self, y_chunk=y_chunk, posteriors_T=posteriors_T, phi_x=phi_x, counts=counts, initial_params=initial_params, optimizer=optimizer, max_iter=max_iter, tolerance=tolerance, learning_rate=learning_rate)

    def _params_to_dataframe(self, params: Dict[str, torch.Tensor], gene_names: np.ndarray) -> pd.DataFrame:
        """See :func:`scritmo.ml.tools.genome_fit._params_to_dataframe`."""
        return _genome._params_to_dataframe(self, params=params, gene_names=gene_names)

    def _convert_to_beta_format(self, params_df: pd.DataFrame) -> Beta:
        """See :func:`scritmo.ml.tools.genome_fit._convert_to_beta_format`."""
        return _genome._convert_to_beta_format(self, params_df=params_df)

    def _compute_posteriors_from_adata(self, adata: anndata.AnnData, layer: Optional[str] = 'spliced', n_theta: Optional[int] = None, device: str = 'cpu') -> np.ndarray:
        """See :func:`scritmo.ml.tools.genome_fit._compute_posteriors_from_adata`."""
        return _genome._compute_posteriors_from_adata(self, adata=adata, layer=layer, n_theta=n_theta, device=device)

    def fit_genome_wide_parallel(self, adata_new: anndata.AnnData, posteriors_c: Optional[np.ndarray] = None, n_jobs: int = -1, **kwargs) -> Beta:
        """See :func:`scritmo.ml.tools.genome_fit.fit_genome_wide_parallel`."""
        return _genome.fit_genome_wide_parallel(self, adata_new=adata_new, posteriors_c=posteriors_c, n_jobs=n_jobs, **kwargs)
