"""glm_gene_fit(backend="batched") against the per-gene statsmodels loop."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from scritmo import glm_gene_fit
from scritmo.glm import benjamini_hochberg_correction

COEF_TOL = 1e-4  # statsmodels stops at a deviance change of 1e-8, not at the optimum


def _toy(n_cells=300, n_genes=40, seed=0, zero_gene=True):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n_cells)
    L = rng.uniform(3e3, 3e4, n_cells)
    a0 = rng.uniform(-9, -4, n_genes)
    a1, b1 = rng.normal(0, 0.6, (2, n_genes))
    mu = L[:, None] * np.exp(a0 + np.outer(np.cos(theta), a1) + np.outer(np.sin(theta), b1))
    alpha = rng.uniform(0.05, 0.5, n_genes)
    Y = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu)).astype(float)
    if zero_gene:
        Y[:, 0] = 0          # all zero: both backends drop it
        Y[:, 1] = 0
        Y[5, 1] = 3          # a single nonzero cell
    genes = [f"g{i}" for i in range(n_genes)]
    adata = ad.AnnData(Y, var=pd.DataFrame(index=genes))
    return adata, theta, L, genes


def _fit(adata, theta, L, backend, **kw):
    kw.setdefault("outlier_threshold", 100)
    return glm_gene_fit(adata, theta, counts=L, layer=None, n_jobs=1, backend=backend, **kw)


def _same(a, b, coef_cols=("a_0", "a_1", "b_1")):
    # statsmodels keeps a diverged fit (one nonzero cell: a_0 ~ -1e6 with a NaN
    # p-value under NB, or a_0 ~ -50 with p ~ 1e-4 under Poisson); the batched
    # backend reports it failed and drops it
    degenerate = a.index[a["pvalue"].isna() | (a["a_0"].abs() > 30)]
    assert list(b.index) == [g for g in a.index if g not in degenerate]
    a = a.drop(degenerate)
    a["pvalue_correctedBH"] = benjamini_hochberg_correction(a["pvalue"].values)
    # statsmodels takes its column order from the first gene's row, which
    # differs when that gene failed
    assert sorted(a.columns) == sorted(b.columns)
    for c in coef_cols:
        np.testing.assert_allclose(b[c], a[c], atol=COEF_TOL, rtol=0, err_msg=c)
    for c in ("r2", "BIC", "disp", "disp_MoM", "log10_mean", "amp"):
        if c in a:
            np.testing.assert_allclose(b[c], a[c], rtol=1e-4, atol=1e-6, err_msg=c)
    lp = lambda p: -np.log10(np.maximum(p.astype(float), 1e-300))
    np.testing.assert_allclose(lp(b["pvalue"]), lp(a["pvalue"]), atol=1e-3, err_msg="pvalue")
    np.testing.assert_allclose(lp(b["pvalue_correctedBH"]), lp(a["pvalue_correctedBH"]), atol=1e-3)


def test_nb_scalar_disp():
    adata, theta, L, _ = _toy()
    a = _fit(adata, theta, L, "statsmodels", fixed_disp=0.1)
    b = _fit(adata, theta, L, "batched", fixed_disp=0.1)
    assert "g0" not in a.index and "g0" not in b.index      # all zero: dropped by both
    assert "g1" in a.index and "g1" not in b.index          # one nonzero cell
    _same(a, b)


def test_nb_per_gene_disp_series():
    adata, theta, L, genes = _toy(seed=1)
    disp = pd.Series(np.linspace(0.05, 0.6, len(genes)), index=genes[::-1])
    _same(_fit(adata, theta, L, "statsmodels", fixed_disp=disp),
          _fit(adata, theta, L, "batched", fixed_disp=disp))


def test_poisson():
    adata, theta, L, _ = _toy(seed=2)
    _same(_fit(adata, theta, L, "statsmodels", noise_model="poisson"),
          _fit(adata, theta, L, "batched", noise_model="poisson"))


def test_two_harmonics_and_slope():
    adata, theta, L, _ = _toy(seed=3)
    kw = dict(n_harmonics=2, add_slope=True)
    a = _fit(adata, theta, L, "statsmodels", **kw)
    b = _fit(adata, theta, L, "batched", **kw)
    _same(a, b, coef_cols=("a_0", "a_1", "b_1", "a_2", "b_2", "slope"))


def test_outlier_trimming():
    adata, theta, L, _ = _toy(seed=4)
    _same(_fit(adata, theta, L, "statsmodels", outlier_threshold=95),
          _fit(adata, theta, L, "batched", outlier_threshold=95))


def test_per_gene_counts_dataframe():
    adata, theta, L, genes = _toy(seed=5, zero_gene=False)
    rng = np.random.default_rng(5)
    C = pd.DataFrame(L[:, None] * rng.uniform(0.8, 1.2, (len(L), len(genes))), columns=genes)
    _same(_fit(adata, theta, C, "statsmodels"), _fit(adata, theta, C, "batched"))


def test_gene_blocks():
    adata, theta, L, _ = _toy(seed=6)
    a = _fit(adata, theta, L, "batched")
    b = _fit(adata, theta, L, "batched", batch_genes=7)
    pd.testing.assert_frame_equal(a, b)


def test_auto_and_errors():
    adata, theta, L, _ = _toy(seed=7, n_genes=5, zero_gene=False)
    _same(_fit(adata, theta, L, "auto"), _fit(adata, theta, L, "batched"))
    g = _fit(adata, theta, L, "auto", noise_model="gaussian")   # falls back to the loop
    assert len(g) == 5
    with pytest.raises(ValueError):
        _fit(adata, theta, L, "batched", fit_disp=True)
    with pytest.raises(ValueError):
        _fit(adata, theta, L, "batched", noise_model="gaussian")
    with pytest.raises(ValueError):
        _fit(adata, theta, L, "fast")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_cuda_matches_cpu():
    adata, theta, L, _ = _toy(seed=8, n_cells=2000)
    a = _fit(adata, theta, L, "batched")
    b = _fit(adata, theta, L, "batched", device="cuda")
    _same(a, b)
    np.testing.assert_allclose(b["a_1"], a["a_1"], atol=1e-10)
