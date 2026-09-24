"""Tests for the identifiable "ratio" parametrization of the unspliced model."""

import numpy as np
import pandas as pd
import torch
import anndata as ad

import scritmo as sr
from scritmo.ml import warmup_and_train, unspliced_lrt, refine_mle

W = 2 * np.pi / 24


def _toy_adata(n_cells=400, seed=0):
    rng = np.random.default_rng(seed)
    genes = ["g0", "g1", "g2"]
    par = sr.Beta(pd.DataFrame(
        {"a_0": [-6.0, -5.5, -7.0], "amp": [0.8, 0.5, 1.2], "phase": [1.0, 3.0, 5.0]},
        index=genes,
    ))
    par.get_cartesian(inplace=True)
    theta = rng.uniform(0, 2 * np.pi, n_cells)
    L = np.full(n_cells, 1e4)
    log_s = par["a_0"].values + np.outer(np.cos(theta), par["a_1"]) + np.outer(np.sin(theta), par["b_1"])
    s = np.exp(log_s)
    h = 0.3 * (1 + 0.5 * np.cos(theta[:, None] - (par["phase"].values - np.pi / 2)))
    adata = ad.AnnData(rng.poisson(L[:, None] * s).astype(np.float32))
    adata.var_names = genes
    adata.layers["spliced"] = adata.X.copy()
    adata.layers["unspliced"] = rng.poisson(L[:, None] * s * h).astype(np.float32)
    return adata, par, theta, L


def _fit(adata, par, theta, L, rd, n_epochs=0, param="ratio", fixed=True):
    cm, *_ = warmup_and_train(
        adata, par.copy(), counts=L, unspliced_layer="unspliced",
        rhythmic_degradation=rd, unspliced_param=param, n_epochs=n_epochs,
        fixed_cell_phases=theta if fixed else None, init_mean=False, k_beta=None,
        device="cpu", learning_rate=0.05, batch_size=400,
    )
    return cm


def test_ratio_null_has_no_delta_parameter():
    adata, par, theta, L = _toy_adata()
    cm0 = _fit(adata, par, theta, L, rd=False)
    cm1 = _fit(adata, par, theta, L, rd=True)
    names0 = {n for n, _ in cm0.named_parameters()}
    names1 = {n for n, _ in cm1.named_parameters()}
    assert names1 - names0 == {"u_delta"}
    assert not hasattr(cm0, "log_k_splice_g")
    # identical at init (delta = 0): the null is nested in the alternative
    X = cm0.X
    assert torch.allclose(cm0._unspliced_formula(X), cm1._unspliced_formula(X))


def test_level_init_matches_pooled_ratio():
    adata, par, theta, L = _toy_adata()
    cm = _fit(adata, par, theta, L, rd=False)
    u, s = adata.layers["unspliced"].sum(0), adata.layers["spliced"].sum(0)
    np.testing.assert_allclose(cm.u_log_level.detach().numpy(), np.log((u + 1) / (s + 1)), rtol=1e-5)


def test_ratio_curve_positive_and_formula():
    adata, par, theta, L = _toy_adata()
    cm = _fit(adata, par, theta, L, rd=True)
    with torch.no_grad():
        cm.u_logit_rho.fill_(8.0)  # rho -> ~1
        cm.u_delta.copy_(torch.tensor([0.3, -2.0, 3.0]))
        u = cm._unspliced_formula(cm.X)
    assert torch.all(u > 0)
    k = cm._get_ratio_tensors()
    s = torch.exp(cm.model_formula()[0])
    th = torch.as_tensor(theta, dtype=torch.float32)[None, :, None]
    h = torch.exp(cm.u_log_level) * (1 + k["rho"] * torch.cos(th - k["psi"]))
    assert torch.allclose(u, s * h, rtol=1e-5)


def test_kinetic_and_ratio_give_same_curve():
    """The same (beta, gamma_mean, G) must map to the same u in both parametrizations."""
    adata, par, theta, L = _toy_adata()
    kin = _fit(adata, par, theta, L, rd=True, param="kinetic")
    rat = _fit(adata, par, theta, L, rd=True, param="ratio")
    k = kin._get_gamma_kinetics_tensors()
    m = k["gamma_mean"] / k["k_splice"]
    r_cos, r_sin = k["R_cos"] / k["k_splice"], k["R_sin"] / k["k_splice"]
    rho = torch.sqrt(r_cos**2 + r_sin**2) / m
    psi = torch.atan2(r_sin, r_cos)
    with torch.no_grad():
        rat.u_log_level.copy_(torch.log(m))
        rat.u_logit_rho.copy_(torch.logit(rho))
        rat.u_delta.copy_(psi - (rat.acrophase - np.pi / 2))
        assert torch.allclose(kin._unspliced_formula(kin.X), rat._unspliced_formula(rat.X), rtol=1e-4)


def test_derived_rates_null():
    adata, par, theta, L = _toy_adata()
    cm = _fit(adata, par, theta, L, rd=False)
    df = cm.get_kinetic_parameters()
    A = cm._amp_s().detach().numpy()
    np.testing.assert_allclose(df["gamma_mean"], W * A / df["rho"], rtol=1e-5)
    assert np.all(df["A_gamma_min"] == 0) and df["feasible"].all()


def test_derived_rates_minimal_rhythm():
    adata, par, theta, L = _toy_adata()
    cm = _fit(adata, par, theta, L, rd=True)
    with torch.no_grad():
        cm.u_delta.copy_(torch.tensor([0.5, 2.0, -0.2]))
    df = cm.get_kinetic_parameters()
    A = cm._amp_s().detach().numpy()
    np.testing.assert_allclose(df["A_gamma_min"], W * A * np.abs(np.sin(df["delta"])), rtol=1e-5)
    # |delta| > 90 deg: minimal-rhythm point infeasible
    assert not df.loc["g1", "feasible"] and np.isnan(df.loc["g1", "gamma_mean"])
    assert df.loc["g0", "feasible"]


def test_lrt_and_fisher_after_training():
    adata, par, theta, L = _toy_adata(n_cells=2000)
    cm0, _, _, d, du = warmup_and_train(
        adata, par.copy(), counts=L, unspliced_layer="unspliced", rhythmic_degradation=False,
        n_epochs=150, fixed_cell_phases=theta, init_mean=False, k_beta=None, device="cpu",
        learning_rate=0.05, batch_size=500, return_data=True,
    )
    refine_mle(cm0, d, du)
    cm1 = _fit(adata, par, theta, L, rd=True, n_epochs=0)
    cm1.load_state_dict(cm0.state_dict())  # warm start: same point, delta = 0
    refine_mle(cm1, d, du)
    res = unspliced_lrt(cm0, cm1, d, du)
    assert (res["ll_alt"] >= res["ll_null"] - 1e-2).all()  # nested + polished
    assert (res["lrt"] >= 0).all() and res["pval"].between(0, 1).all()
    # true delta = 0 here: no gene should be wildly significant
    assert (res["pval"] > 1e-4).all()
    kp = cm0.get_kinetic_parameters()
    np.testing.assert_allclose(kp["rho"], 0.5, atol=0.1)
    fi = cm1.compute_fisher_uncertainty(du)
    assert np.isfinite(fi[["u_level_std", "rho_std", "delta_std"]].values).all()


def test_legacy_pickle_defaults_to_kinetic():
    adata, par, theta, L = _toy_adata()
    kin = _fit(adata, par, theta, L, rd=True, param="kinetic")
    del kin.unspliced_param  # as in a model pickled before the ratio version
    assert kin._unspliced_param() == "kinetic"
    assert "k_splice" in kin.get_kinetic_parameters().columns


if __name__ == "__main__":
    # runnable without pytest: python tests/test_unspliced_ratio.py
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("PASS", _name)
