"""The estimator API of Scritmo, and its compatibility with the legacy entry points."""

import pickle

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

import scritmo as sr
from scritmo.ml import Scritmo, ContextModel, warmup_and_train
from scritmo.ml.utils import assemble_mp


def _toy(n_cells=240, seed=0):
    rng = np.random.default_rng(seed)
    genes = [f"g{i}" for i in range(6)]
    par = sr.Beta(pd.DataFrame(
        {"a_0": rng.uniform(-7, -5, 6), "amp": rng.uniform(0.5, 1.2, 6),
         "phase": rng.uniform(0, 2 * np.pi, 6)},
        index=genes,
    ))
    par.get_cartesian(inplace=True)
    zt = np.repeat([0, 6, 12, 18], n_cells // 4)
    theta = (zt * 2 * np.pi / 24 + rng.normal(0, 0.3, n_cells)) % (2 * np.pi)
    L = rng.uniform(5e3, 2e4, n_cells)
    s = np.exp(par["a_0"].values + np.outer(np.cos(theta), par["a_1"]) + np.outer(np.sin(theta), par["b_1"]))
    obs = pd.DataFrame({"ZTmod": zt, "sample_name": [f"s{z}" for z in zt], "libsize": L},
                       index=[f"c{i}" for i in range(n_cells)])
    adata = ad.AnnData(rng.poisson(L[:, None] * s).astype(np.float32), obs=obs)
    adata.var_names = genes
    adata.layers["spliced"] = adata.X.copy()
    return adata, par, theta, L


FIT = dict(n_epochs=5, learning_rate=0.05, batch_size=120)


def _fit(**hyper):
    adata, par, theta, L = _toy()
    torch.manual_seed(0)
    model = Scritmo(device="cpu", **hyper).fit(adata, par, counts=L, **FIT)
    return model, adata, par, theta, L


def test_fit_matches_warmup_and_train():
    adata, par, theta, L = _toy()
    torch.manual_seed(0)
    model = Scritmo(k_beta=0.5, n_theta=24, device="cpu").fit(adata, par, counts="libsize", **FIT)
    torch.manual_seed(0)
    cm, losses, mad = warmup_and_train(adata, par.copy(), counts=L, k_beta=0.5, device="cpu", **FIT)
    for a, b in zip(model.state_dict().values(), cm.state_dict().values()):
        assert torch.equal(a, b)
    np.testing.assert_array_equal(model.phases_, cm.post_mode_c)
    np.testing.assert_array_equal(model.losses_, losses)
    assert mad is None and model.mad_epochs_ is None


def test_fit_returns_self_and_keeps_input():
    adata, par, theta, L = _toy()
    before = par.copy()
    model = Scritmo(device="cpu")
    assert model.fit(adata, par, counts=L, kill_amps=True, **FIT) is model
    pd.testing.assert_frame_equal(par, before)


def test_results_and_params():
    model, adata, *_ = _fit()
    n = adata.n_obs
    assert model.phases_.shape == model.phases_mean_.shape == model.phases_std_.shape == (n,)
    assert ((model.phases_ >= 0) & (model.phases_ < 2 * np.pi)).all()
    assert list(model.params_.index) == list(model.genes)
    assert {"a_0", "amp", "phase", "disp"} <= set(model.params_.columns)


def test_unfitted_model():
    model = Scritmo(k_beta=2.0)
    assert model.get_params()["k_beta"] == 2.0
    with pytest.raises(AttributeError):
        model.phases_
    with pytest.raises(RuntimeError):
        model.predict(_toy()[0])
    with pytest.raises(ValueError):
        Scritmo(y=torch.zeros(1, 1, 1))


def test_refit_starts_from_scratch():
    model, adata, par, theta, L = _fit()
    model.fit(adata, par, counts=L, n_epochs=2, batch_size=120)
    assert len(model.losses_) == 2


def test_predict_does_not_change_model():
    model, adata, par, theta, L = _fit()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    phases = model.phases_.copy()
    pred = model.predict(adata[:40], counts=L[:40])
    assert list(pred.columns) == ["phase", "phase_mean", "phase_std", "phase_h"]
    assert list(pred.index) == list(adata.obs_names[:40])
    assert model.Nc == adata.n_obs
    np.testing.assert_array_equal(model.phases_, phases)
    for k, v in model.state_dict().items():
        assert torch.equal(v, state[k])


def test_predict_matches_legacy_transfer():
    model, adata, par, theta, L = _fit()
    sub = adata[:50].copy()
    pred = model.predict(sub, counts=L[:50], n_theta=24)
    legacy = Scritmo.from_params_g(model.get_parameter_dataframe())
    data_c, mp = assemble_mp(sub, model.get_parameter_dataframe(), labels=None, counts=L[:50], device="cpu")
    legacy.get_inferred_phases(data_c, counts=mp["counts"], n_theta=24)
    np.testing.assert_allclose(pred["phase"].values, legacy.post_mode_c)


def test_write_obs():
    model, adata, *_ = _fit()
    model.write_obs(adata)
    np.testing.assert_array_equal(adata.obs["scritmo_phase"].values, model.phases_)
    np.testing.assert_allclose(adata.obs["scritmo_phase_h"].values, model.phases_ * sr.rh)
    with pytest.raises(ValueError):
        model.write_obs(adata[:10].copy())


def test_desynchrony_leaves_obs_unchanged():
    model, adata, par, theta, L = _fit()
    cols = list(adata.obs.columns)
    df = model.desynchrony(
        adata, sample_key="sample_name", ext_time_key="ZTmod", ext_phase=theta,
        device="cpu", n_sim_runs=1, library_size_vec=L, n_epochs_training=0,
    )
    assert list(adata.obs.columns) == cols
    assert {"Data_cSTD", "Technical_cSTD", "Bio_cSTD"} <= set(df.columns)
    assert len(df) == adata.obs["sample_name"].nunique()


def test_legacy_constructor_builds_immediately():
    adata, par, theta, L = _toy()
    data_c, mp = assemble_mp(adata, par, labels=None, counts=L, device="cpu")
    model = ContextModel(mp, data_c, "none", False)
    assert model is not None and ContextModel is Scritmo
    assert model._built and model.Nc == adata.n_obs


def test_pickle_roundtrip_and_class_path():
    model, *_ = _fit()
    loaded = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(loaded.phases_, model.phases_)
    assert loaded.config == model.config
    # pickles keep the historical class path, loadable by older scritmo versions
    assert f"{type(model).__module__}.{type(model).__qualname__}" == "scritmo.ml.context_model.Scritmo"


def test_old_layout_pickle_gets_config():
    model, *_ = _fit(fix_phase=True)
    state = model.__dict__.copy()
    del state["config"], state["_built"]
    old = Scritmo.__new__(Scritmo)
    old.__setstate__(state)
    assert old._built is True
    assert old.config["fix_phase"] is True and old.config["noise_model"] == "nb"
    np.testing.assert_array_equal(old.phases_, model.phases_)


def test_old_module_paths():
    from scritmo.ml.context_model import Scritmo as S1, ContextModel as C1
    from scritmo.ml.warmup import warmup_and_train as w1, arc_mesor_correction  # noqa: F401
    from scritmo.ml.trainer import train_ritmo  # noqa: F401
    from scritmo.ml.analysis_utils import desync_results, aggregate_technical_harmonic  # noqa: F401
    from scritmo.ml.deconvolution import solve_exact  # noqa: F401
    from scritmo.ml.simulations.simulate_populations import _infer_phases_for_context  # noqa: F401
    from scritmo.ml.null_model import NullModelMixin  # noqa: F401
    from scritmo.ml.genome_fit import GenomeFitMixin  # noqa: F401
    from scritmo.ml.marginalization import MarginalizationMixin  # noqa: F401

    assert S1 is C1 is Scritmo and w1 is warmup_and_train
    import scritmo.ml.deconvolution as d_old
    import scritmo.ml.desync.deconvolution as d_new
    assert d_old is d_new
