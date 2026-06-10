from xml.parsers.expat import model
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import scritmo as sr
from .utils import nmp
from .context_model import ContextModel
from functools import partial
from .trainer import train_ritmo
import pandas as pd
from scritmo import Beta
from .utils import assemble_mp


def warmup_and_train(
    adata,
    params_g,
    context,
    context_mode,
    fix_phase,
    noise_model="nb",
    fix_disp_val="gene",
    log_amp_fn="logit",
    counts=None,
    # batch parameters
    batch=None,
    phi_b=None,
    fixed_prior=False,
    k_batch=None,
    # model init
    k_beta=None,
    # unspliced parameters
    rhythmic_degradation=True,
    # training
    pretrain_epochs=0,
    pretrain_batch_size=128,
    n_epochs=300,
    layer="spliced",
    unspliced_layer=None,
    n_theta=24,
    batch_size=256,
    learning_rate=0.001,
    true_phase=None,
    init_mean=True,
    kill_amps=False,
    device="cuda",
    return_data=False,
    entropy_factor=None,
    n_theta_post=24,
    weights_g=None,
    fixed_cell_phases=None,
    posterior_cell_chunk=None,
):

    Ng = params_g.shape[0]
    if init_mean == True:
        n_jobs = 1 if Ng < 50 else -1
        par_0 = sr.glm_gene_fit(
            adata,
            phases=np.zeros_like(context),
            genes=params_g.index,
            n_harmonics=0,
            counts=counts,
            outlier_threshold=100,
            layer=layer,
            noise_model=noise_model,
            n_jobs=n_jobs,
        )
        genes = par_0.index
        params_g = params_g.loc[genes]
        params_g["a_0"] = par_0.loc[params_g.index, "a_0"]

        # if unspliced_layer is not None:
        #     par_0_u = sr.glm_gene_fit(
        #         adata,
        #         phases=np.zeros_like(context),
        #         genes=params_g.index,
        #         n_harmonics=0,
        #         counts=counts,
        #         outlier_threshold=100,
        #         layer=unspliced_layer,
        #         noise_model=noise_model,
        #         n_jobs=n_jobs,
        #     )
        #     # not intersect with params_g.index
        #     intersect_genes = np.intersect1d(par_0_u.index, params_g.index)
        #     a_0_u = par_0_u.loc[intersect_genes, "a_0"]
        #     params_g = params_g.loc[intersect_genes]

    if kill_amps:
        params_g.kill_amps()

    _assemble_mp = partial(
        assemble_mp,
        adata=adata,
        params_g=params_g,
        counts=counts,
        labels=context,
        layer=layer,
        n_theta=n_theta,
        device=device,
        weights_g=weights_g,
        fixed_cell_phases=fixed_cell_phases,
    )

    if unspliced_layer is None:
        data_c, mp = _assemble_mp()
        data_u_c = None
    else:
        data_c, mp, data_u_c = _assemble_mp(unspliced_layer=unspliced_layer)
        # mp["a_0_u"] = a_0_u.values

    mp["batch"] = batch
    if batch is not None:
        mp["phi_b"] = phi_b
        mp["kappa_b"] = k_batch
        mp["fixed_prior"] = fixed_prior

    mp["k_beta"] = k_beta
    mp["rhythmic_degradation"] = rhythmic_degradation

    if fixed_cell_phases is not None:
        mp["fixed_cell_phases"] = fixed_cell_phases

    cmodel = ContextModel(
        mp,
        data_c,
        context_mode=context_mode,
        fix_phase=fix_phase,
        noise_model=noise_model,
        fix_disp_val=fix_disp_val,
        log_amp_fn=log_amp_fn,
        entropy_factor=entropy_factor,
    )
    cmodel.to(device)

    print("\ntraining model...")

    train_fn = partial(
        train_ritmo,
        model=cmodel,
        data=data_c,
        data_u=data_u_c,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    # training
    if true_phase is not None and not cmodel.fixed_cell_mode:
        losses, mad_epochs = train_fn(true_phase=true_phase)
    else:
        losses = train_fn()
        mad_epochs = None

    # move cmodel to "cpu"
    cmodel.to("cpu")
    cmodel.dev = "cpu"
    data_c = data_c.to("cpu")

    if data_u_c is not None:
        data_u_c = data_u_c.to("cpu")

    # final inference
    if not cmodel.fixed_cell_mode:
        cmodel.get_inferred_phases(
            data_c, y_u=data_u_c, n_theta=n_theta_post, cell_chunk=posterior_cell_chunk
        )

    cmodel.params_g_inf = cmodel.get_parameter_dataframe()
    if return_data:
        return (cmodel, losses, mad_epochs, data_c, data_u_c)
    else:
        return (cmodel, losses, mad_epochs)
