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
from .utils import assemble_mp, resolve_device


def warmup_and_train(
    adata,
    params_g,
    context=None,
    context_mode="none",
    fix_phase=False,
    k_beta=0.2,
    noise_model="nb",
    fix_disp_val="gene",
    log_amp_fn="logit",
    counts=None,
    # model init
    phi_init=None,
    # batch parameters
    batch=None,
    phi_b=None,
    fixed_prior=False,
    k_batch=None,
    # unspliced parameters
    rhythmic_degradation=True,
    # training
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
    """
    Fit the scRITMO model to an AnnData and infer a phase posterior per cell.

    This is the canonical entry point: it assembles the data bundle, builds a
    :class:`~scritmo.ml.context_model.Scritmo` model, trains it, and runs the final
    posterior inference. Most users never touch the model class directly.

    The model
    ---------
    Each gene is a harmonic (Fourier) function of an unobserved circadian phase,
    with Negative-Binomial (or Poisson) counts. Per-cell phases are not point
    parameters: the likelihood is marginalized over a fixed grid of ``n_theta``
    points on the circle, which yields a full posterior over phase for every cell.
    The gene parameters fit during training are the log-mesor, the harmonic
    acrophase and amplitude, and the dispersion. ``params_g`` supplies the
    reference template these are seeded from (and, via ``k_beta``, anchored to).

    What this function does, in order
    --------------------------------
    1. Mesor warm-start (``init_mean=True``): an intercept-only GLM per gene
       (``sr.glm_gene_fit`` with ``n_harmonics=0``) re-estimates ``a_0``. Genes the
       GLM drops are dropped from ``params_g`` too, so the returned parameter table
       can be a subset of the input.
    2. Data assembly (``assemble_mp``): builds the ``[n_theta, Nc, Ng]`` data tensor
       and the ``mp`` bundle on ``device``.
    3. Model construction and ``.to(device)``.
    4. Training (``train_ritmo``) for ``n_epochs``.
    5. Move model and data back to CPU.
    6. Final posterior inference at the finer ``n_theta_post`` grid
       (``get_inferred_phases``), which populates ``post_mean_c``, ``post_mode_c``,
       ``post_std_c``, ``post_var_c``, ``mle_c`` and ``disp`` on the model.
    7. ``cmodel.params_g_inf = cmodel.get_parameter_dataframe()``.

    Phase regimes
    -------------
    ``fix_phase`` and ``k_beta`` together select how much the gene acrophases
    (phi_g) are allowed to move away from the ``params_g`` template:

    ==========  ============  ==========  ====================================
    regime      ``fix_phase`` ``k_beta``  behaviour
    ==========  ============  ==========  ====================================
    fixed       ``True``      ``None``    phi_g hard-pinned to the template
    soft        ``False``     finite      phi_g refit, Von-Mises anchored
    unfixed     ``False``     ``None``    phi_g fully free
    ==========  ============  ==========  ====================================

    "soft" is the production regime. Amplitudes and mesors are trainable in all
    three. Note that the Von-Mises prior is always centered on
    ``params_g["phase"]`` even when ``phi_init`` starts the acrophases elsewhere.

    Parameters
    ----------
    adata : AnnData
        Cells × genes. Must contain every gene in ``params_g.index``.
    params_g : Beta
        Reference gene parameters: columns "a_0" (log-mesor), "amp", "phase". Seeds
        the fit, sets the harmonic order, and centers the ``k_beta`` prior. An
        optional "disp" column warm-starts the dispersion.
    counts : array-like, optional
        Per-cell library sizes used as the NB offset. Defaults to the summed
        ``layer``. **Pass the genome-wide depth vector whenever ``adata`` holds only
        a gene subset**: the realized sum over a handful of oscillating genes is
        itself phase-dependent, which divides the signal back out. The same vector
        must then go to ``cmodel.estimate_phase_desynchrony(library_size_vec=...)``,
        or the real data and its technical twin sit on different offset bases.
    layer : str, default "spliced"
        AnnData layer to fit. None uses ``adata.X``.
    fix_phase : bool, default False
        If True the acrophases are buffers, not parameters: fixed, not trained.
    k_beta : float or None, default 0.2
        Concentration of the soft Von-Mises prior pulling each acrophase toward
        ``params_g["phase"]``. None disables the prior entirely (see the regime
        table above). The paper runs use 2.0.
    phi_init : pd.Series or array-like, optional
        Cold-start values for the acrophases, decoupled from the prior center: the
        prior stays on ``params_g["phase"]`` while training starts here instead.
        Used by the phi_g-recovery simulations to start away from truth. A
        gene-indexed Series is reindexed to the post-GLM gene order; a plain array
        must already align to ``params_g.index``.
    fixed_cell_phases : array-like, optional
        Per-cell known phases, shape [Nc]. Switches the model to fixed-cell mode
        (``Nx=1``): gene parameters are fit against known phases and no posterior
        inference is run.
    noise_model : {"nb", "poisson"}, default "nb"
        Count likelihood.
    fix_disp_val : {"gene", None} or float, default "gene"
        Dispersion shape. "gene" fits one per gene, None fits a single trainable
        scalar, a float fixes the scalar and does not train it.
    log_amp_fn : {"logit", "log"}, default "logit"
        Amplitude parameterization. "logit" bounds the amplitude below a log2FC of 8.
    n_epochs : int, default 300
        Training epochs. 0 skips training and only runs posterior inference against
        the template (the "oracle" path) — pair it with ``kill_amps=False``.
    batch_size : int, default 256
        Cells per minibatch. Cap it at ``adata.n_obs`` for small populations.
    learning_rate : float, default 0.001
        Adam learning rate.
    n_theta : int, default 24
        Phase-grid points used during training.
    n_theta_post : int, default 24
        Phase-grid points used for the final posterior. Raise this (100 is typical)
        for a smooth per-cell posterior; it costs memory, not training time.
    init_mean : bool, default True
        Run the intercept-only GLM warm-start (step 1 above).
    kill_amps : bool, default False
        Zero the template amplitudes before training, so they are learned from
        scratch. Do not combine with ``n_epochs=0`` — nothing would relearn them.
    true_phase : array-like, optional
        Ground-truth phases. Only used to report a per-epoch MAD; it never enters
        the loss. When given (and not in fixed-cell mode) the second return value
        is accompanied by a per-epoch MAD trace.
    posterior_cell_chunk : int, optional
        Process the final posterior in chunks of this many cells. The
        ``(n_theta_post, Nc, Ng)`` likelihood tensor exceeds GPU/host memory for
        large populations; results are identical either way.
    device : str or torch.device, default "cuda"
        Training device. Resolved with :func:`scritmo.ml.utils.resolve_device`, so
        the default falls back to CPU on a machine without a GPU. The returned
        model is always on CPU regardless.
    return_data : bool, default False
        Also return the assembled data tensors (see Returns).

    Other Parameters
    ----------------
    context, context_mode : optional
        Legacy, from the abandoned cellular-context direction. ``context`` is a
        per-cell label vector; None (the default) means one global context, i.e. a
        vector of ones. ``context_mode`` defaults to "none", the standard model;
        "disp_only" freezes the gene parameters and fits only dispersion. See
        :func:`scritmo.ml.utils.set_context_mode`.
    batch, phi_b, k_batch, fixed_prior : optional
        Batch-effect phase shifts: a per-batch shift ``phi_b`` with concentration
        ``k_batch``, trainable unless ``fixed_prior`` is True.
    unspliced_layer, rhythmic_degradation : optional
        Joint spliced/unspliced modeling. Giving ``unspliced_layer`` enables it and
        adds the unspliced tensor to the returns.
    weights_g : array-like, optional
        Per-gene weights on the likelihood, shape [Ng]. Defaults to ones.
    entropy_factor : float, optional
        Weight of a regularizer that penalizes a peaked marginal distribution of
        cells over the phase grid, i.e. encourages cells to spread around the
        circle. None or 0 disables it.

    Returns
    -------
    cmodel : Scritmo
        The trained model, on CPU, with the posterior attributes and
        ``params_g_inf`` populated.
    losses : list
        Per-epoch training loss.
    mad_epochs : list or None
        Per-epoch MAD against ``true_phase``; None unless ``true_phase`` was given.
    data_c, data_u_c : torch.Tensor
        Only when ``return_data=True``: the spliced tensor and the unspliced one
        (None unless ``unspliced_layer`` was given).

    See Also
    --------
    scritmo.ml.context_model.Scritmo.estimate_phase_desynchrony : biological
        desynchrony from the fitted model, corrected for the technical floor.
    scritmo.ml.context_model.Scritmo.from_params_g : build a model from a saved
        parameter table, without refitting.
    """

    device = resolve_device(device)

    # Per-cell context labels are legacy; None means one global context.
    if context is None:
        context = np.ones(adata.n_obs)

    Ng = params_g.shape[0]
    if init_mean == True:
        n_jobs = 1 if Ng < 50 else -1
        par_0 = sr.glm_gene_fit(
            adata,
            phases=np.zeros(adata.n_obs),
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

    mp["batch"] = batch
    if batch is not None:
        mp["phi_b"] = phi_b
        mp["kappa_b"] = k_batch
        mp["fixed_prior"] = fixed_prior

    mp["k_beta"] = k_beta
    mp["rhythmic_degradation"] = rhythmic_degradation

    if fixed_cell_phases is not None:
        mp["fixed_cell_phases"] = fixed_cell_phases

    # Cold-start init for the acrophase, DECOUPLED from the prior center: the
    # Von-Mises prior stays on params_g["phase"] while training starts here.
    # Reindexed to the post-GLM params_g order (robust to the init_mean subset).
    # See the phi_init entry in the docstring.
    if phi_init is not None:
        _phi_init = (
            phi_init.loc[params_g.index].values
            if isinstance(phi_init, pd.Series)
            else np.asarray(phi_init, dtype=float)
        )
        mp["phi_init"] = np.asarray(_phi_init, dtype=float)

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
