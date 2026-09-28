"""
The Scritmo model: a harmonic Negative-Binomial model of gene expression over
an unobserved circadian phase, marginalized over a phase grid.

Typical use::

    model = Scritmo(k_beta=2.0, n_theta=48)
    model.fit(adata, params_g, n_epochs=600)
    model.phases_                     # per-cell posterior mode (radians)
    model.write_obs(adata)            # scritmo_phase, scritmo_phase_std, ...
    df = model.desynchrony(adata, sample_key="sample_name", ext_time_key="ZTmod")

The maths the class calls lives in :mod:`.likelihood` (grid, observation model,
integration, priors) and :mod:`.parameters` (amplitude transforms, ``Beta``
tables); training in :mod:`.training`; desynchrony in :mod:`scritmo.ml.desync`;
post-hoc analyses in :mod:`scritmo.ml.tools`.
"""

import copy

import numpy as np
import pandas as pd
import anndata
import torch
from torch import tensor as tt
from torch import nn

from scritmo import (
    Beta,
    compute_posterior_statistics,
    rh,
    optimal_shift,
    compute_posterior_mode,
    mean_SE,
    mean_AE,
    cSTD,
)
from . import likelihood as lik
from . import parameters as par
from .likelihood import compute_nb_params, compute_poisson_rate  # noqa: F401
from ..utils import set_context_mode, nmp, resolve_device, assemble_mp
from ..misc.power_spherical.power_spherical import log_von_mises
from ..unspliced.unspliced_deg import UnsplicedMixin
from ..desync.results import create_results_dataframe
from ..desync.technical_sim import simulate_cell_populations, simulate_technical_grid
from ..desync.estimate import (
    estimate_phase_desynchrony as _estimate_phase_desynchrony,
    _attach_deconvolution,
)
from ..tools._compat import NullModelMixin, GenomeFitMixin
from .. import marginalization as _marg

# Model hyperparameters taken by the constructor, in the legacy positional order
# (after ``mp, y``) followed by the keyword-only ones the estimator API added.
_LEGACY_ARGS = (
    "context_mode", "fix_phase", "noise_model", "fix_disp_val", "log_amp_fn",
    "method", "entropy_factor", "phase_range",
)
_DEFAULTS = dict(
    context_mode="none", fix_phase=False, noise_model="nb", fix_disp_val="gene",
    log_amp_fn="logit", method=None, entropy_factor=None, phase_range=None,
    k_beta=0.2, n_theta=24, rhythmic_degradation=True, unspliced_param="ratio",
    device="cuda",
)


def _obs_or_array(adata, value, name):
    """``value`` as a per-cell array: an ``adata.obs`` column name, or array-like."""
    if value is None:
        return None
    if isinstance(value, str):
        if value not in adata.obs:
            raise KeyError(f"{name}={value!r} is not a column of adata.obs")
        return adata.obs[value].values
    return value


class Scritmo(nn.Module, UnsplicedMixin):
    """
    scRITMO circadian phase-inference model for single-cell RNA-seq.

    Each gene is modeled as a harmonic (Fourier) function of an unobserved
    circadian phase, with Negative-Binomial (or Poisson) counts. Per-cell phases
    are not point parameters: the likelihood is marginalized over a fixed phase
    grid (``Nx`` points on the circle, integrated with Simpson's rule or a plain
    sum), which yields a full posterior over phase for every cell. Gene
    parameters fit during training are the log-mesor (``m_g``), the harmonic
    acrophase (``acrophase``) and amplitude (``log_amp``), plus dispersion.

    Use it like an estimator::

        model = Scritmo(k_beta=2.0, n_theta=48)       # hyperparameters only
        model.fit(adata, params_g, n_epochs=600)      # returns the model
        model.phases_, model.phases_std_, model.params_
        model.predict(new_adata)                      # phases for new cells
        model.write_obs(adata)                        # results into adata.obs
        model.desynchrony(adata, sample_key=..., ext_time_key=...)

    ``warmup_and_train(adata, params_g, ...)`` is the same fit through a single
    function call and returns ``(model, losses, mad_epochs)``, as it always has.
    The lower-level form ``Scritmo(mp, y, ...)`` builds the model at once from
    an assembled ``mp`` dict and data tensor (see :meth:`__init__`).

    Structure: the class holds the model state and the likelihood; the pure
    maths is in :mod:`scritmo.ml.model.likelihood` and
    :mod:`scritmo.ml.model.parameters`. ``UnsplicedMixin`` adds joint
    spliced/unspliced modeling (its own parameters, plus Fisher/Cramér–Rao
    uncertainties). Post-hoc analyses (:mod:`scritmo.ml.tools`) and the
    desynchrony pipeline (:mod:`scritmo.ml.desync`) are functions of a fitted
    model; the methods of the same names here forward to them.

    Note
    ----
    ``ContextModel`` is a backward-compatible alias for this class (its historical
    name); the two are the same object. Pickles refer to the class as
    ``scritmo.ml.context_model.Scritmo``, which keeps them loadable by older
    scritmo versions too.

    Legacy
    ------
    The class was originally written to ask whether the cellular context (e.g.
    celltype identity) reshapes a gene's Fourier coefficients, hence the historical
    name and the per-context intercept/amplitude parameters (``m_yg``,
    ``log_lambda_y``, selected by ``context_mode``). That research direction was
    abandoned. Those parameters are initialized to zero and frozen under the
    default ``context_mode="none"``, so the model reduces exactly to the
    no-context form; they survive only so old pickles and the paper scripts keep
    reproducing. See ``context_mode`` in :meth:`__init__` and
    :func:`scritmo.ml.utils.set_context_mode`. Per-cell context labels
    (``mp["context"]``) are likewise optional — ``None`` means one global context.
    The unrelated ``context_col`` argument of the desynchrony methods is still
    live: there it is just the ``adata.obs`` column used to group samples.
    """

    def __init__(
        self,
        mp=None,
        y=None,
        context_mode="none",
        fix_phase=False,
        noise_model="nb",
        fix_disp_val="gene",
        log_amp_fn="logit",
        method=None,
        entropy_factor=None,
        phase_range=None,
        *,
        k_beta=0.2,
        n_theta=24,
        rhythmic_degradation=True,
        unspliced_param="ratio",
        device="cuda",
    ):
        """
        Create a model: either unfitted (estimator form) or built from data.

        **Estimator form** — ``Scritmo(k_beta=2.0, n_theta=48, ...)`` stores the
        hyperparameters only; :meth:`fit` assembles the data and builds the
        parameters. ``k_beta``, ``n_theta``, ``rhythmic_degradation``,
        ``unspliced_param`` and ``device`` are used by :meth:`fit` (see
        :func:`scritmo.ml.warmup_and_train` for their meaning).

        **Built form** — ``Scritmo(mp, y, ...)`` builds the model at once from an
        assembled parameter/data bundle (what ``warmup_and_train`` did before the
        estimator API existed). ``mp`` then carries ``k_beta`` and the other
        per-fit options; the keyword-only hyperparameters are ignored.

        Args:
            mp: Model-parameters dict holding initial values and data. Required keys:
                - "params_g": a ``Beta`` DataFrame with columns "a_0" (log-mesor),
                  "amp", "phase" — seeds the gene parameters and the harmonic order.
                - "counts": per-cell library sizes, shape [Nc].
                Optional keys:
                - "context": per-cell context labels, shape [Nc] (legacy). Absent
                  or None means one global context, i.e. a vector of ones.
                - "phi_init": acrophase initialization decoupled from the prior
                  center (lets phi_g start away from the reference template).
                - "k_beta": concentration of the soft Von-Mises prior on acrophases.
                - "batch", "kappa_b", "phi_b", "fixed_prior": batch-effect phase shifts.
                - "weights_g": per-gene loss weights, shape [Ng].
                - "fixed_cell_phases": if given, run in fixed-cell-phase mode (Nx=1).
                - "unspliced_mode": enable joint spliced/unspliced modeling.
            y: Data tensor of shape [Nx, Nc, Ng] (phase grid × cells × genes).
            context_mode: Which parameter groups are trainable. Two values are in
                current use:
                - "none": the standard model (default). Mesor, acrophase, amplitude
                  and dispersion are fit; the legacy context terms stay frozen at
                  zero, so this is exactly the no-context model.
                - "disp_only": freeze every gene parameter and fit only dispersion.
                Legacy (abandoned cellular-context direction; these emit a
                DeprecationWarning and are kept only for reproducibility):
                "intercept", "lambda", "full", "full_lambda", "context_only".
                See :func:`scritmo.ml.utils.set_context_mode`.
            fix_phase: If True, acrophase parameters are fixed (not trained)
            noise_model: "nb" for Negative Binomial, "poisson" for Poisson
            fix_disp_val: Controls dispersion initialization and shape.
                - "gene": Per-gene dispersion (Ng,), trainable. If params_g has a "disp"
                  column (e.g. from a previous run), warm-starts from those values.
                - "context": Per-context dispersion (Ny, 1), trainable (legacy;
                  equivalent to a single scalar when there is one context).
                - None: Single scalar dispersion, trainable.
                - float: Fixed scalar dispersion, not trained.
            log_amp_fn: "logit" or "log" to control amplitude parameterization
            method: Integration method, "simpson" or "sum". None (default) picks
                "simpson" on the full circle and "sum" on an arc. "simpson" is
                refused on an arc: the periodic rule wraps the last grid point
                onto the first, which is only true on the full circle.
            entropy_factor: Optional weight for the phase-entropy regularizer. When set
                (and not in fixed-phase mode), an extra term is added to the loss that
                penalizes a peaked marginal distribution of cells over the phase grid,
                encouraging cells to spread around the circle (analogous to CoPhaser's
                circular entropy term). None or 0 disables it.
            phase_range: Optional ``(lo, hi)`` in radians, with ``0 < hi - lo < 2π``.
                Restricts cell phases to the arc [lo, hi]: the phase grid covers
                only the arc (midpoint rule, integrated with a plain sum) and the
                cell prior is uniform on it, ``1 / (hi - lo)``. Use it when every
                cell is known to come from part of the cycle, e.g. samples
                collected only during the day. None (default) is the full circle.
            k_beta, n_theta, rhythmic_degradation, unspliced_param, device:
                Estimator form only, see above.
        """
        super().__init__()
        self.config = dict(
            context_mode=context_mode,
            fix_phase=fix_phase,
            noise_model=noise_model,
            fix_disp_val=fix_disp_val,
            log_amp_fn=log_amp_fn,
            method=method,
            entropy_factor=entropy_factor,
            phase_range=phase_range,
            k_beta=k_beta,
            n_theta=n_theta,
            rhythmic_degradation=rhythmic_degradation,
            unspliced_param=unspliced_param,
            device=device,
        )
        self._built = False
        if mp is None:
            if y is not None:
                raise ValueError("Pass both mp and y, or neither (then call fit).")
            return
        if y is None:
            raise ValueError("Scritmo(mp, y): the data tensor y is missing.")
        self._build(mp, y)

    def _build(self, mp, y):
        """Build buffers and parameters from an assembled ``mp`` dict and ``y``."""
        context_mode, fix_phase, noise_model, fix_disp_val, log_amp_fn, method, \
            entropy_factor, phase_range = (self.config[k] for k in _LEGACY_ARGS)

        self.Nx, self.Nc, self.Ng = y.shape
        self.nh = mp["params_g"].num_harmonics()
        self.dev = y.device

        # Per-cell context labels are optional: absent or None means one global
        # context, i.e. a single-column design matrix of ones.
        context = mp.get("context")
        if context is None:
            context = np.ones(self.Nc)

        self.register_buffer("dm", self.design_matrix(context))
        self.Ny = self.dm.shape[1]
        self._set_phase_range(phase_range)
        if method is None:
            method = "simpson" if self.phase_range is None else "sum"
        if method == "simpson" and self.phase_range is not None:
            raise ValueError(
                "method='simpson' assumes a periodic grid; use 'sum' with phase_range."
            )
        self.method = method
        self.entropy_factor = entropy_factor
        self.register_buffer("counts", mp["counts"].clone())
        self.context = context
        self.context_u = np.unique(context)
        self.genes = mp["params_g"].index.values
        self.fix_phase = fix_phase
        self.max_amp = par.MAX_AMP  # log2fc of 8
        self.noise_model = noise_model
        self.fix_disp_val = fix_disp_val
        self.log_amp_fn = log_amp_fn
        self.unspliced_mode = mp.get("unspliced_mode", False)
        self.ll_xcg = None

        if "fixed_cell_phases" in mp and mp["fixed_cell_phases"] is not None:
            fixed_cell_mode = True
        else:
            fixed_cell_mode = False

        X = self.X_matrix(mp=mp, fixed_cell_mode=fixed_cell_mode)
        self.register_buffer("X", X)

        # batch related hyperparameters
        if "batch" in mp and mp["batch"] is not None:
            self.batch_mode = True
            dm_batch = self.design_matrix(mp["batch"]).to(self.dev)
            self.register_buffer("dm_batch", dm_batch)
            self.kappa_b = nn.Parameter(mp["kappa_b"])
            self.phi_b = nn.Parameter(mp["phi_b"])
            self.phi_b.requires_grad = False
            self.kappa_b.requires_grad = False
            if "fixed_prior" in mp and not mp["fixed_prior"]:
                self.phi_b.requires_grad = True
                self.kappa_b.requires_grad = True

        else:
            self.batch_mode = False

        if "weights_g" in mp and mp["weights_g"] is not None:
            self.register_buffer("weights_g", mp["weights_g"].to(self.dev))
        else:
            self.register_buffer(
                "weights_g",
                torch.ones((self.Ng,), dtype=torch.float32, device=self.dev),
            )

        ###############
        # parameters
        ###############

        # Acrophase INITIALIZATION. By default seeded from the reference template
        # (mp["params_g"]["phase"]). An optional mp["phi_init"] decouples the init from
        # the reference so phi_g can start AWAY from truth (e.g. perturbed/random) while
        # the Von-Mises prior below stays centered on the reference (prior_g). Backward
        # compatible: absent/None -> identical to the original behavior.
        if mp.get("phi_init") is not None:
            acrophase_tensor = tt(
                np.asarray(mp["phi_init"], dtype=float), dtype=torch.float32
            )
        else:
            acrophase_tensor = tt(mp["params_g"]["phase"].values, dtype=torch.float32)

        # Original amplitude values
        amp_values = tt(mp["params_g"]["amp"].values, dtype=torch.float32)
        self.log_amp = nn.Parameter(par.raw_amplitude(amp_values, log_amp_fn, self.max_amp))

        # Base parameters
        if fix_phase:
            self.register_buffer("acrophase", acrophase_tensor)
        else:
            self.acrophase = nn.Parameter(acrophase_tensor)

        self.m_g = nn.Parameter(tt(mp["params_g"]["a_0"].values, dtype=torch.float32))

        if self.unspliced_mode:
            self.prepare_unspliced_genes(mp)

        # beta prior parameters
        if "k_beta" in mp and mp["k_beta"] is not None:
            self.register_buffer("k_beta", tt(mp["k_beta"]).to(y.device))
            self.register_buffer(
                "prior_g",
                tt(mp["params_g"]["phase"].values, dtype=torch.float32).to(y.device),
            )
        else:
            self.k_beta = None

        if fix_disp_val is None:
            self.log_disp = nn.Parameter(tt(-1.0))
        elif fix_disp_val == "context":
            self.log_disp = nn.Parameter(-torch.ones(self.Ny, 1))
        elif fix_disp_val == "gene":
            if "disp" in mp["params_g"].columns:
                self.log_disp = nn.Parameter(
                    tt(np.log(mp["params_g"]["disp"].values), dtype=torch.float32)
                )
            else:
                self.log_disp = nn.Parameter(-torch.ones(self.Ng))
        else:
            self.log_disp = nn.Parameter(tt(np.log(fix_disp_val)))
            self.log_disp.requires_grad = False

        # LEGACY: per-context intercepts (m_yg) and amplitude scaling
        # (log_lambda_y) from the abandoned cellular-context direction. Both are
        # zero-initialized -> intercept offset 0 and exp(0) = 1 amplitude scale, and
        # set_context_mode("none") freezes them, so the default model is exactly the
        # no-context model. Kept so old pickles stay loadable and the paper scripts
        # keep reproducing; see the class docstring.
        self.m_yg = (
            nn.Parameter(mp["m_yg"].clone().detach())
            if "m_yg" in mp
            else nn.Parameter(torch.zeros(self.dm.shape[1], self.Ng))
        )

        if context_mode == "full_lambda":
            self.log_lambda_y = nn.Parameter(torch.zeros(self.Ny, self.Ng))
        else:
            self.log_lambda_y = nn.Parameter(torch.zeros(self.Ny, 1))

        # Set the context mode and update parameter gradients accordingly
        self.set_context_mode(context_mode)
        self.context_mode = context_mode
        self._built = True

    def get_params(self):
        """The constructor hyperparameters (a copy of :attr:`config`)."""
        return dict(self.config)

    def __setstate__(self, state):
        # Models pickled before the estimator API have no config/_built: they
        # are always built, and their hyperparameters are on the instance.
        super().__setstate__(state)
        if "_built" not in self.__dict__:
            self._built = True
        if "config" not in self.__dict__:
            cfg = dict(_DEFAULTS)
            for k in _LEGACY_ARGS:
                if k in self.__dict__:
                    cfg[k] = self.__dict__[k]
            cfg["method"] = self.__dict__.get("method")
            self.config = cfg

    def _check_built(self, what="this"):
        if not getattr(self, "_built", True):
            raise RuntimeError(f"Call fit() before {what}.")

    set_context_mode = set_context_mode

    def _set_phase_range(self, phase_range):
        """Store the phase support: None (full circle) or an arc ``(lo, hi)``."""
        if phase_range is None:
            self.phase_range = None
            self.phase_width = 2 * np.pi
            return
        lo, hi = (float(v) for v in phase_range)
        if not 0 < hi - lo < 2 * np.pi:
            raise ValueError(
                f"phase_range needs 0 < hi - lo < 2π, got ({lo:.3f}, {hi:.3f})."
            )
        self.phase_range = (lo, hi)
        self.phase_width = hi - lo

    @classmethod
    def from_params_g(
        cls,
        params_g,
        context_mode="none",
        fix_phase=False,
        noise_model="nb",
        fix_disp_val="gene",
        log_amp_fn="logit",
        device="cpu",
    ):
        """
        Initialize a Scritmo model from gene parameters only, without adata.

        Creates a model with gene parameters loaded from params_g but with
        no real dataset attached.  Dataset-specific buffers (X, dm, counts)
        are populated with dummy values for a single cell and are rebuilt
        automatically when get_inferred_phases is called with real data
        (the Nc-mismatch path in get_phase_posteriors handles this, but
        requires context_mode="none" and explicit counts= argument).

        Typical use:
            params_g = sr.Beta("saved_params.csv")
            cmodel = Scritmo.from_params_g(params_g)
            # later, with real adata:
            data_c, mp = assemble_mp(adata, params_g, labels=None, ...)
            cmodel.get_inferred_phases(data_c, counts=mp["counts"], n_theta=100)
            bic_df = cmodel.fit_null_model(adata, counts=counts)

        Args:
            params_g      : Beta (or DataFrame) with columns a_0, amp, phase
                            (and optionally disp for warm-start dispersion).
            context_mode  : Leave at "none" (required for inference on new data).
            fix_phase     : Whether to fix acrophase parameters.
            noise_model   : "nb" or "poisson".
            fix_disp_val  : Dispersion mode (see __init__ docstring).
            log_amp_fn    : "logit" or "log".
            device        : Torch device string. Resolved with
                            :func:`scritmo.ml.utils.resolve_device`.
        """
        from scritmo import Beta

        device = resolve_device(device)
        params_g = Beta(params_g)
        Ng = len(params_g)

        # minimal mp with a single dummy cell
        mp = {
            "params_g": params_g,
            "context": None,
            "counts": torch.ones(1, 1, dtype=torch.float32, device=device),
            "weights_g": None,
            "k_beta": None,
            "rhythmic_degradation": False,
            "batch": None,
        }
        # dummy data tensor: (Nx=1, Nc=1, Ng)
        y_dummy = torch.zeros(1, 1, Ng, dtype=torch.float32, device=device)

        return cls(
            mp,
            y_dummy,
            context_mode=context_mode,
            fix_phase=fix_phase,
            noise_model=noise_model,
            fix_disp_val=fix_disp_val,
            log_amp_fn=log_amp_fn,
        )

    def design_matrix(self, vec, all_categories=None):
        """
        Creates a one-hot encoded design matrix from a vector.

        Args:
            vec (list or np.array): The input vector of categories to encode.
                None gives a single all-ones column, i.e. one global context.
            all_categories (list, optional): The complete list of all possible
                                             categories. If provided, the output
                                             matrix will have a column for each of
                                             these, ensuring consistent shape.

        Returns:
            torch.Tensor: The resulting one-hot encoded tensor.
        """
        return lik.one_hot(vec, self.Nc, self.dev, all_categories)

    def X_matrix(self, fixed_cell_mode, n_theta=None, mp=None):
        """
        Build the harmonic design matrix over the phase axis, and set ``Nx``.

        Also sets ``self.fixed_cell_mode``, which the loss and the posterior
        methods branch on.

        Args:
            fixed_cell_mode: If True, each cell has one known phase instead of a
                grid: reads ``mp["fixed_cell_phases"]`` (shape [Nc]), registers it
                as the ``phi_c`` buffer and sets ``Nx = 1``. ``mp`` is required in
                this branch.
            n_theta: Number of grid points on the circle (or on the arc, see
                :meth:`phase_grid`). Defaults to ``self.Nx``. Ignored in
                fixed-cell mode.
            mp: The model-parameters dict; only read in fixed-cell mode.

        Returns:
            The design tensor, shape [Nx, Nc, 2*nh]. Also registers the ``phi_x``
            (grid) or ``phi_c`` (fixed) phase buffer.
        """
        if fixed_cell_mode:
            self.fixed_cell_mode = True

            # Expecting shape [Nc]
            phi_c = tt(mp["fixed_cell_phases"], dtype=torch.float32)
            if phi_c.shape[0] != self.Nc:
                raise ValueError(
                    f"fixed_cell_phases length {phi_c.shape[0]} does not match Nc {self.Nc}"
                )

            self.register_buffer("phi_c", phi_c)
            X_tensor = lik.harmonic_dm_torch(self.phi_c, self.nh, False)
            self.Nx = 1
            return X_tensor.unsqueeze(0)

        else:
            if n_theta is None:
                n_theta = self.Nx
            self.fixed_cell_mode = False
            phi_x_tensor = self.phase_grid(n_theta)
            self.register_buffer("phi_x", phi_x_tensor)

            # Nx Np -> Nx Nc Np
            return lik.grid_design(phi_x_tensor, self.nh, self.Nc)

    # ------------------------------------------------------------------
    # estimator API
    # ------------------------------------------------------------------

    def fit(
        self,
        adata,
        params_g,
        *,
        layer="spliced",
        counts=None,
        unspliced_layer=None,
        context=None,
        batch=None,
        phi_b=None,
        k_batch=None,
        fixed_prior=False,
        weights_g=None,
        fixed_cell_phases=None,
        phi_init=None,
        init_mean=True,
        kill_amps=False,
        n_epochs=300,
        batch_size=256,
        learning_rate=0.001,
        true_phase=None,
        n_theta_post=24,
        posterior_cell_chunk=None,
    ):
        """
        Fit the model to ``adata`` and infer a phase posterior for every cell.

        Starts from the template ``params_g`` (a ``Beta`` table with columns
        ``a_0``, ``amp``, ``phase``; optionally ``disp``). Afterwards the results
        are on the model: :attr:`phases_`, :attr:`phases_mean_`,
        :attr:`phases_std_`, :attr:`params_`, :attr:`losses_` and
        :attr:`mad_epochs_`. Calling ``fit`` again starts from scratch.

        ``counts``, ``context`` and ``batch`` accept an ``adata.obs`` column name
        or a per-cell array. ``params_g`` is not modified. Every argument has the
        meaning documented in :func:`scritmo.ml.warmup_and_train`.

        Returns:
            self
        """
        self._fit(
            adata, params_g.copy(), layer=layer, counts=counts,
            unspliced_layer=unspliced_layer, context=context, batch=batch,
            phi_b=phi_b, k_batch=k_batch, fixed_prior=fixed_prior,
            weights_g=weights_g, fixed_cell_phases=fixed_cell_phases,
            phi_init=phi_init, init_mean=init_mean, kill_amps=kill_amps,
            n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate,
            true_phase=true_phase, n_theta_post=n_theta_post,
            posterior_cell_chunk=posterior_cell_chunk,
        )
        return self

    def _fit(self, adata, params_g, **kw):
        """:meth:`fit` without the input copy; returns the data tensors used."""
        from .training import fit_model

        if self._built:
            # start from scratch, as sklearn estimators do: drop every parameter,
            # buffer and result of the previous fit, keep the hyperparameters
            config = self.config
            self.__dict__.clear()
            nn.Module.__init__(self)
            self.config = config
            self._built = False
        kw["counts"] = _obs_or_array(adata, kw.get("counts"), "counts")
        kw["context"] = _obs_or_array(adata, kw.get("context"), "context")
        kw["batch"] = _obs_or_array(adata, kw.get("batch"), "batch")
        return fit_model(self, adata, params_g, **kw)

    @property
    def phases_(self):
        """Posterior mode of each cell's phase, radians (``post_mode_c``)."""
        return self._result("post_mode_c")

    @property
    def phases_mean_(self):
        """Circular posterior mean of each cell's phase, radians (``post_mean_c``)."""
        return self._result("post_mean_c")

    @property
    def phases_std_(self):
        """Circular posterior std of each cell's phase, radians (``post_std_c``)."""
        return self._result("post_std_c")

    @property
    def params_(self):
        """The fitted gene parameters as a ``Beta`` table (see :meth:`get_parameter_dataframe`)."""
        self._check_built("reading params_")
        if "params_g_inf" in self.__dict__:
            return self.__dict__["params_g_inf"]
        return self.get_parameter_dataframe()

    def _result(self, name):
        if name not in self.__dict__:
            raise AttributeError(
                f"No phase posterior yet: fit the model (fixed-phase fits have "
                f"no posterior), or call get_inferred_phases(). Missing: {name}."
            )
        return self.__dict__[name]

    def predict(
        self, adata, *, layer="spliced", counts=None, n_theta=100, cell_chunk=None
    ):
        """
        Infer phases for the cells of ``adata`` with the fitted gene parameters.

        The model itself is not changed. ``adata`` must contain the model's genes
        (:attr:`genes`); ``counts`` (library size per cell, or an ``adata.obs``
        column name) defaults to the same fallback as :meth:`fit`.

        Returns:
            DataFrame indexed like ``adata.obs`` with ``phase`` (posterior mode),
            ``phase_mean``, ``phase_std`` (radians) and ``phase_h`` (hours).
        """
        self._check_built("predict()")
        if self.unspliced_mode:
            raise NotImplementedError("predict() does not support unspliced models.")
        if getattr(self, "fixed_cell_mode", False):
            raise NotImplementedError("predict() needs a model fit with free phases.")
        counts = _obs_or_array(adata, counts, "counts")
        data_c, mp = assemble_mp(
            adata, self.get_parameter_dataframe(), labels=None, counts=counts,
            layer=layer, n_theta=1, device="cpu",
        )
        work = copy.deepcopy(self).to("cpu")
        work.dev = "cpu"
        work.get_inferred_phases(
            data_c, counts=mp["counts"], n_theta=n_theta, cell_chunk=cell_chunk
        )
        return pd.DataFrame(
            {
                "phase": work.post_mode_c,
                "phase_mean": work.post_mean_c,
                "phase_std": work.post_std_c,
                "phase_h": work.post_mode_c * rh,
            },
            index=adata.obs_names,
        )

    def write_obs(self, adata, prefix="scritmo_"):
        """
        Write the per-cell results of the fit into ``adata.obs``.

        Adds ``{prefix}phase`` (posterior mode), ``{prefix}phase_mean``,
        ``{prefix}phase_std`` (radians) and ``{prefix}phase_h`` (hours).
        ``adata`` must be the data the model was fit on (same cells, same order).
        """
        phases = self.phases_
        if adata.n_obs != len(phases):
            raise ValueError(
                f"adata has {adata.n_obs} cells but the model was fit on {len(phases)}."
            )
        adata.obs[f"{prefix}phase"] = phases
        adata.obs[f"{prefix}phase_mean"] = self.phases_mean_
        adata.obs[f"{prefix}phase_std"] = self.phases_std_
        adata.obs[f"{prefix}phase_h"] = phases * rh
        return adata

    def desynchrony(
        self,
        adata,
        *,
        sample_key,
        ext_time_key,
        context_key=None,
        ext_phase=None,
        method="simulation",
        layer="spliced",
        **kwargs,
    ):
        """
        Biological phase desynchrony per sample, corrected for the technical floor.

        The same pipeline as :meth:`estimate_phase_desynchrony`, with the
        ``adata.obs`` keys given explicitly and ``adata`` left unchanged.

        Args:
            adata: The data the model was fit on.
            sample_key: ``adata.obs`` column with the sample / replicate label.
            ext_time_key: ``adata.obs`` column with the external (collection) time, hours.
            context_key: Optional ``adata.obs`` column grouping samples (e.g.
                celltype or condition). None puts every cell in one group.
            ext_phase: Optional reference phase per cell (radians) to align to.
            method: Technical floor, "simulation", "harmonic" or "deconvolution"
                (``sigma_tech_method``).
            layer: Count layer.
            **kwargs: Any other argument of :func:`scritmo.ml.desync.estimate_phase_desynchrony`.

        Returns:
            DataFrame with one row per group (see :meth:`estimate_phase_desynchrony`).
        """
        added = context_key is None and "context" not in adata.obs
        try:
            return _estimate_phase_desynchrony(
                self, adata, ext_phase=ext_phase, context_col=context_key,
                sample_col=sample_key, ext_time_col=ext_time_key, layer=layer,
                sigma_tech_method=method, **kwargs,
            )
        finally:
            if added and "context" in adata.obs:
                del adata.obs["context"]

    # ------------------------------------------------------------------
    # likelihood (maths in likelihood.py)
    # ------------------------------------------------------------------
    def phase_grid(self, n_theta=None, device=None):
        """
        The phase grid the likelihood is evaluated on.

        On the full circle: ``n_theta`` even points on [0, 2π), endpoint excluded.
        On an arc (``phase_range``): the ``n_theta`` midpoints of equal bins of
        [lo, hi], so a plain sum times ``phase_width / n_theta`` is the midpoint rule.

        Args:
            n_theta: Number of grid points. Defaults to ``self.Nx``.
            device: Torch device of the result. Defaults to CPU.

        Returns:
            1D float32 tensor of phases in radians, shape [n_theta].
        """
        n = self.Nx if n_theta is None else n_theta
        # getattr: models pickled before phase_range existed are full-circle
        return lik.phase_grid(n, getattr(self, "phase_range", None), device=device)

    def posterior_grid(self, n_theta):
        """
        Numpy phase grid for the posterior helpers in :mod:`scritmo.circular`.

        None on the full circle, so those helpers use their own float64 [0, 2π)
        grid exactly as before ``phase_range`` existed; the arc grid otherwise.
        """
        if getattr(self, "phase_range", None) is None:
            return None
        return nmp(self.phase_grid(n_theta)).astype(np.float64)

    def model_formula(self, indices=slice(None), counts=None, n_theta=None):
        """
        Computes the expected mean (rate) E_xcg for the model.

        Args:
            indices: Indices for cell selection.
            counts: Optional counts tensor.
            n_theta: Optional number of phase bins.

        Returns:
            E_xcg: Expected mean tensor.
            disp: Dispersion tensor.
        """
        if counts is None:
            counts = self.counts[indices]

        if n_theta is not None:
            phi_x_new = self.phase_grid(n_theta, device=self.dev)
            X = lik.grid_design(phi_x_new, self.nh, self.Nc)[:, indices, :]
        else:
            X = self.X[:, indices, :]

        dm = self.dm[indices, :]

        # m_yg / log_lambda_y are the legacy context terms; frozen at zero under
        # context_mode="none", so these reduce to m_g and 1 respectively.
        intercept_cg = torch.matmul(dm, self.m_yg) + self.m_g
        lambda_cg = torch.matmul(dm, self.log_lambda_y.exp())
        disp = torch.exp(self.log_disp)
        if self.fix_disp_val == "context":
            disp = torch.matmul(dm, disp)
        ab = self._get_ab()

        # E_xcg is the expected mean of the distribution
        E_xcg = (X @ ab) * lambda_cg + intercept_cg

        return E_xcg, disp, counts

    def nb_dist(self, indices=slice(None), counts=None, n_theta=None):
        """
        Computes the likelihood distribution (Negative Binomial or Poisson)
        for the given data.

        Called by several methods.
        """
        E_xcg, disp, counts = self.model_formula(
            indices=indices, counts=counts, n_theta=n_theta
        )

        return lik.count_distribution(self.noise_model, E_xcg, disp, counts)

    def beta_prior(self):
        # prior_g = log_power_spherical(self.acrophase, self.prior_g, self.k_beta)

        if self.k_beta is None:
            return tt(0.0, device=self.dev)
        else:
            prior_g = log_von_mises(self.acrophase, self.prior_g, self.k_beta)
            return prior_g.sum()

    def forward(self, y, indices=slice(None), y_u=None, **kwargs):
        """
        Forward pass with category-specific intercepts.
        the data y is already been batched. But the indices are still
        needed for the celltypes.
        """

        dist = self.nb_dist(indices=indices)
        ll_xcg = dist.log_prob(y)

        if self.unspliced_mode:
            if y_u is None:
                raise ValueError(
                    "y_u (unspliced data) must be provided in unspliced_mode."
                )
            dist_u = self.nb_dist_unspliced(indices=indices)
            ll_u_xcg = dist_u.log_prob(y_u)  # Unspliced log-likelihood
            #  the unspliced loss is added to the spliced one
            ll_xcg = ll_xcg + ll_u_xcg  # Combine likelihoods

        Nb = ll_xcg.shape[0]

        # cells priors
        loss = tt(0.0, device=self.dev)

        # sum over genes, add here a weighted sum?
        ll_xc = (ll_xcg * self.weights_g).sum(2)

        if self.fixed_cell_mode:
            loss_like = -ll_xc.squeeze(0).sum()
            # So here we probably want negative sum
        else:
            l_prior_xc = self.cell_prior(indices)
            # Expected marginalized log likelihood
            l_c, max_c, l_xc = self.marginalize_theta(
                ll_xc, l_prior_xc, method=self.method, return_integrand=True
            )
            loss_like = -self.log_like_loss(l_c, max_c)

            # optional phase-entropy regularizer: penalize a peaked marginal
            # distribution of cells over the phase grid (encourages cells to
            # spread around the circle, like CoPhaser's circular entropy term)
            if self.entropy_factor:
                # per-cell posterior over the grid (max_c shift cancels here)
                p_xc = l_xc / (l_xc.sum(dim=0, keepdim=True) + 1e-10)  # (Nx, Nc)
                q_x = p_xc.mean(dim=1)  # (Nx,) marginal phase distribution
                q_x = q_x / (q_x.sum() + 1e-10)
                H = -(q_x * (q_x + 1e-10).log()).sum()  # 0 .. log(Nx)
                Nc = ll_xc.shape[1]
                # scale by Nc so the term is commensurate with the cell-summed
                # likelihood and entropy_factor is batch-size invariant
                loss = loss - self.entropy_factor * Nc * H
                self.last_entropy = H.item()

        loss_beta = -self.beta_prior()
        loss += loss_like + loss_beta * Nb

        return loss

    vectorized_simpson = staticmethod(lik.periodic_simpson)

    marginalize_theta_svi = _marg.MarginalizationMixin.marginalize_theta_svi

    log_like_loss = staticmethod(lik.log_like_loss)

    def _phase_width(self):
        # getattr: models pickled before phase_range existed are full-circle
        return getattr(self, "phase_width", 2 * torch.pi)

    def marginalize_theta(
        self, ll_xc_, log_prior, method="simpson", return_integrand=False
    ):
        r"""
        Log marginal likelihood per cell, :math:`\int P(D|\theta) P(\theta) d\theta`,
        integrated over the phase grid (Simpson or plain sum). Returns
        ``(l_c, max_c)``, plus the integrand ``l_xc`` if ``return_integrand``.
        """
        l_c, max_c, l_xc = lik.marginalize(
            ll_xc_, log_prior, method, self.phi_x, self._phase_width(), self.Nx
        )
        if return_integrand:
            return l_c, max_c, l_xc
        return l_c, max_c

    def cell_prior(self, indices=None, n_theta=None):
        """
        Log prior over the phase grid: flat on the support, or a Von-Mises around
        each cell's batch phase in batch mode. ``n_theta`` is ignored.
        """
        if indices is None:
            indices = slice(None)
        if self.batch_mode:
            return lik.batch_log_prior(
                self.phi_x, self.phi_b, self.kappa_b, self.dm_batch[indices, :]
            )
        return lik.flat_log_prior(self._phase_width())

    def normalize_log_dist(self, ll_xc, method="simpson"):
        """
        this method gets a log likelihood with format
        xc (where x is the phase and c the cell index)
        and it normalizes w.r.t. the x variable
        """
        return lik.normalize_log_dist(
            ll_xc,
            method,
            self.phi_x,
            getattr(self, "phase_width", 2 * np.pi),
            self.Nx,
            on_arc=getattr(self, "phase_range", None) is not None,
        )

    # ------------------------------------------------------------------
    # phase posteriors
    # ------------------------------------------------------------------
    def get_phase_posteriors(
        self,
        y,
        y_u=None,
        method="sum",
        return_all=False,
        counts=None,
        n_theta=None,
        cell_chunk=None,
    ):
        """
        It gives you the posterior distribution of the phase
        for each cell given the fitted model parameters and data

        Args:
            y: Data tensor
            cell_chunk: if not None, process cells in chunks of this size to
                bound peak GPU memory. The full-population log-likelihood tensor
                scales as (n_theta, Nc, Ng); for large cell types this exceeds
                GPU memory, so chunking over cells keeps it tractable. Results
                are identical to the unchunked path (default None).
        Returns:
        """
        if self.fixed_cell_mode:
            raise RuntimeError("Cannot compute phase posteriors in fixed-phase mode.")

        # in case I use a smaller or different dataset (different Nc): rebuild the
        # Nc-dependent buffers (cheap) before any per-chunk forward pass.
        if y.shape[1] != self.Nc:
            self.Nc = y.shape[1]

            if self.context_mode != "none":
                raise ValueError(
                    "Transfer learning with context effects is not supported."
                )
            elif counts is None:
                raise ValueError(
                    "Counts must be provided when evaluating on new cells."
                )
            else:
                # adjust the Nc dependent parameters, first X. Without an explicit
                # n_theta, keep the model's own grid size (y then carries Nx rows).
                n_grid = n_theta if n_theta is not None else self.Nx
                phi_x_tensor = self.phase_grid(n_grid, device=self.m_g.device)
                self.register_buffer("X", lik.grid_design(phi_x_tensor, self.nh, self.Nc))
                # adjust dm
                self.register_buffer("dm", self.design_matrix(np.ones(self.Nc)))

        Nc_total = y.shape[1]
        if cell_chunk is None or cell_chunk >= Nc_total:
            cell_chunk = Nc_total

        post_chunks, lmle_chunks = [], []

        # Run forward calculation without computing gradients, chunking over cells
        with torch.no_grad():
            for c0 in range(0, Nc_total, cell_chunk):
                c1 = min(c0 + cell_chunk, Nc_total)
                idx = slice(c0, c1)

                # build the (n_theta, chunk, Ng) data slice for this chunk
                if n_theta is not None:
                    y_c = y[0, idx, :].unsqueeze(0).repeat(n_theta, 1, 1)
                    y_u_c = (
                        y_u[0, idx, :].unsqueeze(0).repeat(n_theta, 1, 1)
                        if y_u is not None
                        else None
                    )
                else:
                    y_c = y[:, idx, :]
                    y_u_c = y_u[:, idx, :] if y_u is not None else None
                counts_c = counts[idx] if counts is not None else None

                dist = self.nb_dist(indices=idx, counts=counts_c, n_theta=n_theta)
                ll_xcg = dist.log_prob(y_c)

                if self.unspliced_mode:
                    if y_u_c is None:
                        raise ValueError(
                            "y_u (unspliced data) must be provided in unspliced_mode."
                        )
                    dist_u = self.nb_dist_unspliced(
                        indices=idx, counts=counts_c, n_theta=n_theta
                    )
                    ll_xcg = ll_xcg + dist_u.log_prob(y_u_c)

                log_posterior_xc = (ll_xcg * self.weights_g).sum(2)  # + log_prior_xc
                log_mle_c = log_posterior_xc.max(0).values
                posterior_xc = self.normalize_log_dist(log_posterior_xc, method=method)

                post_chunks.append(nmp(posterior_xc))
                lmle_chunks.append(nmp(log_mle_c))

        posterior_xc = np.concatenate(post_chunks, axis=1)
        log_mle_c = np.concatenate(lmle_chunks, axis=0)

        if return_all:
            # with a flat phase prior l_xc is identical to posterior_xc; prior_xc
            # is kept for API compatibility (unused by get_inferred_phases).
            l_xc = posterior_xc
            prior_xc = nmp(self.normalize_log_dist(self.cell_prior(), method=method))
            return (posterior_xc, l_xc, prior_xc, log_mle_c)

        else:
            return posterior_xc

    def get_inferred_phases(
        self, y, y_u=None, method="sum", counts=None, n_theta=None, cell_chunk=None
    ):
        """
        Wrapper around get_phase_posteriors to return just the posterior mean phases.
        it also stores the posterior mean, std and var as attributes
        """

        posterior_xc, l_xc, prior_xc, log_mle_c = self.get_phase_posteriors(
            y,
            y_u=y_u,
            method=method,
            return_all=True,
            counts=counts,
            n_theta=n_theta,
            cell_chunk=cell_chunk,
        )
        phi_post = self.posterior_grid(posterior_xc.shape[0])
        post_mean_c, post_var_c, post_std_c = compute_posterior_statistics(
            posterior_xc, phi_x=phi_post
        )
        self.disp = nmp(self.log_disp.exp())
        self.post_mean_c = post_mean_c
        self.post_std_c = post_std_c
        self.post_var_c = post_var_c
        self.mle_c = log_mle_c / self.Ng
        self.post_mode_c = compute_posterior_mode(posterior_xc, phi_x=phi_post)
        # self.posterior_xc = posterior_xc  # full (Nx, Nc) posterior array

        return post_mean_c

    def compute_mad(self, true_phase, estimator="mode", metric="median_AE"):
        """
        Returns the mad in radians
        """
        if estimator == "mode":
            phi = self.post_mode_c
        else:
            phi = self.post_mean_c
        aligned_phases, aligned_mad = optimal_shift(phi, true_phase, verbose=False)
        self.aligned_phases = aligned_phases
        if metric == "median_AE":
            self.mad = aligned_mad
            return aligned_mad
        elif metric == "mean_AE":
            self.mad = mean_AE(aligned_phases, true_phase, period=2 * np.pi)
            return self.mad
        elif metric == "mean_SE":
            self.mad = mean_SE(aligned_phases, true_phase, period=2 * np.pi)
            return self.mad
        else:
            raise ValueError(
                f"Unknown metric '{metric}'. Use 'median_AE', 'mean_AE' or 'mean_SE'."
            )

    # ------------------------------------------------------------------
    # gene parameters (tables in parameters.py)
    # ------------------------------------------------------------------
    def get_parameter_dataframe(self, unspliced=False):
        """
        The fitted gene parameters as a ``Beta`` table, one row per gene.

        Columns: ``a_0`` (log-mesor), ``a_i``/``b_i`` per harmonic, the derived
        ``amp``/``phase`` columns and ``disp``. Same table as :attr:`params_`.

        Args:
            unspliced: Not supported (the per-gene unspliced table is
                :meth:`extract_params_u`); kept so old call sites fail clearly.
        """
        if unspliced:
            raise NotImplementedError(
                "get_parameter_dataframe(unspliced=True) is not supported; "
                "use extract_params_u() for the unspliced parameters."
            )
        return par.parameter_table(self.m_g, self._get_ab(), self.genes, self.log_disp)

    def _amp_s(self):
        return par.amplitude(self.log_amp, self.log_amp_fn, self.max_amp)

    def _get_ab(self):
        return par.harmonic_coefficients(self._amp_s(), self.acrophase)

    def get_parameter_dataframe_context(self, gene_names=None):
        """
        LEGACY. Per-context gene parameters, one Beta table per context label.

        Folds the context-dependent parts into :meth:`get_parameter_dataframe`:
        the intercept gains ``m_yg[i]`` and the harmonic coefficients and
        amplitude are scaled by ``exp(log_lambda_y[i])``. Under the default
        ``context_mode="none"`` those terms are frozen at their zero init, so every
        returned table is identical to :meth:`get_parameter_dataframe`.

        Args:
            gene_names: Ignored (kept so older call sites keep working).

        Returns:
            dict mapping each label in ``self.context_u`` to a ``Beta`` table.
        """
        return par.context_parameter_tables(
            self.get_parameter_dataframe(), self.context_u, self.m_yg, self.log_lambda_y
        )

    # ------------------------------------------------------------------
    # desynchrony: functions in scritmo.ml.desync
    # ------------------------------------------------------------------
    def simulate_cell_populations(
        self,
        adata,
        context_col: str | None = None,
        n_cells=300,  # cells per set
        layer_to_use="spliced",
        ext_time_label="ZT",
        sample_label="sample_name",
        kappa=np.inf,
        period=24,
        device="cuda",
        return_sim_data=False,
        n_epochs_training=0,
        n_replicates=None,
        seed_replicates: int = 42,
        seed_sim: int | None = None,
        library_size_vec=None,
        n_sim_runs=1,
        use_circular_mean=False,
        posterior_cell_chunk=None,
    ):
        """
        Wrapper around the simulate_cell_populations function.

        ``context_col`` is the ``adata.obs`` column used to group samples; ``device``
        is resolved with :func:`scritmo.ml.utils.resolve_device`, so the "cuda"
        default falls back to CPU on a machine without a GPU.
        """
        device = resolve_device(device)
        return simulate_cell_populations(
            cmodel=self,
            adata=adata,
            context_col=context_col,
            n_cells=n_cells,
            layer_to_use=layer_to_use,
            ext_time_label=ext_time_label,
            sample_label=sample_label,
            kappa=kappa,
            period=period,
            device=device,
            return_sim_data=return_sim_data,
            n_epochs_training=n_epochs_training,
            n_replicates=n_replicates,
            seed_replicates=seed_replicates,
            seed_sim=seed_sim,
            library_size_vec=library_size_vec,
            n_sim_runs=n_sim_runs,
            use_circular_mean=use_circular_mean,
            posterior_cell_chunk=posterior_cell_chunk,
        )

    def simulate_technical_grid(
        self,
        adata,
        context_col: str | None = None,
        layer_to_use="spliced",
        n_grid: int = 24,
        n_cells_per_gridpoint: int = 1000,
        period=24,
        device="cuda",
        n_sim_runs: int = 1,
        library_size_vec=None,
        seed_sim: int | None = None,
        posterior_cell_chunk=None,
    ):
        """Wrapper around the simulate_technical_grid function (harmonic σ_tech floor).

        ``device`` is resolved with :func:`scritmo.ml.utils.resolve_device`.
        """
        device = resolve_device(device)
        return simulate_technical_grid(
            cmodel=self,
            adata=adata,
            context_col=context_col,
            layer_to_use=layer_to_use,
            n_grid=n_grid,
            n_cells_per_gridpoint=n_cells_per_gridpoint,
            period=period,
            device=device,
            n_sim_runs=n_sim_runs,
            library_size_vec=library_size_vec,
            seed_sim=seed_sim,
            posterior_cell_chunk=posterior_cell_chunk,
        )

    def create_results_df(
        self,
        adata: anndata.AnnData,
        ext_phase: None | np.ndarray = None,
        context_col: str | None = None,
        sample_col: str = "sample_name",
        ext_time_col: str = "ZTmod",
        post_estimator: str = "post_mode",
        layer="spliced",
        other_obs_cols: list = [],
        allow_flip: bool = False,
    ):
        """
        One row per cell: the inferred phase and its posterior spread, joined to
        the sample annotation.

        Thin wrapper around :func:`scritmo.ml.analysis_utils.create_results_dataframe`.
        Requires :meth:`get_inferred_phases` to have run (``warmup_and_train`` does
        it). ``ext_phase`` is an optional reference phase per cell (radians) that
        the posterior is aligned to; ``context_col``/``sample_col``/``ext_time_col``
        are ``adata.obs`` columns carried into the frame for grouping.

        Returns:
            pandas.DataFrame, also the input to :func:`desync_results`.
        """
        return create_results_dataframe(
            cmodel=self,
            adata=adata,
            ext_phase=ext_phase,
            context_col=context_col,
            sample_col=sample_col,
            ext_time_col=ext_time_col,
            post_estimator=post_estimator,
            layer=layer,
            other_obs_cols=other_obs_cols,
            allow_flip=allow_flip,
        )

    def estimate_phase_desynchrony(
        self,
        adata,
        ext_phase: None | np.ndarray = None,
        # --- Shared Data/Column Arguments ---
        context_col: str | None = None,
        sample_col: str = "sample_name",
        ext_time_col: str = "ZTmod",
        layer: str = "spliced",
        post_estimator: str = "post_mode",
        # --- Simulation Arguments (simulate_cell_populations) ---
        n_cells: int | None = None,
        period: float = 24.0,
        device: str = "cuda",
        n_epochs_training: int = 0,
        n_replicates_sim: int | None = None,
        library_size_vec=None,
        n_sim_runs: int = 5,
        posterior_cell_chunk: int | None = None,
        # --- Real Data Arguments (create_results_df) ---
        other_obs_cols: list = [],
        allow_flip: bool = False,
        # --- Desynchrony Calculation Arguments (desync_results) ---
        group_cols: list | None = None,
        disp_function=cSTD,
        metrics: dict | None = None,
        n_replicates_real: int | None = None,
        seed_real: int = 42,
        seed_sim: int | None = None,
        # --- Technical floor method ---
        sigma_tech_method: str = "simulation",
        # --- Harmonic floor arguments ---
        n_grid: int = 24,
        n_cells_per_gridpoint: int = 1000,
        return_harmonic_diagnostics: bool = False,
        harmonic_orders=(1, 2, 3),
        harmonic_eval: str = "sample",
        # --- Deconvolution floor arguments ---
        deconv_form: str = "exact",
        return_deconv_diagnostics: bool = False,
        tech_grid=None,
        # --- Cell filtering / weighting ---
        post_std_threshold: float = np.inf,
        weight_by_post_std: bool = False,
        # --- Simulation mean estimation ---
        use_circular_mean: bool = False,
        # --- Over-subtraction policy ---
        clamp_bio_variance: bool = True,
    ):
        """
        Estimate biological phase desynchrony, correcting for the technical floor.

        See :func:`scritmo.ml.desync.estimate_phase_desynchrony` for the full
        description of the method and of every argument; this method forwards
        to it unchanged. :meth:`desynchrony` is the same pipeline with explicit
        ``adata.obs`` keys.
        """
        return _estimate_phase_desynchrony(
            self,
            adata,
            ext_phase=ext_phase,
            context_col=context_col,
            sample_col=sample_col,
            ext_time_col=ext_time_col,
            layer=layer,
            post_estimator=post_estimator,
            n_cells=n_cells,
            period=period,
            device=device,
            n_epochs_training=n_epochs_training,
            n_replicates_sim=n_replicates_sim,
            library_size_vec=library_size_vec,
            n_sim_runs=n_sim_runs,
            posterior_cell_chunk=posterior_cell_chunk,
            other_obs_cols=other_obs_cols,
            allow_flip=allow_flip,
            group_cols=group_cols,
            disp_function=disp_function,
            metrics=metrics,
            n_replicates_real=n_replicates_real,
            seed_real=seed_real,
            seed_sim=seed_sim,
            sigma_tech_method=sigma_tech_method,
            n_grid=n_grid,
            n_cells_per_gridpoint=n_cells_per_gridpoint,
            return_harmonic_diagnostics=return_harmonic_diagnostics,
            harmonic_orders=harmonic_orders,
            harmonic_eval=harmonic_eval,
            deconv_form=deconv_form,
            return_deconv_diagnostics=return_deconv_diagnostics,
            tech_grid=tech_grid,
            post_std_threshold=post_std_threshold,
            weight_by_post_std=weight_by_post_std,
            use_circular_mean=use_circular_mean,
            clamp_bio_variance=clamp_bio_variance,
        )

    _attach_deconvolution = staticmethod(_attach_deconvolution)

    # ------------------------------------------------------------------
    # post-hoc analyses: functions in scritmo.ml.tools
    # ------------------------------------------------------------------

    fit_null_model = NullModelMixin.fit_null_model
    rhythmic_evidence_per_cell = NullModelMixin.rhythmic_evidence_per_cell
    per_gene_residuals = NullModelMixin.per_gene_residuals
    fit_genome_wide = GenomeFitMixin.fit_genome_wide
    fit_genome_wide_parallel = GenomeFitMixin.fit_genome_wide_parallel


# the private helpers of the old GenomeFitMixin, kept as methods for old callers
for _name, _fn in vars(GenomeFitMixin).items():
    if _name.startswith("_") and not _name.startswith("__") and callable(_fn):
        setattr(Scritmo, _name, _fn)
del _name, _fn

# Pickle the class under its historical module path: that path still resolves
# (scritmo.ml.context_model re-exports it), and older scritmo versions define
# the class there, so pickles stay loadable in both directions.
Scritmo.__module__ = "scritmo.ml.context_model"

# Backward-compatible alias (historical name). Keep for old pickles & existing imports.
ContextModel = Scritmo
