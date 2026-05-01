import numpy as np
import torch
from torch import tensor as tt
from tqdm import tqdm
from torch import nn
from sklearn.preprocessing import OneHotEncoder
from functools import partial
from .marginalization import MarginalizationMixin
import anndata
import pandas as pd
from .utils import harmonic_dm_torch, circ_std_P, set_context_mode, nmp
from .misc.power_spherical.power_spherical import (
    log_von_mises,
)

from scritmo import (
    Beta,
    compute_posterior_statistics,
    rh,
    w,
    optimal_shift,
    compute_posterior_mode,
    mean_SE,
    mean_AE,
)
from scritmo import median_dispersion, cSTD
from .simulations.estimate_desynchrony import estimate_desynchrony
from .simulations.simulate_populations import simulate_cell_populations
from .ensemble import EnsembleMixin
from .unspliced.unspliced_deg import UnsplicedMixin
from .unspliced.fisher import FisherUncertaintyMixin
from .analysis_utils import (
    create_results_dataframe,
    desync_results,
    desync_means,
    desync_results_posterior,
)
from .genome_fit import GenomeFitMixin
from .null_model import NullModelMixin

circSTD = partial(cSTD, adjust=True)


class ContextModel(
    nn.Module,
    EnsembleMixin,
    UnsplicedMixin,
    MarginalizationMixin,
    FisherUncertaintyMixin,
    GenomeFitMixin,
    NullModelMixin,
):
    """
    A deterministic model that uses Simpson's rule for integration
    and focuses only on the loss term involving marginalize_theta.
    """

    def __init__(
        self,
        mp,
        y,
        context_mode="none",
        fix_phase=False,
        noise_model="nb",
        fix_disp_val="gene",
        log_amp_fn="logit",
        method="simpson",
    ):
        """
        Initialize the context model with flexible context effects.

        Args:
            mp: Model parameters dictionary with initial values
            y: Data tensor of shape [Nx, Nc, Ng]
            context_mode: String controlling which context effects to use
                - "none": No context effects (baseline model)
                - "intercept": Only use category-specific intercepts (m_yg)
                - "full": Use both intercepts and amplitude scaling (m_yg and lambda)
                - "full_lambda" Lambda is also gene dependent here
                - "context_only" Keeps Betas fixed, fits intercept and scaling
            fix_phase: If True, acrophase parameters are fixed (not trained)
            noise_model: "nb" for Negative Binomial, "poisson" for Poisson
            fix_disp_val: Controls dispersion initialization and shape.
                - "gene": Per-gene dispersion (Ng,), trainable. If params_g has a "disp"
                  column (e.g. from a previous run), warm-starts from those values.
                - "context": Per-context dispersion (Ny, 1), trainable.
                - None: Single scalar dispersion, trainable.
                - float: Fixed scalar dispersion, not trained.
            log_amp_fn: "logit" or "log" to control amplitude parameterization
            method: Integration method, either "simpson" or "sum"
        """
        super().__init__()

        self.Nx, self.Nc, self.Ng = y.shape
        self.nh = mp["params_g"].num_harmonics()
        self.dev = y.device
        # self.dm = self.design_matrix(mp["context"])
        self.register_buffer("dm", self.design_matrix(mp["context"]))
        self.Ny = self.dm.shape[1]
        self.method = method
        self.register_buffer("counts", mp["counts"].clone())
        self.context = mp["context"]
        self.context_u = np.unique(mp["context"])
        self.genes = mp["params_g"].index.values
        self.fix_phase = fix_phase
        self.max_amp = 8 / (2 * np.log2(np.e))  # log2fc of 8
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

        acrophase_tensor = tt(mp["params_g"]["phase"].values, dtype=torch.float32)

        # Original amplitude values
        amp_values = tt(mp["params_g"]["amp"].values, dtype=torch.float32)
        if log_amp_fn == "logit":
            safe_amp = torch.clamp(amp_values, min=1e-2, max=self.max_amp - 1e-2)
            # The ratio is now guaranteed to be in a safe sub-interval of (0, 1)
            log_amp = torch.logit(safe_amp / self.max_amp)
            self.log_amp = nn.Parameter(log_amp)
        elif log_amp_fn == "log":
            log_amp = torch.log(amp_values)
            self.log_amp = nn.Parameter(log_amp)

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

        # context related parameters
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

    set_context_mode = set_context_mode

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
        Initialize a ContextModel from gene parameters only, without adata.

        Creates a model with gene parameters loaded from params_g but with
        no real dataset attached.  Dataset-specific buffers (X, dm, counts)
        are populated with dummy values for a single cell and are rebuilt
        automatically when get_inferred_phases is called with real data
        (the Nc-mismatch path in get_phase_posteriors handles this, but
        requires context_mode="none" and explicit counts= argument).

        Typical use:
            params_g = sr.Beta("saved_params.csv")
            cmodel = ContextModel.from_params_g(params_g)
            # later, with real adata:
            data_c, mp = assemble_mp(adata, params_g, labels=np.ones(Nc), ...)
            cmodel.get_inferred_phases(data_c, counts=mp["counts"], n_theta=100)
            bic_df = cmodel.fit_null_model(adata, counts=counts)

        Args:
            params_g      : Beta (or DataFrame) with columns a_0, amp, phase
                            (and optionally disp for warm-start dispersion).
            context_mode  : Must be "none" for inference on new data.
            fix_phase     : Whether to fix acrophase parameters.
            noise_model   : "nb" or "poisson".
            fix_disp_val  : Dispersion mode (see __init__ docstring).
            log_amp_fn    : "logit" or "log".
            device        : Torch device string.
        """
        from scritmo import Beta

        params_g = Beta(params_g)
        Ng = len(params_g)

        # minimal mp with a single dummy cell
        mp = {
            "params_g": params_g,
            "context": np.array([1]),
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

        # extra: to be removed later
        ll_e_xc = torch.zeros((self.Nx, self.Nc), dtype=torch.float32, device=y.device)[
            :, indices
        ]

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
                ll_xc, l_prior_xc, ll_e_xc, method=self.method, return_integrand=True
            )
            loss_like = -self.log_like_loss(l_c, max_c)

        loss_beta = -self.beta_prior()
        loss += loss_like + loss_beta * Nb

        return loss

    def beta_prior(self):
        # prior_g = log_power_spherical(self.acrophase, self.prior_g, self.k_beta)

        if self.k_beta is None:
            return tt(0.0, device=self.dev)
        else:
            prior_g = log_von_mises(self.acrophase, self.prior_g, self.k_beta)
            return prior_g.sum()

    def get_phase_posteriors(
        self,
        y,
        y_u=None,
        method="sum",
        return_all=False,
        counts=None,
        n_theta=None,
    ):
        """
        It gives you the posterior distribution of the phase
        for each cell given the fitted model parameters and data

        Args:
            y: Data tensor
        Returns:
        """
        if self.fixed_cell_mode:
            raise RuntimeError("Cannot compute phase posteriors in fixed-phase mode.")
        # in case I want to use a higher n_theta for phase resolution (differrent Nx)
        if n_theta is not None:
            # Use a higher n_theta for phase resolution
            Nx = n_theta

            # Repeat the data along the phase dimension
            y = y[0, :, :].unsqueeze(0).repeat(n_theta, 1, 1)
            if y_u is not None:
                y_u = y_u[0, :, :].unsqueeze(0).repeat(n_theta, 1, 1)
        else:
            Nx = self.Nx

        # in case I use a smaller or different dataset (different Nc)
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
                # adjust the Nc dependent parameters, first X
                phi_x_tensor = torch.linspace(
                    0, 2 * torch.pi, n_theta + 1, dtype=torch.float32
                )[:-1]
                X_tensor = harmonic_dm_torch(phi_x_tensor, self.nh, False)
                X_tensor = X_tensor.unsqueeze(1).expand(n_theta, self.Nc, self.nh * 2)
                self.register_buffer("X", X_tensor)
                # adjust dm
                self.register_buffer("dm", self.design_matrix(np.ones(self.Nc)))

        # Run forward calculation without computing gradients
        with torch.no_grad():
            dist = self.nb_dist(counts=counts, n_theta=n_theta)
            ll_xcg = dist.log_prob(y)

            if self.unspliced_mode:
                if y_u is None:
                    raise ValueError(
                        "y_u (unspliced data) must be provided in unspliced_mode."
                    )
                dist_u = self.nb_dist_unspliced(counts=counts, n_theta=n_theta)
                ll_u_xcg = dist_u.log_prob(y_u)  # Unspliced log-likelihood
                # the unspliced loss is added to the spliced one
                ll_xcg = ll_xcg + ll_u_xcg  # Combine likelihoods

            # cells priors
            log_prior_xc = self.cell_prior()

            # add here a weighted sum?
            log_posterior_xc = (ll_xcg * self.weights_g).sum(2)  # + log_prior_xc
            log_mle_c = (ll_xcg * self.weights_g).sum(2).max(0).values
            posterior_xc = self.normalize_log_dist(log_posterior_xc, method=method)

        if return_all:
            l_xc = self.normalize_log_dist(
                (ll_xcg * self.weights_g).sum(2), method=method
            )
            prior_xc = self.normalize_log_dist(log_prior_xc, method=method)
            return (nmp(posterior_xc), nmp(l_xc), nmp(prior_xc), nmp(log_mle_c))

        else:
            return nmp(posterior_xc)

    def get_inferred_phases(self, y, y_u=None, method="sum", counts=None, n_theta=None):
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
        )
        post_mean_c, post_var_c, post_std_c = compute_posterior_statistics(posterior_xc)
        self.disp = nmp(self.log_disp.exp())
        self.post_mean_c = post_mean_c
        self.post_std_c = post_std_c
        self.post_var_c = post_var_c
        self.mle_c = log_mle_c / self.Ng
        self.post_mode_c = compute_posterior_mode(posterior_xc)
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

    def normalize_log_dist(self, ll_xc, method="simpson"):
        """
        this method gets a log likelihood with format
        xc (where x is the phase and c the cell index)
        and it normalizes w.r.t. the x variable
        """
        max_c = torch.max(ll_xc, dim=0, keepdim=True).values
        # numerical stability
        ll_xc = ll_xc - max_c
        l_xc = torch.exp(ll_xc)

        if method == "simpson":
            l_c = self.vectorized_simpson(l_xc, self.phi_x)
        elif method == "sum":
            l_c = torch.sum(l_xc, dim=0) * (2 * torch.pi / self.Nx)

        return l_xc / l_c

    def get_parameter_dataframe(self, unspliced=False):
        """
        Create a DataFrame with harmonic coefficients for each gene.
        It is the core parameters shared by all celltypes.

        Parameters
        ----------
        genes : list, optional
            useless, It's there just to not brake previous code.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per gene and columns:
            'a_0', 'a_1', 'b_1', ..., 'a_n', 'b_n'
        """
        # Convert tensors to numpy arrays

        if unspliced:
            a_0_np = nmp(self.m_u_g).squeeze()
            ab_np = nmp(self._get_ab_u()).T
            gene_names = self.genes
        else:
            a_0_np = nmp(self.m_g).squeeze()
            ab_np = nmp(self._get_ab()).T
            # ab_np = self.ab.detach().cpu().numpy().squeeze().T
            gene_names = self.genes

        # Determine number of harmonics
        n_harmonics = ab_np.shape[1] // 2

        # Create column names
        col_names = ["a_0"] + [
            f"{c}_{i+1}" for i in range(n_harmonics) for c in ("a", "b")
        ]

        # Concatenate a_0 and ab
        full_array = np.concatenate([a_0_np[:, None], ab_np], axis=1)

        # Create the DataFrame
        params_g = pd.DataFrame(full_array, columns=col_names, index=gene_names)
        params_g = Beta(params_g)
        params_g.get_amp(inplace=True)
        params_g["disp"] = nmp(self.log_disp.exp()).squeeze()

        return params_g

    def _amp_s(self):
        if self.log_amp_fn == "logit":
            amp = torch.sigmoid(self.log_amp) * self.max_amp
            return amp
        elif self.log_amp_fn == "log":
            amp = torch.exp(self.log_amp)
            return amp

    def _get_ab(self):
        amp = self._amp_s()
        cos = amp * torch.cos(self.acrophase).unsqueeze(0)
        sin = amp * torch.sin(self.acrophase).unsqueeze(0)
        ab = torch.cat([cos, sin], dim=0)
        return ab

    def get_parameter_dataframe_context(self, gene_names=None):
        """
        Calls self.get_parameter_dataframe adds the context
        dependent parts
        """
        params_g = self.get_parameter_dataframe()
        gene_names = params_g.index.values

        params_y = {}
        for i, ct in enumerate(self.context_u):
            par = params_g.copy()
            par.a_0 += self.m_yg[i, :].cpu().detach().numpy()
            col_names = par.get_ab_column_names(keep_a_0=False)
            lambda_y = nmp(self.log_lambda_y[i].exp().squeeze())
            par[col_names] = par[col_names] * lambda_y
            par["amp"] = par["amp"] * lambda_y
            par.get_cartesian(inplace=True)
            params_y[ct] = par

        return params_y

    def design_matrix(self, vec, all_categories=None):
        """
        Creates a one-hot encoded design matrix from a vector.

        Args:
            vec (list or np.array): The input vector of categories to encode.
            all_categories (list, optional): The complete list of all possible
                                             categories. If provided, the output
                                             matrix will have a column for each of
                                             these, ensuring consistent shape.

        Returns:
            torch.Tensor: The resulting one-hot encoded tensor.
        """
        if vec is None:
            # Your original logic for when no vector is passed
            return torch.ones((self.Nc, 1), dtype=torch.float32)

        # Reshape the input vector for the encoder
        data = np.array(vec).reshape(-1, 1)

        # Initialize the encoder
        if all_categories is not None:
            encoder = OneHotEncoder(
                categories=[all_categories],
                sparse_output=False,
                handle_unknown="ignore",
            )
        else:
            # Default behavior: infer categories directly from the input 'vec'
            encoder = OneHotEncoder(sparse_output=False)

        # Create the one-hot encoded matrix
        one_hot_matrix = encoder.fit_transform(data)

        # Convert to a PyTorch tensor
        return torch.tensor(one_hot_matrix, dtype=torch.float32, device=self.dev)

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
            phi_x_new = torch.linspace(
                0, 2 * torch.pi, n_theta + 1, dtype=torch.float32, device=self.dev
            )[:-1]
            X_new = harmonic_dm_torch(phi_x_new, self.nh, False)
            X = X_new.unsqueeze(1).expand(n_theta, self.Nc, self.nh * 2)
            X = X[:, indices, :]
        else:
            X = self.X[:, indices, :]

        dm = self.dm[indices, :]

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

        # --- Select distribution based on the noise model ---
        if self.noise_model == "nb":
            # Use JIT-compiled NB parameter computation
            r, p = compute_nb_params(E_xcg, disp, counts)
            return torch.distributions.NegativeBinomial(total_count=r, probs=p)

        elif self.noise_model == "poisson":
            # Use JIT-compiled Poisson rate computation
            rate = compute_poisson_rate(E_xcg, counts)
            return torch.distributions.Poisson(rate=rate)

        elif self.noise_model == "gaussian":
            # Gaussian distribution with mean E_xcg and fixed std dev
            std_dev = 1.0
            return torch.distributions.Normal(loc=E_xcg, scale=std_dev)

        else:
            raise NotImplementedError(
                f"Noise model '{self.noise_model}' is not implemented."
            )

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
        return_posteriors=False,
        use_circular_mean=False,
    ):
        """
        Wrapper around the simulate_cell_populations function.
        """
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
            return_posteriors=return_posteriors,
            use_circular_mean=use_circular_mean,
        )

    def create_results_df(
        self,
        adata: anndata.AnnData,
        ext_phase: None | np.ndarray = None,
        context_col: None = None,
        sample_col: str = "sample_name",
        ext_time_col: str = "ZTmod",
        post_estimator: str = "post_mode",
        layer="spliced",
        other_obs_cols: list = [],
        allow_flip: bool = False,
    ):
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
        # --- Mode ---
        mode: str = "point_estimate",
        # --- Cell filtering / weighting ---
        post_std_threshold: float = np.inf,
        weight_by_post_std: bool = False,
        # --- Simulation mean estimation ---
        use_circular_mean: bool = False,
    ):
        """
        Orchestrates the estimation of phase desynchrony by:
        1. Creating a results DataFrame from real data.
        2. Simulating cell populations with no phase variance (technical).
        3. Computing biological desynchrony by comparing real and simulated dispersion.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix.
        ext_phase : np.ndarray
            External phase array for the real data.
        context_col : str, optional
            Column in adata.obs defining the context (e.g., condition/genotype).
        sample_col : str, default "sample_name"
            Column in adata.obs defining the sample identifier.
        ext_time_col : str, default "ZTmod"
            Column name for external time. Passed as 'ext_time_label' to simulation.
        layer : str, default "spliced"
            The layer to use for processing and simulation.
        n_replicates_sim : int, optional
            Number of replicates to generate during simulation.
        n_replicates_real : int, optional
            Number of bootstrap replicates for the real data aggregation.
        """

        if context_col is None:
            context_col = "context"
            if len(self.context_u) == 1:
                context_val = self.context_u[0]
                adata.obs[context_col] = context_val
            else:
                raise ValueError(
                    "context_col cannot be None as there are multiple possible contexts. "
                    "Please specify the context_col argument."
                )

        # For posterior_mixture: run inference first so that post_std_c in df_real
        # is always aligned with the passed adata (handles pre-trained loaded models).
        if mode == "posterior_mixture":
            self._infer_on_adata(adata, layer=layer, device=device)

        # 1. Generate Real Results DataFrame
        df_real = self.create_results_df(
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
        self.result_df = df_real

        # 1b. Filter cells by posterior uncertainty
        if post_std_threshold < np.inf and "post_std_c" in df_real.columns:
            mask = df_real["post_std_c"] <= post_std_threshold
            n_before = len(df_real)
            df_real = df_real[mask].copy()
            cell_keep_idx = np.where(mask.values)[0]
            print(
                f"  post_std_threshold={post_std_threshold:.3f} rad: "
                f"kept {len(df_real)}/{n_before} cells"
            )
            if len(df_real) == 0:
                raise ValueError(
                    f"post_std_threshold={post_std_threshold} filtered out all cells."
                )
        else:
            cell_keep_idx = None

        if mode == "point_estimate":
            # 2a. Simulate (point estimates only)
            df_sim = self.simulate_cell_populations(
                adata=adata,
                context_col=context_col,
                n_cells=n_cells,
                layer_to_use=layer,
                ext_time_label=ext_time_col,
                sample_label=sample_col,
                kappa=np.inf,
                period=period,
                device=device,
                return_sim_data=True,
                n_epochs_training=n_epochs_training,
                n_replicates=n_replicates_sim,
                seed_sim=seed_sim,
                library_size_vec=library_size_vec,
                n_sim_runs=n_sim_runs,
                use_circular_mean=use_circular_mean,
            )

            # 3a. Compute desynchrony from point estimates
            df_final = desync_results(
                df_real=df_real,
                df_sim=df_sim,
                group_cols=group_cols,
                disp_function=disp_function,
                post_estimator=post_estimator,
                metrics=metrics,
                n_replicates=n_replicates_real,
                seed=seed_real,
                weight_col="post_std_c" if weight_by_post_std else None,
            )

        elif mode == "posterior_mixture":
            # posterior_xc is already populated by _infer_on_adata called above
            real_posterior_xc = self.posterior_xc
            if cell_keep_idx is not None:
                real_posterior_xc = real_posterior_xc[:, cell_keep_idx]

            # 2b. Simulate and return full posteriors
            df_sim, sim_posteriors_dict = self.simulate_cell_populations(
                adata=adata,
                context_col=context_col,
                n_cells=n_cells,
                layer_to_use=layer,
                ext_time_label=ext_time_col,
                sample_label=sample_col,
                kappa=np.inf,
                period=period,
                device=device,
                return_sim_data=True,
                n_epochs_training=n_epochs_training,
                n_replicates=n_replicates_sim,
                seed_sim=seed_sim,
                library_size_vec=library_size_vec,
                n_sim_runs=n_sim_runs,
                return_posteriors=True,
                use_circular_mean=use_circular_mean,
            )

            # 3b. Compute desynchrony from posterior mixtures
            df_final = desync_results_posterior(
                df_real=df_real,
                real_posterior_xc=real_posterior_xc,
                df_sim=df_sim,
                sim_posteriors_dict=sim_posteriors_dict,
                group_cols=group_cols,
                n_replicates=n_replicates_real,
                seed=seed_real,
            )

        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Choose 'point_estimate' or 'posterior_mixture'."
            )

        return df_final

    def _infer_on_adata(
        self, adata, layer: str = "spliced", device: str = "cpu", n_theta: int = 100
    ):
        """
        Run get_inferred_phases on an AnnData object, populating posterior_xc.

        Used by estimate_phase_desynchrony(mode='posterior_mixture') to ensure
        self.posterior_xc is aligned with the provided adata.

        Parameters
        ----------
        adata : AnnData
            Must contain all genes in self.genes (or a superset).
        layer : str
            Layer to use for count data.
        device : str
            Torch device.
        n_theta : int
            Phase grid resolution for posteriors.
        """
        import scipy.sparse

        # Extract count matrix for model genes
        missing = np.setdiff1d(self.genes, adata.var_names)
        if len(missing) > 0:
            raise ValueError(
                f"adata is missing {len(missing)} model genes (e.g. {missing[:3]}). "
                "Ensure adata contains all genes the model was trained on."
            )
        adata_sub = adata[:, self.genes]

        if layer in adata_sub.layers:
            X = adata_sub.layers[layer]
        else:
            X = adata_sub.X
        if scipy.sparse.issparse(X):
            X = X.toarray()
        X = np.array(X, dtype=np.float32)

        lib_sizes = X.sum(axis=1, keepdims=True).astype(np.float32)

        data_c = torch.tensor(X, dtype=torch.float32, device=device)
        data_c = data_c.unsqueeze(0).expand(n_theta, -1, -1)
        counts_c = torch.tensor(lib_sizes, dtype=torch.float32, device=device)

        self.to(device)
        self.get_inferred_phases(data_c, n_theta=n_theta, counts=counts_c)

    def X_matrix(self, fixed_cell_mode, n_theta=None, mp=None):

        if fixed_cell_mode:
            self.fixed_cell_mode = True

            # Expecting shape [Nc]
            phi_c = tt(mp["fixed_cell_phases"], dtype=torch.float32)
            if phi_c.shape[0] != self.Nc:
                raise ValueError(
                    f"fixed_cell_phases length {phi_c.shape[0]} does not match Nc {self.Nc}"
                )

            self.register_buffer("phi_c", phi_c)
            X_tensor = harmonic_dm_torch(self.phi_c, self.nh, False)
            self.Nx = 1
            return X_tensor.unsqueeze(0)

        else:
            if n_theta is None:
                n_theta = self.Nx
            self.fixed_cell_mode = False
            # EXISTING LOGIC
            phi_x_tensor = torch.linspace(
                0, 2 * torch.pi, n_theta + 1, dtype=torch.float32
            )[:-1]
            self.register_buffer("phi_x", phi_x_tensor)

            # Nx Np -> Nx Nc Np
            X_tensor = harmonic_dm_torch(phi_x_tensor, self.nh, False)
            X_tensor = X_tensor.unsqueeze(1).expand(n_theta, self.Nc, self.nh * 2)
            return X_tensor


def compute_nb_params(
    E_xcg: torch.Tensor, disp: torch.Tensor, counts: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    JIT-compiled Negative Binomial parameter computation.

    Args:
        E_xcg: Expected mean values (before exp transform)
        disp: Dispersion parameter
        counts: Library size counts
        eps: Epsilon for numerical stability

    Returns:
        r: Total count parameter
        p: Success probability parameter (clamped)
    """
    E_xcg_exp = torch.exp(E_xcg) * counts
    r = 1.0 / disp
    p = disp * E_xcg_exp / (1.0 + disp * E_xcg_exp)
    p = p.clamp(min=eps, max=1.0 - eps)
    return r, p


def compute_poisson_rate(E_xcg: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    return torch.exp(E_xcg) * counts
