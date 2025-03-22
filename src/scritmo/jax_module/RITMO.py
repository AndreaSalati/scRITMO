import numpy as np
import numpyro
import scanpy as sc
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from tqdm import tqdm

from scritmo.basics import df2dict, dict2df, ccg
from scritmo.circular import *
from scritmo.linear_regression import polar_genes_pandas

from scritmo.jax_module.posterior import *
from scritmo.jax_module.numpyro_models import (
    model_MLE_NB,
    guide_MLE,
    model_null,
)

from scritmo.jax_module.numpyro_models_handles import *


from scritmo.jax_module.RITMO_base import DataLoader
from scritmo.jax_module.RITMO_helper import (
    _eval_performance,
    _filter_genes_and_prepare_data,
    check_shapes,
    _configure_jax_device,
    _compute_posterior_statistics,
    eval_fit_genes,
)

numpyro.set_platform("gpu")


class RITMO(DataLoader):

    def __init__(self):
        super().__init__()
        pass  # No attributes initially

    def fit_genes(
        self,
        genes_sc,
        ph,
        model=model_MLE_NB,
        remove_genes=True,
        n_steps=10000,
        save_path=None,
        counts=None,
        params_g=None,
        use_PB=False,
        jax_device="gpu",
        get_stats=True,
    ):
        """
        This function fits the gene coefficients to the external time values (ph).
        It is basically a wrapper around the fit_gene_coeff function.

        Parameters:
        - genes_sc: list of genes to analyze
        - ph: external time values, in radians
        - model: the numpyro model to use
        - remove_genes: whether to remove genes with large amplitudes
        - n_steps: number of steps for the inference
        - save_path: path to save the inference results. if None, it will not save
        - counts: total counts for each cell, if None, it will be calculated from the data
        - params_g: a dictionary containing initial estimates for gene coefficients
            needs to be indexed by the name of the parameters of 'model'
        - use_PB: whether to use the pseudobulked data
        """
        self._configure_jax_device(jax_device)
        # Remove genes with no expression, create Nan values
        genes_sc, data_c, counts, _ = self._filter_genes_and_prepare_data(
            genes_sc, counts, params_g, use_PB=use_PB
        )

        svi_g = fit_gene_coeff(
            model=model,
            guide=guide_MLE,
            y=data_c,
            true_phase=ph,
            counts=counts,
            n_steps=n_steps,
            guess=params_g,
        )

        if get_stats:
            svi_null = fit_gene_coeff(
                model=model_null,
                guide=guide_MLE,
                y=data_c,
                true_phase=None,
                counts=counts,
                n_steps=n_steps,
                guess=params_g,
            )
            model_stats = self.eval_fit_genes(svi_g, svi_null, data_c, ph)
        else:
            model_stats = None

            # Stuff to plot the results
        params_g = {key: np.array(svi_g.params[key]) for key in ["a_g", "b_g", "m_g"]}
        params_g = dict2df(params_g, genes_sc)
        # add dispersion parameter column to the df
        params_g["disp"] = np.array(svi_g.params["disp"]).squeeze()
        params_g_polar = polar_genes_pandas(params_g)

        self.add_attributes_and_save(
            save_path=save_path,
            params_g_pol=params_g_polar,
            params_g=params_g,
            genes_sc=genes_sc,
            svi_g=svi_g,
            ph=ph,
            model_stats=model_stats,
            external_ph=ph,
        )

    def fit_genes_fast(
        self,
        genes_sc,
        ph,
        save_path=None,
        counts=None,
        use_PB=True,
        fit_disp=False,
        fixed_disp=0.1,
    ):
        """
        This function fits the gene coefficients to the external time values (ph).
        It is a faster version of the fit_genes method, but less flexible.
        It is based on statsmodels' NegativeBinomial class.

        Parameters:
        - genes_sc: list of genes to analyze
        - ph: external time values, in radians
        - model: the numpyro model to use
        - save_path: path to save the inference results. if None, it will not save
        - counts: total counts for each cell, if None, it will be calculated from the data
        - params_g: a dictionary containing initial estimates for gene coefficients
            needs to be indexed by the name of the parameters of 'model'
        - use_PB: whether to use the pseudobulked data
        - fit_disp: whether to fit the dispersion parameter. It uses different
            models from statsmodels, always try False first, as it is more stable
        - disp: the dispersion parameter, if fit_disp is False
        """
        # Remove genes with no expression, create Nan values
        params_g = None
        genes_sc, data_c, counts, _ = self._filter_genes_and_prepare_data(
            genes_sc, counts, params_g, use_PB=use_PB
        )
        ph, counts = ph.squeeze(), counts.squeeze()

        X = np.ones((data_c.shape[0], 3))
        X = pd.DataFrame(X, columns=["cos", "sin", "intercept"])
        X.cos = np.cos(ph)
        X.sin = np.sin(ph)

        X_null = np.ones((ph.shape[0], 1))

        results_list = []
        pvals = []
        for gene_index, g in enumerate(genes_sc):
            # Extract the gene counts
            gene_counts = data_c[:, gene_index]

            # Fit the Negative Binomial model using statsmodels.discrete, display=False
            # or using sm.GLM
            if fit_disp:
                model = NegativeBinomial(gene_counts, X, offset=np.log(counts))
                model_null = NegativeBinomial(
                    gene_counts, X_null, offset=np.log(counts)
                )
            else:
                model = sm.GLM(
                    gene_counts,
                    X,
                    family=sm.families.NegativeBinomial(alpha=fixed_disp),
                    offset=np.log(counts),
                )
                model_null = sm.GLM(
                    gene_counts,
                    X_null,
                    family=sm.families.NegativeBinomial(alpha=fixed_disp),
                    offset=np.log(counts),
                )
            result = model.fit(disp=False)
            result_null = model_null.fit(disp=False)

            # pvalue
            llr = 2 * (result.llf - result_null.llf)
            pval = 1 - chi2.cdf(llr, 2)

            # Store the results, including the dispersion parameter (alpha)
            result_dict = {
                "gene": g,
                "a_g": result.params.iloc[0],
                "b_g": result.params.iloc[1],
                "m_g": result.params.iloc[2],
            }

            if fit_disp:
                result_dict["disp"] = result.params.iloc[3]

            results_list.append(result_dict)
            pvals.append(pval)

        # Convert the list of results into a DataFrame
        params_g = pd.DataFrame(results_list)
        # first column is index
        params_g = params_g.set_index("gene")
        params_g_pol = polar_genes_pandas(params_g)
        model_stats = pd.DataFrame(pvals, index=genes_sc, columns=["pvalue"])

        self.add_attributes_and_save(
            save_path=save_path,
            params_g_pol=params_g_pol,
            params_g=params_g,
            genes_sc=genes_sc,
            ph=ph,
            model_stats=model_stats,
            external_ph=ph,
        )

    def MLE_scan(
        self,
        genes_sc,
        params_g,
        disp="gene",
        model=model_MLE_NB,
        Nx=100,
        jax_device="cpu",
        use_PB=False,
        save_path=None,
        counts=None,
    ):
        """
        This method scans the phase space. To use if you are cells are
        statistically independent. It is equlivalent to the second step
        of the sc_inference method, but without the gradient descent.
        It is also used by the sc_posterior method.
        Parameters:
        - params_g: gene coefficients, has to be a dictionary with keys "a_g", "b_g", "m_g"
        - disp: dispersion parameter, if None, it will be set to a small value, it can also
          be a vector as long as it respects the shape: (1, Nc, Ng). Both (1, Nc, 1)
          and (1, 1, Ng) are valid shapes
        - model: the model to use, if None, the default model is used
        - Nx: number of points to scan
        - jax_device: jax device to use, default gpu
        - usep_PB: weather to compute the phase estimation on PB data
        - save_path: saving attributes as dict. If None they will not be saved
        - counts: Passing explictely the toal RNA counts that cells have
            if None, it will be computed from self.adata hence MAKE SURE that all
            genes are present in self.adata!
        """

        # use check_shapes to make sure the number of genes in the gene coefficients
        genes_sc, params_g = self.check_shapes(genes_sc, params_g)

        if disp == "gene":
            disp = params_g["disp"].squeeze()
            disp = disp.reshape(1, 1, -1)
        elif disp is None:
            disp = 0.1  # small value, poisson limit
        else:
            disp = np.array(disp).reshape(1, 1, -1)

        self.genes_sc = genes_sc
        self._configure_jax_device(jax_device)

        genes_sc, data_c, counts, params_g = self._filter_genes_and_prepare_data(
            genes_sc, counts, params_g, use_PB=use_PB
        )

        Nc, Ng = data_c.shape
        # preparing an array with size Nx x Nc x 1
        phi_x = np.linspace(0, 2 * np.pi, Nx)
        # repeat phi_range Nc times
        phi_xc = np.tile(phi_x, (Nc, 1)).T
        phi_xc = phi_xc.reshape(Nx, Nc, 1)

        # preparing the parameters for the model
        model_params = params_g.copy()
        model_params["phi_c"] = phi_xc
        if type(disp) == float:
            model_params["disp"] = disp

        ll_xcg = get_ll(model, model_params, data_c, counts=counts)
        ll_xc = np.array(ll_xcg.sum(axis=2))

        phi_MLE_c = fix_phases(ll_xc, phi_x)

        # saving results in the object
        self.add_attributes_and_save(
            save_path=save_path,
            params_g=params_g,
            disp=disp,
            phi_MLE_c=phi_MLE_c,
            ll_xc=ll_xc,
            ll_xcg=ll_xcg,
        )

    def sc_posterior(
        self,
        genes_sc,
        params_g,
        disp="gene",
        model=model_MLE_NB,
        Nx=100,
        save_path=None,
        jax_device="cpu",
        use_PB=False,
        counts=None,
    ):
        """
        This method computes the posterior distribution for the single-cell data.
        It calls the MLE_scan method and performs some additional statistics.
        Numerically approximates the posterior distribution of the phases.
        Parameters:
        - genes_sc: list of genes to analyze
        - params_g = dictionary with genes parameters, used for the posterior estimation
            if using sc_inference first, do not specify
        - disp: dispersion parameters used for the posterior estimation
        - Nx: number of steps for the posterior calculation
        - load_path: path to the inference results
        - save_path: path to save the posterior statistics
        - jax_device: device to use for the posterior calculation, cpu is recomended here
        """

        self.MLE_scan(
            genes_sc=genes_sc,
            params_g=params_g,
            disp=disp,
            model=model,
            Nx=Nx,
            jax_device=jax_device,
            use_PB=use_PB,
            counts=counts,
        )

        l_xc = normalize_ll_xc(self.ll_xc)
        # Compute posterior statistics
        post_mean_c, post_var_c, post_std_c = self._compute_posterior_statistics(
            l_xc, Nx
        )

        # turn inf into nan
        post_std_c[np.isinf(post_std_c)] = np.nan
        nn = ~np.isnan(post_std_c)
        print(f"median std of posteriors = {np.median(post_std_c[nn]) * self.rh:.2f}")

        self.add_attributes_and_save(
            save_path=save_path,
            l_xc=l_xc,
            post_std_c=post_std_c * self.rh,
            post_mean_c=post_mean_c * self.rh,
            post_var_c=post_var_c,
            nn=nn,
            average_post_std=post_std_c[nn].mean() * self.rh,
        )

    def tandem_fit(
        self,
        genes_sc,
        ph_init,
        save_path=None,
        counts=None,
        use_PB=False,
        fit_disp=False,
        fixed_disp=0.1,
        n_iterations=10,
        Nx=100,
    ):
        """
        This function is a recursive function that fits gene coefficients
        with fit_genes_fast and than fits the cells MLE_scan.
        It repeats the process n_interations times.
        """
        ph = ph_init

        if fit_disp:
            disp_scan = "gene"
        else:
            disp_scan = None

        # loading bar
        for i in tqdm(range(n_iterations), desc="Processing"):

            self.fit_genes_fast(
                genes_sc=genes_sc,
                ph=ph,
                save_path=save_path,
                counts=counts,
                use_PB=use_PB,
                fit_disp=fit_disp,
                fixed_disp=fixed_disp,
            )

            self.MLE_scan(
                genes_sc=genes_sc,
                params_g=self.params_g,
                disp=disp_scan,
                Nx=Nx,
                use_PB=use_PB,
                save_path=save_path,
                counts=counts,
            )

            ph = self.phi_MLE_c

    def fit_cells(
        self,
        genes_sc,
        init_phase,
        params_g,
        model=model_MLE_NB,
        guide=guide_MLE,
        n_steps=10000,
        save_path=None,
        counts=None,
        block_list=["a_g", "b_g", "m_g"],
        lr=0.001,
        jax_device="gpu",
    ):
        """
        This function, fits cells to the gene coefficients. It is a wrapper around the fit_phases function.
        Here potentially everything can be fit togetjer
        Through the block list parameter, it is possible to fix some parameters during the optimization.
        Parameters:
        - genes_sc: list of genes to analyze
        - init_phase: initial phase values,
        - params_g: a dictionary containing the gene coefficients, and other parameters
        - model: the model to use
        - n_steps: number of steps for the inference
        - save_path: path to save the inference results. if None, it will not save
        - counts: total counts for each cell, if None, it will be calculated from the data
        - block_list: list of parameters to block/fix during the optimization
        """
        # use check shapes
        self._configure_jax_device(jax_device)

        genes_sc, params_g = self.check_shapes(genes_sc, params_g)

        genes_sc, data_c, counts, params_g = self._filter_genes_and_prepare_data(
            genes_sc, counts, params_g
        )

        svi_c = fit_phases(
            model=model,
            guide=guide,
            y=data_c,
            init_phase=init_phase.reshape(-1, 1),
            params_g=params_g,
            n_steps=n_steps,
            counts=counts,
            block_list=block_list,
            lr=lr,
        )

        self.add_attributes_and_save(
            save_path=save_path,
            svi_c=svi_c,
            params_g=params_g,
            genes_sc=genes_sc,
            phi_c=np.array(svi_c.params["phi_c"]).squeeze() % (2 * np.pi),
        )


# add all the methods from RITMO_helper.py
RITMO.check_shapes = check_shapes
RITMO._eval_performance = _eval_performance
RITMO._filter_genes_and_prepare_data = _filter_genes_and_prepare_data
RITMO.eval_fit_genes = eval_fit_genes
RITMO._configure_jax_device = _configure_jax_device
RITMO._compute_posterior_statistics = _compute_posterior_statistics
