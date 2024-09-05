import numpy as np
import numpyro
import jax
import scanpy as sc
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from .posterior import *
from .circular import *
from .numpyro_models import (
    model_MLE_NB2,
    model_MLE_NB,
    guide_MLE,
)

from .numpyro_models_handles import *
from .linear_regression import genes_polar_parameters

# from .pseudobulk import pseudobulk_new, pseudo_bulk_time
from .RITMO_base import DataLoader, Pseudobulk

numpyro.set_platform("gpu")


class RITMO(DataLoader):
    def __init__(self):
        super().__init__()
        pass  # No attributes initially

    def logistic_regression(
        self,
        gene_list,
        layer="s_norm",
        test_size=0.2,
        random_state=42,
        max_iter=1000,
        n_jobs=-1,
        save_path=None,
        time_obs_field="ZTmod",
    ):
        """
        Performs a logistic regression analysis to predict the
        external time label. It uses the gene_list as features. It leverages
        sklearn's LogisticRegression class.
        Parameters:
        - gene_list: list of genes to use as features
        - layer: layer to use for the analysis
        - test_size: proportion of the data to use as test set
        - random_state: random seed
        - max_iter: maximum number of iterations for the logistic regression
        - n_jobs: number of jobs to run in parallel, -1 means all processors
        """
        self.LR_genes = np.intersect1d(gene_list, self.adata.var_names)

        X = self.adata[:, self.LR_genes].layers[layer].toarray()
        Y = self.adata.obs[time_obs_field].values

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state
        )

        clf = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=max_iter, n_jobs=n_jobs
        )
        # running the logistic regression
        clf.fit(X_train, Y_train)
        # storing a bunch of stuff
        Y_pred_test = clf.predict(X_test)
        Y_pred_total = clf.predict(X)
        accuracy = accuracy_score(Y_test, Y_pred_test)
        print(f"LR accuracy: {accuracy:.2f}")
        cm = confusion_matrix(Y_test, Y_pred_test, normalize="true")
        unique_labels = np.unique(Y)

        # evaluating the performance with different metrics
        LR_median_abs_error, LR_mean_abs_error, LR_root_mean_sq_error = (
            self._eval_performance(Y_pred_total.squeeze(), Y.squeeze(), period=24)
        )
        # save LR results
        self.add_attributes_and_save(
            save_path=save_path,
            gene_list=gene_list,
            clf=clf,
            Y_pred_total=Y_pred_total,
            Y_pred_test=Y_pred_test,
            accuracy=accuracy,
            confusion_matrix=cm,
            unique_labels=unique_labels,
            LR_median_abs_error=LR_median_abs_error,
            LR_mean_abs_error=LR_mean_abs_error,
            LR_root_mean_sqrt_error=LR_root_mean_sq_error,
        )

    def fit_genes(
        self,
        genes_sc,
        ph,
        model=model_MLE_NB,
        remove_genes=True,
        n_steps=10000,
        save_path=None,
        counts=None,
        block_list=["a_g", "b_g", "m_g"],
    ):
        """
        This function fits the gene coefficients to the external time values (ph).
        It is basically a wrapper around the fit_gene_coeff function.
        """
        # this first part should be a method by itself that can be celled independently
        print(f"counts per cell: {np.round(self.adata.obs['s_counts'].mean(), 2)}")

        # Remove genes with no expression, create Nan values
        genes_sc, data_c, counts, _ = self._filter_genes_and_prepare_data(
            genes_sc, counts, None
        )

        svi_g = fit_gene_coeff(
            model,
            guide_MLE,
            data_c,
            ph,
            counts=counts,
            n_steps=n_steps,
            block_list=block_list,
        )

        # Stuff to plot the results
        res_GLM_sc, params_g_polar = self._prepare_plot_results(svi_g)
        params_g = {key: svi_g.params[key] for key in ["a_g", "b_g", "m_g"]}

        if remove_genes:
            genes_sc, data_c, params_g, params_g_polar = (
                self._remove_large_amplitude_genes(
                    genes_sc, data_c, params_g, params_g_polar
                )
            )

        self.add_attributes_and_save(
            save_path=save_path,
            params_g_pol=params_g_polar,
            params_g=params_g,
            genes_sc=genes_sc,
            a_g=np.array(params_g["a_g"]),
            b_g=np.array(params_g["b_g"]),
            m_g=np.array(params_g["m_g"]),
            svi_g=svi_g,
            ph=ph,
        )

    def fit_cells(
        self,
        genes_sc,
        init_phase,
        params_g,
        model=model_MLE_NB,
        n_steps=10000,
        save_path=None,
        counts=None,
        block_list=["a_g", "b_g", "m_g"],
        lr=0.001,
    ):
        """
        This function, fits cells to the gene coefficients. It is a wrapper around the fit_phases function.
        Through the block list parameter, it is possible to fix some parameters during the optimization.
        Parameters:
        - genes_sc: list of genes to analyze
        - init_phase: initial phase values,
        - params_g: a dictionary containing the gene coefficients
        - model: the model to use
        - n_steps: number of steps for the inference
        - save_path: path to save the inference results. if None, it will not save
        - counts: total counts for each cell, if None, it will be calculated from the data
        """
        genes_sc, data_c, counts, params_g = self._filter_genes_and_prepare_data(
            genes_sc, counts, params_g
        )
        svi_c = fit_phases(
            model=model,
            guide=guide_MLE,
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

    def sc_posterior(
        self,
        genes_sc,
        params_g,
        disp=None,
        Nx=100,
        load_path=None,
        save_path=None,
        jax_device="cpu",
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
        # Load inference results if a path is provided
        # PROBABLY IT'S GONNA GIVE PROBLEMS WITH THE DISPERSION PARAMETER WITH params_g
        if load_path is not None:
            self._load_inference_results(load_path)

        if params_g["a_g"].shape[1] != genes_sc.shape[0]:
            raise ValueError(
                "The number of genes in the gene coefficients does not match the number of genes in the data"
            )
        self.genes_sc = genes_sc

        if disp is not None:
            self.disp = disp

        self.MLE_scan(
            genes_sc=genes_sc,
            params_g=params_g,
            disp=self.disp,
            Nx=100,
            jax_device=jax_device,
        )

        l_xc = normalize_ll_xc(self.ll_xc)
        # Compute posterior statistics
        deltaH_c, post_mean_c, post_var_c, post_std_c = (
            self._compute_posterior_statistics(l_xc, Nx)
        )

        # Handle NaN values
        nn = ~np.isnan(deltaH_c)
        print(f"mean std of posteriors = {post_std_c[nn].mean() * self.rh:.2f}")

        self.add_attributes_and_save(
            save_path=save_path,
            l_xc=l_xc,
            post_std_c=post_std_c * self.rh,
            post_mean_c=post_mean_c * self.rh,
            deltaH_c=deltaH_c,
            post_var_c=post_var_c,
            nn=nn,
            average_post_std=post_std_c[nn].mean() * self.rh,
        )

    def MLE_scan(
        self,
        genes_sc,
        params_g,
        disp=None,
        model=None,
        Nx=100,
        jax_device="cpu",
    ):
        """
        This method scans the phase space. To use if you are cells are
        statistically independent. It is equlivalent to the second step
        of the sc_inference method, but without the gradient descent.
        It also basically performs the first step of the sc_posterior method.
        Parameters:
        - params_g: gene coefficients, has to be a dictionary with keys "a_g", "b_g", "m_g"
        - disp: dispersion parameter, if None, it will be set to a small value, it can also
          be a vector as long as it respects the shape: (1, Nc, Ng). Both (1, Nc, 1)
          and (1, 1, Ng) are valid shapes
        - model: the model to use, if None, the default model is used
        - Nx: number of points to scan
        - return_ll_xcg: whether to return the log likelihoods for the cells
        """

        if model is None:
            model = model_MLE_NB2

        if disp is None:
            disp = 0.00001  # small value, poisson limit

        if params_g["a_g"].shape[1] != genes_sc.shape[0]:
            raise ValueError(
                "The number of genes in the gene coefficients does not match the number of genes in the data"
            )
        self.genes_sc = genes_sc

        self._configure_jax_device(jax_device)

        data_c, counts = self._prepare_data_for_posterior()
        Nc, Ng = data_c.shape
        # preparing an array with size Nx x Nc x 1
        phi_x = np.linspace(0, 2 * np.pi, Nx)
        # repeat phi_range Nc times
        phi_xc = np.tile(phi_x, (Nc, 1)).T
        phi_xc = phi_xc.reshape(Nx, Nc, 1)

        model_sub = numpyro.handlers.substitute(model, params_g)
        model_sub = numpyro.handlers.substitute(
            model_sub, {"disp": disp, "phi_c": phi_xc}
        )

        trace = numpyro.handlers.trace(model_sub).get_trace(data_c, counts=counts)
        ll_xcg = trace["obs"]["fn"].log_prob(data_c)
        ll_xc = np.array(ll_xcg.sum(axis=2))

        phi_MLE_c = fix_phases(ll_xc, phi_x)

        # saving results in the object
        self.add_attributes_and_save(
            save_path=None,
            params_g=params_g,
            disp=disp,
            phi_MLE_c=phi_MLE_c,
            ll_xc=ll_xc,
        )

    def add_attributes_and_save(self, save_path, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        if save_path is not None:
            self.save_results(save_path, **kwargs)

    def _eval_performance(self, pred_ph, true_ph, period=2 * np.pi):
        """
        This method evaluates the performance of the model by comparing the predicted
        phases with the true phases. Remember to sepcify what is the period
        of the circular variable (2pi or 24h). The output is in hours
        """
        median_abs_error = circular_median_absolute_deviation(pred_ph, true_ph, period)
        mean_abs_error = circular_mean_absolute_deviation(pred_ph, true_ph, period)
        root_mean_sq_error = circular_sqrt_mean_squared_error(pred_ph, true_ph, period)

        if period != 24:
            median_abs_error *= self.rh
            mean_abs_error *= self.rh
            root_mean_sq_error *= self.rh

        return median_abs_error, mean_abs_error, root_mean_sq_error

    ########################################################
    # sc_inference method helper functions
    ########################################################

    def _filter_genes_and_prepare_data(self, genes_sc, counts, params_g):
        """
        This function filters the genes that are notexpressed in the subset of
        the adata object that will be used for the inference.
        Also, it prepared data_c and counts for the inference.
        CAREFUL: counts should be the total counts for each cell, including
        ALL genes, and not only the subset of genes used for the inference.
        """
        genes_sc = np.intersect1d(genes_sc, self.adata.var_names)
        circ = self.adata[:, genes_sc]

        # getting rid of genes with no expression
        mask_g = np.array(circ.layers["spliced"].sum(0) != 0).squeeze()
        genes_sc = genes_sc[mask_g]

        if params_g is not None:
            for key in params_g.keys():
                if key == "disp":
                    continue
                params_g[key] = params_g[key][:, mask_g]
        circ = circ[:, genes_sc]

        data_c = circ.layers["spliced"].toarray()

        if counts is None:
            counts = self.adata[:, :].layers["spliced"].toarray().sum(1).reshape(-1, 1)
        self.counts = counts
        self.genes_sc = genes_sc

        return genes_sc, data_c, counts, params_g

    def _prepare_plot_results(self, svi_g):
        res_GLM_sc = np.concatenate(
            [
                svi_g.params["a_g"],
                svi_g.params["b_g"],
                svi_g.params["m_g"],
            ],
            axis=0,
        ).T
        res_GLM_sc = res_GLM_sc[None, :, :]
        params_g_polar = genes_polar_parameters(res_GLM_sc, rel_ampl=False)
        return res_GLM_sc, params_g_polar

    def _remove_large_amplitude_genes(self, genes_sc, data_c, params_g, params_g_polar):
        amps = params_g_polar[0, :, 0]
        too_big_amps = genes_sc[amps > 4]
        print(f"Removing genes with too big amplitudes: {too_big_amps}")
        mask_amp = np.array(amps < 4)
        for key in params_g.keys():
            params_g[key] = params_g[key][:, mask_amp]
        genes_sc = genes_sc[mask_amp]
        data_c = data_c[:, mask_amp]
        return genes_sc, data_c, params_g, params_g_polar

    def _align_phases_and_print_mad(self, svi_c, ph):
        phi_MLE_NB = svi_c.params["phi_c"].squeeze() % (2 * np.pi)
        phi_MLE_NB, mad_MLE_NB = optimal_shift(phi_MLE_NB, ph)
        print(f"mad = {mad_MLE_NB * self.rh:.2f}h ")
        return phi_MLE_NB, mad_MLE_NB

    ########################################################
    # sc_posterior method helper functions
    ########################################################

    def _load_inference_results(self, load_path):
        """Load inference results from a file and assign to attributes."""
        inf_dict = np.load(load_path, allow_pickle=True).item()
        self.CT = inf_dict["CT"]
        self.svi_c = inf_dict["svi_c"]
        self.svi_g = inf_dict["svi_g"]
        self.genes_sc = inf_dict["gene_list"]
        self.phi_c = self.svi_c["phi_c"]
        self.disp = self.svi_c["disp"]

    def _configure_jax_device(self, jax_device):
        """Configure JAX device."""
        if jax_device == "cpu":
            jax.config.update("jax_platform_name", "cpu")

    def _prepare_data_for_posterior(self):
        """Prepare data for posterior calculation."""
        circ = self.adata[:, self.genes_sc]
        data_c = circ.X.toarray()
        counts = self.adata[:, :].layers["spliced"].toarray().sum(1).reshape(-1, 1)
        return data_c, counts

    def _compute_posterior_statistics(self, l_xc, Nx):
        """Compute posterior statistics."""
        delta_phi = 2 * np.pi / Nx
        deltaH_c = np.apply_along_axis(delta_entropy, axis=0, arr=l_xc * delta_phi)
        post_mean_c = np.apply_along_axis(circ_mean_P, 0, l_xc * delta_phi)
        post_var_c = np.apply_along_axis(circ_var_P, 0, l_xc * delta_phi)
        post_std_c = np.apply_along_axis(circ_std_P, 0, l_xc * delta_phi)
        return deltaH_c, post_mean_c, post_var_c, post_std_c
