import numpy as np
import torch
from torch import tensor as tt
from scipy.stats import nbinom
from ..utils import nmp
from scritmo import (
    Beta,
    compute_posterior_statistics,
    rh,
    w,
    optimal_shift,
    circular_deviation,
    median_AE,
    mean_AE,
    median_SE,
    mean_SE,
)
from ..utils import assemble_mp
from ..context_model import ContextModel


class GeneSearch:

    def __init__(
        self,
        params_init,
        params_candidate,  # new candidate genes
        adata,
        true_phase,
        disp=0.1,
        noise_model="nb",
        align=False,
        metric="mean_SE",
        n_theta=24,
        phase_estimator="mode",
    ):

        self.params_candidate = params_candidate
        self.disp = disp
        self.noise_model = noise_model
        self.adata = adata
        self.true_phase = true_phase
        self.align = align
        self.n_theta = n_theta
        self.metric = metric
        self.core_genes = params_init.index
        self.Nc = adata.shape[0]
        self.new_genes = []
        self.phase_estimator = phase_estimator

        # initializes everything
        self.update_model(params_init)

    def nb_dist_df(self, g):
        """
        Computes the likelihood distribution (Negative Binomial or Poisson)
        for the given data.

        Called by several methods.
        """
        par = self.params_candidate.loc[g]
        a_0 = tt(par.a_0, device=self.dev, dtype=torch.float32)
        ab = tt(
            par[["a_1", "b_1"]].values.reshape(-1, 1),
            device=self.dev,
            dtype=torch.float32,
        )

        E_xcg = (self.X @ ab) + a_0
        E_xcg = torch.exp(E_xcg) * self.counts

        # --- Select distribution based on the noise model ---
        if self.noise_model == "nb":
            # Negative Binomial distribution
            r = 1 / self.disp
            eps = 1e-6
            p = self.disp * E_xcg / (1 + self.disp * E_xcg)
            p = p.clamp(min=eps, max=1 - eps)

            return torch.distributions.NegativeBinomial(total_count=r, probs=p)

    def forward(self, g, update_model=False, return_mad=False, verbose=True):
        """
        Evaluates the log-likelihood of the data for a new gene `g`.
        """
        datum = self._expand_gene_vector(g)
        nb_dist_new = self.nb_dist_df(g=g)
        ll_new_xcg = nb_dist_new.log_prob(datum)

        ll_xcg = torch.cat((self.ll_old_xcg, ll_new_xcg), dim=-1)
        ll_xc = ll_xcg.sum(dim=-1)

        # get new phi indices
        phi_idx = ll_xc.argmax(dim=0)
        phi_c = nmp(self.cmodel.phi_x[phi_idx])

        # compute new mad
        mad = self.evaluate_phases(phi_c)

        if mad < self.old_mad:
            if verbose:
                print(f"{g} Improved MAD: {self.old_mad:.4f} -> {mad:.4f}")
            if update_model:
                self.ll_old_xcg = ll_xcg
                self.old_mad = mad
                self.old_phi = phi_c
                self.new_genes.append(g)
                self.update_model(g)
            if return_mad:
                return mad
        else:
            if verbose:
                print(f"{g} No improvement: {self.old_mad:.4f} -> {mad:.4f}")

    def evaluate_phases(self, phi):
        """
        Evaluates the mean absolute deviation (MAD) between the inferred phases
        and the true phases.
        """
        if self.align:
            aligned_phases, mad = optimal_shift(phi, self.true_phase, verbose=False)
        else:
            metric = self._metric_name(self.metric)
            m = metric(phi, self.true_phase, period=2 * np.pi)
        return m

    def update_model(self, new_gene, add=True):
        """
        Updates the context model with the newly inferred phases.
        Also initlializes all the attributes needed for the next gene search
        when gene is None.
        """
        # to init we pass the params_init
        if type(new_gene) != str:
            par_new = new_gene
        else:
            par = self.cmodel.get_parameter_dataframe()
            if add:
                gene_row = self.params_candidate.loc[new_gene]
                par_new = par
                # add new gene parameters to the context model
                par_new.loc[new_gene] = gene_row
            else:
                par_new = par.drop(index=new_gene)

        # than here call assamble_mp
        data_c, mp = assemble_mp(
            self.adata, par_new, labels=np.zeros(self.Nc), n_theta=self.n_theta
        )
        self.data_c = data_c
        cmodel = ContextModel(
            mp=mp,
            y=data_c,
        )
        cmodel.to(device="cuda")
        self.cmodel = cmodel

        # if first time running copy some attributes
        if type(new_gene) != str:
            self.X = self.cmodel.X
            self.phi_x = self.cmodel.phi_x
            self.counts = self.cmodel.counts
            self.dev = self.cmodel.dev
            self.old_phi = self.cmodel.get_inferred_phases(self.data_c)

            if self.phase_estimator == "mean":
                self.old_phi = self.cmodel.post_mean_c
            elif self.phase_estimator == "mode":
                self.old_phi = self.cmodel.post_mode_c

            self.old_mad = self.evaluate_phases(self.old_phi)
            # update the log-likelihood distribution, to save computation time
            nb_old = self.cmodel.nb_dist()
            self.ll_old_xcg = nb_old.log_prob(self.data_c)

        self.genes = par_new.index
        self.Ng = len(self.genes)

    def _expand_gene_vector(self, g):
        data = self.adata[:, g].layers["spliced"].toarray()
        data_c = torch.tensor(data, dtype=torch.float32, device=self.dev)
        if type(g) is str:
            Ng = 1
        else:
            Ng = len(g)
        data_c = data_c.unsqueeze(0).expand(self.n_theta, self.Nc, Ng)
        return data_c

    def gene_removal(self, gene_to_remove, update=False, verbose=True):
        """
        Evaluates if removing a specific gene from the current set improves the MAD score.

        This method works by masking the specified gene out of the pre-computed
        log-likelihood tensor and re-evaluating the phases and the resulting metric.
        It does not modify the model state.

        Args:
            gene_to_remove (str): The name of the gene to consider for removal.
        """
        # --- Safety checks ---
        if gene_to_remove in self.core_genes:
            print(
                f"⚠️ Cannot evaluate removal of '{gene_to_remove}': It is a core gene."
            )
            return

        if gene_to_remove not in self.genes:
            print(f"⚠️ Gene '{gene_to_remove}' is not in the current set.")
            return

        # --- Find the index of the gene to mask out ---
        try:
            # self.genes is a pandas Index from the parameter dataframe
            gene_idx_to_remove = self.genes.get_loc(gene_to_remove)
        except KeyError:
            print(f"Error: Could not find index for gene '{gene_to_remove}'.")
            return

        # mask
        mask = [i != gene_idx_to_remove for i in range(self.Ng)]
        ll_xcg_masked = self.ll_old_xcg[:, :, mask]
        ll_xc = ll_xcg_masked.sum(dim=-1)

        # Get the new optimal phases (phi)
        phi_idx = ll_xc.argmax(dim=0)
        phi_c = nmp(self.cmodel.phi_x[phi_idx])

        # Calculate the new MAD
        new_mad = self.evaluate_phases(phi_c)

        # --- Print the result ---
        if new_mad < self.old_mad:
            if verbose:
                print(
                    f"✅ Removing '{gene_to_remove}' improves MAD: {self.old_mad:.4f} -> {new_mad:.4f}"
                )
            if update:
                # Update the model by actually removing the gene
                self.ll_old_xcg = ll_xcg_masked
                self.old_mad = new_mad
                self.old_phi = phi_c
                # pop from new_genes if present
                self.new_genes.remove(gene_to_remove)
                # self.params_g = self.params_g.drop(index=gene_to_remove)
                self.update_model(gene_to_remove, add=False)
                if verbose:
                    print(f"Model updated: '{gene_to_remove}' has been removed.")

        else:
            if verbose:
                print(
                    f"❌ Removing '{gene_to_remove}' does not improve MAD: {self.old_mad:.4f} -> {new_mad:.4f}"
                )

        return new_mad

    def mad_hours(self, metric="mean_AE"):
        metric = self._metric_name(metric)
        return metric(self.old_phi, self.true_phase, period=(2 * np.pi)) * rh

    def _metric_name(self, metric_name):
        if metric_name == "mean_SE":
            return mean_SE
        elif metric_name == "median_SE":
            return median_SE
        elif metric_name == "mean_AE":
            return mean_AE
        elif metric_name == "median_AE":
            return median_AE
        else:
            raise ValueError("Metric not recognized")
