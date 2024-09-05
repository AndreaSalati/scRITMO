import numpy as np
import numpyro
import scanpy as sc

from .posterior import *
from .circular import *
from .numpyro_models_handles import *
from .pseudobulk import pseudobulk_new, pseudo_bulk_time, change_shape, normalize_log_PB


class DataLoader:
    def __init__(self):
        pass  # No attributes initially

    def load_data(self, data_path, time_obs_field="ZTmod", celltype_field="celltype"):
        """
        This method loads the data from a .h5ad file and stores it in the object.
        Also adds some useful attributes. This method always needs to be called
        before other ones.
        """
        self.adata = sc.read_h5ad(data_path)

        # Load the celltypes .npy file
        if celltype_field is not None:
            self.celltype = self.adata.obs[celltype_field].values
            self.ctu = np.unique(self.celltype)
        self.w = 2 * np.pi / 24
        self.rh = self.w**-1
        if time_obs_field is not None:
            self.time = self.adata.obs[time_obs_field].values
            self.phase = self.time * self.w
            self.adata.obs["phase"] = self.phase

        ccg = np.array(
            [
                "Arntl",
                "Npas2",
                "Cry1",
                "Cry2",
                "Per1",
                "Per2",
                "Nr1d1",
                "Nr1d2",
                "Tef",
                "Dbp",
                "Ciart",
                "Per3",
                "Bmal1",
            ]
        )
        ccg = np.intersect1d(self.adata.var_names, ccg)
        self.ccg = np.unique(ccg)

    def subselect_data(
        self, item, obs_column, time_obs_field="ZTmod", celltype_field="celltype"
    ):
        """
        This method subselects the data based on a specific column in the obs
        attribute. It is useful for selecting a specific celltype or sample.
        """
        self.adata = self.adata[self.adata.obs[obs_column] == item]
        if celltype_field is not None:
            self.celltype = self.adata.obs[celltype_field].values
        if time_obs_field is not None:
            self.time = self.adata.obs[time_obs_field].values
            self.phase = self.time * self.w


class Pseudobulk(DataLoader):

    def __init__(self):
        super().__init__()
        pass

    def pseudobulk(
        self,
        groupby_obs_list,
        pseudobulk_layer="spliced",
        n_groups=1,
        keep_obs=["ZTmod", "ZT"],
    ):
        """
        This method is a wrapper around the pseudobulk_new
        function.
        """
        self.n_groups = n_groups
        self.sample_name = self.adata.obs["sample_name"]
        self.samples_u = np.unique(self.sample_name)
        self.NC, self.NG = self.adata.shape
        self.PB = pseudobulk_new(
            self.adata,
            groupby_obs_list,
            pseudobulk_layer=pseudobulk_layer,
            n_groups=n_groups,
            keep_obs=keep_obs,
        )
        self.NS = self.samples_u.shape[0]

    def change_shape_and_preprocess(
        self, groupby_obs_list, n_groups=1, eps=None, base=2.0
    ):
        """
        This function tranforms the 2dim (n_samples x n_genes)
        adata object into 3dim (n_celltypes x n_genes x n_samples)
        This is necessary for compatibility with functions in
        the linear_regression module. It also normalizes the data
        and applies a log2 transformation. It's a wrapper around
        the change_shape function and normalize_log_PB in pseudobulk.py
        """
        self.n_ygt = change_shape(self.PB, groupby_obs_list, n_groups=n_groups)
        self.f_ygt, self.gamma_ygt = normalize_log_PB(self.n_ygt, eps=eps, base=base)

    def pb_time(self, ZT_obs="ZTmod"):
        """
        This method is a wrapper around pseudo_bulk_time
        """
        self.pb_time = pseudo_bulk_time(
            self.adata, ZT_obs=ZT_obs, n_groups=self.n_groups
        )
