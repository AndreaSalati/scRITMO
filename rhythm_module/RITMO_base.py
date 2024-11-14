import numpy as np
import scanpy as sc

from .posterior import *
from .circular import *
from .numpyro_models_handles import *
from .pseudobulk import pseudobulk_new, pseudo_bulk_time, change_shape, normalize_log_PB
from .basics import ccg
import pickle


class DataLoader:
    def __init__(self):
        self.w = 2 * np.pi / 24
        self.rh = self.w**-1

    def load_data(
        self,
        adata,
        time_obs_field="ZTmod",
        celltype_field="celltype",
    ):
        """
        This method loads the adata file
        Also adds some useful attributes from adata.obs to the object

        Parameters:
        adata: str or AnnData object
        time_obs_field: the obs field that contains the time information
        celltype_field: the obs field that contains the celltype information
        """
        if type(adata) == str:
            self.adata = sc.read_h5ad(adata)
        else:
            self.adata = adata

        # Load the celltypes .npy file
        if celltype_field is not None:
            self.celltype = self.adata.obs[celltype_field].values
            self.ctu = np.unique(self.celltype)

        if time_obs_field is not None:
            self.time = self.adata.obs[time_obs_field].values
            self.phase = self.time * self.w
            self.adata.obs["phase"] = self.phase

        ccg_ = np.intersect1d(self.adata.var_names, ccg)
        self.ccg = np.unique(ccg_)

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

    def pseudobulk(
        self,
        groupby_obs_list,
        pseudobulk_layer="spliced",
        n_groups=1,
        keep_obs=["ZT", "ZTmod"],
        inplace=False,
        base=2,
    ):
        """
        A wrapper for the pseudobulk_new function. It creates a pseudobulk object
        and it makes sure that the object attributes are properly set.

        Parameters:
        - adata: anndata object
        - groupby_obs_list: list of the 2 obs columns to group the cells
            One should be the sample column, discriminating between biological replicates
            The other is whatever you want (usually celltype)
        - pseudobulk_layer: layer of the adata object to be used
        - n_groups: number of groups to split the cells in each timepoint, it's a pseudo-pseudo-bulk
        - keep_obs: list of strings
            The obs columns to keep in the pseudobulk object
        - inplace: bool
            Whether to overwrite seld.adata or create self.PB
        """
        PB = pseudobulk_new(
            self.adata,
            groupby_obs_list=groupby_obs_list,
            pseudobulk_layer=pseudobulk_layer,
            n_groups=n_groups,
            keep_obs=keep_obs,
        )
        time = PB.obs.ZTmod
        phase = PB.obs.ZTmod.values * self.w
        celltype = PB.obs.celltype

        # PB.layers["spliced"] = PB.X.copy()
        # # add normalized layer
        # PB.layers["s_norm"] = PB.layers["spliced"] / PB.layers["spliced"].sum(
        #     1, keepdims=True
        # )
        PB = self._log_transform_PB(PB, layer=pseudobulk_layer, base=base)

        if inplace:
            self.adata = PB
            self.time = time
            self.phase = phase
            self.celltype = celltype
        else:
            self.PB = PB
            self.PB_phase = phase

    def add_attributes_and_save(self, save_path, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        if save_path is not None:
            # save all kwargs in the save_path into .pkl file
            with open(save_path, "wb") as f:
                pickle.dump(kwargs, f)

    def _log_transform_PB(self, PB, layer="spliced", base=2):
        """
        This method log-transforms the PB object
        """
        PB.layers["spliced"] = PB.X.copy()
        # add normalized layer
        Nc = PB.layers["spliced"].sum(1)
        PB.layers["s_norm"] = PB.layers["spliced"] / Nc[:, None]
        # find best offset for log transformation
        cc = np.nanmedian(Nc)
        eps = cc**-1
        PB.layers["s_log"] = np.log(PB.layers["s_norm"] + eps) / np.log(base)
        return PB
