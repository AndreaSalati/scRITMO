import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re


class Beta(pd.DataFrame):
    """
    Class to handle the beta values with different harmonics.
    The columns are assumed to be ordered as "a_0", "a_1", "b_1", "a_2", "b_2", etc.
    """

    # tell pandas how to construct new instances of *this* class
    @property
    def _constructor(self):
        # note: you could also write: return Beta
        return lambda *args, **kwargs: Beta(*args, **kwargs)

    def __init__(self, data=None, **kwargs):
        """
        data : DataFrame-like OR filesystem path (str / Path)
        kwargs: passed to pd.DataFrame (if data is a DF) or to pd.read_csv (if data is a path)
        """
        # if they passed a path, load it first
        if isinstance(data, (str, Path)):
            data = pd.read_csv(data, **kwargs, index_col=0)
            kwargs = {}  # clear out CSV args so super() only sees DF kwargs

        # now init the DataFrame
        super().__init__(data, **kwargs)

    def convert_old_notation(self):
        """
        Rename m_g → a_0, a_g → a_1, b_g → b_1
        then reorder so that the first three columns are [a_0, a_1, b_1], with all data aligned.
        """
        # 1) build your rename mapping
        mapping = {}
        for col in self.columns:
            if col.startswith("m_"):
                mapping[col] = "a_0"
            elif col.startswith(("a_", "b_")):
                prefix, old_idx = col.split("_", 1)
                new_idx = "1" if old_idx == "g" else old_idx
                mapping[col] = f"{prefix}_{new_idx}"

        # 2) apply the rename in one shot
        if mapping:
            self.rename(columns=mapping, inplace=True)

    def get_ab(self, nh=None, keep_a_0=True):
        """
        Get the beta values from the DataFrame. I.e. get all the columns that start with "a_" or "b_".
        nh: number of harmonics.
        Stop when i reaches nh, e.g. return "a_0", "a_1", "b_1" if nh=1
        """
        # get the columns that start with "a_i" or "b_i", but stop when i reaches nh, e.g. "a_0", "b_0", "a_1", "b_1" if nh=1
        # if nh is None, get all the columns that start with "a_" or "b_"
        if nh is None:
            keep = [
                col
                for col in self.columns
                if col.startswith("a_") or col.startswith("b_")
            ]
        elif nh > -1:
            # i need a regex that matches "a_i" or "b_i" where i is a number from 0 to nh
            keep = [
                col
                for col in self.columns
                if (col.startswith("a_") or col.startswith("b_"))
                and int(col.split("_")[1]) <= nh
            ]
        else:
            raise ValueError("nh must be a positive integer or None")

        # if a0_only is True, only return the a_0 column
        if not keep_a_0:
            keep = [col for col in keep if col != "a_0"]

        if len(keep) == 0:
            raise ValueError("no columns left after filtering")

        # --- NEW REORDERING STEP ---
        # Sort by harmonic index, then 'a' before 'b':
        keep = sorted(
            keep, key=lambda col: (int(col.split("_", 1)[1]), col.split("_", 1)[0])
        )

        beta = self.loc[:, keep].copy()
        return beta

    def trace_det(self, nh=None):
        """
        Compute the trace of the determinant of the covariance matrix.
        """
        A = self.get_ab(nh, a0=False).values
        cov = A.T @ A
        # compute the determinant of the covariance matrix
        det = np.linalg.det(cov)
        # compute the trace of the covariance matrix
        trace = np.trace(cov)
        return trace, det

    def get_ab_se(self, nh=None):
        """
        Get the uncertainty of the beta values. The convention is that the columns start with "sig_a_" or "sig_b_", etc
        In this version the standard errors are diagonal, i.e. the covariance matrix is diagonal.
        nh: number of harmonics to return
        """
        # get the columns that start with "sig_a_i" or "sig_b_i", but stop when i reaches nh, e.g. "sig_a_0", "sig_b_0", "sig_a_1", "sig_b_1" if nh=1
        # if nh is None, get all the columns that start with "sig_a_" or "sig_b_"
        if nh is None:
            keep = [
                col
                for col in self.columns
                if col.startswith("sig_a_") or col.startswith("sig_b_")
            ]
        elif nh > -1:
            # i need a regex that matches "sig_a_i" or "sig_b_i" where i is a number from 0 to nh
            keep = [
                col
                for col in self.columns
                if (col.startswith("sig_a_") or col.startswith("sig_b_"))
                and int(col.split("_")[2]) <= nh
            ]
        else:
            raise ValueError("nh must be a positive integer or None")

        return self.loc[:, keep].copy()

    def predict(self, phi, exp_base=None):
        """
        Predict the values using the beta coefficients.
        t: time points
        """
        beta = self.get_ab(None)
        nh = int((beta.shape[1] - 1) / 2)
        X = make_X(phi, nh=nh)

        if exp_base == None:
            return X @ beta.T.values
        else:
            return exp_base ** (X @ beta.T.values)

    def get_amp(self, Ndense=1000, inplace=False):
        """
        Get the amplitude of the beta values.
        """
        if self is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        phi_dense = np.linspace(0, 2 * np.pi, Ndense, endpoint=False)
        y_dense = self.predict(phi_dense)

        out = []
        for gene in range(y_dense.shape[1]):
            y_mean = np.mean(y_dense[:, gene])
            y_max = np.max(y_dense[:, gene])
            y_min = np.min(y_dense[:, gene])
            amp_abs = y_max - y_min
            amp = amp_abs / 2
            amp_fc = y_max / y_min if y_min != 0 else np.inf
            phase = phi_dense[np.argmax(y_dense[:, gene])]

            out.append([y_mean, y_min, y_max, amp_abs, amp_fc, phase, amp])
        cols = ["y_mean", "y_min", "y_max", "amp_abs", "amp_fc", "phase", "amp"]

        out_df = pd.DataFrame(out, columns=cols, index=self.index)

        if inplace:
            self["y_mean"] = out_df["y_mean"]
            self["y_min"] = out_df["y_min"]
            self["y_max"] = out_df["y_max"]
            self["amp_abs"] = out_df["amp_abs"]
            self["amp_fc"] = out_df["amp_fc"]
            self["phase"] = out_df["phase"]
            self["amp"] = out_df["amp"]
            return
        else:
            return out_df

    def plot_circular(self, nh, beta2=None, mode="max-min", legend=True):
        """
        Make a circular plot for the betas. Use the get_amp function to get the amplitude
        so that it works for all numbers of harmonics.
        """

        amp_tmp = self.get_amp(nh=nh)
        # [y_mean, y_min, y_max, amp_abs, amp_fc, phi_peak]
        if mode == "max-min":
            phi = amp_tmp["phi_peak"]
            r = (amp_tmp["y_max"] - amp_tmp["y_min"]) / 2
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            lim = np.max(r) * 1.1

        elif mode == "rel-amp":
            phi = amp_tmp["phi_peak"]
            mean = amp_tmp[:, 0]
            assert np.all(
                mean >= 0
            ), "mean must be positive in plot_circular if mode=rel-amp"
            r = amp_tmp[:, 3] / amp_tmp[:, 0] / 2
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            lim = np.max(np.log2(amp_tmp[:, 4])) * 1.1

        else:
            raise ValueError("mode must be either 'max-min' or 'rel-amp'")

        plt.gca().set_aspect("equal")
        plt.title(mode)
        for i in range(len(self.index)):
            plt.plot(x[i], y[i], "o", label=self.index[i])
        # plot legend outside the plot
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1)) if legend else None
        # add vertical and horizontal lines
        plt.axhline(0, color="gray", lw=0.5)
        plt.axvline(0, color="gray", lw=0.5)

        plt.gca().set_prop_cycle(None)

        if beta2 is not None:
            # plot the second beta values using the same colors but different markers
            # get the beta values
            amp_tmp2 = beta2.get_amp(nh=nh).values
            if mode == "max-min":
                phi2 = amp_tmp2[:, 5]
                r2 = amp_tmp2[:, 3] / 2  # CAREFUL
                x2 = r2 * np.cos(phi2)
                y2 = r2 * np.sin(phi2)
                lim2 = np.max(r2) * 1.1
                if lim2 > lim:
                    lim = lim2
            elif mode == "rel-amp":
                phi2 = amp_tmp2[:, 5]
                mean2 = amp_tmp2[:, 0]
                assert np.all(
                    mean2 >= 0
                ), "mean must be positive in plot_circular if mode=rel-amp"
                r2 = amp_tmp2[:, 3] / amp_tmp2[:, 0] / 2
                x2 = r2 * np.cos(phi2)
                y2 = r2 * np.sin(phi2)
                lim2 = np.max(np.log2(amp_tmp2[:, 4])) * 1.1
                if lim2 > lim:
                    lim = lim2
            else:
                raise ValueError("mode must be either 'max-min' or 'rel-amp'")
            for i in range(len(self.index)):
                plt.plot(x2[i], y2[i], "x", label=self.index[i])
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)

        plt.show()

    def plot_circular2(
        self,
        genes_to_plot=None,
        title="",
        amp_lim=[0.0, 5.0],
        s=20,
        fontisize=12,
        col_names=["amp", "phase"],
    ):
        """
        Takes as input a pandas dataframe with columns "amp_abs", "phphi_peak"
        doesn't metter the order of the columns
        """

        # polar plot stuff
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
        ax.set_xticklabels(np.arange(24))
        ax.set_title(title)

        if genes_to_plot is None:
            genes_to_plot = self.index

        for j, gene in enumerate(self.index):

            amp, phase = self[col_names].iloc[j]

            if gene not in genes_to_plot:
                continue

            if amp < amp_lim[0] or amp > amp_lim[1]:
                continue

            ax.scatter(phase, amp, s=s)
            # annotate

            ax.annotate(gene, (phase, amp), fontsize=fontisize)

    def plot_means(self):
        """
        Plots barplots of the means (a_0), ordered by value
        """

        # get the means
        means = self.a_0
        # sort the means
        means = means.sort_values(ascending=False)
        # plot the means
        plt.bar(means.index, means.values)
        plt.xticks(rotation=90)
        plt.xlabel("Gene")
        plt.ylabel("Mean")
        # plt.title("Means of the beta values")
        plt.show()

    def plot_genes(self, nh=None, legend=True, exp_base=None):
        """
        plot the gene profiles
        """
        phi = np.linspace(0, 2 * np.pi, 100)
        profiles = self.predict(phi, exp_base=exp_base)
        plt.plot(phi, profiles)
        plt.xlabel("Phase")
        plt.ylabel("Gene profiles")
        (
            plt.legend(self.index, loc="upper left", bbox_to_anchor=(1, 1))
            if legend
            else None
        )
        plt.show()

    def plot_gene(self, gene, exp_base=None, legend=True):
        """
        Plot the expression profile for a single gene over [0, 2π].
        Args:
            gene: gene name (index label) or integer index.
            nh: number of harmonics.
            exp_base: optional exponent base for predict.
            legend: whether to show legend.
        """
        # Create a phase grid
        phi = np.linspace(0, 2 * np.pi, 100)
        # Predict profiles: shape (num_points, n_genes)
        profiles = self.predict(phi, exp_base=exp_base)
        # Resolve gene index
        if isinstance(gene, str):
            idx = list(self.index).index(gene)
        else:
            idx = int(gene)
        # Extract profile for the specified gene
        y = profiles[:, idx]
        # Plot
        plt.figure()
        plt.plot(phi, y, label=str(gene))
        plt.xlabel("Phase")
        plt.ylabel("Expression")
        plt.title(str(gene))
        if legend:
            plt.legend()
        plt.tight_layout()
        plt.show()

    def num_harmonics(self):
        """
        Get the number of harmonics from the beta values.
        """
        keep = [
            col for col in self.columns if col.startswith("a_") or col.startswith("b_")
        ]
        # get the number of harmonics
        nh = int((len(keep) - 1) / 2)
        return nh

    def add_harmonics(self, n_tot):
        """
        Ensure the DataFrame has harmonics 1…n_tot (i.e. columns a_k, b_k for k=1..n_tot).
        If you already have nh = self.num_harmonics(), then this adds only
        the missing ones, and does nothing if nh >= n_tot.
        """
        nh = self.num_harmonics()  # how many harmonics you currently have
        # how many new ones to add
        n_new = n_tot - nh
        if n_new <= 0:
            # nothing to do: already have at least n_tot harmonics
            return self

        # indices of the new harmonics to add: (nh+1) … n_tot
        new_idx = np.arange(nh + 1, n_tot + 1)

        # build the new column names: a_{k}, b_{k} for each k
        new_cols = [name for k in new_idx for name in (f"a_{k}", f"b_{k}")]

        # add them filled with zeros
        for col in new_cols:
            self[col] = 0.0

        # reorder
        self.sort_data_metadata()

    def rotate(self, phi):
        """
        Rotate the beta values by a given phase.
        """
        rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

        # assumes that the columns are ordered as "a_0", "a_1", "b_1", "a_2", "b_2", etc
        # check the order of the columns
        nh = self.num_harmonics()

        if self.columns[0] != "a_0":
            raise ValueError("columns must start with 'a_0'")
        for i in range(nh):
            if self.columns[2 * i + 1] != "a_" + str(i + 1) or self.columns[
                2 * i + 2
            ] != "b_" + str(i + 1):
                print(self.columns[2 * i + 1], self.columns[2 * i + 2])
                raise ValueError(
                    "columns must be ordered as 'a_0', 'a_1', 'b_1', 'a_2', etc"
                )

        for i in range(nh):
            self.iloc[:, 2 * i + 1 : 2 * i + 3] = (
                self.iloc[:, 2 * i + 1 : 2 * i + 3].values @ rot.T
            )
            rot = rot @ rot

        # Update phi_peak column if present
        if "phase" in self.columns:
            self["phase"] = (self["phase"] + phi) % (2 * np.pi)

    def rescale_amp(self, kappa):
        """
        Rescale the amplitude of the beta values.
        kappa can be a scalar or a vector of the same length as the number of genes.
        """
        if np.any(kappa < 0):
            raise ValueError("kappa must be positive")
        # loop over the columns of beta
        for col in self.columns:
            # if indeex doe not contain "a_0" or "sig_a_0" then rescale the amplitude
            if "a_0" not in col and "sig_a_0" not in col:
                self[col] *= kappa
        return self

    def optimal_rotate(self, beta_ref, nphi=200, apply=True):
        """
        rotate to beta_ref
        """
        phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
        scores = []
        for phi in phis:
            beta_rot = self.copy()
            beta_rot.rotate(phi)
            beta_rot_vals = beta_rot.get_ab().values
            amp1 = np.sqrt(np.sum(beta_rot_vals[:, 1:] ** 2))  # no a_0
            beta_ref_vals = beta_ref.get_ab().values
            amp2 = np.sqrt(np.sum(beta_ref_vals[:, 1:] ** 2))
            # compute the score
            score = np.sum((beta_rot.values / amp1 - beta_ref.values / amp2) ** 2)
            scores.append(score)

        # find the minimum score
        min_phi = phis[np.argmin(scores)]
        # rotate the beta values to the minimum score
        if apply:
            self.rotate(min_phi)
        return min_phi

    def copy(self, deep=True):
        """
        Override copy method to preserve Beta class.
        """
        data = super().copy(deep=deep)
        return Beta(data)

    def sort_data_metadata(self):
        """
        Reorder columns so that:
          1) all 'data' columns (a_k, b_k) come first, sorted by (k, letter)
          2) all other 'metadata' columns follow in their original relative order
        Returns a new Beta with columns reordered.
        """

        # 1) identify data columns via regex
        data_cols = [c for c in self.columns if re.fullmatch(r"[ab]_[0-9]+", c)]

        # 2) define a sort key: parse 'a_3' → (3, 'a'), 'b_12' → (12, 'b')
        def _harmonic_key(col_name):
            letter, idx = col_name.split("_")
            return (int(idx), letter)

        # 3) sort them
        data_cols_sorted = sorted(data_cols, key=_harmonic_key)

        # 4) metadata = everything else, in original order
        meta_cols = [c for c in self.columns if c not in data_cols_sorted]

        # 5) reindex and return
        return self.reindex(columns=data_cols_sorted + meta_cols)

    # def sort_data_metadata(self):
    #     """
    #     Reorder columns in place so that:
    #     1) all 'data' columns (a_k, b_k) come first, sorted by (k, letter)
    #     2) all other 'metadata' columns follow in their original relative order

    #     Modifies self.columns directly and returns self for convenience.
    #     """
    #     # 1) pick up exactly a_#, b_# names
    #     data_cols = [c for c in self.columns if re.fullmatch(r"[ab]_[0-9]+", c)]

    #     # 2) sort by harmonic index, then letter
    #     def _key(col):
    #         letter, idx = col.split("_")
    #         return (int(idx), letter)
    #     data_cols_sorted = sorted(data_cols, key=_key)

    #     # 3) metadata are everything else in original order
    #     meta_cols = [c for c in self.columns if c not in data_cols_sorted]

    #     # 4) reset the columns index in place
    #     self.columns = data_cols_sorted + meta_cols

    #     return self


def make_X(phi, nh):
    """
    Create the design matrix for Fourier fitting.
    phi: phase points
    """
    X = [np.ones(len(phi))]
    for n in range(nh):
        X.append(np.cos((n + 1) * phi))
        X.append(np.sin((n + 1) * phi))
    return np.array(X).T
