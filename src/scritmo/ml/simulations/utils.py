from scipy.special import i0, i1
from scipy.stats import vonmises, circstd, circmean
import pandas as pd
import numpy as np
import warnings
from ..utils import df2dict, nmp
import torch
import scritmo as sr

rh = sr.rh


def circular_std(kappa):
    """
    Compute the circular standard deviation (cStd) of a von Mises distribution
    given the concentration parameter kappa.

    Parameters:
        kappa (float): The concentration parameter (kappa > 0)

    Returns:
        float: Circular standard deviation
    """
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    if kappa > 700:
        return 0
    R = i1(kappa) / i0(kappa)
    cstd = np.sqrt(-2 * np.log(R))
    return cstd


def kappa2circular_std(kappa):
    """return circular standard deviation given kappa in radians"""
    return circular_std(kappa)


def circular_std2kappa(cstd):
    """
    Compute the approximate concentration parameter (kappa) given
    the circular standard deviation using a fitted Power Law inverse.

    Model: cstd = a * kappa^b
    Inverse: kappa = (cstd / a)^(1/b)

    Fitted Parameters:
    a = 1.0142
    b = -0.4952

    Parameters:
        cstd (float or np.array): The circular standard deviation.

    Returns:
        float or np.array: The estimated kappa.
    """
    # Hardcoded fitted parameters
    a = 1.0142
    b = -0.4952

    # Avoid division by zero errors if cstd is 0 (implies infinite kappa)
    # Using numpy's maximum to prevent negative inputs if working with noisy data
    cstd_safe = np.maximum(cstd, 1e-9)

    return np.power(cstd_safe / a, 1 / b)


def get_df_clock(ab, m_g):
    df = pd.DataFrame(ab.T, columns=["a_g", "b_g"])
    df["m_g"] = m_g
    return df


def get_ext_time(ext_time_unformatted, period=24, convert_rad=True):

    try:
        ext_time = ext_time_unformatted.astype(float)
    except ValueError:
        warnings.warn(
            "ZT was automatically transformed in float by removing all non digit character"
        )
        ext_time = (
            ext_time_unformatted.astype(str)
            .str.extract(r"(\d+(?:\.\d+)?)")
            .astype(float)
        )
        if ext_time.isna().any().values[0]:
            raise ValueError(
                f"Could not extract a numeric time from some values in the ext time."
            )
    if convert_rad:
        ext_time = (ext_time % period / period * (2 * np.pi)).values.squeeze()
    else:
        ext_time = (ext_time % period).values.squeeze()
    return ext_time


def results_to_df(results, trained=False):
    df = {
        "cad": [],
        "seq_depth": [],
        "circadian_counts": [],
        "true_phase": [],
        "posterior_std": [],
        "mean_posterior_phase": [],
        "mode_posterior_phase": [],
        "population": [],
    }
    if not trained:
        key = "res_not_trained"
    else:
        key = "res_trained"

    for res in results:
        cad = res[key]["cad"]
        n_cells = len(cad)
        df["cad"].extend(cad)
        df["seq_depth"].extend([res["seq_depth"]] * n_cells)
        df["circadian_counts"].extend(res["circadian_counts"])
        df["true_phase"].extend(res["phi_sim"])
        df["population"].extend(
            res["population"] if res["population"] is not None else [np.nan] * n_cells
        )
        df["posterior_std"].extend(res[key]["post_std_c"])
        df["mean_posterior_phase"].extend(res[key]["post_mean_c"] * rh)
        df["mode_posterior_phase"].extend(res[key]["post_mode_c"] * rh)

    df = pd.DataFrame(df)
    df["seq_depth"] = df["seq_depth"].astype("category")
    df["true_phase_binned"] = pd.cut(
        df["true_phase"],
        bins=np.linspace(0, 2.01 * np.pi, 24 + 1),
        include_lowest=True,
    ).apply(lambda x: x.mid * rh)
    return df


def assemble_mp_simulation(data, counts, labels, params_g_init, use_data_mean=True):
    """
    Assemble the model parameters for the simulation.
    The input is data nd not adata
    """
    use_data_mean = True
    mp = df2dict(params_g_init)

    for key in mp.keys():
        mp[key] = torch.tensor(mp[key]).float()
    mp["params_g"] = params_g_init

    if use_data_mean:
        f_cg = data / counts[:, None]
        log_f_cg = np.log(f_cg + 1e-6)
        means_g = log_f_cg.mean(0)
        params_g_init["m_g"] = means_g

    mp["counts"] = torch.tensor(counts[:, None])
    mp["context"] = np.array(labels)
    return mp


def get_df_var_inference(df_plot):
    """
    Computes the standard deviation of posterior phase estimates for each population and sequencing depth,
    and returns the results as a pandas DataFrame.
    Parameters
    ----------
    df_plot : pandas.DataFrame
        Input DataFrame containing at least the following columns:
        - 'population': Identifies the population group (as integers or strings).
        - 'seq_depth': Sequencing depth values.
        - 'mean_posterior_phase': Posterior phase estimates (float).
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - 'population_mean': The mean phase value for each population.
        - 'cStd': The computed standard deviation of the posterior phase distribution for each population and sequencing depth.
        - 'seq_depth': The sequencing depth corresponding to each entry.
    Notes
    -----
    - The function assumes that the number of unique populations determines the phase means, which are spaced evenly between 0 and 24.
    - Uses a histogram with 24 bins over the range [0, 24] to estimate the posterior phase distribution.
    - Relies on an external function `sr.compute_posterior_statistics` to compute statistics from the histogram.
    """

    n_pop = len(df_plot["population"].astype(int).unique())
    phases_means = np.linspace(0, 24, n_pop + 1)
    df_var_inference = {"population_mean": [], "cStd": [], "seq_depth": []}
    for seq_depth in df_plot["seq_depth"].unique():
        std_phases = []
        population_mean = []
        for g in range(n_pop):
            g = str(g)
            df_tmp = df_plot[
                (df_plot["population"] == g) & (df_plot["seq_depth"] == seq_depth)
            ]
            pdf, _ = np.histogram(
                df_tmp["mean_posterior_phase"].values,
                bins=24,
                range=(0, 24),
                density=True,
            )
            _, _, std = sr.compute_posterior_statistics(pdf)
            std_phases.append(float(std))
            population_mean.append(phases_means[int(g)])
        df_var_inference["population_mean"].extend(population_mean)
        df_var_inference["cStd"].extend(std_phases)
        df_var_inference["seq_depth"].extend([str(seq_depth)] * len(population_mean))
    return pd.DataFrame(df_var_inference)


def get_df_var_inference2(df_plot, post_estimator="mean_posterior_phase"):
    """
    Computes the standard deviation of posterior phase estimates for each population and sequencing depth,
    and returns the results as a pandas DataFrame.
    Parameters
    ----------
    df_plot : pandas.DataFrame
        Input DataFrame containing at least the following columns:
        - 'population': Identifies the population group (as integers or strings).
        - 'seq_depth': Sequencing depth values.
        - 'mean_posterior_phase': Posterior phase estimates (float).

    post_estimator : str, optional
        Either 'mean_posterior_phase' or 'mode_posterior_phase' to specify which
        posterior phase estimate to use for the calculations. Default is 'mean_posterior_phase'.
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - 'population_mean': The mean phase value for each population.
        - 'cStd': The computed standard deviation of the posterior phase distribution for each population and sequencing depth.
        - 'seq_depth': The sequencing depth corresponding to each entry.
    Notes
    -----
    - The function assumes that the number of unique populations determines the phase means, which are spaced evenly between 0 and 24.
    - Uses a histogram with 24 bins over the range [0, 24] to estimate the posterior phase distribution.
    - Relies on an external function `sr.compute_posterior_statistics` to compute statistics from the histogram.
    """

    n_pop = len(df_plot["population"].astype(int).unique())
    df_var_inference = {
        "inf_population_mean": [],
        "inf_population_std": [],
        "sim_population_mean": [],
        "sim_population_std": [],
        "seq_depth": [],
        "population": [],
    }
    for seq_depth in df_plot["seq_depth"].unique():
        std_phases = []
        population_mean = []
        population = []
        sim_means = []
        sim_cstds = []
        for g in range(n_pop):
            g = str(g)
            df_tmp = df_plot[
                (df_plot["population"] == g) & (df_plot["seq_depth"] == seq_depth)
            ]
            phis = df_tmp[post_estimator].values * sr.w
            true_phis = df_tmp["true_phase"].values
            cstd = circstd(phis)
            cmean = circmean(phis)
            sim_mean = circmean(true_phis)
            sim_cstd = circstd(true_phis)
            sim_means.append(sim_mean)
            sim_cstds.append(sim_cstd)
            population.append(g)
            std_phases.append(float(cstd))
            population_mean.append(cmean)
        df_var_inference["inf_population_mean"].extend(population_mean)
        df_var_inference["inf_population_std"].extend(std_phases)
        df_var_inference["sim_population_mean"].extend(sim_means)
        df_var_inference["sim_population_std"].extend(sim_cstds)
        df_var_inference["seq_depth"].extend([str(seq_depth)] * len(population_mean))
        df_var_inference["population"].extend(population)
    return pd.DataFrame(df_var_inference)


def assign_replicates(
    obs: pd.DataFrame, group_by_cols: list, n_replicates: int, seed: int = 42
) -> pd.Series:
    """
    Assigns cells to reproducible replicate groups.

    Args:
        obs: The DataFrame (e.g., adata.obs) containing cell metadata.
        group_by_cols: List of columns to group by (e.g., ['celltype', 'sample_name']).
        n_replicates: The number of subsets to create.
        seed: The random seed for reproducible shuffling.

    Returns:
        A pandas Series with the replicate number (0 to n_replicates-1) for each cell.
    """
    # Create a new Series to store the replicate assignments
    replicate_assignments = pd.Series(index=obs.index, dtype=int)

    # Use a fixed seed for reproducibility
    rng = np.random.RandomState(seed)

    # Group by the specified columns
    for _, group_df in obs.groupby(group_by_cols):
        # Get the indices of the cells in this group
        group_indices = group_df.index

        # Shuffle the indices reproducibly
        shuffled_indices = rng.permutation(group_indices)

        # Create an array of replicate numbers (e.g., [0, 1, 2, 0, 1, 2, ...])
        rep_numbers = np.tile(
            np.arange(n_replicates), len(shuffled_indices) // n_replicates + 1
        )
        rep_numbers = rep_numbers[: len(shuffled_indices)]

        # Assign the replicate numbers to the shuffled indices
        replicate_assignments.loc[shuffled_indices] = rep_numbers

    return replicate_assignments
