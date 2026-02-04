import pandas as pd
import os
import numpy as np
from tempo import unsupervised_alg
import scritmo as sr
from scipy.stats import vonmises
from scritmo import w, rh


def gene_parameters_prior(
    params_g,
    par_folder_out: str,
    fix_phase: bool = False,
    prior_95_interval: float = 0.39,
):
    path = par_folder_out + "/gene_priors/"
    os.makedirs(path, exist_ok=True)

    genes = params_g.index.tolist()

    # save as .txt file the ccg
    gene_path = path + "core_clock_genes.txt"
    with open(gene_path, "w") as f:
        for gene in genes:
            f.write(f"{gene}\n")

    if fix_phase:
        prior_95_interval = 0.01

    acrophases = params_g["phase"].values

    # Compute 95% interval for each gene's prior acrophase
    intervals = []

    df = pd.DataFrame(
        {
            "gene": genes,
            "prior_acrophase_loc": acrophases,
            "prior_acrophase_95_interval": prior_95_interval,
        }
    )
    gene_acrophase_prior_path = path + "core_clock_acrophase_prior.csv"
    df.to_csv(gene_acrophase_prior_path, index=False)

    # return paths
    return gene_acrophase_prior_path, gene_path


def tempo_wrapper(
    adata,
    folder_out,
    params_g,
    reference_gene="Bmal1",
    use_clock_input_only=True,
    use_de_novo_cycler_detection=False,
    num_phase_grid_points=24,
    pass_layer2X="spliced",
    fix_phase=True,
):
    """
    Wrapper for tempo

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the data.
    folder_out : str
        Output folder path.
    params_g : either a pd.DataFrame or str
        Gene parameters or path to gene parameters file.
    """

    if pass_layer2X is not None:
        adata.X = adata.layers[pass_layer2X].copy()

    params_g = sr.Beta(params_g)

    # create ccg file and gene acrophase prior file files from params_g
    # than save them and pass the path to tempo, also use fix_phase do
    # detrmine the width of the gene acrophase prior
    gene_acrophase_prior_path, gene_path = gene_parameters_prior(
        params_g=params_g,
        par_folder_out=folder_out,
        fix_phase=fix_phase,
    )

    unsupervised_alg.run(
        adata=adata,
        folder_out=folder_out,
        gene_acrophase_prior_path=gene_acrophase_prior_path,
        core_clock_gene_path=gene_path,
        reference_gene=reference_gene,
        use_clock_input_only=use_clock_input_only,
        use_de_novo_cycler_detection=use_de_novo_cycler_detection,
        num_phase_grid_points=num_phase_grid_points,
    )


def get_param_dataframe(folder_out):
    path = folder_out + "tempo_results/opt/cycler_gene_prior_and_posterior.tsv"
    params_df = pd.read_csv(path, sep="\t", index_col=0)
    return params_df


def transform_tempo_df(folder_out, return_tempo_df=False):

    par = get_param_dataframe(folder_out)
    theta_x, theta_y = par["phi_euclid_cos"], par["phi_euclid_sin"]
    phi = np.arctan2(theta_y, theta_x) % (2 * np.pi)
    amp = par.A_loc
    mu = par.mu_loc

    new_df = pd.DataFrame(
        {
            "a_0": mu,
            "phase": phi,
            "amp": amp,
        },
        index=par.index,
    )
    new_df = sr.Beta(new_df)
    new_df.get_cartesian(inplace=True)
    new_df.get_amp(inplace=True)
    if return_tempo_df:
        return new_df, par
    else:
        return new_df


def get_cell_posterior(folder_out):
    path = folder_out + "tempo_results/0/cell_phase_estimation/"
    cell_posterior = pd.read_csv(path + "cell_posterior.tsv", sep="\t", index_col=0)
    Nc, Nbins = cell_posterior.iloc[:, :].values.shape
    l_xc = cell_posterior.values.T
    # make such that l_xc takes into account the bin size
    delta_x = 2 * np.pi / Nbins
    l_xc = l_xc / delta_x
    return l_xc


def create_tempo_results_dataframe(
    l_xc,
    adata,
    context_col: str,
    ext_phase: np.ndarray,
    genes: list,
    sample_col: str = "sample_name",
    zt_col: str = "ZTmod",
    post_estimator: str = "post_mode",
    layer="spliced",
    other_obs_cols: list = [],
):
    """
    Creates the main results DataFrame (df_res) from a trained ContextModel.
    (This function is unchanged)
    """

    post_mean_c, post_var_c, post_std_c = sr.circular.compute_posterior_statistics(l_xc)
    post_mode_c = sr.compute_posterior_mode(l_xc)

    if post_estimator == "post_mode":
        phi = post_mode_c
    else:
        phi = post_mean_c

    # Align phases and calculate MAE (cad)
    phi_aligned, best_mad = sr.optimal_shift(phi, ext_phase)
    cad = sr.circular_deviation(ext_phase, phi_aligned, period=2 * np.pi) * rh

    # Re-compute ccounts
    # Use intersection to be safe
    ccg_genes = np.intersect1d(genes, adata.var_names)
    ccounts = np.array(adata[:, ccg_genes].layers[layer].sum(axis=1)).squeeze()

    # Create the DataFrame
    df_res = pd.DataFrame()
    # Add index to df_res to be able to call assign_replicates
    df_res.index = adata.obs.index

    df_res["true_phase"] = ext_phase
    df_res["context"] = adata.obs[context_col].values
    df_res["pred_phase"] = phi
    df_res["pred_phase_h"] = phi * rh
    df_res["post_std_c"] = post_std_c
    df_res["MAE"] = cad
    df_res["counts"] = np.array(adata.layers[layer].sum(1)).squeeze()
    df_res["ccounts"] = ccounts
    df_res[zt_col] = adata.obs[zt_col].values
    df_res["sample_name"] = adata.obs[sample_col].values
    df_res["method"] = "tempo"

    for col in other_obs_cols:
        df_res[col] = adata.obs[col].values

    # Add ZT_sample for easier grouping if needed
    df_res["ZT_sample"] = (
        df_res[zt_col].astype(str) + "_" + df_res["sample_name"].astype(str)
    )

    print(f"Created df_res. Median MAE: {best_mad*rh:.2f} hours")
    return df_res
