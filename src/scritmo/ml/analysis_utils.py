import pandas as pd
import numpy as np
import scritmo as sr
import anndata
from scritmo import w, rh
from .simulations.utils import assign_replicates
from scritmo import cstd2R, R2cstd
import matplotlib.pyplot as plt
import seaborn as sns


def create_results_dataframe(
    cmodel,
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
    """
    Creates the main results DataFrame (df_res) from a trained ContextModel.
    (This function is unchanged)
    """

    # Create the DataFrame
    df_res = pd.DataFrame()
    df_res.index = adata.obs.index

    if post_estimator == "post_mode":
        phi = cmodel.post_mode_c
    else:
        phi = cmodel.post_mean_c

    post_std_c = cmodel.post_std_c

    if ext_phase is not None:
        # Align phases and calculate MAE (cad)
        df_res["true_phase"] = ext_phase
        phi_aligned, best_mad = sr.optimal_shift(phi, ext_phase, allow_flip=allow_flip)
        cad = sr.circular_deviation(ext_phase, phi_aligned, period=2 * np.pi) * rh
        df_res["MAE"] = cad
        print(f"Created df_res. Median MAE: {best_mad*rh:.2f} hours")

    # Re-compute ccounts
    # Use intersection to be safe
    ccg_genes = np.intersect1d(cmodel.genes, adata.var_names)
    # ccounts = np.array(adata[:, ccg_genes].layers[layer].sum(axis=1)).squeeze()

    if context_col is not None and context_col in adata.obs.columns:
        df_res["context"] = adata.obs[context_col].values
    else:
        print("WARNING: context column not present")

    df_res["pred_phase"] = phi
    df_res["pred_phase_h"] = phi * rh
    df_res["post_mode"] = cmodel.post_mode_c
    df_res["post_mean"] = cmodel.post_mean_c
    df_res["post_std_c"] = post_std_c
    # df_res["counts"] = np.array(adata.layers[layer].sum(1)).squeeze()
    # df_res["ccounts"] = ccounts
    if ext_time_col in adata.obs.columns:
        df_res["ext_time_hours"] = adata.obs[ext_time_col].values
    if sample_col in adata.obs.columns:
        df_res["sample_name"] = adata.obs[sample_col].values

    for col in other_obs_cols:
        df_res[col] = adata.obs[col].values

    return df_res


def desync_results(
    df_real,
    df_sim,
    group_cols: list = None,
    disp_function=sr.cSTD,
    post_estimator: str = "post_mode",
    # real arguments
    metrics: dict = None,
    n_replicates: int | None = None,
    seed: int = 42,
    # sim arguments
    ext_time_col: str = "ext_time",
):
    """
    First it aggregates data by calling aggregate_real_results and aggregate_simulated_results,
    then fuses the 2 in one dataframe. Finally it computes the
    biological desynchrony with the quadrature difference.

    TO BE FIXED: Currently the group cols can only be two: context and sample_name.
    columns with another names will create problems
    """
    # 0. Handle Defaults
    if group_cols is None:
        group_cols = ["context", "sample_name"]
    else:
        print(
            "Warning: Currently only ['context', 'sample_name'] are supported as group_cols."
        )
    # 1. Aggregate Data
    real_agg = aggregate_real_results(
        df_real,
        group_cols=group_cols,
        disp_function=disp_function,
        post_estimator=post_estimator,
        metrics=metrics,
        n_replicates=n_replicates,
        seed=seed,
    )

    sim_agg = aggregate_simulated_results(
        df_sim,
        # group_cols=group_cols,
        disp_function=disp_function,
        post_estimator=post_estimator,
        ext_time_col=ext_time_col,
    )

    # 2. Initialize Mixed DataFrame
    df_mixed = real_agg.copy()

    # 3. Create Lookup Logic
    # We index the simulated data by the grouping columns for fast mapping
    try:
        sim_lookup = sim_agg.set_index(group_cols)
    except KeyError:
        print(f"Error: 'sim_agg' is missing one of the group_cols: {group_cols}")
        return df_mixed

    # Create the mapping index from the target dataframe (df_mixed)
    # This ensures alignment even if the sort order differs
    try:
        map_index = pd.MultiIndex.from_frame(df_mixed[group_cols])
    except KeyError:
        print(f"Error: 'df_mixed' is missing one of the group_cols: {group_cols}")
        return df_mixed

    # 4. Map Technical Component
    # We pull 'Technical_cSTD' from the sim results into the real results
    if "Technical_cSTD" in sim_lookup.columns:
        df_mixed["Technical_cSTD"] = map_index.map(sim_lookup["Technical_cSTD"])
    if "Technical_R" in sim_lookup.columns:
        df_mixed["Technical_R"] = map_index.map(sim_lookup["Technical_R"])

    # multiply by rh both Technical and Data cSTD to convert to hours
    df_mixed["Technical_cSTD"] = df_mixed["Technical_cSTD"] * rh
    df_mixed["Data_cSTD"] = df_mixed["Data_cSTD"] * rh

    # 5. Compute Biological Desynchrony (Quadrature Difference)

    df_mixed["Bio_cSTD"] = np.sqrt(
        df_mixed["Data_cSTD"] ** 2 - df_mixed["Technical_cSTD"] ** 2
    )
    df_mixed["Bio_R"] = cstd2R(df_mixed["Bio_cSTD"] / rh)

    return df_mixed


def desync_means(
    df_desync,
):
    context_u = df_desync["context"].unique()
    df_desync["Technical_cSTD2"] = df_desync["Technical_cSTD"] ** 2
    df_desync["Data_cSTD2"] = df_desync["Data_cSTD"] ** 2

    results = []
    for ct in context_u:
        df_ct = df_desync[df_desync["context"] == ct]
        if "organ" in df_ct.columns:
            organ = df_ct["organ"].iloc[0]
        else:
            organ = None
        if "celltype" in df_ct.columns:
            celltype = df_ct["celltype"].iloc[0]
        else:
            celltype = None
        if "condition" in df_ct.columns:
            condition = df_ct["condition"].iloc[0]
        else:
            condition = None

        weighted_mean_technical_var = np.average(
            df_ct["Technical_cSTD2"], weights=df_ct["group_size"]
        )
        weighted_mean_data_var = np.average(
            df_ct["Data_cSTD2"], weights=df_ct["group_size"]
        )

        final_bio_cSTD = np.sqrt(
            np.maximum(weighted_mean_data_var - weighted_mean_technical_var, 0)
        )
        final_technical_cSTD = np.sqrt(weighted_mean_technical_var)
        final_data_cSTD = np.sqrt(weighted_mean_data_var)

        results.append(
            {
                "ct": ct,
                "Technical_cSTD": final_technical_cSTD,
                "Bio_cSTD": final_bio_cSTD,
                "Data_cSTD": final_data_cSTD,
                "organ": organ,
                "celltype": celltype,
                "condition": condition,
            }
        )

    df_summary = pd.DataFrame(results).set_index("ct")[
        ["Technical_cSTD", "Bio_cSTD", "Data_cSTD", "organ", "celltype", "condition"]
    ]
    return df_summary


def aggregate_real_results(
    df_res: pd.DataFrame,
    group_cols: list = None,
    disp_function=sr.cSTD,
    post_estimator: str = "post_mode",
    metrics: dict = None,
    n_replicates: int | None = None,
    seed: int = 42,
):
    """
    Aggregates the 'df_res' DataFrame.
    If n_replicates is provided, it first splits the data into
    reproducible subsets and aggregates over them.
    """
    if group_cols is None:
        group_cols = ["context", "sample_name"]

    if metrics is None:
        metrics = {
            "MAE": "median",
            "post_std_c": "median",
            # "ccounts": "median",
            # "counts": "median",
            post_estimator: disp_function,
            "ext_time_hours": "first",
        }

    # --- NEW REPLICATE LOGIC ---
    df_to_agg = df_res.copy()

    # Ensure columns are string type for grouping
    for col in group_cols:
        df_to_agg[col] = df_to_agg[col].astype(str)

    if n_replicates is not None:
        # Use the same reproducible subsetting
        df_to_agg["replicate"] = assign_replicates(
            df_to_agg, group_cols, n_replicates, seed
        )
        # Add 'replicate' to the grouping
        group_cols_with_rep = group_cols + ["replicate"]
    else:
        group_cols_with_rep = group_cols

    # Aggregate the metrics
    agg_df = df_to_agg.groupby(group_cols_with_rep).agg(metrics).reset_index()
    # change name from pred_phase to Data_cSTD if present
    if post_estimator in agg_df.columns:
        agg_df = agg_df.rename(columns={post_estimator: "Data_cSTD"})
        agg_df["Data_R"] = cstd2R(agg_df["Data_cSTD"])

    # Get group sizes
    group_sizes = (
        df_to_agg.groupby(group_cols_with_rep).size().reset_index(name="group_size")
    )

    # Merge metrics and group sizes
    agg_df = agg_df.merge(group_sizes, on=group_cols_with_rep)

    # Create the replicate name in the output
    if n_replicates is not None:
        sample_col = group_cols[-1]  # Assumes sample_name is the last
        agg_df[sample_col] = (
            agg_df[sample_col].astype(str) + "_" + (agg_df["replicate"] + 1).astype(str)
        )
        agg_df = agg_df.drop(columns=["replicate"])

    return agg_df


def aggregate_simulated_results(
    df_sim: pd.DataFrame,
    disp_function=sr.cSTD,
    post_estimator: str = "post_mode",
    ext_time_col: str = "ext_time",
):
    # 1. Parse the 'run' out of the sample_name
    # Assuming sample_name format is "SampleName_runX" or similar
    # We want to group by ["context", "original_sample_name", "run_id"] first

    # Regex to split sample_name from the run suffix if you added one
    # Or, if you added a dedicated 'run_id' column in simulate_cell_populations, use that.
    # If you didn't, we can extract it:

    # Extract base sample name (removing _runX)
    df_sim["base_sample"] = df_sim["sample_name"].str.replace(
        r"_run\d+$", "", regex=True
    )
    df_sim["run_id"] = df_sim["sample_name"].str.extract(r"(run\d+)$")

    # 2. First Aggregation: Calculate Variance PER RUN
    # This collapses the 300 cells -> 1 variance value per run
    run_level_stats = (
        df_sim.groupby(["context", "base_sample", "run_id"])
        .agg(
            {
                post_estimator: disp_function,  # This computes cSTD for one run
                "post_std": "mean",
            }
        )
        .reset_index()
    )

    # Rename to clear names
    run_level_stats = run_level_stats.rename(columns={post_estimator: "cSTD_run"})

    # Convert to Variance (Statistics must be done on Variance, not STD)
    run_level_stats["Var_run"] = run_level_stats["cSTD_run"] ** 2

    # 3. Second Aggregation: Average the VARIANCES across runs
    # This collapses N runs -> 1 final technical value
    final_stats = (
        run_level_stats.groupby(["context", "base_sample"])
        .agg(
            {
                "Var_run": "mean",  # Mean of Variances
                "cSTD_run": "mean",  # (Optional) Mean of STDs for reference, but don't use for math
                "post_std": "mean",
            }
        )
        .reset_index()
    )

    # 4. Convert back to cSTD for the final output (if needed for the dataframe format)
    # But remember the "Twin" logic: we want the squared term.
    final_stats["Technical_cSTD"] = np.sqrt(final_stats["Var_run"])

    # Cleanup for merging
    final_stats = final_stats.rename(columns={"base_sample": "sample_name"})

    # Add the R conversion if needed
    final_stats["Technical_R"] = cstd2R(final_stats["Technical_cSTD"])

    return final_stats


def append_first_timepoint_periodic(df_desync, time_col: str = "ext_time_hours"):
    # (This function is unchanged)
    # Find the minimum ext_time_hours (first timepoint)
    zt_min = df_desync[time_col].min()
    # Select all rows corresponding to the first timepoint
    first_rows = df_desync[df_desync.ext_time_hours == zt_min].copy()
    # Set ext_time_hours to 24 + zt_min for these rows
    first_rows.ext_time_hours = 24 + zt_min
    # Concatenate to the original dataframe
    df_periodic = pd.concat([df_desync, first_rows], ignore_index=True)
    return df_periodic


def summarize_desync_results_one_ct(
    df_desync, plot=True, context: str = "", palette="Set1"
):
    weighted_mean_technical_R = np.average(
        df_desync["Technical_R"], weights=df_desync["group_size"]
    )
    weighted_mean_data_R = np.average(
        df_desync["Data_R"], weights=df_desync["group_size"]
    )

    final_technical_cSTD = sr.R2cstd(weighted_mean_technical_R) * rh
    final_data_cSTD = sr.R2cstd(weighted_mean_data_R) * rh
    final_bio_cSTD = np.sqrt(final_data_cSTD**2 - final_technical_cSTD**2)

    print("Final Technical cSTD (hours):", f"{final_technical_cSTD:.2f}h")
    print("Final Data cSTD (hours):", f"{final_data_cSTD:.2f}h")
    print("Final Bio cSTD (hours):", f"{final_bio_cSTD:.2f}h")

    if plot:
        equation_string2 = (
            "\n$\\sigma_{Bio} = \\sqrt{\\sigma_{Data}^2 - \\sigma_{Technical}^2}$"
        )
        plt.figure(figsize=(6, 5))
        bar_labels = [r"$\sigma_{Technical}$", r"$\sigma_{Bio}$", r"$\sigma_{Data}$"]
        bar_values = [final_technical_cSTD, final_bio_cSTD, final_data_cSTD]
        ax = sns.barplot(x=bar_labels, y=bar_values, hue=bar_labels, palette=palette)

        # Remove top frame
        ax.spines["top"].set_visible(False)

        # Add value labels on top of bars
        for i, v in enumerate(bar_values):
            ax.text(i, v + 0.05, f"{v:.2f}h", ha="center", va="bottom", fontsize=12)

        plt.ylabel("Circular STD [h]")
        plt.title(f"Desynchrony summary {context} \n" + equation_string2)
        plt.tight_layout()
        plt.show()

    return final_technical_cSTD, final_bio_cSTD, final_data_cSTD
