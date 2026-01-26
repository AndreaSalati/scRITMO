import pandas as pd
import numpy as np
from .dryseq_utils import create_matrix_list, compute_BICW, create_model_labels
from .dryseq_fitting import estimate_dispersions, run_iterative_fitting


def dryseq(count_data, group, time, period=24, n_cores=4):
    """
    Main execution function.

    Args:
        count_data (pd.DataFrame): (Samples x Genes) Raw counts.
        group (list/array): Condition labels for each sample (row).
        time (list/array): Time points for each sample (row).
        period (int): Oscillation period.
        n_cores (int): Number of CPU cores.

    Returns:
        dict: 'results' dataframe (Genes x Stats) and intermediate data.
    """

    # 1. Validation and Setup
    count_data = count_data.dropna()
    samples = count_data.index

    # Metadata for PyDESeq2
    metadata = pd.DataFrame({"group": group, "time": time}, index=samples)

    # 2. Estimate Dispersions (PyDESeq2)
    print("Estimating Dispersions with PyDESeq2...")
    # Count data is already Samples x Genes, perfect for PyDESeq2
    dds, dispersions, offsets = estimate_dispersions(
        count_data, metadata, design_formula="group"
    )

    # 3. Generate Models (Design Matrices)
    print("Generating Rhythmic Models...")
    n_groups = len(np.unique(group))
    models = create_matrix_list(time, group, n_groups, period)
    model_labels = create_model_labels(group, n_groups)
    print(f"Generated {len(models)} models to test per gene.")

    # 4. Iterative Fitting
    print("Fitting models (Parallel)...")
    # Pass the (Samples x Genes) matrix directly
    fit_results = run_iterative_fitting(
        count_data, models, dispersions, offsets, n_jobs=n_cores
    )

    # 5. Compile Results
    print("Compiling results...")
    final_stats = []
    n_samples = len(samples)
    k_params = np.array([m.shape[1] for m in models])

    for res in fit_results:
        gene = res["gene"]
        deviances = np.array(res["deviances"])

        # Calculate BIC: Deviance + ln(n)*k
        bic_values = deviances + np.log(n_samples) * k_params

        # Calculate BICW
        bicw = compute_BICW(bic_values)

        try:
            # Find the winner (lowest BIC)
            chosen_idx = np.nanargmin(bic_values)
            chosen_bicw = bicw[chosen_idx]
            winner_params = res["params"][chosen_idx]

            row = {
                "gene": gene,
                "chosen_model": chosen_idx + 1,  # 1-based index to match R
                "choesen_model_label": model_labels[int(chosen_idx + 1)],
                "chosen_model_BICW": chosen_bicw,
            }

            # Map parameters to their column names
            winner_cols = models[chosen_idx].columns
            for pname, pval in zip(winner_cols, winner_params):
                row[pname] = pval

            final_stats.append(row)

        except ValueError:
            # Handles cases where all fits failed (all NaNs)
            continue

    results_df = pd.DataFrame(final_stats)
    if not results_df.empty:
        results_df = results_df.set_index("gene")

    return {
        "results": results_df,
        "dispersions": dispersions,
        "models": models,  # Useful to debug which ID maps to which matrix
        "dds": dds,  # The raw PyDESeq2 object
        "model_labels": model_labels,
    }


# --- Quick Test Block ---
if __name__ == "__main__":
    print("Running test...")
    # Mock Data: 12 Samples, 100 Genes
    samples = [f"S{i}" for i in range(12)]
    genes = [f"Gene{i}" for i in range(100)]

    # (Samples x Genes)
    counts = pd.DataFrame(
        np.random.randint(0, 1000, size=(12, 100)), index=samples, columns=genes
    )

    group = ["WT"] * 4 + ["KO1"] * 4 + ["KO2"] * 4
    time = [0, 6, 12, 18] * 3

    out = dryseq(counts, group, time, n_cores=2)
    print("Top 5 Results:")
    print(out["results"].head())
