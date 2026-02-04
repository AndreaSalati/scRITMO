import numpy as np
from scipy.special import i0, i1
from scipy.sparse import csr_matrix
from .utils import get_ext_time
from .simulations import simulate_data_no_context
from ..utils import nmp
import torch
import scritmo as sr


def simulate_synchronized_populations(
    cmodel,
    adata,
    context_col: str | None = None,
    layer_to_use="spliced",
    ext_time_label="ZT",
    kappa=np.inf,
    period=24,
    device="cuda",
    inplace=True,
    output_column="simulation_perfect_synchro",
    return_sim_data=False,
):
    extra = cmodel.extra_gene_mode
    # check inputs
    if cmodel.context_mode == "none" or len(cmodel.context_u) == 1:
        context_col = None
        context = ["none"]
    elif context_col is not None:
        labels1, counts1 = np.unique(
            adata.obs[context_col].astype(str), return_counts=True
        )
        labels2, counts2 = np.unique(
            np.array(cmodel.context, dtype=str), return_counts=True
        )
        if not (np.all(labels1 == labels2) and np.all(counts1 == counts2)):
            raise ValueError(
                "The context given and the context given to the model do not match"
            )
        context = labels1
    else:
        raise ValueError("No context given, but the model uses a context")

    library_size = csr_matrix(adata.layers[layer_to_use]).sum(axis=1).A1
    ext_time = get_ext_time(adata.obs[ext_time_label], period=period)
    if np.all(np.isfinite(kappa)):
        ext_time = np.random.vonmises(ext_time, kappa)

    # get the clock
    fourier_coefficients_y = cmodel.get_parameter_dataframe_context(
        np.array(range(cmodel.Ng))
    )

    # simulate data according to the clock of each context
    if inplace:
        df_obs = adata.obs
    else:
        df_obs = adata.obs.copy()
    df_obs[output_column] = np.nan
    data = np.empty((cmodel.Nc, cmodel.Ng))

    if extra:
        fourier_coefficients_e_y = cmodel.get_parameter_dataframe(get_extra=True)
        data_e = np.zeros((cmodel.Nc, cmodel.Ne_g))

    i_df_total = []
    i_model_total = []
    Ng = cmodel.Ng

    for i, y in enumerate(context):
        if context_col is not None:
            i_df = (df_obs[context_col] == y).values
            i_model = cmodel.context == y
            fourier_coefficients = fourier_coefficients_y[y]
            if extra:
                mask_g = nmp(cmodel.dm_e_gy[:, i])
                par_e_temp = fourier_coefficients_e_y[mask_g]
                # concatenate par_e_temp and fourier_coefficients along index
                fourier_coefficients = pd.concat(
                    [fourier_coefficients, par_e_temp], axis=0
                )
        else:
            i_df = np.array([True] * len(df_obs))
            i_model = i_df
            fourier_coefficients = list(fourier_coefficients_y.values())[0]
        data_tmp_ = simulate_data_no_context(
            phases=ext_time[i_df],
            seq_depths=library_size[i_df],
            fourier_coefficients=fourier_coefficients,
            context_label=y,
        )

        data_tmp = data_tmp_[:, :Ng]
        data[i_model] = data_tmp.copy()
        i_df_total.extend(np.where(i_df)[0])
        i_model_total.extend(np.where(i_model)[0])

        if extra:
            data_tmp_e = data_tmp_[:, Ng:]
            data_e[i_df, :][:, mask_g] = data_tmp_e.copy()

    # infer phases on the synthetic data
    data_c = torch.tensor(data, dtype=torch.float32, device=device)
    data_c = data_c.unsqueeze(0).expand(cmodel.Nx, cmodel.Nc, cmodel.Ng)

    if extra:
        data_e_c = torch.tensor(data_e, dtype=torch.float32, device=device)
        data_e_c = data_e_c.unsqueeze(0).expand(cmodel.Nx, cmodel.Nc, cmodel.Ne_g)
        posterior_xc = cmodel.get_phase_posteriors(
            data_c, method="simpson", y_e=data_e_c
        )
    else:
        posterior_xc = cmodel.get_phase_posteriors(data_c, method="simpson")
    post_mean_c, _, _ = sr.compute_posterior_statistics(
        posterior_xc,
    )
    col_idx = df_obs.columns.get_loc(output_column)
    df_obs.iloc[i_df_total, col_idx] = np.array(post_mean_c)[i_model_total]
    _, _, shift = sr.optimal_shift(
        df_obs[output_column].values, ext_time, return_shift=True
    )
    df_obs[output_column] = (df_obs[output_column] - shift) % (2 * np.pi)
    if not inplace:
        if return_sim_data:
            return df_obs[output_column], data
        return df_obs[output_column]


def estimate_phase_desynchrony(
    cmodel,
    adata: anndata.AnnData,
    inferred_phases: list,
    context_col: str | None = None,
    context_match_cStd_col: str | None = None,
    layer_to_use="spliced",
    ext_time_label="ZT",
    max_iteration=30,
    epsilon=0.5,
    period=24,
    device="cuda",
    max_kappa=1000,
):
    def get_inferred_phases(kappas, context_col=context_col):
        return simulate_synchronized_populations(
            cmodel=cmodel,
            adata=adata,
            context_col=context_col,
            layer_to_use=layer_to_use,
            ext_time_label=ext_time_label,
            kappa=kappas,
            period=period,
            device=device,
            inplace=False,
        )

    if context_match_cStd_col is not None:
        context_col = context_match_cStd_col
    inferred_phases = np.array(inferred_phases)
    if context_col is not None:
        zt_contexts = adata.obs[[ext_time_label, context_col]].copy()
        zt_contexts[ext_time_label] = get_ext_time(
            zt_contexts[ext_time_label], period=period, convert_rad=False
        )
        unique_zt_context = zt_contexts.drop_duplicates()
    else:
        zt_contexts = adata.obs[ext_time_label].copy().to_frame()
        zt_contexts[ext_time_label] = get_ext_time(
            zt_contexts[ext_time_label], period=period, convert_rad=False
        )
        unique_zt_context = zt_contexts.drop_duplicates().to_frame()

    i_zt_contexts = [
        (zt_contexts == zt_context[1]).all(axis=1)
        for zt_context in unique_zt_context.iterrows()
    ]
    cStds_dataset = [circstd(inferred_phases[i]) for i in i_zt_contexts]

    kappas_fitted, cStds_fitted = find_kappas_matching_cStds(
        cStds_dataset,
        i_zt_contexts,
        get_inferred_phases,
        max_kappa=max_kappa,
        epsilon=epsilon / period * (2 * np.pi),
        max_iter=max_iteration,
    )
    rad_unit = period / (2 * np.pi)
    unique_zt_context = unique_zt_context.reset_index().drop("index", axis=1)
    res = {"simulated_phases_cStd": [], "cStd_inferred": [], "true_cStd": []}
    for i in range(len(unique_zt_context)):
        res["simulated_phases_cStd"].append(circular_std(kappas_fitted[i]) * rad_unit),
        res["cStd_inferred"].append(cStds_fitted[i] * rad_unit),
        res["true_cStd"].append(cStds_dataset[i] * rad_unit),
    res = pd.DataFrame(res, index=unique_zt_context.index)
    return res.merge(unique_zt_context, left_index=True, right_index=True)


def find_kappas_matching_cStds(
    cStds, i_zt_contexts, get_angles, max_kappa=1000, epsilon=1e-1, max_iter=30
):
    """
    Finds von Mises kappa values that match the target circular standard deviations
    for multiple independent contexts using a bisection method.

    Parameters:
        cStds (list of float): Target circular standard deviations (in radians).
        i_zt_contexts (list of boolean arrays): Boolean masks per context over the same data.
        get_angles (function): Function that accepts a vector of kappas and returns the full angle array.
        max_kappa (float): Upper bound for kappa.
        epsilon (float): Convergence threshold.
        max_iter (int): Maximum number of iterations.

    Returns:
        kappas (np.ndarray): Estimated kappa values per context.
        cstds  (np.ndarray): Corresponding estimated cStd values.
    """
    assert len(cStds) == len(
        i_zt_contexts
    ), "cStds and i_zt_contexts must match in length"
    n = len(cStds)

    def to_log(kappa):
        return np.log(kappa + 1)

    def from_log(y):
        return np.exp(y) - 1

    # Initial bounds in log-space
    y_lows = np.full(n, to_log(0))
    y_highs = np.full(n, to_log(max_kappa))
    kappas_high = from_log(y_highs)

    def extend_kappas(kappas, i_zt_contexts=i_zt_contexts):
        extended = np.empty(len(i_zt_contexts[0]))
        for kappa, i in zip(kappas, i_zt_contexts):
            extended[i] = kappa
        return extended

    # CALLING the function !!
    angles_high = get_angles(extend_kappas(kappas_high))
    cstds_high = np.array([circstd(angles_high[mask]) for mask in i_zt_contexts])
    # Identify indices that already failed to reach target cStd at max_kappa
    too_low_at_high_kappa = cstds_high > np.array(cStds)
    active = np.array(~too_low_at_high_kappa, dtype=bool)
    pbar = tqdm(range(max_iter))
    for _ in pbar:
        y_mids = (y_lows + y_highs) / 2
        kappas_mid = from_log(y_mids)

        angles = get_angles(extend_kappas(kappas_mid))
        cstds_mid = np.array([circstd(angles[mask]) for mask in i_zt_contexts])
        #                        current value
        active = active & (np.abs(cstds_mid - np.array(cStds)) > epsilon)
        if np.all(~active):
            pbar.set_description(f"converged {sum(~active)}/{len(active)}")
            break

        for i in range(n):
            if not active[i]:
                continue
            if cstds_mid[i] > cStds[i]:
                y_lows[i] = y_mids[i]  # cStd too high → kappa too low
            else:
                y_highs[i] = y_mids[i]  # cStd too low → kappa too high
        pbar.set_description(f"converged {sum(~active)}/{len(active)}")

    # Final estimates
    y_final = (y_lows + y_highs) / 2
    kappas_final = from_log(y_final)

    # Use max_kappa for the failed ones
    kappas_final[too_low_at_high_kappa] = max_kappa
    angles_final = get_angles(extend_kappas(kappas_final))
    cstds_final = np.array([circstd(angles_final[mask]) for mask in i_zt_contexts])

    return kappas_final, cstds_final
