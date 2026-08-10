import numpy as np
import torch
from scritmo import Beta, optimal_shift, check_library_size_fallback
import pandas as pd
import json
from scipy.stats import circstd, circmean
from scritmo.beta import cSVD_beta, polar_module


def df2dict(df, return_index=False):
    """
    converts a pandas dataframe to a dictionary.
    """

    index = df.index.values
    d = df.to_dict(orient="list")

    for k in d.keys():
        d[k] = np.array(d[k])[None, :]

    if return_index:
        return d, index
    else:
        return d


def harmonic_dm_torch(phases, n_harmonics=1, add_intercept=True):
    """
    Constructs a design matrix for harmonic regression using PyTorch tensors.

    Parameters:
    -----------
    phases : numpy.ndarray
        1D array of phases in radians (0 to 2π)
    n_harmonics : int, default=1
        Number of harmonics to include
    add_intercept : bool, default=True
        Whether to add a column of ones as intercept

    Returns:
    --------
    X : torch.Tensor
        Design matrix (N, 2 * n_harmonics [+ 1 if intercept])
    """

    terms = [torch.cos(h * phases).unsqueeze(-1) for h in range(1, n_harmonics + 1)] + [
        torch.sin(h * phases).unsqueeze(-1) for h in range(1, n_harmonics + 1)
    ]

    X = torch.cat(terms, dim=-1)

    # might be wrongl;y implemented

    if add_intercept:
        intercept = torch.ones_like(phases).unsqueeze(1)
        X = torch.cat([intercept, X], dim=-1)

    return X


def circ_std_P(P):
    """
    Compute the circular standard deviation from a discrete probability distribution over angles.

    Parameters:
    - P: 1D tensor of shape (N,), assumed to be a probability distribution over [0, 2π)

    Returns:
    - std: scalar tensor, circular standard deviation
    """
    N = P.shape[0]
    phis = torch.linspace(0, 2 * torch.pi, N, device=P.device, dtype=P.dtype)

    # complex exponentials e^{i phi}
    e_iphi = torch.cos(phis) + 1j * torch.sin(phis)

    # weighted mean resultant vector
    R = torch.sum(P * e_iphi)

    std = torch.sqrt(-2 * torch.log(torch.abs(R)))
    return std


def set_context_mode(self, context_mode):
    """
    Method to change the context mode during training/inference.

    Args:
        context_mode: String, one of "none", "intercept", or "full"
    """
    self.context_mode = context_mode

    # Update requires_grad based on context_mode
    # m_g is tunred off with context to mantain model identifiability
    if context_mode == "none":
        # Freeze both context parameters
        self.m_g.requires_grad = True
        self.m_yg.requires_grad = False
        self.log_lambda_y.requires_grad = False

    elif context_mode == "intercept":
        # Only allow intercept adjustments
        self.m_g.requires_grad = False
        self.m_yg.requires_grad = True
        self.log_lambda_y.requires_grad = False

    elif context_mode == "lambda":
        # Only allow amp rescaling adjustments
        self.m_g.requires_grad = True
        self.m_yg.requires_grad = False
        self.log_lambda_y.requires_grad = True

    elif context_mode == "full":
        # Allow both intercept and amplitude adjustments
        self.m_g.requires_grad = False
        self.m_yg.requires_grad = True
        self.log_lambda_y.requires_grad = True

    elif context_mode == "full_lambda":
        # Allow both intercept and amplitude adjustments
        self.m_g.requires_grad = False
        self.m_yg.requires_grad = True
        self.log_lambda_y.requires_grad = True

    elif context_mode == "context_only":
        # Only allow context adjustments
        self.m_g.requires_grad = False
        self.log_amp.requires_grad = False
        self.acrophase.requires_grad = False
        self.m_yg.requires_grad = True
        self.log_lambda_y.requires_grad = True

    elif context_mode == "disp_only":
        # block everything, fit only disp
        self.m_g.requires_grad = False
        self.log_amp.requires_grad = False
        self.m_yg.requires_grad = False
        self.log_lambda_y.requires_grad = False
        self.acrophase.requires_grad = False

    else:
        raise ValueError(f"Unknown context_mode: {context_mode}")


def get_shift_y(ext_time, ph, context, center=False):
    """
    This function aligns all cells belonging to the
    same context class to the ext_time vector.
    """
    context_u = np.unique(context)
    shift_y = pd.Series(index=context_u)

    for i, ct in enumerate(context_u):
        mask = context == ct
        tp_ct = ext_time[mask]
        delta = optimal_shift(ph[mask], tp_ct, return_shift_only=True, allow_flip=False)
        if delta > np.pi:
            delta -= 2 * np.pi

        shift_y[ct] = delta

    shift_y = -shift_y

    if center:
        mean_shift = circmean(shift_y)
        shift_y -= mean_shift

    # for some reason we need minus here
    return -shift_y


def calculate_aposteriori_lambda_y(
    par_dict,
    amp_col="amp",
    mean_col="a_0",
    gene_col="gene",
    context_col="context",
    scale_factor=1.0,
):
    """
    Calculates a weighted rhythmicity score for each cell type (context).

    Logic:
    1. Center amps by gene (Amp - Median).
    2. Weight by expression (exp(mean)).
    3. Aggregate and exponentiate back to linear scale.
    """

    def extract_amps(par_dict):
        """
        Converts the dictionary of parameters into a single DataFrame.
        """
        context_u = list(par_dict.keys())
        par_list = []
        for ct in context_u:
            par = par_dict[ct]
            # Copy to avoid SettingWithCopy warnings
            par = par.copy()
            par["gene"] = par.index
            par = par.loc[:, ["gene", "amp", "a_0"]]
            par.index = np.arange(len(par))
            par["context"] = ct
            par_list.append(par)
        df_par = pd.concat(par_list, axis=0)
        return df_par

    # Work on a copy to avoid modifying the original df
    df = extract_amps(par_dict).copy()

    # 1. Calculate Median Amp per Gene
    df["median_amp"] = df.groupby(gene_col)[amp_col].transform("median")

    # 2. Center Amplitudes (Log Space)
    # Result: >0 if stronger than median, <0 if weaker
    df["centered_amp"] = df[amp_col] - df["median_amp"]

    # 3. Define Weights (Linear Space)
    # Transform log-mean to linear abundance for weighting
    if mean_col is not None:
        df["weight"] = np.exp(df[mean_col])
    else:
        df["weight"] = 1.0

    # 4. Compute Weighted Average per Context
    # Formula: Sum(Centered_Amp * Weight) / Sum(Weight)
    df["weighted_term"] = df["centered_amp"] * df["weight"]

    # Group by context and calculate the weighted mean
    # We use a custom apply or summation approach
    grouped = df.groupby(context_col)
    raw_scores = grouped["weighted_term"].sum() / grouped["weight"].sum()

    # 5. Final Transformation
    # Exponentiate to get a positive score centered at 1.0
    final_vector = np.exp(raw_scores * scale_factor)

    return final_vector.sort_values(ascending=False)


def nmp(tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.detach().numpy()


def get_ccg_pars(path=None):
    if path is None:
        path = "/home/maxine/Documents/andrea/context_repo/data/params_g/ccg_zhang_context.csv"
    params_g = Beta(path)
    return params_g


def get_params_g(
    organ,
    ct,
    pb=False,
    filter=True,
    fname_list="gene_set_BIG_withImmune.json",
    main_path="/home/maxine/Documents/andrea/context_repo/data/params_g/",
):
    if pb:
        main_path += "PB/"
    if " " in ct:
        ct = ct.replace(" ", "_")
    name = organ + "_" + ct
    gene_list_path = main_path + fname_list
    par_df_path = main_path + f"{name}.csv" if not pb else main_path + f"{name}_pb.csv"

    # load json file gene_list_path
    with open(gene_list_path, "r") as f:
        gene_list_ = json.load(f)

    gene_list = gene_list_[name]
    params_g = Beta(par_df_path)
    if filter:
        params_g = params_g.loc[gene_list]
    return params_g


def assemble_mp(
    adata,
    params_g,
    labels,
    counts=None,
    layer="spliced",
    n_theta=24,
    device="cuda",
    unspliced_layer=None,
    weights_g=None,
    fixed_cell_phases=None,
):
    """
    Assemble the model parameters and data for the model.
    Parameters
    ----------
    adata : AnnData
        AnnData object with the data.
    params_g_init : pd.DataFrame
        DataFrame with the model parameters for each gene.
    counts : np.ndarray
        Array with the counts for each cell.
    labels : np.ndarray
        Array with the labels for each cell.
    layer : str
        Layer to use for the data. If None, use adata.X.
    n_theta : int
        Number of grid points to use in the unit circle.
    device : str
        Device to use for the model. 'gpu' or 'cpu'.
    unspliced_layer : str
        Layer to use for unspliced data. If None, no unspliced counts.
    Returns
    -------
    data_c : torch.Tensor
        Data tensor for the model.
    mp : dict
        Model parameters for the model.
    """

    # send error if params_g.index is not all included in adata.var_names
    if not np.all(np.isin(params_g.index.values, adata.var_names)):
        raise ValueError("params_g.index is not all included in adata.var_names")

    genes = params_g.index.values
    Nc, Ng = adata.shape[0], len(genes)
    mp = {}

    if layer is None:
        data = adata[:, genes].X.toarray()
    else:
        data = adata[:, genes].layers[layer].toarray()
    # create training data tensor
    data_c = torch.tensor(data, dtype=torch.float32, device=device)

    if fixed_cell_phases is None:
        data_c = data_c.unsqueeze(0).expand(n_theta, Nc, Ng)
    else:
        data_c = data_c.unsqueeze(0)

    if unspliced_layer is not None:
        data_u = adata[:, genes].layers[unspliced_layer].toarray()
        data_u_c = torch.tensor(data_u, dtype=torch.float32, device=device)
        if fixed_cell_phases is None:
            data_u_c = data_u_c.unsqueeze(0).expand(n_theta, Nc, Ng)
        else:
            data_u_c = data_u_c.unsqueeze(0)
        mp["unspliced_mode"] = True

    if counts is None:
        counts = check_library_size_fallback(adata, layer=layer, context="assemble_mp")

    if counts.ndim == 1:
        counts = counts.squeeze()[:, None]
    if type(counts) is pd.DataFrame:
        counts = counts.values

    mp["counts"] = torch.tensor(counts, dtype=torch.float32, device=device)
    mp["weights_g"] = (
        torch.tensor(weights_g, dtype=torch.float32, device=device)
        if weights_g is not None
        else None
    )
    mp["params_g"] = Beta(params_g)

    mp["context"] = np.array(labels)

    if unspliced_layer is not None:
        mp["counts_u"] = None
        return data_c, mp, data_u_c
    
    return data_c, mp
