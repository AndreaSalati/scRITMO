import torch
import pyro
from pyro import distributions as dist


def model_tempo(y, mp):
    """
    Model for the tempo-like inference
    """
    Nx, Nc, Ng = y.shape

    disp = pyro.param("disp", torch.tensor(0.1), constraint=dist.constraints.positive)

    a_1 = pyro.sample(
        "a_1",
        dist.Normal(mp["a_1"].clone().to(mp["dev"]), mp["std_a"].clone().to(mp["dev"])),
    )
    b_1 = pyro.sample(
        "b_1",
        dist.Normal(mp["b_1"].clone().to(mp["dev"]), mp["std_b"].clone().to(mp["dev"])),
    )
    a_0 = pyro.sample(
        "a_0",
        dist.Normal(mp["a_0"].clone().to(mp["dev"]), mp["std_m"].clone().to(mp["dev"])),
    )

    # scan all possible angles - precompute this once and reuse
    phi_x = torch.linspace(0, 2 * torch.pi, Nx)
    # Use reshape and expand instead of tile for better performance
    phi_xc = phi_x.reshape(Nx, 1, 1).expand(Nx, Nc, 1).to(mp["dev"])

    # define the model
    E_xcg = a_1 * torch.cos(phi_xc) + b_1 * torch.sin(phi_xc) + a_0
    E_xcg = torch.exp(E_xcg) * mp["counts"].to(mp["dev"])

    r = 1 / disp
    p = disp * E_xcg / (1 + disp * E_xcg)

    # define the likelihood
    pyro.sample("obs", dist.NegativeBinomial(r, p), obs=y)


def guide_tempo(y, mp):
    """Guide function for tempo inference"""
    a_loc_g = pyro.param(
        "a_loc_g", mp["a_1"].clone().to(mp["dev"]), constraint=dist.constraints.real
    )
    a_scale_g = pyro.param(
        "a_scale_g",
        mp["std_a"].clone().to(mp["dev"]),
        constraint=dist.constraints.positive,
    )
    b_loc_g = pyro.param(
        "b_loc_g", mp["b_1"].clone().to(mp["dev"]), constraint=dist.constraints.real
    )
    b_scale_g = pyro.param(
        "b_scale_g",
        mp["std_b"].clone().to(mp["dev"]),
        constraint=dist.constraints.positive,
    )
    m_loc_g = pyro.param(
        "m_loc_g", mp["a_0"].clone().to(mp["dev"]), constraint=dist.constraints.real
    )
    m_scale_g = pyro.param(
        "m_scale_g",
        mp["std_m"].clone().to(mp["dev"]),
        constraint=dist.constraints.positive,
    )

    a_1 = pyro.sample(
        "a_1", dist.Normal(a_loc_g, a_scale_g)
    )  # a_g is the amplitude of the cosine
    b_1 = pyro.sample(
        "b_1", dist.Normal(b_loc_g, b_scale_g)
    )  # b_g is the amplitude of the sine
    a_0 = pyro.sample(
        "a_0", dist.Normal(m_loc_g, m_scale_g)
    )  # m_g is the mean of the distribution
