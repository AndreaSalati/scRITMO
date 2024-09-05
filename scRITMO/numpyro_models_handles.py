import numpy as np
import numpyro
import jax
import jax.numpy as jnp
from numpyro import distributions as dist

# from .numpyro_models import model, guide
# numpyro.set_platform('gpu')


def fit_gene_coeff(model, guide, y, true_phase, n_steps=10000, lr=0.001, **kwargs):
    """
    Fit the gene coefficients using SVI:
    Args:
    model: the model function
    guide: the guide function
    y: the data Nc x Ng
    counts: the counts Nc x 1
    n_steps: number of steps for training
    lr: learning rate
    guess: a dictionary containing the initial guess for the parameters

    Returns:
    The SVI object, containing the trained parameters, and much more
    """
    # preparing an educated guess for mean parameter
    counts = kwargs.get("counts", None)
    guess = kwargs.get("guess", None)
    dm = kwargs.get("dm", None)

    m_g = (y / counts).mean(axis=0, keepdims=True)
    m_g = np.log(m_g)

    block_cells_model = numpyro.handlers.substitute(
        model, {"phi_c": true_phase.reshape(-1, 1)}
    )
    block_cells_model = numpyro.handlers.block(block_cells_model, hide=["phi_c"])
    # VERY IMPORTANT! Always inisitalize m_g to good values, other wise it's not gonna work

    block_cells_model = numpyro.handlers.substitute(block_cells_model, {"m_g": m_g})

    if guess is not None:
        a_g = guess[:, 0]
        b_g = guess[:, 1]
        m_g = guess[:, 2]
        block_cells_model = numpyro.handlers.substitute(
            block_cells_model, {"a_g": a_g, "b_g": b_g, "m_g": m_g}
        )

    optimizer = numpyro.optim.Adam(step_size=lr)

    # define the inference algorithm
    svi = numpyro.infer.SVI(
        block_cells_model, guide, optimizer, loss=numpyro.infer.Trace_ELBO()
    )

    # run the inference
    print(counts.shape, y.shape)
    svi_g = svi.run(jax.random.PRNGKey(0), n_steps, y, counts=counts, dm=dm)
    return svi_g


# def fit_phases(model, guide, y, init_phase, svi_g, n_steps=10000, lr=0.001, **kwargs):
#     """
#     Fit the phases using SVI:
#     Args:
#     model: the model function
#     guide: the guide function
#     y: the data Nc x Ng
#     init_phase: initial phases Nc x 1,
#     svi_g: a dictionary containing the trained gene coefficients
#     n_steps: number of steps for training
#     lr: learning rate

#     Returns:
#     The SVI object, containing the trained parameters, and much more
#     """
#     counts = kwargs.get("counts", None)
#     # preparing an educated guess for mean parameter
#     model_ = numpyro.handlers.substitute(model, svi_g)
#     model_ = numpyro.handlers.block(model_, hide=["a_g", "b_g", "m_g"])
#     # substitute the true phases
#     model_ = numpyro.handlers.substitute(model_, {"phi_c": init_phase})

#     optimizer = numpyro.optim.Adam(step_size=lr)
#     svi = numpyro.infer.SVI(model_, guide, optimizer, loss=numpyro.infer.Trace_ELBO())

#     # run the inference
#     svi_c = svi.run(jax.random.PRNGKey(0), n_steps, y, counts=counts)
#     return svi_c


def fit_phases(
    model,
    guide,
    y,
    init_phase,
    params_g,
    n_steps=10000,
    lr=0.001,
    block_list=["a_g", "b_g", "m_g"],
    **kwargs
):
    """
    Fit the phases using SVI:
    Args:
    model: the model function
    guide: the guide function
    y: the data Nc x Ng
    init_phase: initial phases Nc x 1,
    params_g: a dictionary containing the trained gene coefficients
    n_steps: number of steps for training
    lr: learning rate
    block_list: list of parameters to fix

    Returns:
    The SVI object, containing the trained parameters, and much more
    """
    counts = kwargs.get("counts", None)
    model_ = numpyro.handlers.substitute(model, params_g)
    model_ = numpyro.handlers.block(model_, hide=block_list)
    # substitute the true phases
    model_ = numpyro.handlers.substitute(model_, {"phi_c": init_phase})

    optimizer = numpyro.optim.Adam(step_size=lr)
    svi = numpyro.infer.SVI(model_, guide, optimizer, loss=numpyro.infer.Trace_ELBO())

    # run the inference
    # svi_c = svi.run(jax.random.PRNGKey(0), n_steps, y, counts=counts)
    key = jax.random.PRNGKey(0)
    svi_c = svi.run(key, n_steps, y, counts=counts)

    return svi_c


def fit_gene_coeff_batcheffect(model, guide, y, mp, n_steps=10000, lr=0.001, **kwargs):
    """
    This function is similar to the older version but uses the dictionaries
    to bring in informations about the model and the data.

    Fit the gene coefficients using SVI:
    Args:
    model: the model function
    guide: the guide function
    y: the data Nc x Ng
    mp: a dictionary containing the design matrix, the counts, the phase, the genes
    n_steps: number of steps for training
    lr: learning rate
    guess: a dictionary containing the initial guess for the parameters

    Returns:
    The SVI object, containing the trained parameters, and much more
    """
    # preparing an educated guess for mean parameter

    block_cells_model = numpyro.handlers.substitute(model, {"phi_c": mp["ph"]})
    block_cells_model = numpyro.handlers.block(block_cells_model, hide=["phi_c"])

    block_cells_model = numpyro.handlers.substitute(
        block_cells_model, {"m_g": mp["m_g"]}
    )

    optimizer = numpyro.optim.Adam(step_size=lr)

    # define the inference algorithm
    svi = numpyro.infer.SVI(
        block_cells_model, guide, optimizer, loss=numpyro.infer.Trace_ELBO()
    )

    # # run the inference
    svi_g = svi.run(jax.random.PRNGKey(0), n_steps, y, mp)
    return svi_g
