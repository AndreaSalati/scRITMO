import numpy as np
import numpyro
import jax
import jax.numpy as jnp
from numpyro import distributions as dist


def model_MLE_NB(y, **kwargs):
    """
    Base model for MLE estimation. All the parameters are set
    thorugh the handlers in the numpyro_models_handles.py
    PARAMETERS:
    y: the data Nc x Ng, matrix of integer values
    counts: the counts Nc x 1, matrix of integer values, is passed as kwarg
    """
    Nc = y.shape[0]
    Ng = y.shape[1]

    counts = kwargs.get("counts", None)

    omega = 1.0
    # define gene parameters
    disp = numpyro.param("disp", 0.01, constraint=dist.constraints.positive)
    a_g = numpyro.param("a_g", jnp.zeros((1, Ng)))
    b_g = numpyro.param("b_g", jnp.ones((1, Ng)))
    m_g = numpyro.param("m_g", -2 * jnp.ones((1, Ng)))
    phi_c = numpyro.param("phi_c", jnp.zeros((Nc, 1)))

    E = a_g * jnp.cos(omega * phi_c) + b_g * jnp.sin(omega * phi_c) + m_g
    E = jnp.exp(E) * counts

    conc = 1 / disp

    numpyro.sample("obs", dist.NegativeBinomial2(E, conc), obs=y)


def model_MLE_NB_constraint_strong(y, **kwargs):
    """
    Here every gene is fixed to a phase, but can chose one common
    scaling factor for all the genes. (and the mean)
    """
    Nc = y.shape[0]
    Ng = y.shape[1]

    counts = kwargs.get("counts", None)
    omega = 1.0
    # define gene parameters
    disp = numpyro.param("disp", 0.01, constraint=dist.constraints.positive)
    # remember to fix both a_g and b_g with handles
    a_g = numpyro.param("a_g", jnp.zeros((1, Ng)))
    b_g = numpyro.param("b_g", jnp.ones((1, Ng)))
    m_g = numpyro.param("m_g", -2 * jnp.ones((1, Ng)))
    phi_c = numpyro.param("phi_c", jnp.zeros((Nc, 1)))
    kappa = numpyro.param("kappa", 1.0)

    E = (
        kappa * a_g * jnp.cos(omega * phi_c)
        + kappa * b_g * jnp.sin(omega * phi_c)
        + m_g
    )
    E = jnp.exp(E) * counts

    conc = 1 / disp
    obs = numpyro.sample("obs", dist.NegativeBinomial2(E, conc), obs=y)


# def model_MAP_NB(y, **kwargs):
#     """
#     In this model we just peform MAP,
#     """
#     Nc = y.shape[0]
#     Ng = y.shape[1]

#     counts = kwargs.get("counts", None)

#     omega = 1.0
#     # define gene parameters
#     disp = numpyro.param("disp", 0.01, constraint=dist.constraints.positive)
#     a_g = numpyro.param("a_g", jnp.zeros((1, Ng)))
#     b_g = numpyro.param("b_g", jnp.ones((1, Ng)))
#     m_g = numpyro.param("m_g", -2 * jnp.ones((1, Ng)))

#     phi = numpyro.param("phi_c", jnp.zeros((Nc, 1)))
#     E = a_g * jnp.cos(omega * phi) + b_g * jnp.sin(omega * phi) + m_g
#     E = jnp.exp(E) * counts

#     conc = 1 / disp
#     numpyro.sample("obs", dist.NegativeBinomial2(E, conc), obs=y)


def model_MLE_NB2(y, **kwargs):
    """
    Cell dependent dispersion parameter
    """
    Nc = y.shape[0]
    Ng = y.shape[1]

    counts = kwargs.get("counts", None)
    omega = 1.0
    # cell dependent dispersion parameter
    disp = numpyro.param(
        "disp", 0.01 * jnp.ones((Nc, 1)), constraint=dist.constraints.positive
    )
    a_g = numpyro.param("a_g", jnp.zeros((1, Ng)))
    b_g = numpyro.param("b_g", jnp.ones((1, Ng)))
    m_g = numpyro.param("m_g", -2 * jnp.ones((1, Ng)))
    phi = numpyro.param("phi_c", jnp.zeros((Nc, 1)))

    E = a_g * jnp.cos(omega * phi) + b_g * jnp.sin(omega * phi) + m_g
    E = jnp.exp(E) * counts

    conc = 1 / disp

    numpyro.sample("obs", dist.NegativeBinomial2(E, conc), obs=y)


def model_NB_ygs(y, phi, **kwargs):
    """
    This model basically performs the same as harmonic_regression functions
    but it uses a negative binomial likelihood, and takes into account which
    samples have more counts
    """

    N_ct = y.shape[0]
    N_g = y.shape[1]
    N_s = y.shape[2]

    counts = kwargs.get("counts", None)
    guess = kwargs.get("guess", None)

    omega = 1.0
    # define gene parameters
    disp = numpyro.param(
        "disp", 0.01 * np.ones((N_ct, 1, 1)), constraint=dist.constraints.positive
    )

    # condense all aprameters i one only
    m_g = (y / counts).mean(axis=-1)
    m_g = np.log2(m_g)
    res = np.zeros((N_ct, N_g, 3))
    res[:, :, 2] = m_g

    # in case we have a first estiamte we override
    if guess is not None:
        res = guess

    # actual parameter definition
    res = numpyro.param("res", res)

    E = (
        res[:, :, 0][:, :, None] * jnp.cos(omega * phi)
        + res[:, :, 1][:, :, None] * jnp.sin(omega * phi)
        + res[:, :, 2][:, :, None]
    )

    E = jnp.exp2(E) * counts
    conc = 1 / disp

    numpyro.sample("obs", dist.NegativeBinomial2(E, conc), obs=y)


def model_MLE_disp_g(y, **kwargs):
    """
    In this model we just peform MLE, we dont care about the phases dispersion
    we just overparametrize the model, and fit them all with MLE
    strongly raccomanded to initalize parametrs with reasonable guesses,
    and also to
    GENE DEPENDENT DISPERSION
    """
    Nc = y.shape[0]
    Ng = y.shape[1]

    counts = kwargs.get("counts", None)

    cell_plate = numpyro.plate("cells", size=Nc, dim=-2)
    gene_plate = numpyro.plate("genes", size=Ng, dim=-1)

    omega = 1.0
    # define gene parameters
    disp = numpyro.param(
        "disp", 0.01 * jnp.ones((1, Ng)), constraint=dist.constraints.positive
    )
    a_g = numpyro.param("a_g", jnp.zeros((1, Ng)))
    b_g = numpyro.param("b_g", jnp.ones((1, Ng)))
    m_g = numpyro.param("m_g", -2 * jnp.ones((1, Ng)))
    phi = numpyro.param("phi_c", jnp.zeros((Nc, 1)))

    E = a_g * jnp.cos(omega * phi) + b_g * jnp.sin(omega * phi) + m_g
    E = jnp.exp(E) * counts

    conc = 1 / disp

    with cell_plate:
        with gene_plate:
            numpyro.sample("obs", dist.NegativeBinomial2(E, conc), obs=y)


def model_MLE_NB_batcheffect(y, mp, **kwargs):
    """
    In this model we just peform MLE, using  NB distribution.
    We introduce batch effect as the mean of the gene expression
    for each gene in each batch/sample.
    """

    Nc = y.shape[0]
    Ng = y.shape[1]
    Ns = mp["dm"].shape[1]

    omega = 1.0
    # define gene parameters
    disp = numpyro.param("disp", 0.01, constraint=dist.constraints.positive)
    a_g = numpyro.param("a_g", jnp.zeros((1, Ng)))
    b_g = numpyro.param("b_g", jnp.ones((1, Ng)))
    m_g = numpyro.param("delta_s", jnp.zeros((Ns, Ng)))
    # m_g = numpyro.param("m_g", -2 * jnp.ones((1, Ng)))
    # delta_s = numpyro.param("delta_s", jnp.zeros((Ns, Ng)))

    phi = numpyro.param("phi_c", jnp.zeros((Nc, 1)))

    E = a_g * jnp.cos(omega * phi) + b_g * jnp.sin(omega * phi) + mp["dm"] @ m_g
    E = jnp.exp(E) * mp["counts"]
    print(E.shape)
    conc = 1 / disp

    numpyro.sample("obs", dist.NegativeBinomial2(E, conc), obs=y)


def guide_MLE_mp(y, mp, **kwargs):
    pass


def guide_MLE(y, **kwargs):
    pass


# training functions


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
    svi_g = svi.run(jax.random.PRNGKey(0), n_steps, y, counts=counts, dm=dm)
    return svi_g


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

    # if mp["guess"] is not None:
    #     a_g = guess[:, 0]
    #     b_g = guess[:, 1]
    #     m_g = guess[:, 2]
    #     block_cells_model = numpyro.handlers.substitute(
    #         block_cells_model, {"a_g": a_g, "b_g": b_g, "m_g": m_g}
    #     )

    optimizer = numpyro.optim.Adam(step_size=lr)

    # define the inference algorithm
    svi = numpyro.infer.SVI(
        block_cells_model, guide, optimizer, loss=numpyro.infer.Trace_ELBO()
    )

    # # run the inference
    svi_g = svi.run(jax.random.PRNGKey(0), n_steps, y, mp)
    return svi_g


def fit_phases(model, guide, y, init_phase, svi_g, n_steps=10000, lr=0.001, **kwargs):
    """
    Fit the phases using SVI:
    Args:
    model: the model function
    guide: the guide function
    y: the data Nc x Ng
    init_phase: initial phases Nc x 1,
    svi_g: a dictionary containing the trained gene coefficients
    n_steps: number of steps for training
    lr: learning rate

    Returns:
    The SVI object, containing the trained parameters, and much more
    """
    counts = kwargs.get("counts", None)
    # preparing an educated guess for mean parameter
    model_ = numpyro.handlers.substitute(model, svi_g)
    model_ = numpyro.handlers.block(model_, hide=["a_g", "b_g", "m_g"])
    # substitute the true phases
    model_ = numpyro.handlers.substitute(model_, {"phi_c": init_phase})

    optimizer = numpyro.optim.Adam(step_size=lr)
    svi = numpyro.infer.SVI(model_, guide, optimizer, loss=numpyro.infer.Trace_ELBO())

    # run the inference
    svi_c = svi.run(jax.random.PRNGKey(0), n_steps, y, counts=counts)
    return svi_c


#########################
# stuff Ido not use
#########################


def model_SVI_NB(y, **kwargs):
    """
    In this model SVI is performed. It uses priors and onviously needs a guide.
    It is not optimal as I do not use yet Vonmises distributions, but only gaussians

    """
    Nc = y.shape[0]
    Ng = y.shape[1]

    cell_plate = numpyro.plate("cells", size=Nc, dim=-2)
    gene_plate = numpyro.plate("genes", size=Ng, dim=-1)

    counts = kwargs.get("counts", None)
    mp = kwargs.get("mp", None)

    omega = 1
    disp = numpyro.param("disp", 0.01, constraint=dist.constraints.positive)
    # define gene parameters

    with gene_plate:
        a_g = numpyro.sample("a_g", dist.Normal(mp["a_g"], 1))
        b_g = numpyro.sample("b_g", dist.Normal(mp["b_g"], 1))
        m_g = numpyro.sample("m_g", dist.Normal(mp["m_g"], 1))

    with cell_plate:
        phi_c = numpyro.sample("phi_c", dist.Normal(mp["phi_c"], 1))

    # # define the model
    E = a_g * jnp.cos(omega * phi_c) + b_g * jnp.sin(omega * phi_c) + m_g
    E = jnp.exp(E) * counts

    conc = 1 / disp

    # define the likelihood
    with cell_plate:
        with gene_plate:
            numpyro.sample("obs", dist.NegativeBinomial2(E, conc), obs=y)


def guide_SVI_NB(y, **kwargs):

    Nc = y.shape[0]
    Ng = y.shape[1]

    mp = kwargs.get("mp", None)

    cell_plate = numpyro.plate("cells", size=Nc, dim=-2)
    gene_plate = numpyro.plate("genes", size=Ng, dim=-1)
    # define gene parameters
    a_g_loc = numpyro.param("a_g_loc", mp["a_g"])
    a_g_scale = numpyro.param(
        "a_g_scale", np.ones((1, Ng)), constraint=dist.constraints.positive
    )
    b_g_loc = numpyro.param("b_g_loc", mp["b_g"])
    b_g_scale = numpyro.param(
        "b_g_scale", np.ones((1, Ng)), constraint=dist.constraints.positive
    )
    m_g_loc = numpyro.param("m_g_loc", mp["m_g"])
    m_g_scale = numpyro.param(
        "m_g_scale", np.ones((1, Ng)), constraint=dist.constraints.positive
    )

    with gene_plate:
        a_g = numpyro.sample("a_g", dist.Normal(a_g_loc, a_g_scale))
        b_g = numpyro.sample("b_g", dist.Normal(b_g_loc, b_g_scale))
        m_g = numpyro.sample("m_g", dist.Normal(m_g_loc, m_g_scale))

    phi_c_loc = numpyro.param("phi_c_loc", mp["phi_c"])
    phi_c_scale = numpyro.param(
        "phi_c_scale", np.ones((Nc, 1)), constraint=dist.constraints.positive
    )

    with cell_plate:
        phi_c = numpyro.sample("phi_c", dist.Normal(phi_c_loc, phi_c_scale))


def model_MLE_NB_constraint(y, mp, **kwargs):
    """
    Here every gene is fixed to a phase, but can chose its own amplitude
    """
    Nc = y.shape[0]
    Ng = y.shape[1]

    counts = mp["counts"]
    # it's teh phase at which the gene peaks
    phi_g = mp["phi_g"]

    # cell_plate = numpyro.plate("cells", size=Nc, dim=-2)
    # gene_plate = numpyro.plate("genes", size=Ng, dim=-1)

    omega = 1.0
    # define gene parameters
    disp = numpyro.param("disp", 0.01, constraint=dist.constraints.positive)
    a_g = numpyro.param("a_g", jnp.zeros((1, Ng)))
    b_g = jnp.tan(phi_g) * a_g
    m_g = numpyro.param("m_g", mp["m_g"])

    phi_c = numpyro.param("phi_c", mp["phi_c"])

    #  print all shapes
    # print(f"a_g shape: {a_g.shape} b_g shape: {b_g.shape} m_g shape: {m_g.shape} phi shape: {phi_c.shape} counts shape: {counts.shape} phi_g shape: {phi_g.shape}")
    E = a_g * jnp.cos(omega * phi_c) + b_g * jnp.sin(omega * phi_c) + m_g
    E = jnp.exp(E) * counts

    conc = 1 / disp

    # with cell_plate:
    #     with gene_plate:
    obs = numpyro.sample("obs", dist.NegativeBinomial2(E, conc), obs=y)


# def model_MLE_NB_constraint2(y, mp, **kwargs):
#     """
#     Here every gene is fixed to a phase, but can chose its own amplitude
#     plus we take care of some numerical instability
#     """
#     Nc = y.shape[0]
#     Ng = y.shape[1]

#     omega = 1.0
#     # define gene parameters
#     disp = numpyro.param("disp", 0.01, constraint=dist.constraints.positive)

#     # initialize on the unit circle
#     a_g = numpyro.param("a_g", mp["a_g"])
#     # a_g = numpyro.param("a_g", jnp.ones((1, Ng)))
#     b_g = mp["slope"] * a_g
#     m_g = numpyro.param("m_g", mp["m_g"])
#     phi_c = numpyro.param("phi_c", mp["phi_c"])

#     E = a_g * jnp.cos(omega * phi_c[:, :]) + b_g * jnp.sin(omega * phi_c[:, :]) + m_g
#     # clip the E to avoid numerical instability
#     E = jnp.clip(E, -30, 30)
#     E = jnp.exp(E) * mp["counts"][:, :]

#     print(E.shape, y.shape)
#     conc = 1 / disp

#     obs = numpyro.sample("obs", dist.NegativeBinomial2(E, conc), obs=y[:, :])


def NB_pdf(mu, disp, x):
    """
    Negative binomial pdf, based on the numpyro implementation
    The distribution is already normalized, and gives values also
    for non integer values
    Parameters:
    mu: the mean
    disp: the dispersion
    x: the value for which we want to compute the pdf, flaot of vector
    """

    pdf = dist.NegativeBinomial2(mu, 1 / disp).log_prob(x)
    pdf = np.array(np.exp(pdf))

    return pdf
