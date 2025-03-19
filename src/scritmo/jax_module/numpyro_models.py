import numpy as np
import numpyro
import jax
import jax.numpy as jnp
from numpyro import distributions as dist
from numpyro import handlers
from numpyro.infer.reparam import Reparam
from numpyro.infer.reparam import CircularReparam


def model_MLE_NB(y, **kwargs):
    """
    Base model for MLE estimation. All the parameters are set
    thorugh the handlers in the numpyro_models_handles.py

    PARAMETERS:
    y: the data Nc x Ng, matrix of integer values
    counts: the counts Nc x 1, matrix of integer values, is passed as kwarg
    """

    counts = kwargs.get("counts", None)

    Nc = counts.shape[0]
    # here the number of genes is random, always use handelrs to set the parameters
    Ng = 12

    omega = 1.0
    # define gene parameters
    disp = numpyro.param(
        "disp", 0.1 * jnp.ones((1, Ng)), constraint=dist.constraints.positive
    )
    a_g = numpyro.param("a_g", jnp.zeros((1, Ng)))
    b_g = numpyro.param("b_g", jnp.ones((1, Ng)))
    m_g = numpyro.param("m_g", -2 * jnp.ones((1, Ng)))
    phi_c = numpyro.param("phi_c", jnp.zeros((Nc, 1)))

    E = a_g * jnp.cos(omega * phi_c) + b_g * jnp.sin(omega * phi_c) + m_g
    E = jnp.exp(E) * counts

    conc = 1 / disp
    numpyro.sample("obs", dist.NegativeBinomial2(E, conc), obs=y)


def model_MLE_G(y, **kwargs):
    """
    Base model for MLE estimation. All the parameters are set
    thorugh the handlers in the numpyro_models_handles.py

    PARAMETERS:
    y: the data Nc x Ng, matrix of log-transformed values
    """
    Nc = y.shape[0]
    Ng = y.shape[1]

    omega = 1.0
    # define gene parameters
    a_g = numpyro.param("a_g", jnp.zeros((1, Ng)))
    b_g = numpyro.param("b_g", jnp.ones((1, Ng)))
    m_g = numpyro.param("m_g", -2 * jnp.ones((1, Ng)))
    phi_c = numpyro.param("phi_c", jnp.zeros((Nc, 1)))

    E = a_g * jnp.cos(omega * phi_c) + b_g * jnp.sin(omega * phi_c) + m_g

    numpyro.sample("obs", dist.Normal(E, 1), obs=y)


def model_null(y, **kwargs):
    """
    Model that fits every gene with a constant value.
    Its used to compute pvalues for the genes

    PARAMETERS:
    y: the data Nc x Ng, matrix of integer values
    counts: the counts Nc x 1, matrix of integer values, is passed as kwarg
    """
    Ng = y.shape[1]

    counts = kwargs.get("counts", None)

    # define gene parameters
    disp = numpyro.param(
        "disp", 0.01 * jnp.ones((1, Ng)), constraint=dist.constraints.positive
    )
    m_g = numpyro.param("m_g", -2 * jnp.ones((1, Ng)))

    E = jnp.exp(m_g) * counts
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


def model_MCMC_NB(y, priors, **kwargs):
    """
    Base model for MCMC estimation. All the parameters are now defined as samples
    with priors to enable MCMC sampling.

    PARAMETERS:
    y: the data Nc x Ng, matrix of integer values
    counts: the counts Nc x 1, matrix of integer values, passed as kwarg
    """
    Nc = y.shape[0]
    Ng = y.shape[1]

    counts = kwargs.get("counts", None)
    omega = 1.0

    # Define priors for gene parameters
    disp = numpyro.sample(
        "disp", dist.Gamma(priors["disp"], 1.0)  # Positive prior for dispersion
    )
    a_g = numpyro.sample("a_g", dist.Normal(0, 1))
    b_g = numpyro.sample("b_g", dist.Normal(1, 1))
    m_g = numpyro.sample("m_g", dist.Normal(-2, 1))
    phi_c = numpyro.sample("phi_c", dist.Normal(0, 1))

    # Compute expected values
    E = a_g * jnp.cos(omega * phi_c) + b_g * jnp.sin(omega * phi_c) + m_g
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


#########################
# stuff I do not use
#########################


def model_SVI(y, **kwargs):
    """
    In this  SVI is performed. It uses priors and onviously needs a guide.
    It is not optmodelimal as I do not use yet Vonmises distributions, but only gaussians
    """

    counts = kwargs.get("counts", None)
    mp = kwargs.get("mp", None)
    # phase_mode = kwargs.get("phase_mode", "param")

    Nc, Ng = y.shape

    omega = 1
    disp = numpyro.param("disp", 0.1, constraint=dist.constraints.positive)

    a_g = numpyro.sample("a_g", dist.Normal(mp["a_g"], mp["std_ab"]))
    b_g = numpyro.sample("b_g", dist.Normal(mp["b_g"], mp["std_ab"]))
    m_g = numpyro.sample("m_g", dist.Normal(mp["m_g"], mp["std_m"]))

    phi_xy = numpyro.sample("phi_xy", dist.Normal(np.zeros((Nc, 2)), 1.0))
    cos_sin = phi_xy / jnp.linalg.norm(phi_xy, axis=1)[:, None]

    # define the model
    # E = a_g * jnp.cos(omega * cos_sin[:, 0]) + b_g * jnp.sin(omega * cos_sin[:, 1]) + m_g
    E = a_g * cos_sin[:, 0][:, jnp.newaxis] + b_g * cos_sin[:, 1][:, jnp.newaxis] + m_g
    E = jnp.exp(E) * counts

    conc = 1 / disp

    # define the likelihood
    numpyro.sample("obs", dist.NegativeBinomial2(E, conc), obs=y)


def guide_SVI(y, **kwargs):

    Nc = y.shape[0]
    Ng = y.shape[1]

    mp = kwargs.get("mp", None)
    # phase_mode = kwargs.get("phase_mode", "param")

    a_g_loc = numpyro.param("a_g_loc", mp["a_g"])
    b_g_loc = numpyro.param("b_g_loc", mp["b_g"])
    m_g_loc = numpyro.param("m_g_loc", mp["m_g"])

    # scales
    a_g_scale = numpyro.param(
        "a_g_scale", np.ones((1, Ng)) * 0.5, constraint=dist.constraints.positive
    )
    b_g_scale = numpyro.param(
        "b_g_scale", np.ones((1, Ng)) * 0.5, constraint=dist.constraints.positive
    )
    m_g_scale = numpyro.param(
        "m_g_scale", np.ones((1, Ng)) * 0.5, constraint=dist.constraints.positive
    )

    phi_xy_loc = numpyro.param("phi_xy_loc", np.zeros((Nc, 2)))
    phi_xy_scale = numpyro.param(
        "phi_xy_scale", np.ones((Nc, 2)), constraint=dist.constraints.positive
    )
    phi_xy = numpyro.sample("phi_xy", dist.Normal(phi_xy_loc, phi_xy_scale))

    # hard assignment by using Delta distributions
    a_g = numpyro.sample("a_g", dist.Normal(a_g_loc, a_g_scale))
    b_g = numpyro.sample("b_g", dist.Normal(b_g_loc, b_g_scale))
    m_g = numpyro.sample("m_g", dist.Normal(m_g_loc, m_g_scale))


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
