import numpy as np


def bin_phi(phi, n_bins=24):
    """
    Assigns phases to bins,

    returns:
    - a vector with mebership to specfic bin
    - bin fill
    """
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    phi_binned = np.digitize(phi, bin_edges) - 1
    phi_binned[phi_binned == n_bins] = 0
    bin_fill = np.zeros(n_bins)
    for i in range(n_bins):
        bin_fill[i] = np.sum(phi_binned == i)
    bin_fill = bin_fill / bin_fill.sum()
    return phi_binned, bin_fill


def bin_z(z):
    n_cell = z.shape[0]
    cat_u = np.unique(z)

    n_cat = cat_u.shape[0]
    pdf_z = np.zeros(n_cat)
    z_binned = np.zeros_like(z)

    for i, cat in enumerate(cat_u):
        z_binned[z == cat] = i
        pdf_z[i] = (z == cat).sum() / n_cell
    return z_binned, pdf_z


def joint_pdf(phi_bins, z_bins):
    """
    Computes the joint pdf of phi and z
    """
    n_phi = phi_bins.max() + 1
    n_z = z_bins.max() + 1
    joint_pdf = np.zeros((n_phi, n_z))

    for i in range(n_phi):
        for j in range(n_z):
            joint_pdf[i, j] = np.sum((phi_bins == i) & (z_bins == j))

    joint_pdf /= joint_pdf.sum()
    return joint_pdf


def MI(phi, z, n_bins_phi=24):
    """
    Implementation of the MI for the case where vector phi is
    a vector of phases on the unit circle and z is a categorical varibale
    (the celltype). In this case binning is rather easy because both domains
    are compact?
    """
    phi_binned, phi_pdf = bin_phi(phi, n_bins=n_bins_phi)
    z_binned, z_pdf = bin_z(z)

    jpdf = joint_pdf(phi_binned, z_binned)
    # now ge the multiplication of the marginal
    marg_mult_pdf = phi_pdf[:, None] * z_pdf[None, :]

    # now get the MI
    # MI = sum(p(x,y) * log(p(x,y) / p(x)p(y)))
    MI = 0
    for i in range(jpdf.shape[0]):
        for j in range(jpdf.shape[1]):
            if jpdf[i, j] > 0:
                MI += jpdf[i, j] * np.log(jpdf[i, j] / marg_mult_pdf[i, j])
    return MI
