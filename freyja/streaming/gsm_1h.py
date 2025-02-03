import numpy as np
import sys

from scipy.integrate import simps
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy import stats, signal
from scipy.special import legendre

from astropy.cosmology import FlatLambdaCDM, Planck15
from halotools.empirical_models import halo_mass_to_virial_velocity

from hmd import Occupation, HaloModel
from hmd.occupation import Zheng07Centrals, Zheng07Sats
from hmd.galaxy import Galaxy
from hmd.profiles import FixedCosmologyNFW
from jax_fht.cosmology import FFTLog




def tpcf_multipole(s_mu_tcpf_result, mu_bins, order=0):
    r"""
    copied from from halotools.mock_observables.tpcf_multipole
    https://halotools.readthedocs.io/en/latest/_modules/halotools/mock_observables/two_point_clustering/tpcf_multipole.html#tpcf_multipole

    Calculate the multipoles of the two point correlation function
    after first computing `~halotools.mock_observables.s_mu_tpcf`.

    Parameters
    ----------
    s_mu_tcpf_result : np.ndarray
        2-D array with the two point correlation function calculated in bins
        of :math:`s` and :math:`\mu`.  See `~halotools.mock_observables.s_mu_tpcf`.

    mu_bins : array_like
        array of :math:`\mu = \cos(\theta_{\rm LOS})`
        bins for which ``s_mu_tcpf_result`` has been calculated.
        Must be between [0,1].

    order : int, optional
        order of the multpole returned.

    Returns
    -------
    xi_l : np.array
        multipole of ``s_mu_tcpf_result`` of the indicated order.

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a
    periodic cube of length 250 Mpc/h.

    >>> Npts = 100
    >>> Lbox = 250.

    >>> x = np.random.uniform(0, Lbox, Npts)
    >>> y = np.random.uniform(0, Lbox, Npts)
    >>> z = np.random.uniform(0, Lbox, Npts)

    We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> sample1 = np.vstack((x,y,z)).T

    First, we calculate the correlation function using
    `~halotools.mock_observables.s_mu_tpcf`.

    >>> from halotools.mock_observables import s_mu_tpcf
    >>> s_bins  = np.linspace(0.01, 25, 10)
    >>> mu_bins = np.linspace(0, 1, 15)
    >>> xi_s_mu = s_mu_tpcf(sample1, s_bins, mu_bins, period=Lbox)

    Then, we can calculate the quadrapole of the correlation function:

    >>> xi_2 = tpcf_multipole(xi_s_mu, mu_bins, order=2)
    """

    # process inputs
    s_mu_tcpf_result = np.atleast_1d(s_mu_tcpf_result)
    mu_bins = np.atleast_1d(mu_bins)
    order = int(order)

    # calculate the center of each mu bin
    mu_bin_centers = (mu_bins[:-1]+mu_bins[1:])/(2.0)

    # get the Legendre polynomial of the desired order.
    Ln = legendre(order)

    # numerically integrate over mu
    result = (2.0*order + 1.0)/2.0 * np.sum(s_mu_tcpf_result * np.diff(mu_bins) *\
        (Ln(mu_bin_centers) + Ln(-1.0*mu_bin_centers)), axis=1)

    return result


def pdf_vlos_1h_cs_func(vlos, sigma_vir, alpha_c=0.0, alpha_s=1.0,):
    vlos = vlos[..., np.newaxis]

    stddev = np.sqrt(alpha_c**2 + alpha_s**2) * sigma_vir
    return stats.norm.pdf(
        vlos,
        loc=0.0,
        scale=stddev,
    )


def pdf_vlos_1h_ss_func(vlos, sigma_vir, alpha_s=1.0,):
    vlos = vlos[..., np.newaxis]

    stddev = np.sqrt(2) * alpha_s * sigma_vir
    return stats.norm.pdf(
        vlos,
        loc=0.0,
        scale=stddev,
    )



def get_xi_S_1h_cs(
    s_binedge,
    mu_binedge,
    r_xiR1h,
    xiR1h,
    eps=1e-6,
    nps=int(1000),
    r_parallel_max=80.0, # don't need large
    return_multipoles=True,
):  
    log10xiR1h_func = ius(np.log10(r_xiR1h), np.log10(xiR1h), ext='const')

    s_bincentre = 0.5 * (s_binedge[1:] + s_binedge[:-1])
    mu_bincentre = 0.5 * (mu_binedge[1:] + mu_binedge[:-1])

    # shape: (len(r_perp), len(r_parallel), len(M))

    ss = s_bincentre.reshape(-1, 1)
    mu = mu_bincentre.reshape(1, -1)
    s_parallel = ss * mu 
    s_perp = ss * np.sqrt(1.0 - mu**2)
    s_parallel = s_parallel.reshape(-1, 1)
    s_perp = s_perp.reshape(-1, 1)
    r_perp = s_perp 
    

    # r_parallel_integrand = np.linspace(-r_parallel_max, -eps, nps)
    r_parallel_integrand = np.geomspace(eps, r_parallel_max, nps)
    r_parallel_integrand = -1.0 * r_parallel_integrand[::-1]

    rr = np.sqrt(r_perp**2 + r_parallel_integrand**2)
    vlos = (s_parallel - r_parallel_integrand) * np.sign(r_parallel_integrand)
    pdf_vlos = pdf_vlos_1h_cs_func(
        vlos, 
        self.sigma_vir_halo, 
        alpha_c=self.HOD_params.v_bias_centrals,
        alpha_s=self.HOD_params.v_bias_satellites,
    )
    xiS_smu_left = simps(
        simps(
            (1.0 + 10**log10xiR1h_func(10**rr)) * pdf_vlos,
            r_parallel_integrand,
            axis=1,
        ) * self.dndlog10M * self.N_sat,
        self.log10M_bincentre,
        axis=-1,
    ).reshape((s_bincentre.shape[0], mu_bincentre.shape[0])) / (self.n_cen * self.n_sat)


    r_parallel_integrand = np.geomspace(eps, r_parallel_max, nps)
    # r_parallel_integrand = np.linspace(eps, r_parallel_max, nps)

    rr = np.sqrt(r_perp**2 + r_parallel_integrand**2)
    vlos = (s_parallel - r_parallel_integrand) * np.sign(r_parallel_integrand)
    pdf_vlos = pdf_vlos_1h_cs_func(
        vlos, 
        self.sigma_vir_halo, 
        alpha_c=self.HOD_params.v_bias_centrals,
        alpha_s=self.HOD_params.v_bias_satellites,
    )
    xiS_smu_right = simps(
        simps(
            (1.0 + 10**log10xiR1h_func(10**rr)) * pdf_vlos,
            r_parallel_integrand,
            axis=1,
        ) * self.dndlog10M * self.N_sat,
        self.log10M_bincentre,
        axis=-1,
    ).reshape((s_bincentre.shape[0], mu_bincentre.shape[0])) / (self.n_cen * self.n_sat)

    self.xiS_1h_cs = xiS_smu_right + xiS_smu_left

    if return_multipoles:
        self.xiS0_1h_cs = tpcf_multipole(self.xiS_1h_cs, mu_binedge, order=0)
        self.xiS2_1h_cs = tpcf_multipole(self.xiS_1h_cs, mu_binedge, order=2)
        self.xiS4_1h_cs = tpcf_multipole(self.xiS_1h_cs, mu_binedge, order=4)
        return s_bincentre, self.xiS0_1h_cs, self.xiS2_1h_cs, self.xiS4_1h_cs

    return self.xiS_1h_cs


