import numpy as np
import sys

from scipy.integrate import simps
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy import stats

from hmd import Occupation, HaloModel
from hmd.occupation import Zheng07Centrals, Zheng07Sats
from hmd.galaxy import Galaxy
from hmd.profiles import FixedCosmologyNFW
from hmd.cosmology_abcs import Cosmology_abcs

# from mimicus.cosmology import JaxCosmology
from jax_fht.cosmology import FFTLog

from gsm.models.skewt.from_radial_transverse import moments2skewt
from gsm.streaming_integral import real2redshift
from scipy import stats, signal
from scipy.special import legendre


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


def pdf_vlos_1h_cs_func(
    vlos, 
    sigma_vir, 
    alpha_c=1.0, 
    alpha_s=1.0,
):
    """
    Parameters:
    vlos: 
    sigma_vir:

    Returns:

    Raises:
    """
    vlos = vlos[..., np.newaxis]

    stddev = np.sqrt(alpha_c**2 + alpha_s**2) * sigma_vir
    return stats.norm.pdf(
        vlos,
        loc=0.0,
        scale=stddev,
    )



def pdf_vlos_1h_ss_func(
    vlos, 
    sigma_vir, 
    alpha_s=1.0,
):
    """
    Parameters:
    vlos: in the unit of km/s
    sigma_vir: velocity dispersion, in the unit of km/s

    Returns:

    Raises:
    """
    vlos = vlos[..., np.newaxis]

    stddev = np.sqrt(2) * alpha_s * sigma_vir
    return stats.norm.pdf(
        vlos,
        loc=0.0,
        scale=stddev,
    )



class xigg_hm_hod:
    def __init__(
        self, 
        log10M_bincentre: np.array,
        dlog10M: np.array,
        dndlog10M: np.array,
        log10M_binleftedge: np.array,
        cHMF: np.array,
        redshift: float,
        dataflag: str = 'test',
        cosmology: Cosmology_abcs = Cosmology_abcs.from_cosidx(0),
        central_occupation: Occupation = Zheng07Centrals(),
        satellite_occupation: Occupation = Zheng07Sats(),
        HOD_params: Galaxy = Galaxy(
            logM_min = 13.62,
            sigma_logM = 0.6915,
            kappa = 0.51,
            logM1 = 14.42,
            alpha = 0.9168,
            M_cut = 10**12.26,
            M_sat = 10**14.87,
            concentration_bias = 1.0,
            v_bias_centrals = 0.1,
            v_bias_satellites = 1.0,
            B_cen = 0.0,
            B_sat = 0.0,
        ),
        sat_profile=FixedCosmologyNFW(
            redshift=0.25,
            cosmology=Cosmology_abcs.from_cosidx(0),
            mdef="200m",
        ),
        fft_num: int = 1,
        fft_logrmin: float = -5.0,
        fft_logrmax: float = 3.0,
        sigma_vir_halo = None,
        cen_vel_bias = False,
        verbose=False,
    ):
        self.redshift = redshift
        self.dataflag = dataflag
        self.cosmology = cosmology
        self.kms_to_Mpch = (1 + self.redshift) / (100 * self.cosmology.efunc(self.redshift))
        
        self.log10M_bincentre = log10M_bincentre
        self.log10M = self.log10M_bincentre
        self.dlog10M = dlog10M
        self.dndlog10M = dndlog10M
        self.log10M_binleftedge = log10M_binleftedge
        self.cHMF = cHMF

        self.central_occupation = central_occupation
        self.satellite_occupation = satellite_occupation
        self.HOD_params = HOD_params
        self.galaxy = HOD_params
        self.sat_profile = sat_profile
        self.cen_vel_bias = cen_vel_bias


        # halo occupation number N
        self.N_cen = self.central_occupation.get_n(
            10**self.log10M_bincentre, 
            galaxy=self.HOD_params,
        )
        self.N_sat = self.satellite_occupation.get_n(
            10**self.log10M_bincentre, 
            n_centrals=self.N_cen, 
            galaxy=self.HOD_params,
        )
        # galaxy number density n
        self.n_cen = simps(
            self.dndlog10M * self.N_cen,
            x=self.log10M_bincentre,
        )
        self.n_sat = simps(
            self.dndlog10M * self.N_sat,
            x=self.log10M_bincentre,
        )
        self.n_gal = self.n_cen + self.n_sat
        self.n_g = self.n_gal
        self.n_c = self.n_cen
        self.n_s = self.n_sat
        self.f_c = self.n_c / self.n_g
        self.f_s = self.n_s / self.n_g
        if verbose:
            print(f'basic information of the current HOD setting:')
            print(f'log10(n_gal) = {np.log10(self.n_g):.4f}')
            print(f'log10(n_cen) = {np.log10(self.n_c):.4f}')
            print(f'log10(n_sat) = {np.log10(self.n_s):.4f}')
            print(f'satellite fraction n_sat / n_gal = {self.f_s:.4f}\n')

        self.fft = FFTLog(
            fft_num, 
            log_r_min=fft_logrmin, 
            log_r_max=fft_logrmax, 
            kr=1.0,
        )

        self.sigma_vir_halo = sigma_vir_halo


    def get_xiR_1h_cs(
        self,
        r_bincentre,
    ):
        # shape of arrays: 
        # (len(r), len(mass))

        u_sat_r_M = self.sat_profile.u_r_M_profile(
            r=r_bincentre,
            mass=10**self.log10M_bincentre,
            redshift=self.redshift,
            cosmology=self.cosmology,
        )
        self.xiR_1h_cs_plus1 = simps(
            self.dndlog10M * self.N_sat * u_sat_r_M,
            self.log10M_bincentre,
            axis=-1,
        ) / (self.n_cen * self.n_sat)
        self.xiR_1h_cs = self.xiR_1h_cs_plus1 - 1.0

        return self.xiR_1h_cs


    def get_xiR_1h_ss(
        self,
        r_bincentre,
    ):
        # shape of arrays: 
        # (len(r), len(mass))

        self.u_sat_k_M = self.sat_profile.fourier_mass_density(
            k=self.fft.k,
            mass=10**self.log10M_bincentre,
            cosmology=self.cosmology,
            redshift=self.redshift,
        )

        xiR_1h_ss_forinterp = simps(
            self.dndlog10M * self.N_sat**2 / self.N_cen * self.fft.pk2xi(self.u_sat_k_M**2).T,
            self.log10M_bincentre,
            axis=-1,
        ) / (self.n_sat**2)

        self.xiR_1h_ss_plus1 = ius(
            np.log10(self.fft.r),
            xiR_1h_ss_forinterp,
            ext='zeros',
        )(np.log10(r_bincentre))

        self.xiR_1h_ss = self.xiR_1h_ss_plus1 - 1.0

        return self.xiR_1h_ss




    def get_xiS_1h_cs(
        self,
        s_binedge,
        mu_binedge=np.linspace(0, 1, 256),
        return_multipoles=False,
        eps=1e-4,
        nps=int(200),
        r_parallel_max=20.0, # for 1-halo terms, don't need large values
    ):
        # shape: (len(r_perp), len(r_parallel), len(M))

        s_bincentre = 0.5 * (s_binedge[1:] + s_binedge[:-1])
        mu_bincentre = 0.5 * (mu_binedge[1:] + mu_binedge[:-1])

        ss = s_bincentre.reshape(-1, 1)
        mu = mu_bincentre.reshape(1, -1)
        s_parallel = ss * mu 
        s_perp = ss * np.sqrt(1.0 - mu**2)
        s_parallel = s_parallel.reshape(-1, 1)
        s_perp = s_perp.reshape(-1, 1)
        r_perp = s_perp 
        

        r_parallel_integrand = np.linspace(-r_parallel_max, -eps, nps)

        rr = np.sqrt(r_perp**2 + r_parallel_integrand**2)
        u_sat_r_M = self.sat_profile.u_r_M_profile(
            r=rr,
            mass=10**self.log10M_bincentre,
            redshift=self.redshift,
            cosmology=self.cosmology,
        )
        vlos = (s_parallel - r_parallel_integrand) * np.sign(r_parallel_integrand)
        pdf_vlos = pdf_vlos_1h_cs_func(
            vlos, 
            self.sigma_vir_halo, 
            alpha_c=self.galaxy.v_bias_centrals,
            alpha_s=self.galaxy.v_bias_satellites,
        )
        xiS_smu_left = simps(
            simps(
                u_sat_r_M * pdf_vlos,
                r_parallel_integrand,
                axis=1,
            ) * self.dndlog10M * self.N_sat,
            self.log10M_bincentre,
            axis=-1,
        ).reshape((s_bincentre.shape[0], mu_bincentre.shape[0])) / (self.n_cen * self.n_sat)


        r_parallel_integrand = np.linspace(eps, r_parallel_max, nps)

        rr = np.sqrt(r_perp**2 + r_parallel_integrand**2)
        u_sat_r_M = self.sat_profile.u_r_M_profile(
            r=rr,
            mass=10**self.log10M_bincentre,
            redshift=self.redshift,
            cosmology=self.cosmology,
        )
        vlos = (s_parallel - r_parallel_integrand) * np.sign(r_parallel_integrand)
        pdf_vlos = pdf_vlos_1h_cs_func(
            vlos, 
            self.sigma_vir_halo, 
            alpha_c=self.galaxy.v_bias_centrals,
            alpha_s=self.galaxy.v_bias_centrals,
        )
        xiS_smu_right = simps(
            simps(
                u_sat_r_M * pdf_vlos,
                r_parallel_integrand,
                axis=1,
            ) * self.dndlog10M * self.N_sat,
            self.log10M_bincentre,
            axis=-1,
        ).reshape((s_bincentre.shape[0], mu_bincentre.shape[0])) / (self.n_cen * self.n_sat)

        self.xiS_1h_cs = np.nan_to_num(xiS_smu_right + xiS_smu_left)

        if return_multipoles:
            xiS0 = tpcf_multipole(self.xiS_1h_cs, mu_binedge, order=0)
            xiS2 = tpcf_multipole(self.xiS_1h_cs, mu_binedge, order=2)
            xiS4 = tpcf_multipole(self.xiS_1h_cs, mu_binedge, order=4)
            return s_bincentre, xiS0, xiS2, xiS4
        return self.xiS_1h_cs


