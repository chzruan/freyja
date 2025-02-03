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


def pdf_vlos_1h_cs_func(
    vlos, 
    sigma_vir, 
    alpha_c=0.0, 
    alpha_s=1.0,
):
    r"""
    Gaussian velocity PDF of the 1-halo central-satellite galaxy pairs

    Parameters
    ----------
    vlos: np.ndarray
        numpy array with the PDF calculated in bins of :math:`v_{\mathrm{los}}`.
        Should have the same unit with ``sigma_vir``.
    
    sigma_vir: np.ndarray
        the velocity dispersion of the host haloes. 
        Should have the same unit with ``vlos``.

    alpha_c: float
        the velocity bias parameter of central galaxies

    alpha_s: float
        the velocity bias parameter of satellite galaxies

    Returns
    -------
    xi_l : np.array
        multipole of ``s_mu_tcpf_result`` of the indicated order.
    """



    vlos = vlos[..., np.newaxis]

    stddev = np.sqrt(alpha_c**2 + alpha_s**2) * sigma_vir
    return stats.norm.pdf(
        vlos,
        loc=0.0,
        scale=stddev,
    )


def pdf_vlos_1h_ss_func(vlos, sigma_vir, alpha_s=1.0,):
    vlos = vlos[..., np.newaxis]

    stddev = np.sqrt(2.0) * alpha_s * sigma_vir
    return stats.norm.pdf(
        vlos,
        loc=0.0,
        scale=stddev,
    )



class xi_1h_analy:
    def __init__(
        self, 
        log10M_bincentre: np.array,
        dlog10M: np.array,
        dndlog10M: np.array,
        redshift: float = 0.25,
        cosmology=FlatLambdaCDM(H0=70, Om0=0.3,),
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
            cosmology=FlatLambdaCDM(H0=70, Om0=0.3,),
            mdef="200m",
        ),
        mdef="vir",
        fft_num: int = 1,
        fft_logrmin: float = -5.0,
        fft_logrmax: float = 3.0,
        sigma_vir_halo=None,
        conc=None,
        cen_vel_bias=False,
        verbose=False,
    ):
        self.redshift = redshift
        self.cosmology = cosmology
        self.kms_to_Mpch = (1 + self.redshift) / (100 * self.cosmology.efunc(self.redshift))
        
        self.log10M_bincentre = log10M_bincentre
        self.log10M = self.log10M_bincentre
        self.dlog10M = dlog10M
        self.dndlog10M = dndlog10M
        self.log10M_binleftedge = self.log10M_bincentre - self.dlog10M/2.0

        self.central_occupation = central_occupation
        self.satellite_occupation = satellite_occupation
        self.HOD_params = HOD_params
        self.galaxy = HOD_params
        self.sat_profile = sat_profile
        self.cen_vel_bias = cen_vel_bias
        self.mdef = mdef
        self.conc = conc

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

        self.fft = FFTLog(
            fft_num, 
            log_r_min=fft_logrmin, 
            log_r_max=fft_logrmax, 
            kr=1.0,
        )

        if sigma_vir_halo is not None:
            self.sigma_vir_halo = sigma_vir_halo
        else:
            self.sigma_vir_halo = (
                halo_mass_to_virial_velocity(
                    10**self.log10M_bincentre, 
                    cosmology=self.cosmology, 
                    redshift=self.redshift, 
                    mdef=self.mdef,
                ) / (np.sqrt(2) * np.sqrt(1 + redshift))
            ) / np.sqrt(3) * self.kms_to_Mpch

        if verbose:
            print(f'basic information of the current HOD setting:')
            print(f'log10(n_gal) = {np.log10(self.n_g):.4f}')
            print(f'log10(n_cen) = {np.log10(self.n_c):.4f}')
            print(f'log10(n_sat) = {np.log10(self.n_s):.4f}')
            print(f'satellite fraction n_sat / n_gal = {self.f_s:.4f}\n')



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
        self.xiR_1h_cs = simps(
            self.dndlog10M * self.N_sat * u_sat_r_M,
            self.log10M_bincentre,
            axis=-1,
        ) / (self.n_cen * self.n_sat)

        self.xiR_1h_cs[
            np.where((self.xiR_1h_cs) < 1e-6*np.max(self.xiR_1h_cs))
        ] = 0.0

        # self.u_sat_k_M = self.sat_profile.fourier_mass_density(
        #     k=self.fft.k,
        #     mass=10**self.log10M_bincentre,
        #     cosmology=self.cosmology,
        #     redshift=self.redshift,
        #     conc=self.conc,
        # )
        # xiR_1h_cs_forinterp = simps(
        #     self.dndlog10M * self.N_sat * self.fft.pk2xi(self.u_sat_k_M).T,
        #     self.log10M_bincentre,
        #     axis=-1,
        # ) / (self.n_cen * self.n_sat)
        # self.xiR_1h_cs = ius(
        #     np.log10(self.fft.r),
        #     xiR_1h_cs_forinterp
        # )(np.log10(r_bincentre))


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
            conc=self.conc,
        )

        xiR_1h_ss_forinterp = simps(
            self.dndlog10M * self.N_sat**2 / self.N_cen * self.fft.pk2xi(self.u_sat_k_M**2).T,
            self.log10M_bincentre,
            axis=-1,
        ) / (self.n_sat**2)

        self.xiR_1h_ss = ius(
            np.log10(self.fft.r),
            xiR_1h_ss_forinterp,
            ext='zeros',
        )(np.log10(r_bincentre))
        return self.xiR_1h_ss



    def get_xiS_1h_cs(
        self,
        s_binedge,
        mu_binedge,
        eps=1e-6,
        nps=int(1000),
        r_parallel_max=80.0, # don't need large
        return_multipoles=True,
    ):
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
            alpha_c=self.HOD_params.v_bias_centrals,
            alpha_s=self.HOD_params.v_bias_satellites,
        )
        xiS_smu_left = simps(
            simps(
                (0.0 + u_sat_r_M) * pdf_vlos,
                r_parallel_integrand,
                axis=1,
            ) * self.dndlog10M * self.N_sat,
            self.log10M_bincentre,
            axis=-1,
        ).reshape((s_bincentre.shape[0], mu_bincentre.shape[0])) / (self.n_cen * self.n_sat)


        r_parallel_integrand = np.geomspace(eps, r_parallel_max, nps)
        # r_parallel_integrand = np.linspace(eps, r_parallel_max, nps)

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
            alpha_c=self.HOD_params.v_bias_centrals,
            alpha_s=self.HOD_params.v_bias_satellites,
        )
        xiS_smu_right = simps(
            simps(
                (0.0 + u_sat_r_M) * pdf_vlos,
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



    def get_xiS_1h_ss(
        self,
        s_binedge,
        mu_binedge,
        eps=1e-8,
        nps=int(1000),
        r_parallel_max=80.0, # don't need large
        return_multipoles=True,
    ):
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
        


        u_sat_k_M = self.sat_profile.fourier_mass_density(
            k=self.fft.k,
            mass=10**self.log10M_bincentre,
            cosmology=self.cosmology,
            redshift=self.redshift,
            conc=self.conc,
        )
        Akari = self.fft.pk2xi(u_sat_k_M**2).T



        # r_parallel_integrand = np.linspace(-r_parallel_max, -eps, nps)
        r_parallel_integrand = np.geomspace(eps, r_parallel_max, nps)
        r_parallel_integrand = -1.0 * r_parallel_integrand[::-1]

        rr = np.sqrt(r_perp**2 + r_parallel_integrand**2)
        Akari_integrand = np.zeros((
            rr.shape[0],
            rr.shape[1],
            len(self.log10M_bincentre),
        ))
        for idx_mass in range(len(self.log10M_bincentre)):
            Akari_integrand[:, :, idx_mass] = ius(
                np.log10(self.fft.r),
                Akari[:, idx_mass],
                ext='zeros',
            )(np.log10(rr))

        # vlos PDF
        vlos = (s_parallel - r_parallel_integrand) * np.sign(r_parallel_integrand)
        pdf_vlos = pdf_vlos_1h_ss_func(
            vlos, 
            self.sigma_vir_halo, 
            alpha_s=self.HOD_params.v_bias_satellites,
        )

        xiS_smu_left = simps(
            simps(
                Akari_integrand * pdf_vlos,
                r_parallel_integrand,
                axis=1,
            ) * self.dndlog10M * self.N_sat**2 / self.N_cen,
            self.log10M_bincentre,
            axis=-1,
        ).reshape((s_bincentre.shape[0], mu_bincentre.shape[0])) / (self.n_sat**2)



        r_parallel_integrand = np.geomspace(eps, r_parallel_max, nps)
        # r_parallel_integrand = np.linspace(eps, r_parallel_max, nps)

        rr = np.sqrt(r_perp**2 + r_parallel_integrand**2)
        Akari_integrand = np.zeros((
            rr.shape[0],
            rr.shape[1],
            len(self.log10M_bincentre),
        ))
        for idx_mass in range(len(self.log10M_bincentre)):
            Akari_integrand[:, :, idx_mass] = ius(
                np.log10(self.fft.r),
                Akari[:, idx_mass],
                ext='zeros',
            )(np.log10(rr))

        # vlos PDF
        vlos = (s_parallel - r_parallel_integrand) * np.sign(r_parallel_integrand)
        pdf_vlos = pdf_vlos_1h_ss_func(
            vlos, 
            self.sigma_vir_halo, 
            alpha_s=self.HOD_params.v_bias_satellites,
        )

        xiS_smu_right = simps(
            simps(
                Akari_integrand * pdf_vlos,
                r_parallel_integrand,
                axis=1,
            ) * self.dndlog10M * self.N_sat**2 / self.N_cen,
            self.log10M_bincentre,
            axis=-1,
        ).reshape((s_bincentre.shape[0], mu_bincentre.shape[0])) / (self.n_sat**2)


        self.xiS_1h_ss = xiS_smu_right + xiS_smu_left

        if return_multipoles:
            self.xiS0_1h_ss = tpcf_multipole(self.xiS_1h_ss, mu_binedge, order=0)
            self.xiS2_1h_ss = tpcf_multipole(self.xiS_1h_ss, mu_binedge, order=2)
            self.xiS4_1h_ss = tpcf_multipole(self.xiS_1h_ss, mu_binedge, order=4)
            return s_bincentre, self.xiS0_1h_ss, self.xiS2_1h_ss, self.xiS4_1h_ss

        return self.xiS_1h_ss





    def get_xi_S_1h_cs_fromsplit(
        self,
        s_binedge,
        mu_binedge,
        r_xiR1h,
        xiR1h,
        eps=1e-2,
        nps=int(1000),
        r_parallel_max=60.0, # don't need large
        return_multipoles=True,
    ):  
        xiR1h_func = ius(
            np.log10(r_xiR1h), 
            xiR1h, 
            ext='const',
        )

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
        pdf_vlos_diffM = pdf_vlos_1h_cs_func(
            vlos, 
            self.sigma_vir_halo, 
            alpha_c=self.HOD_params.v_bias_centrals,
            alpha_s=self.HOD_params.v_bias_satellites,
        )
        pdf_vlos = simps(
            pdf_vlos_diffM * self.dndlog10M * self.N_sat,
            self.log10M_bincentre,
            axis=-1,
        ) / simps(
            self.dndlog10M * self.N_sat,
            self.log10M_bincentre,
            axis=-1,
        )
        xiS_smu_left = simps(
            (xiR1h_func(np.log10(rr))) * pdf_vlos,
            r_parallel_integrand,
            axis=1,
        ).reshape((s_bincentre.shape[0], mu_bincentre.shape[0]))
        


        r_parallel_integrand = np.geomspace(eps, r_parallel_max, nps)
        # r_parallel_integrand = np.linspace(eps, r_parallel_max, nps)

        rr = np.sqrt(r_perp**2 + r_parallel_integrand**2)
        vlos = (s_parallel - r_parallel_integrand) * np.sign(r_parallel_integrand)
        pdf_vlos_diffM = pdf_vlos_1h_cs_func(
            vlos, 
            self.sigma_vir_halo, 
            alpha_c=self.HOD_params.v_bias_centrals,
            alpha_s=self.HOD_params.v_bias_satellites,
        )
        pdf_vlos = simps(
            pdf_vlos_diffM * self.dndlog10M * self.N_sat,
            self.log10M_bincentre,
            axis=-1,
        ) / simps(
            self.dndlog10M * self.N_sat,
            self.log10M_bincentre,
            axis=-1,
        )
        xiS_smu_right = simps(
            (xiR1h_func(np.log10(rr))) * pdf_vlos,
            r_parallel_integrand,
            axis=1,
        ).reshape((s_bincentre.shape[0], mu_bincentre.shape[0]))

        self.xiS_1h_cs = xiS_smu_right + xiS_smu_left - 1.0

        if return_multipoles:
            self.xiS0_1h_cs = tpcf_multipole(self.xiS_1h_cs, mu_binedge, order=0)
            self.xiS2_1h_cs = tpcf_multipole(self.xiS_1h_cs, mu_binedge, order=2)
            self.xiS4_1h_cs = tpcf_multipole(self.xiS_1h_cs, mu_binedge, order=4)
            return s_bincentre, self.xiS0_1h_cs, self.xiS2_1h_cs, self.xiS4_1h_cs

        return self.xiS_1h_cs
