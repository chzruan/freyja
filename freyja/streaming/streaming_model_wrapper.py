import numpy as np
import sys
from scipy.interpolate import InterpolatedUnivariateSpline

from gsm.models.skewt.from_radial_transverse import moments2skewt
from gsm.models.skewt.from_radial_transverse_gcs import moments2skewt_cc

from gsm.models.gaussian.from_radial_transverse import moments2gaussian
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


class streaming_model:
    def __init__(self, model):
        self.model = model # ['gsm', 'stsm',]

    def __call__(
        self,
        r_velmom, 
        m10,
        c20, c02,
        c12, c30,
        c40, c04, c22,
        r_xiR, xiR,
        s_output_binedge,
        mu_output_binedge,
        kms_to_Mpch=1.0,
        return_multipoles=True,
        sigma_vir_M1=1.0,
        sigma_vir_M2=1.0,
        alpha_c=1.0,
        alpha_s=1.0,
    ):
        r'''
        Parameters
        ----------
        r_velmom : np.ndarray
            in the unit of Mpc/h

        m10 : np.ndarray
            in the unit of km/s

        c20 : np.ndarray
            in the unit of km/s

        c02 : np.ndarray
            in the unit of km/s

        c12 : np.ndarray
            in the unit of km/s

        c30 : np.ndarray
            in the unit of km/s

        c40 : np.ndarray
            in the unit of km/s

        c04 : np.ndarray
            in the unit of km/s

        c22 : np.ndarray
            in the unit of km/s

        Returns
        -------
        s_output_bincentre : np.array
        xiS0 : np.array
        xiS2 : np.array
        xiS4 : np.array

        '''


        s_output_bincentre = 0.5 * (s_output_binedge[1:] + s_output_binedge[:-1])
        mu_output_bincentre = 0.5 * (mu_output_binedge[1:] + mu_output_binedge[:-1])


        m10 *= kms_to_Mpch
        c20 *= kms_to_Mpch**2
        c02 *= kms_to_Mpch**2
        c12 *= kms_to_Mpch**3
        c30 *= kms_to_Mpch**3
        c40 *= kms_to_Mpch**4
        c04 *= kms_to_Mpch**4
        c22 *= kms_to_Mpch**4

        self.xiR_func = InterpolatedUnivariateSpline(
            r_xiR,
            xiR,
            ext=0,
        )

        r = r_velmom
        if (self.model == 'gsm'):
            self.vlos_pdf_func = moments2gaussian(
                m_10=InterpolatedUnivariateSpline(r, m10, ext=1),
                c_20=InterpolatedUnivariateSpline(r, c20, ext=1),
                c_02=InterpolatedUnivariateSpline(r, c02, ext=1),
            )
        elif (self.model == 'stsm'):
            self.vlos_pdf_func = moments2skewt(
                m_10=InterpolatedUnivariateSpline(r, m10, ext=1),
                c_20=InterpolatedUnivariateSpline(r, c20, ext=1),
                c_02=InterpolatedUnivariateSpline(r, c02, ext=1),
                c_12=InterpolatedUnivariateSpline(r, c12, ext=1),
                c_30=InterpolatedUnivariateSpline(r, c30, ext=1),
                c_22=InterpolatedUnivariateSpline(r, c22, ext=1),
                c_40=InterpolatedUnivariateSpline(r, c40, ext=1),
                c_04=InterpolatedUnivariateSpline(r, c04, ext=1)
            )
        else:
            raise ValueError('wrong model!')

        xiS_s_mu = real2redshift.simps_integrate(
            s_c=s_output_bincentre, 
            mu_c=mu_output_bincentre, 
            twopcf_function=self.xiR_func,
            los_pdf_function=self.vlos_pdf_func,
            limit = 120.0,
            epsilon = 0.01,
            n = 300,
        )

        if return_multipoles:
            xiS0 = tpcf_multipole(xiS_s_mu, mu_output_binedge, order=0)
            xiS2 = tpcf_multipole(xiS_s_mu, mu_output_binedge, order=2)
            xiS4 = tpcf_multipole(xiS_s_mu, mu_output_binedge, order=4)
            return s_output_bincentre, xiS0, xiS2, xiS4
        return xiS_s_mu

