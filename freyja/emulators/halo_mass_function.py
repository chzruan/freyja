from __future__ import annotations

from typing import Callable, Tuple, Dict

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from tinygp import GaussianProcess, kernels
from tinygp.helpers import dataclass

from hmf import MassFunction
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import curve_fit


@dataclass
class RBFKernel(kernels.Kernel):
    """
    Radial Basis Function (RBF) kernel with anisotropic length-scales.

    This kernel implements the squared exponential covariance:
    .. math::
        k(x_1, x_2) = A \exp\left(-0.5 \sum_i \frac{(x_{1,i} - x_{2,i})^2}{\ell_i^2}\right)

    Attributes
    ----------
    log_amp : jnp.ndarray
        Logarithm of the kernel amplitude (variance).
    log_scale : jnp.ndarray
        Logarithm of the length-scales for each input dimension.
    """

    log_amp: jnp.ndarray
    log_scale: jnp.ndarray

    def evaluate(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the kernel between two vectors."""
        amp = jnp.exp(self.log_amp)
        ell = jnp.exp(self.log_scale)
        r = (X1 - X2) / ell
        r2 = jnp.dot(r, r)
        return amp * jnp.exp(-0.5 * r2)

    def evaluate_diag(self, X: jnp.ndarray) -> jnp.ndarray:
        """Return the diagonal of the kernel matrix."""
        return jnp.exp(self.log_amp)


def build_gp(params: Dict[str, jnp.ndarray], X: jnp.ndarray) -> GaussianProcess:
    """
    Construct a tinygp Gaussian Process object.

    Parameters
    ----------
    params : dict
        Dictionary containing 'log_amp' and 'log_scale'.
    X : jnp.ndarray
        The training input data.

    Returns
    -------
    tinygp.GaussianProcess
        The initialized GP model.
    """
    kernel = RBFKernel(
        log_amp=params["log_amp"],
        log_scale=params["log_scale"],
    )
    return GaussianProcess(kernel, X, diag=1e-5)


def halo_mass_function_analytical(
    z: float = 0.25,
    Om0: float = 0.3,
    Ob0: float = 0.02,
    H0: float = 70.0,
    sigma8: float = 0.81,
    hmf_model: str = "ST",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the analytical Halo Mass Function using the hmf library.

    Parameters
    ----------
    z : float
        Redshift.
    Om0 : float, optional
        Omega Matter at z=0.
    Ob0 : float, optional
        Omega Baryon at z=0. Fixed by default for Durham wide-sample emulators.
    H0 : float, optional
        Hubble constant in km/s/Mpc.
    sigma8 : float, optional
        Root-mean-square fluctuations in spheres of 8 Mpc/h.
    hmf_model : str, optional
        The HMF fitting function (default "ST" for Sheth-Tormen).

    Returns
    -------
    m : np.ndarray
        Mass grid in Msun/h.
    dndlnm : np.ndarray
        Differential mass function dn/dlnM.
    ngtm : np.ndarray
        Cumulative mass function n(>M).
    """
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0, Tcmb0=2.7255)
    mf = MassFunction(
        z=z,
        cosmo_model=cosmo,
        sigma_8=sigma8,
        hmf_model=hmf_model,
    )
    return mf.m, mf.dndlnm, mf.ngtm


def schechter_log_form(
    M: np.ndarray, phi_star: float, M1: float, M2: float, alpha: float, beta: float
) -> np.ndarray:
    """
    Modified Schechter function for dn/dlog10M or high-mass tail fitting.

    Parameters
    ----------
    M : np.ndarray
        Halo mass in Msun/h.
    phi_star : float
        Normalization.
    M1, M2 : float
        Characteristic mass scales.
    alpha : float
        Low-mass power law slope.
    beta : float
        High-mass exponential cutoff steepness.

    Returns
    -------
    np.ndarray
        The function value at M.
    """
    return phi_star * (M / M1) ** (-alpha) * np.exp(-((M / M2) ** beta))


class HMFEmulator:
    """
    Gaussian Process emulator for the Halo Mass Function.

    This class emulates the ratio between simulation-measured cHMF and
    analytical predictions. It utilizes a high-mass Schechter extrapolation
    to extend beyond the training limits of the GP.

    The emulator is specifically designed for flat LCDM cosmologies as part
    of the Freyja project.
    """

    def __init__(
        self,
        gp_emulator_path: str,
        z: float,
    ) -> None:
        """
        Initialize the emulator by loading trained GP weights.

        Parameters
        ----------
        gp_emulator_path : str
            Path to the .npz file containing GP parameters and training data.
        z : float
            Operating redshift.
        """
        self.gp_bundle = np.load(gp_emulator_path, allow_pickle=True)
        self.gp_params = {
            "log_amp": jnp.array(self.gp_bundle["log_amp"]),
            "log_scale": jnp.array(self.gp_bundle["log_scale"]),
        }
        self.X_train = jnp.array(self.gp_bundle["X_train"])
        self.Y_train = jnp.array(self.gp_bundle["Y_train"])
        self.gp = build_gp(self.gp_params, self.X_train)
        self.z = z

    def _build_input(
        self,
        cosmo_params: np.ndarray,
        log10M_binleftedges: np.ndarray,
    ) -> jnp.ndarray:
        """Construct the 5D input array for the GP [Om0, h, S8, ns, log10M]."""
        cosmo_params = np.asarray(cosmo_params)
        log10M_binleftedges = np.asarray(log10M_binleftedges)
        cosmo_block = np.repeat(cosmo_params[None, :], len(log10M_binleftedges), axis=0)
        mass_block = log10M_binleftedges.reshape(-1, 1)
        return jnp.array(np.hstack([cosmo_block, mass_block]))

    def _predict_ratio(
        self, cosmo_params: np.ndarray, log10M_binleftedges: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the cHMF ratio and uncertainty from the GP."""
        X_test = self._build_input(cosmo_params, log10M_binleftedges)
        cond = self.gp.condition(self.Y_train, X_test)
        return np.asarray(cond.gp.mean), np.sqrt(np.asarray(cond.gp.variance))

    def _analytic_cHMF(self, cosmo_params: np.ndarray) -> InterpolatedUnivariateSpline:
        """Compute the Sheth-Tormen cumulative HMF and return a spline interpolator."""
        Om0, h, S8, ns = cosmo_params
        sigma8 = S8 / np.sqrt(Om0 / 0.3)
        m_ana, _, cHMF_analytic = halo_mass_function_analytical(
            z=self.z, Om0=Om0, H0=h * 100.0, sigma8=sigma8
        )
        return InterpolatedUnivariateSpline(
            np.log10(m_ana), np.log10(cHMF_analytic), k=3
        )

    def _schechter_fit(
        self,
        log10M_input,
        cHMF_input,
        cHMF_err_input,
        log10M_extend,
        n_halo_threshold: float = 100.0 / 1024.0**3,
    ):
        """Fit high-mass tail to Schechter form and return extrapolated values."""
        mask_fit = cHMF_input < 2000 * n_halo_threshold
        p0 = [1e-4, 1e14, 1e14, 1.0, 1.0]
        popt, _ = curve_fit(
            schechter_log_form,
            10 ** log10M_input[mask_fit],
            cHMF_input[mask_fit],
            p0=p0,
            sigma=cHMF_err_input[mask_fit],
            absolute_sigma=True,
        )
        self.schechter_params = popt
        return schechter_log_form(10**log10M_extend, *popt)

    def cumulative_hmf(
        self, cosmo_params: np.ndarray, log10M_binleftedges: np.ndarray
    ) -> np.ndarray:
        """
        Compute the emulated cumulative HMF n(>M).

        Parameters
        ----------
        cosmo_params : array_like
            [Om0, h, S8, ns].
        log10M_binleftedges : array_like
            Logarithm of mass bin edges.

        Returns
        -------
        cHMF_total : np.ndarray
            Reconstructed cumulative HMF including Schechter extension.
        """
        log10M_binleftedges = np.asarray(log10M_binleftedges)
        mask = log10M_binleftedges <= 14.8

        ratio_mean, ratio_std = self._predict_ratio(
            cosmo_params, log10M_binleftedges[mask]
        )
        spline = self._analytic_cHMF(cosmo_params)

        cHMF_analytic_gp = 10.0 ** spline(log10M_binleftedges[mask])
        cHMF_emu_gp = ratio_mean * cHMF_analytic_gp
        cHMF_err_emu_gp = ratio_std * cHMF_analytic_gp

        if np.any(~mask):
            cHMF_extend = self._schechter_fit(
                log10M_binleftedges[mask],
                cHMF_emu_gp,
                cHMF_err_emu_gp,
                log10M_binleftedges[~mask],
            )
            cHMF_total = np.concatenate([cHMF_emu_gp, cHMF_extend])
        else:
            cHMF_total = cHMF_emu_gp

        return cHMF_total

    def dndlog10M(
        self,
        cosmo_params: np.ndarray,
        log10M_bincentres: np.ndarray,
        dlog10M: float = 0.10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute differential HMF dn/dlog10M via numerical differentiation.

        Parameters
        ----------
        cosmo_params : array_like
            [Om0, h, S8, ns].
        log10M_bincentres : np.ndarray
            Central mass values for prediction.
        dlog10M : float, optional
            Step size in log10 mass.

        Returns
        -------
        dndlog10M : np.ndarray
            Differential mass function.
        """
        log10M_bincentres = np.asarray(log10M_bincentres)
        _lmble = log10M_bincentres - 0.5 * dlog10M
        _lmble = np.append(_lmble, log10M_bincentres[-1] + 0.5 * dlog10M)

        cHMF_emu = self.cumulative_hmf(cosmo_params, _lmble)
        return -np.diff(cHMF_emu) / dlog10M

    def differential_hmf(
        self,
        cosmo_params: np.ndarray,
        log10M_bincentres: np.ndarray,
        dlog10M: float = 0.10,
    ) -> np.ndarray:
        """Wrapper for dndlog10M."""
        val = self.dndlog10M(cosmo_params, log10M_bincentres, dlog10M)
        return val
