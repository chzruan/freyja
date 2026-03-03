from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from tinygp import GaussianProcess, kernels
from tinygp.helpers import dataclass

from hmf import MassFunction
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import curve_fit

# --- 1. Define Default Path ---
MODULE_DIR = Path(__file__).parent
DEFAULT_CKPT_PATH = MODULE_DIR / "checkpoints" / "gp_cHMFratio_LCDM_wide64_z0.25.npz"


@dataclass
class RBFKernel(kernels.Kernel):
    r"""
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

    _GP_MAX_LOG10M = 14.7
    _SCHECHTER_P0 = (1e-4, 1e14, 1e14, 1.0, 1.0)
    _SCHECHTER_BOUNDS = (
        (1e-30, 1e8, 1e8, -20.0, 1e-6),
        (np.inf, np.inf, np.inf, 20.0, 20.0),
    )
    _MIN_SIGMA = 1e-30

    def __init__(
        self,
        gp_emulator_path: Path = DEFAULT_CKPT_PATH,
        z: float = 0.25,
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
        self._analytic_chmf_spline_cache: Dict[
            Tuple[float, float, float, float], InterpolatedUnivariateSpline
        ] = {}

    @staticmethod
    def _normalize_cosmo_params(cosmo_params: np.ndarray) -> np.ndarray:
        """Return cosmology parameters as a 1D float array [Om0, h, S8, ns]."""
        arr = np.asarray(cosmo_params, dtype=float).reshape(-1)
        if arr.size != 4:
            raise ValueError(
                "cosmo_params must contain exactly 4 values: [Om0, h, S8, ns]"
            )
        return arr

    def _build_input(
        self,
        cosmo_params: np.ndarray,
        log10M_binleftedges: np.ndarray,
    ) -> jnp.ndarray:
        """Construct the 5D input array for the GP [Om0, h, S8, ns, log10M]."""
        cosmo = self._normalize_cosmo_params(cosmo_params)
        masses = np.asarray(log10M_binleftedges, dtype=float).reshape(-1)

        X = np.empty((masses.size, 5), dtype=float)
        X[:, :4] = cosmo
        X[:, 4] = masses
        return jnp.asarray(X)

    def _predict_ratio(
        self, cosmo_params: np.ndarray, log10M_binleftedges: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the cHMF ratio and uncertainty from the GP."""
        X_test = self._build_input(cosmo_params, log10M_binleftedges)
        cond = self.gp.condition(self.Y_train, X_test)
        variance = np.asarray(cond.gp.variance)
        return np.asarray(cond.gp.mean), np.sqrt(np.maximum(variance, 0.0))

    def _analytic_cHMF(self, cosmo_params: np.ndarray) -> InterpolatedUnivariateSpline:
        """Compute the Sheth-Tormen cumulative HMF and return a spline interpolator."""
        cosmo = self._normalize_cosmo_params(cosmo_params)
        key = tuple(float(x) for x in cosmo)
        cached = self._analytic_chmf_spline_cache.get(key)
        if cached is not None:
            return cached

        Om0, h, S8, ns = cosmo
        sigma8 = S8 / np.sqrt(Om0 / 0.3)
        m_ana, _, cHMF_analytic = halo_mass_function_analytical(
            z=self.z, Om0=Om0, H0=h * 100.0, sigma8=sigma8
        )
        spline = InterpolatedUnivariateSpline(
            np.log10(m_ana), np.log10(cHMF_analytic), k=3
        )
        self._analytic_chmf_spline_cache[key] = spline
        return spline

    def _schechter_fit(
        self,
        log10M_input: np.ndarray,
        cHMF_input: np.ndarray,
        cHMF_err_input: np.ndarray,
        log10M_extend: np.ndarray,
        n_halo_threshold: float = 100.0 / 1024.0**3,
    ) -> np.ndarray:
        """Fit high-mass tail to Schechter form and return extrapolated values."""
        mask_fit = cHMF_input < 2000 * n_halo_threshold
        sigma = np.maximum(cHMF_err_input[mask_fit], self._MIN_SIGMA)
        popt, _ = curve_fit(
            schechter_log_form,
            np.power(10.0, log10M_input[mask_fit]),
            cHMF_input[mask_fit],
            p0=self._SCHECHTER_P0,
            bounds=self._SCHECHTER_BOUNDS,
            sigma=sigma,
            absolute_sigma=True,
        )
        self.schechter_params = popt
        return schechter_log_form(np.power(10.0, log10M_extend), *popt)

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
        log10M_binleftedges = np.asarray(log10M_binleftedges, dtype=float)
        if log10M_binleftedges.size == 0:
            return np.array([], dtype=float)

        mask = log10M_binleftedges <= self._GP_MAX_LOG10M

        ratio_mean, ratio_std = self._predict_ratio(
            cosmo_params, log10M_binleftedges[mask]
        )
        spline = self._analytic_cHMF(cosmo_params)

        cHMF_analytic_gp = 10.0 ** spline(log10M_binleftedges[mask])
        cHMF_emu_gp = ratio_mean * cHMF_analytic_gp
        cHMF_err_emu_gp = ratio_std * cHMF_analytic_gp

        cHMF_total = np.empty(log10M_binleftedges.shape, dtype=float)
        cHMF_total[mask] = cHMF_emu_gp

        if np.any(~mask):
            cHMF_total[~mask] = self._schechter_fit(
                log10M_binleftedges[mask],
                cHMF_emu_gp,
                cHMF_err_emu_gp,
                log10M_binleftedges[~mask],
            )

        return cHMF_total

    def dndlog10M(
        self,
        cosmo_params: np.ndarray,
        log10M_bincentres: np.ndarray,
        dlog10M: float = 0.10,
    ) -> np.ndarray:
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
        log10M_bincentres = np.asarray(log10M_bincentres, dtype=float)
        if log10M_bincentres.size == 0:
            return np.array([], dtype=float)

        _lmble = log10M_bincentres - 0.5 * dlog10M
        _lmble = np.concatenate([_lmble, [log10M_bincentres[-1] + 0.5 * dlog10M]])

        cHMF_emu = self.cumulative_hmf(cosmo_params, _lmble)
        return -np.diff(cHMF_emu) / dlog10M

    def get_dndlog10M(
        self,
        cosmo_params: np.ndarray,
        log10M_bincentres: np.ndarray,
        dlog10M: float = 0.10,
    ) -> np.ndarray:
        """Wrapper for dndlog10M."""
        val = self.dndlog10M(cosmo_params, log10M_bincentres, dlog10M)
        return val

    def get_dHMF(
        self,
        cosmo_params: np.ndarray,
        log10M_bincentres: np.ndarray,
        dlog10M: float = 0.10,
    ) -> np.ndarray:
        """Wrapper for dndlog10M."""
        val = self.dndlog10M(cosmo_params, log10M_bincentres, dlog10M)
        return val
