from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from numba import njit
from scipy.integrate import simpson
from scipy.interpolate import InterpolatedUnivariateSpline as ius, interp1d
from scipy.special import legendre

from astropy.cosmology import FlatLambdaCDM
from halotools.empirical_models import halo_mass_to_virial_velocity

from hmd import Occupation, HaloModel  # noqa: F401 (kept for external API parity)
from hmd.occupation import Zheng07Centrals, Zheng07Sats
from hmd.galaxy import Galaxy
from hmd.profiles import FixedCosmologyNFW
from jax_fht.cosmology import FFTLog


# ------------------------------- #
#    Small, hot Numba kernels     #
# ------------------------------- #

@njit(cache=True, fastmath=True)
def _gaussian_pdf(v: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Vectorized Gaussian pdf with broadcasting (Numba)."""
    inv = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    return inv * np.exp(-0.5 * (v / sigma) ** 2)


@njit(cache=True, fastmath=True)
def _pdf_vlos_1h_cs(vlos: np.ndarray, sigma_vir: np.ndarray,
                    alpha_c: float, alpha_s: float) -> np.ndarray:
    # sigma_vir[..., mass], vlos[..., 1]
    stddev = np.sqrt(alpha_c * alpha_c + alpha_s * alpha_s) * sigma_vir
    return _gaussian_pdf(vlos, stddev)


@njit(cache=True, fastmath=True)
def _pdf_vlos_1h_ss(vlos: np.ndarray, sigma_vir: np.ndarray,
                    alpha_s: float) -> np.ndarray:
    # std = sqrt(2) * alpha_s * sigma_vir
    stddev = 1.4142135624 * alpha_s * sigma_vir
    return _gaussian_pdf(vlos, stddev)


# ------------------------------- #
#         Helper functions        #
# ------------------------------- #

def tpcf_multipole(s_mu: np.ndarray, mu_bins: np.ndarray, order: int = 0) -> np.ndarray:
    """
    Multipole of xi(s, mu). Compatible with halotools' convention.
    """
    s_mu = np.atleast_2d(s_mu)
    mu_bins = np.atleast_1d(mu_bins).astype(float)
    Ln = legendre(int(order))
    mu_c = 0.5 * (mu_bins[1:] + mu_bins[:-1])
    dmu = np.diff(mu_bins)

    # (2ℓ+1)/2 ∫_{-1}^{1} xiℓ(s) Lℓ(mu) dmu ≈ sum over [0,1] using symmetry
    # Lℓ is even for even ℓ and odd for odd ℓ; using Ln(mu)+Ln(-mu) doubles even, cancels odd.
    prefac = (2.0 * order + 1.0) * 0.5
    return prefac * np.sum(s_mu * dmu * (Ln(mu_c) + Ln(-mu_c)), axis=1)


def _sym_r_parallel_grid(eps: float, rmax: float, nps: int) -> np.ndarray:
    """
    Symmetric line-of-sight grid: [-rmax, ..., -eps, +eps, ..., +rmax] (geometric spacing in |r|).
    """
    # strictly positive side
    rp = np.geomspace(max(eps, 1e-12), rmax, nps).astype(float)
    # mirror to negative side
    rm = -rp[::-1]
    return np.concatenate([rm, rp], axis=0)


def _safe_div(a: np.ndarray, b: float, eps: float = 0.0) -> np.ndarray:
    if b == 0.0:
        return np.zeros_like(a)
    return a / (b + eps)


# ------------------------------- #
#        Main analysis class      #
# ------------------------------- #

class xi_1h_analy:
    """
    1-halo analytic xi in real- and redshift-space for (cen-sat) and (sat-sat).

    Notes on shapes
    --------------
    - Mass dimension: (nM,)
    - r-parallel integration dimension: (nRpar,)
    - Output xi_S(s, mu): (ns, nmu)
    """

    def __init__(
        self,
        log10M_bincentre: np.ndarray,
        dlog10M: np.ndarray,
        dndlog10M: np.ndarray,
        *,
        redshift: float = 0.25,
        cosmology: FlatLambdaCDM = FlatLambdaCDM(H0=70, Om0=0.3),
        central_occupation: Occupation = Zheng07Centrals(),
        satellite_occupation: Occupation = Zheng07Sats(),
        HOD_params: Galaxy = Galaxy(
            logM_min=13.62, sigma_logM=0.6915, kappa=0.51, logM1=14.42,
            alpha=0.9168, M_cut=10 ** 12.26, M_sat=10 ** 14.87,
            concentration_bias=1.0, v_bias_centrals=0.1, v_bias_satellites=1.0,
            B_cen=0.0, B_sat=0.0,
        ),
        sat_profile: FixedCosmologyNFW = FixedCosmologyNFW(
            redshift=0.25, cosmology=FlatLambdaCDM(H0=70, Om0=0.3), mdef="200m"
        ),
        mdef: str = "vir",
        fft_num: int = 1,
        fft_logrmin: float = -5.0,
        fft_logrmax: float = 3.0,
        sigma_vir_halo: Optional[np.ndarray] = None,
        conc: Optional[np.ndarray] = None,
        cen_vel_bias: bool = False,
        verbose: bool = False,
    ):
        # cosmology / units
        self.redshift = float(redshift)
        self.cosmology = cosmology
        self.kms_to_Mpch = (1.0 + self.redshift) / (100.0 * self.cosmology.efunc(self.redshift))

        # mass grids & HMF bits
        self.log10M = np.asarray(log10M_bincentre, dtype=float)
        self.dlog10M = np.asarray(dlog10M, dtype=float)
        self.dndlog10M = np.asarray(dndlog10M, dtype=float)
        self.M = 10.0 ** self.log10M
        self.conc = conc
        self.mdef = mdef

        # HOD & profile
        self.central_occupation = central_occupation
        self.satellite_occupation = satellite_occupation
        self.galaxy = HOD_params
        self.sat_profile = sat_profile
        self.cen_vel_bias = cen_vel_bias

        # FFTLog for xi(r)
        self.fft = FFTLog(fft_num, log_r_min=fft_logrmin, log_r_max=fft_logrmax, kr=1.0)

        # Occupation numbers (per halo)
        self.N_cen = self.central_occupation.get_n(self.M, galaxy=self.galaxy)
        self.N_sat = self.satellite_occupation.get_n(self.M, n_centrals=self.N_cen, galaxy=self.galaxy)

        # Number densities
        # ∫ dlog10M (dn/dlog10M) N
        self.n_cen = simpson(self.dndlog10M * self.N_cen, x=self.log10M)
        self.n_sat = simpson(self.dndlog10M * self.N_sat, x=self.log10M)
        self.n_gal = self.n_cen + self.n_sat
        self.n_c = self.n_cen
        self.n_s = self.n_sat
        self.n_g = self.n_gal
        self.f_c = _safe_div(self.n_cen, self.n_gal)
        self.f_s = _safe_div(self.n_sat, self.n_gal)

        # Virial velocity dispersion per halo mass (converted to Mpc/h)
        if sigma_vir_halo is None:
            # Vvir / sqrt(2*(1+z)) / sqrt(3) → 1D disp, then convert km/s → Mpc/h
            vvir = halo_mass_to_virial_velocity(
                self.M, 
                cosmology=self.cosmology,
                redshift=self.redshift, 
                mdef=self.mdef
            )
            self.sigma_vir_halo = (vvir / (np.sqrt(2.0 * (1.0 + self.redshift)) * np.sqrt(3.0))) * self.kms_to_Mpch
        else:
            self.sigma_vir_halo = np.asarray(sigma_vir_halo, dtype=float)

        if verbose:
            print("HOD summary:")
            print(f"log10 n_gal = {np.log10(max(self.n_gal, 1e-99)):.4f}")
            print(f"log10 n_cen = {np.log10(max(self.n_cen, 1e-99)):.4f}")
            print(f"log10 n_sat = {np.log10(max(self.n_sat, 1e-99)):.4f}")
            print(f"satellite fraction f_s = {self.f_s:.4f}")

        # small eps to protect divisions
        self._eps_norm = 0.0

        # scratch/cache
        self._Akari_interp: Optional[interp1d] = None
        self._Akari_logr_fft: Optional[np.ndarray] = None

    # ------------- Real-space xi_1h ------------- #

    def get_xiR_1h_cs(self, r_bincentre: np.ndarray) -> np.ndarray:
        """
        Real-space central–satellite xi_1h(r).
        """
        r = np.asarray(r_bincentre, dtype=float)
        # u_sat(r|M) shape: (nr, nM)
        u_sat_r_M = self.sat_profile.u_r_M_profile(
            r=r, mass=self.M, redshift=self.redshift, cosmology=self.cosmology
        )
        # ∫ dlog10M (dn/dlog10M) N_sat u(r|M) / (n_cen n_sat)
        num = simpson(self.dndlog10M * self.N_sat * u_sat_r_M, x=self.log10M, axis=-1)
        xi = _safe_div(num, (self.n_cen * self.n_sat), self._eps_norm)

        # hard zeroing tiny tails (optional)
        thresh = 1e-8 * np.max(xi) if xi.size else 0.0
        xi[xi < thresh] = 0.0
        self.xiR_1h_cs = xi
        return xi

    def get_xiR_1h_ss(self, r_bincentre: np.ndarray) -> np.ndarray:
        """
        Real-space satellite–satellite xi_1h(r).
        """
        r = np.asarray(r_bincentre, dtype=float)

        # Fourier-space u(k|M), then xi(r) from pk2xi; square profile for ss
        u_sat_k_M = self.sat_profile.fourier_mass_density(
            k=self.fft.k, mass=self.M, cosmology=self.cosmology,
            redshift=self.redshift, conc=self.conc
        )
        xi_u2_r_M = self.fft.pk2xi((u_sat_k_M ** 2)).T  # (nr_fft, nM)

        # integrate over mass, then interpolate to desired r
        num_fft = simpson(self.dndlog10M * (self.N_sat ** 2) / np.maximum(self.N_cen, 1e-30) * xi_u2_r_M,
                          x=self.log10M, axis=-1)
        xi_fft_r = self.fft.r
        xi_interp = ius(np.log10(xi_fft_r), num_fft, ext="zeros")(np.log10(r))
        xi = _safe_div(xi_interp, (self.n_sat ** 2), self._eps_norm)

        self.xiR_1h_ss = xi
        return xi

    # --------- Redshift-space xi_1h(s, mu) --------- #

    def _prepare_s_mu(self, s_edges: np.ndarray, mu_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s_edges = np.asarray(s_edges, dtype=float)
        mu_edges = np.asarray(mu_edges, dtype=float)

        s_c = 0.5 * (s_edges[1:] + s_edges[:-1])
        mu_c = 0.5 * (mu_edges[1:] + mu_edges[:-1])

        ss = s_c.reshape(-1, 1)                    # (ns, 1)
        mu = mu_c.reshape(1, -1)                    # (1, nmu)
        s_par = (ss * mu).reshape(-1, 1)            # (ns*nmu, 1)
        s_perp = (ss * np.sqrt(np.maximum(1.0 - mu * mu, 0.0))).reshape(-1, 1)

        return s_c, mu_c, s_par, s_perp

    def _vlos_pdf_cs(self, vlos: np.ndarray) -> np.ndarray:
        # Broadcast sigma to mass dim
        # vlos shape: (..., 1) ; sigma_vir_halo shape: (nM,)
        return _pdf_vlos_1h_cs(vlos, self.sigma_vir_halo, self.galaxy.v_bias_centrals, self.galaxy.v_bias_satellites)

    def _vlos_pdf_ss(self, vlos: np.ndarray) -> np.ndarray:
        return _pdf_vlos_1h_ss(vlos, self.sigma_vir_halo, self.galaxy.v_bias_satellites)

    def get_xiS_1h_cs(
        self,
        s_binedge: np.ndarray,
        mu_binedge: np.ndarray,
        *,
        eps: float = 1e-6,
        nps: int = 500,
        r_parallel_max: float = 80.0,
        return_multipoles: bool = True,
    ):
        """
        Redshift-space 1-halo CS contribution xi(s, mu).
        """
        s_c, mu_c, s_par, s_perp = self._prepare_s_mu(s_binedge, mu_binedge)

        # symmetric r_parallel grid: (-, +) combined; shape (nRpar,)
        rpar = _sym_r_parallel_grid(eps=eps, rmax=r_parallel_max, nps=int(nps))  # (nRpar,)
        # geometry
        rr = np.sqrt(s_perp ** 2 + rpar[None, :] ** 2)  # (ns*nmu, nRpar)

        # u(r|M): (ns*nmu, nRpar, nM)
        u_sat_r_M = self.sat_profile.u_r_M_profile(
            r=rr, mass=self.M, redshift=self.redshift, cosmology=self.cosmology
        )

        # vlos = s_parallel - r_parallel (with sign on r_parallel)
        vlos = (s_par - rpar[None, :]) * np.sign(rpar)[None, :]  # (ns*nmu, nRpar)

        # expand vlos to (..., 1) for broadcasting with mass dim in Numba
        vlos_exp = vlos[..., None]
        pdf = self._vlos_pdf_cs(vlos_exp)  # (ns*nmu, nRpar, nM)

        # integrate over r_parallel, then over mass
        inner = simpson(u_sat_r_M * pdf, x=rpar, axis=1)  # (ns*nmu, nM)
        outer = simpson(inner * self.dndlog10M * self.N_sat, x=self.log10M, axis=-1)  # (ns*nmu,)

        xi_smu = _safe_div(outer, (self.n_cen * self.n_sat), self._eps_norm).reshape(s_c.size, mu_c.size)
        self.xiS_1h_cs = xi_smu

        if return_multipoles:
            xi0 = tpcf_multipole(xi_smu, mu_binedge, order=0)
            xi2 = tpcf_multipole(xi_smu, mu_binedge, order=2)
            xi4 = tpcf_multipole(xi_smu, mu_binedge, order=4)
            self.xiS0_1h_cs, self.xiS2_1h_cs, self.xiS4_1h_cs = xi0, xi2, xi4
            return s_c, xi0, xi2, xi4
        return xi_smu

    def _prepare_Akari_interp(self) -> None:
        """
        Precompute Akari(r, M) = xi[r; u(k|M)^2] and set an axis-wise interp1d over log10 r.
        """
        if self._Akari_interp is not None:
            return
        u_sat_k_M = self.sat_profile.fourier_mass_density(
            k=self.fft.k, mass=self.M, cosmology=self.cosmology,
            redshift=self.redshift, conc=self.conc
        )
        Akari = self.fft.pk2xi(u_sat_k_M ** 2).T  # (nr_fft, nM)
        self._Akari_logr_fft = np.log10(self.fft.r)
        # vectorized interpolation across mass via axis=0
        self._Akari_interp = interp1d(
            self._Akari_logr_fft, Akari,
            axis=0, bounds_error=False, fill_value=0.0, assume_sorted=True
        )

    def get_xiS_1h_ss(
        self,
        s_binedge: np.ndarray,
        mu_binedge: np.ndarray,
        *,
        eps: float = 1e-8,
        nps: int = 500,
        r_parallel_max: float = 80.0,
        return_multipoles: bool = True,
    ):
        """
        Redshift-space 1-halo SS contribution xi(s, mu).
        """
        s_c, mu_c, s_par, s_perp = self._prepare_s_mu(s_binedge, mu_binedge)
        rpar = _sym_r_parallel_grid(eps=eps, rmax=r_parallel_max, nps=int(nps))

        # radii and Akari(r, M)
        rr = np.sqrt(s_perp ** 2 + rpar[None, :] ** 2)  # (ns*nmu, nRpar)
        self._prepare_Akari_interp()
        Akari_integrand = self._Akari_interp(np.log10(rr))  # (ns*nmu, nRpar, nM)

        # vlos pdf
        vlos = (s_par - rpar[None, :]) * np.sign(rpar)[None, :]
        pdf = self._vlos_pdf_ss(vlos[..., None])  # (ns*nmu, nRpar, nM)

        # integrate over r_parallel then mass
        inner = simpson(Akari_integrand * pdf, x=rpar, axis=1)  # (ns*nmu, nM)
        weight = self.dndlog10M * (self.N_sat ** 2) / np.maximum(self.N_cen, 1e-30)
        outer = simpson(inner * weight, x=self.log10M, axis=-1)  # (ns*nmu,)

        xi_smu = _safe_div(outer, (self.n_sat ** 2), self._eps_norm).reshape(s_c.size, mu_c.size)
        self.xiS_1h_ss = xi_smu

        if return_multipoles:
            xi0 = tpcf_multipole(xi_smu, mu_binedge, order=0)
            xi2 = tpcf_multipole(xi_smu, mu_binedge, order=2)
            xi4 = tpcf_multipole(xi_smu, mu_binedge, order=4)
            self.xiS0_1h_ss, self.xiS2_1h_ss, self.xiS4_1h_ss = xi0, xi2, xi4
            return s_c, xi0, xi2, xi4
        return xi_smu
