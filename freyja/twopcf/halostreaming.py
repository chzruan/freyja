from __future__ import annotations

from typing import Dict, Tuple
import copy

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.special import legendre

from astropy.cosmology import FlatLambdaCDM
from hmd import Occupation
from hmd.occupation import Zheng07Centrals, Zheng07Sats
from hmd.galaxy import Galaxy
from hmd.profiles import FixedCosmologyNFW

from gsm.models.skewt.from_radial_transverse import moments2skewt
from gsm.models.gaussian.from_radial_transverse import moments2gaussian
from gsm.streaming_integral import real2redshift

from ..analytical.xi_1h_gcs import xi_1h_analy


def tpcf_multipole(s_mu_tpcf: np.ndarray, mu_bins: np.ndarray, order: int = 0) -> np.ndarray:
    """
    Multipole of the two-point function measured on (s, mu) bins.
    Adapted from halotools.mock_observables.tpcf_multipole.

    Parameters
    ----------
    s_mu_tpcf : (Ns, Nmu) array
        2D xi(s, mu) evaluated on s- and mu-bin centers (Nmu corresponds to mu-bin centers).
    mu_bins : (Nmu+1,) array
        Mu bin edges in [0, 1].
    order : int
        Legendre order (0, 2, 4, ...).

    Returns
    -------
    xi_l : (Ns,) array
        The multipole of the given order.
    """
    s_mu_tpcf = np.atleast_2d(s_mu_tpcf)
    mu_bins = np.asarray(mu_bins, dtype=float)
    order = int(order)

    mu_centers = 0.5 * (mu_bins[1:] + mu_bins[:-1])
    L_n = legendre(order)
    # symmetric in mu -> average L(mu)+L(-mu)
    L_sym = L_n(mu_centers) + L_n(-mu_centers)
    dmu = np.diff(mu_bins)

    # (2l+1)/2 ∑_mu xi(s,mu) Δmu [L(mu)+L(-mu)]
    prefac = (2.0 * order + 1.0) / 2.0
    return prefac * np.sum(s_mu_tpcf * dmu[None, :] * L_sym[None, :], axis=1)


class twopcf_halostreaming(xi_1h_analy):
    """
    Two-point correlation function in redshift space using a streaming model.
    - CC, CS can use Gaussian at large s and Skew-t at small s.
    - SS uses GSM

    """

    # -------------------------
    # Construction / utilities
    # -------------------------
    def __init__(
        self,
        log10M_bincentre: np.ndarray,
        dlog10M: np.ndarray,
        dndlog10M,
        r_xiR: np.ndarray,
        xiR_cc: np.ndarray,
        xiR_cs: np.ndarray,
        xiR_ss: np.ndarray,
        dict_vm: Dict[str, Dict[str, np.ndarray]],
        redshift: float = 0.25,
        cosmology: FlatLambdaCDM = FlatLambdaCDM(H0=70, Om0=0.3),
        central_occupation: Occupation = Zheng07Centrals(),
        satellite_occupation: Occupation = Zheng07Sats(),
        HOD_params: Galaxy = Galaxy(
            logM_min=13.62,
            sigma_logM=0.6915,
            kappa=0.51,
            logM1=14.42,
            alpha=0.9168,
            M_cut=10**12.26,
            M_sat=10**14.87,
            concentration_bias=1.0,
            v_bias_centrals=0.1,
            v_bias_satellites=1.0,
            B_cen=0.0,
            B_sat=0.0,
        ),
        sat_profile: FixedCosmologyNFW = FixedCosmologyNFW(
            redshift=0.25, cosmology=FlatLambdaCDM(H0=70, Om0=0.3), mdef="vir"
        ),
        mdef: str = "vir",
        fft_num: int = 1,
        fft_logrmin: float = -5.0,
        fft_logrmax: float = 3.0,
        sigma_vir_halo=None,
        conc=None,
        cen_vel_bias: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            log10M_bincentre=log10M_bincentre,
            dlog10M=dlog10M,
            dndlog10M=dndlog10M,
            redshift=redshift,
            cosmology=cosmology,
            central_occupation=central_occupation,
            satellite_occupation=satellite_occupation,
            HOD_params=HOD_params,
            sat_profile=sat_profile,
            mdef=mdef,
            fft_num=fft_num,
            fft_logrmin=fft_logrmin,
            fft_logrmax=fft_logrmax,
            sigma_vir_halo=sigma_vir_halo,
            conc=conc,
            cen_vel_bias=cen_vel_bias,
            verbose=verbose,
        )

        # Real-space xi inputs (e.g., from emulators)
        self.r_xiR = np.asarray(r_xiR, dtype=float)
        self.xiR_cc = np.asarray(xiR_cc, dtype=float)
        self.xiR_cs = np.asarray(xiR_cs, dtype=float)
        self.xiR_ss = np.asarray(xiR_ss, dtype=float)

        # Two-halo cc is just xiR_cc in your original
        self.xiR_2h_cc = self.xiR_cc

        # Copy + convert velocity moments (to avoid mutating caller's dict)
        self.dict_vm = copy.deepcopy(dict_vm)
        for pairs in ("cc", "cs", "ss"):
            self.dict_vm[pairs]["m10"] *= self.kms_to_Mpch
            for key in ("c20", "c02"):
                self.dict_vm[pairs][key] *= self.kms_to_Mpch**2
            for key in ("c12", "c30"):
                self.dict_vm[pairs][key] *= self.kms_to_Mpch**3
            for key in ("c40", "c04", "c22"):
                self.dict_vm[pairs][key] *= self.kms_to_Mpch**4

    # Placeholder
    def get_xiR_cc(self, r_bincentre: np.ndarray) -> None:
        pass

    def get_xiR_cs(self, r_bincentre: np.ndarray) -> None:
        pass

    def get_xiR_ss(self, r_bincentre: np.ndarray) -> None:
        pass

    # -------------------------
    # Internal helpers
    # -------------------------
    @staticmethod
    def _bin_centers(edges: np.ndarray) -> np.ndarray:
        edges = np.asarray(edges, dtype=float)
        return 0.5 * (edges[1:] + edges[:-1])

    def _xiR_spline(self, r: np.ndarray, xiR: np.ndarray) -> IUS:
        return IUS(r, xiR, ext="zeros")

    def _gaussian_pdf_from_dict(self, pairs: str):
        r_vm = self.dict_vm[pairs]["r_vm"]
        return moments2gaussian(
            m_10=IUS(r_vm, self.dict_vm[pairs]["m10"], ext="const"),
            c_20=IUS(r_vm, self.dict_vm[pairs]["c20"], ext="const"),
            c_02=IUS(r_vm, self.dict_vm[pairs]["c02"], ext="const"),
        )

    def _skewt_pdf_from_dict(self, pairs: str):
        r_vm = self.dict_vm[pairs]["r_vm"]
        return moments2skewt(
            m_10=IUS(r_vm, self.dict_vm[pairs]["m10"], ext="const"),
            c_20=IUS(r_vm, self.dict_vm[pairs]["c20"], ext="const"),
            c_02=IUS(r_vm, self.dict_vm[pairs]["c02"], ext="const"),
            c_12=IUS(r_vm, self.dict_vm[pairs]["c12"], ext="const"),
            c_30=IUS(r_vm, self.dict_vm[pairs]["c30"], ext="const"),
            c_22=IUS(r_vm, self.dict_vm[pairs]["c22"], ext="const"),
            c_40=IUS(r_vm, self.dict_vm[pairs]["c40"], ext="const"),
            c_04=IUS(r_vm, self.dict_vm[pairs]["c04"], ext="const"),
        )

    def _integrate_streaming(
        self,
        s_centers: np.ndarray,
        mu_centers: np.ndarray,
        xiR_func,
        vlos_pdf_func,
        limit: float = 120.0,
        epsilon: float = 0.01,
        n: int = 300,
    ) -> np.ndarray:
        return real2redshift.simps_integrate(
            s_c=s_centers,
            mu_c=mu_centers,
            twopcf_function=xiR_func,
            los_pdf_function=vlos_pdf_func,
            limit=limit,
            epsilon=epsilon,
            n=n,
        )

    @staticmethod
    def _assemble_multipoles(
        xi_smu: np.ndarray, mu_edges: np.ndarray, orders=(0, 2, 4)
    ) -> Dict[int, np.ndarray]:
        return {l: tpcf_multipole(xi_smu, mu_edges, order=l) for l in orders}

    # -------------------------
    # 2-halo CC
    # -------------------------
    def get_xiS_2h_cc(
        self,
        s_binedge: np.ndarray,
        mu_binedge: np.ndarray,
        return_multipoles: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        2-halo xi^S for CC pairs:
        - small s: Skew-t
        - large s: Gaussian
        """
        pairs = "cc"
        xiR_func = self._xiR_spline(self.r_xiR, self.xiR_2h_cc)
        gau_pdf = self._gaussian_pdf_from_dict(pairs)
        st_pdf = self._skewt_pdf_from_dict(pairs)

        s_centers = self._bin_centers(s_binedge)
        mu_centers = self._bin_centers(mu_binedge)

        # Split regime
        mask_small = s_centers < 30.1
        mask_large = ~mask_small

        xi_smu_large = self._integrate_streaming(s_centers[mask_large], mu_centers, xiR_func, gau_pdf)
        xi_smu_small = self._integrate_streaming(s_centers[mask_small], mu_centers, xiR_func, st_pdf)

        mp_large = self._assemble_multipoles(xi_smu_large, mu_binedge)
        mp_small = self._assemble_multipoles(xi_smu_small, mu_binedge)

        # Recompose in original s order (small first, then large, since s is ascending)
        self.xiS0_2h_cc = np.concatenate([mp_small[0], mp_large[0]])
        self.xiS2_2h_cc = np.concatenate([mp_small[2], mp_large[2]])
        self.xiS4_2h_cc = np.concatenate([mp_small[4], mp_large[4]])

        return s_centers, self.xiS0_2h_cc, self.xiS2_2h_cc, self.xiS4_2h_cc

    # -------------------------
    # 2-halo CS
    # -------------------------
    def get_xiS_2h_cs(
        self,
        s_binedge: np.ndarray,
        mu_binedge: np.ndarray,
        return_multipoles: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pairs = "cs"

        # Subtract 1-halo from total to get 2-halo
        self.get_xiR_1h_cs(self.r_xiR)  # sets self.xiR_1h_cs
        # numerical hygiene as in your original
        self.xiR_1h_cs[self.xiR_1h_cs < 1e-10 * np.max(self.xiR_1h_cs)] = 0.0
        self.xiR_2h_cs = self.xiR_cs - self.xiR_1h_cs
        self.xiR_2h_cs[np.abs(self.xiR_2h_cs) < 1e-10] = 0.0

        xiR_func = self._xiR_spline(self.r_xiR, self.xiR_2h_cs)
        gau_pdf = self._gaussian_pdf_from_dict(pairs)
        st_pdf = self._skewt_pdf_from_dict(pairs)

        s_centers = self._bin_centers(s_binedge)
        mu_centers = self._bin_centers(mu_binedge)

        mask_small = s_centers < 30.1
        mask_large = ~mask_small

        xi_smu_large = self._integrate_streaming(s_centers[mask_large], mu_centers, xiR_func, gau_pdf)
        xi_smu_small = self._integrate_streaming(s_centers[mask_small], mu_centers, xiR_func, st_pdf)

        mp_large = self._assemble_multipoles(xi_smu_large, mu_binedge)
        mp_small = self._assemble_multipoles(xi_smu_small, mu_binedge)

        self.xiS0_2h_cs = np.concatenate([mp_small[0], mp_large[0]])
        self.xiS2_2h_cs = np.concatenate([mp_small[2], mp_large[2]])
        self.xiS4_2h_cs = np.concatenate([mp_small[4], mp_large[4]])

        return s_centers, self.xiS0_2h_cs, self.xiS2_2h_cs, self.xiS4_2h_cs

    # -------------------------
    # 2-halo SS (Gaussian-only here)
    # -------------------------
    def get_xiS_2h_ss(
        self,
        s_binedge: np.ndarray,
        mu_binedge: np.ndarray,
        return_multipoles: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pairs = "ss"

        self.get_xiR_1h_ss(self.r_xiR)  # sets self.xiR_1h_ss
        self.xiR_1h_ss[self.xiR_1h_ss < 1e-6 * np.max(self.xiR_1h_ss)] = 0.0
        self.xiR_2h_ss = self.xiR_ss - self.xiR_1h_ss
        self.xiR_2h_ss[self.xiR_2h_ss < 1e-8] = 0.0

        xiR_func = self._xiR_spline(self.r_xiR, self.xiR_2h_ss)
        gau_pdf = self._gaussian_pdf_from_dict(pairs)

        s_centers = self._bin_centers(s_binedge)
        mu_centers = self._bin_centers(mu_binedge)

        xi_smu = self._integrate_streaming(s_centers, mu_centers, xiR_func, gau_pdf)
        mp = self._assemble_multipoles(xi_smu, mu_binedge)

        self.xiS0_2h_ss = mp[0]
        self.xiS2_2h_ss = mp[2]
        self.xiS4_2h_ss = mp[4]
        return s_centers, self.xiS0_2h_ss, self.xiS2_2h_ss, self.xiS4_2h_ss

    # -------------------------
    # Driver
    # -------------------------
    def __call__(
        self,
        s_binedge: np.ndarray,
        mu_binedge: np.ndarray,
        return_multipoles: bool = True,
    ):
        # 2-halo parts
        self.get_xiS_2h_cc(s_binedge=s_binedge, mu_binedge=mu_binedge, return_multipoles=return_multipoles)
        self.get_xiS_2h_cs(s_binedge=s_binedge, mu_binedge=mu_binedge, return_multipoles=return_multipoles)
        self.get_xiS_2h_ss(s_binedge=s_binedge, mu_binedge=mu_binedge, return_multipoles=return_multipoles)

        # 1-halo parts (your originals)
        self.get_xiS_1h_cs(s_binedge=s_binedge, mu_binedge=mu_binedge, return_multipoles=return_multipoles)
        self.get_xiS_1h_ss(s_binedge=s_binedge, mu_binedge=mu_binedge, return_multipoles=return_multipoles)

        # Galaxy auto multipoles
        self.xiS0_gg = (
            self.f_c**2 * self.xiS0_2h_cc
            + 2.0 * self.f_c * self.f_s * (self.xiS0_1h_cs + self.xiS0_2h_cs)
            + self.f_s**2 * (self.xiS0_1h_ss + self.xiS0_2h_ss)
        )
        self.xiS2_gg = (
            self.f_c**2 * self.xiS2_2h_cc
            + 2.0 * self.f_c * self.f_s * (self.xiS2_1h_cs + self.xiS2_2h_cs)
            + self.f_s**2 * (self.xiS2_1h_ss + self.xiS2_2h_ss)
        )

        return (
            self.xiS0_2h_cc,
            self.xiS0_2h_cs,
            self.xiS0_1h_cs,
            self.xiS0_2h_ss,
            self.xiS0_1h_ss,
            self.xiS0_gg,
            self.xiS2_gg,
        )
