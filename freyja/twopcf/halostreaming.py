import numpy as np
import sys

from scipy import stats, signal
from scipy.integrate import simpson
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.special import legendre

from astropy.cosmology import FlatLambdaCDM, Planck15
from halotools.empirical_models import halo_mass_to_virial_velocity

from hmd import Occupation, HaloModel
from hmd.occupation import Zheng07Centrals, Zheng07Sats
from hmd.galaxy import Galaxy
from hmd.profiles import FixedCosmologyNFW
from jax_fht.cosmology import FFTLog

from ..analytical.xi_1h_gcs import xi_1h_analy
from ..streaming import streaming_model
stsm = streaming_model(model='stsm')


from scipy.optimize import curve_fit
def func_powerlaw(logr, k, b):
    return b + k * logr



def xi_R_cs_split(
    r_bincentre,
    xiR_cs,
    rfit_min = 3.0,
    rfit_max = 5.0,
):
    mask = (r_bincentre < rfit_max) & (r_bincentre > rfit_min)
    popt, pcov = curve_fit(
        func_powerlaw, 
        np.log10(r_bincentre[mask]), 
        np.log10(xiR_cs[mask]), 
        p0=[-1,1], 
    )
    k_optim, b_optim = popt
    xi2h_powerlaw = 10**(k_optim * np.log10(r_bincentre) + b_optim)

    xi1h_approx = xiR_cs - (
        10**(k_optim * np.log10(r_bincentre) + b_optim)
    )
    xi1h_approx[r_bincentre > rfit_max] = 1e-8
    xi2h_approx = xiR_cs - xi1h_approx

    xi1h_approx[xi1h_approx < 0.0] = 1e-8
    return xi1h_approx, xi2h_approx




class twopcf_halostreaming(xi_1h_analy):
    def __init__(
        self, 
        log10M_bincentre: np.array,
        dlog10M: np.array,
        dndlog10M,
        r_xiR,
        xiR_cc,
        xiR_cs,
        xiR_ss,
        r_vm,
        vm_cc,
        vm_cs,
        vm_ss,
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
            mdef="vir",
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


        # the full xi-R term, will from emulators
        self.r_xiR = r_xiR
        self.xiR_cc = xiR_cc
        self.xiR_cs = xiR_cs
        self.xiR_ss = xiR_ss

        self.xiR_cc_2h = self.xiR_cc

        # the pairwise velocity moments, will from emulators
        self.r_vm = r_vm 
        self.vm_cc = vm_cc
        self.vm_cs = vm_cs
        self.vm_ss = vm_ss


        # self.r_binedge = r_binedge
        # self.s_binedge = s_binedge
        # self.mu_binedge = mu_binedge
        # self.r_bincentre = (self.r_binedge[1:] + self.r_binedge[:-1]) / 2.0
        # self.s_bincentre = (self.s_binedge[1:] + self.s_binedge[:-1]) / 2.0
        # self.mu_bincentre = (self.mu_binedge[1:] + self.mu_binedge[:-1]) / 2.0



    # def get_xiR_1h(self, r_bincentre):
    #     self.get_xiR_1h_cs(r_bincentre=r_bincentre) # self.xiR_1h_cs
    #     self.get_xiR_1h_ss(r_bincentre=r_bincentre) # self.xiR_1h_ss
    #     return self.xiR_1h_cs, self.xiR_1h_ss

    def get_xiR_cc(self, r_bincentre):
        # from emulator 
        # self.xiR_cc_2h = self.xiR_cc
        pass

    def get_xiR_cs(self, r_bincentre):
        # from emulator 
        # self.xiR_cs
        pass

    def get_xiR_ss(self, r_bincentre):
        # from emulator 
        # self.xiR_ss
        pass

    
    # self.get_xiS_1h_cs(
    #     s_binedge=s_binedge,
    #     mu_binedge=mu_binedge,
    #     return_multipoles=True,
    # ) # self.xiS0_1h_cs, self.xiS2_1h_cs, self.xiS4_1h_cs
    # self.get_xiS_1h_ss(
    #     s_binedge=s_binedge,
    #     mu_binedge=mu_binedge,
    #     return_multipoles=True,
    # ) # self.xiS0_1h_ss, self.xiS2_1h_ss, self.xiS4_1h_ss



    def get_xiS_2h_cc(
        self,
        s_binedge,
        mu_binedge,
        return_multipoles=True,
    ):
        s_stsm, self.xiS0_2h_cc, self.xiS2_2h_cc, self.xiS4_2h_cc = stsm(
            r_velmom=self.r_vm, 
            m10=self.vm_cc[:, 1],
            c20=self.vm_cc[:, 2], 
            c02=self.vm_cc[:, 3],
            c12=self.vm_cc[:, 4], 
            c30=self.vm_cc[:, 5],
            c40=self.vm_cc[:, 6], 
            c04=self.vm_cc[:, 7], 
            c22=self.vm_cc[:, 8],
            r_xiR=self.r_xiR, 
            xiR=self.xiR_cc_2h,
            s_output_binedge=s_binedge,
            mu_output_binedge=mu_binedge,
            kms_to_Mpch=self.kms_to_Mpch,
            return_multipoles=True,
        )
        return s_stsm, self.xiS0_2h_cc, self.xiS2_2h_cc, self.xiS4_2h_cc



    def get_xiS_2h_cs(
        self,
        s_binedge,
        mu_binedge,
        return_multipoles=True,
    ):
        self.get_xiR_1h_cs(self.r_xiR)
        self.xiR_1h_cs[
            np.where((self.xiR_1h_cs) < 1e-6*np.max(self.xiR_1h_cs))
        ] = 0.0
        self.xiR_2h_cs = self.xiR_cs - self.xiR_1h_cs
        self.xiR_2h_cs[
            np.where(np.abs(self.xiR_2h_cs) < 1e-8)
        ] = 0.0

        s_stsm, self.xiS0_2h_cs, self.xiS2_2h_cs, self.xiS4_2h_cs = stsm(
            r_velmom=self.r_vm, 
            m10=self.vm_cs[:, 1],
            c20=self.vm_cs[:, 2], 
            c02=self.vm_cs[:, 3],
            c12=self.vm_cs[:, 4], 
            c30=self.vm_cs[:, 5],
            c40=self.vm_cs[:, 6], 
            c04=self.vm_cs[:, 7], 
            c22=self.vm_cs[:, 8],
            r_xiR=self.r_xiR, 
            xiR=self.xiR_2h_cs,
            s_output_binedge=s_binedge,
            mu_output_binedge=mu_binedge,
            kms_to_Mpch=self.kms_to_Mpch,
            return_multipoles=True,
        )
        return s_stsm, self.xiS0_2h_cs, self.xiS2_2h_cs, self.xiS4_2h_cs




    def get_xiS_2h_ss(
        self,
        s_binedge,
        mu_binedge,
        return_multipoles=True,
    ):
        self.get_xiR_1h_ss(self.r_xiR)
        self.xiR_1h_ss[
            np.where((self.xiR_1h_ss) < 1e-6*np.max(self.xiR_1h_ss))
        ] = 0.0
        self.xiR_2h_ss = self.xiR_ss - self.xiR_1h_ss
        self.xiR_2h_ss[
            np.where(self.xiR_2h_ss < 1e-6)
        ] = 0.0

        s_stsm, self.xiS0_2h_ss, self.xiS2_2h_ss, self.xiS4_2h_ss = stsm(
            r_velmom=self.r_vm, 
            m10=self.vm_ss[:, 1],
            c20=self.vm_ss[:, 2], 
            c02=self.vm_ss[:, 3],
            c12=self.vm_ss[:, 4], 
            c30=self.vm_ss[:, 5],
            c40=self.vm_ss[:, 6], 
            c04=self.vm_ss[:, 7], 
            c22=self.vm_ss[:, 8],
            r_xiR=self.r_xiR, 
            xiR=self.xiR_2h_ss,
            s_output_binedge=s_binedge,
            mu_output_binedge=mu_binedge,
            kms_to_Mpch=self.kms_to_Mpch,
            return_multipoles=True,
        )
        return s_stsm, self.xiS0_2h_ss, self.xiS2_2h_ss, self.xiS4_2h_ss

    def __call__(
        self,
        s_binedge,
        mu_binedge,
        return_multipoles=True,
    ):
        self.get_xiS_2h_cc(
            s_binedge=s_binedge,
            mu_binedge=mu_binedge,
            return_multipoles=return_multipoles,
        )
        self.get_xiS_2h_cs(
            s_binedge=s_binedge,
            mu_binedge=mu_binedge,
            return_multipoles=return_multipoles,
        )
        self.get_xiS_1h_cs(
            s_binedge=s_binedge,
            mu_binedge=mu_binedge,
            return_multipoles=return_multipoles,
        )
        self.get_xiS_2h_ss(
            s_binedge=s_binedge,
            mu_binedge=mu_binedge,
            return_multipoles=return_multipoles,
        )
        self.get_xiS_1h_ss(
            s_binedge=s_binedge,
            mu_binedge=mu_binedge,
            return_multipoles=return_multipoles,
        )
        self.xiS0_gg = self.f_c**2 * (self.xiS0_2h_cc) + \
            2.0 * self.f_c * self.f_s * (self.xiS0_1h_cs + self.xiS0_2h_cs) + \
            self.f_s**2 * (self.xiS0_1h_ss + self.xiS0_2h_ss)
            
        self.xiS2_gg = self.f_c**2 * (self.xiS2_2h_cc) + \
            2.0 * self.f_c * self.f_s * (self.xiS2_1h_cs + self.xiS2_2h_cs) + \
            self.f_s**2 * (self.xiS2_1h_ss + self.xiS2_2h_ss)


        return self.xiS0_2h_cc, self.xiS0_2h_cs, self.xiS0_1h_cs, self.xiS0_2h_ss, self.xiS0_1h_ss, self.xiS0_gg, self.xiS2_gg
