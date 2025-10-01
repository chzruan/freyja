import numpy as np 
import h5py
import pandas as pd
import sys
import argparse
from pathlib import Path
import yaml

from pycorr import TwoPointCorrelationFunction
from freyja.utils.config import ConfigHOD, ConfigMeasure


from scipy.interpolate import InterpolatedUnivariateSpline as ius 
from astropy.cosmology import Planck15
from freyja.twopcf import twopcf_halostreaming
from hmd.occupation import Zheng07Centrals, Zheng07Sats
from hmd.galaxy import Galaxy
from hmd.profiles import FixedCosmologyNFW
from freyja.utils.cofm import c_Duffy, c_Prada, c_DuttonMaccio
from pysam import VariantFile


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path", 
    type=Path,
    default="config.yaml",
)
parser.add_argument(
    "--xiR_path", 
    type=Path,
)
parser.add_argument(
    "--velmom_path", 
    type=Path,
)
parser.add_argument(
    "--output_path", 
    type=Path,
)

pp = parser.parse_args()
with open(pp.config_path, "r") as f:
    config = yaml.safe_load(f)

config_basic = ConfigHOD(**config["basic"])

gravity = config_basic.gravity
dataflag = config_basic.dataflag
boxsize = config_basic.boxsize
snapnum = config_basic.snapnum
redshift = config_basic.redshift
log10M_halo_min = config_basic.log10M_halo_range[0]
log10M_halo_max = config_basic.log10M_halo_range[1]
datasets = config_basic.datasets
cosmology = Planck15



s_binedge = np.arange(0.5, 101.5, 0.5)
mu_binedge=np.linspace(0, 1, 256)
s_bincentre = (s_binedge[1:] + s_binedge[:-1]) / 2.0
mu_bincentre = (mu_binedge[1:] + mu_binedge[:-1]) / 2.0




log10M_bincentre, dndlog10M = np.loadtxt(
    f'./data/hmf_{gravity}_{dataflag}_z{redshift:.2f}.dat',
    unpack=True,
    usecols=(0, 1),
)
dlog10M=log10M_bincentre[1] - log10M_bincentre[0]
log10M_binleftedge = log10M_bincentre - dlog10M/2.0


dict_xi = {}
file_xiR = h5py.File(
    pp.xiR_path,
    'r',
)
for pairs in ['cc', 'cs', 'ss',]:
    _lst = []
    for ibox in range(1, 100+1):
        group_xiR = file_xiR[f'box{ibox}']
        r_xiR = group_xiR[f'r_bincentre_{pairs}'][...]
        _lst.append(group_xiR[f'xiR_{pairs}'][...])
    xiR_all = np.array(_lst)
    xiR_mean = np.mean(
        xiR_all,
        axis=0,
    )
    dict_xi[pairs] = xiR_mean


dict_vm = {}
file_vm = h5py.File(
    pp.velmom_path,
    "r",
)
for pairs in ['cc', 'cs', 'ss']:
    _lst = []
    for i, keybox in enumerate(file_vm.keys()):
        group_velmom = file_vm[keybox]
        r_vm = group_velmom[f"r_velmom_{pairs}"][...]
        _lst.append(
            np.vstack((
                group_velmom[f"r_velmom_{pairs}"][...],
                group_velmom[f"m10_{pairs}"][...],
                group_velmom[f"c20_{pairs}"][...],
                group_velmom[f"c02_{pairs}"][...],
                group_velmom[f"c12_{pairs}"][...],
                group_velmom[f"c30_{pairs}"][...],
                group_velmom[f"c40_{pairs}"][...],
                group_velmom[f"c04_{pairs}"][...],
                group_velmom[f"c22_{pairs}"][...],
            )).T
        )
    vm_all = np.mean(
        np.array(_lst),
        axis=0,
    )
    dict_vm[pairs] = vm_all
    r_vm = vm_all[:, 0]



hs = twopcf_halostreaming(
    log10M_bincentre=log10M_bincentre,
    dlog10M=log10M_bincentre[1] - log10M_bincentre[0],
    dndlog10M=dndlog10M,
    r_xiR=r_xiR,
    xiR_cc=dict_xi['cc'],
    xiR_cs=dict_xi['cs'],
    xiR_ss=dict_xi['ss'],
    r_vm=r_vm,
    vm_cc=dict_vm['cc'],
    vm_cs=dict_vm['cs'],
    vm_ss=dict_vm['ss'],
    redshift=redshift,
    cosmology=cosmology,
    HOD_params=Galaxy(
        logM_min = 13.62,
        sigma_logM = 0.6915,
        kappa = 0.51,
        logM1 = 14.42,
        alpha = 0.9168,
        M_cut = 10 ** 12.26,
        M_sat = 10 ** 14.87,
        concentration_bias = 1.0,
        v_bias_centrals = 0.25,
        v_bias_satellites = 0.88,
        B_cen = 0.,
        B_sat = 0.,
    ),
    sat_profile=FixedCosmologyNFW(
        redshift=redshift,
        cosmology=cosmology,
        mdef="vir",
    ),
    mdef="vir",
    sigma_vir_halo=None,
    conc=c_Prada(redshift, 10**log10M_bincentre,),
    cen_vel_bias=True,
    verbose=False,
)
# hs.get_xiS_2h_cc(
#     s_binedge,
#     mu_binedge,
# )
# hs.get_xiS_2h_cs(
#     s_binedge,
#     mu_binedge,
# )
# hs.get_xiS_1h_cs(
#     s_binedge,
#     mu_binedge,
# )
# hs.get_xiS_2h_ss(
#     s_binedge,
#     mu_binedge,
# )
# hs.get_xiS_1h_ss(
#     s_binedge,
#     mu_binedge,
# )
hs(
    s_binedge,
    mu_binedge,
)


with h5py.File(
    pp.output_path,
    'w',
) as file_xiStheory:
    file_xiStheory.create_dataset(
        's_bincentre',
        data=s_bincentre,
    )

    file_xiStheory.create_dataset(
        'xiS0_gg',
        data=hs.xiS0_gg,
    )
    file_xiStheory.create_dataset(
        'xiS2_gg',
        data=hs.xiS2_gg,
    )

    file_xiStheory.create_dataset(
        'xiS0_2h_cc',
        data=hs.xiS0_2h_cc,
    )
    file_xiStheory.create_dataset(
        'xiS2_2h_cc',
        data=hs.xiS2_2h_cc,
    )
    file_xiStheory.create_dataset(
        'xiS4_2h_cc',
        data=hs.xiS4_2h_cc,
    )


    file_xiStheory.create_dataset(
        'xiS0_2h_cs',
        data=hs.xiS0_2h_cs,
    )
    file_xiStheory.create_dataset(
        'xiS2_2h_cs',
        data=hs.xiS2_2h_cs,
    )
    file_xiStheory.create_dataset(
        'xiS4_2h_cs',
        data=hs.xiS4_2h_cs,
    )

    file_xiStheory.create_dataset(
        'xiS0_1h_cs',
        data=hs.xiS0_1h_cs,
    )
    file_xiStheory.create_dataset(
        'xiS2_1h_cs',
        data=hs.xiS2_1h_cs,
    )
    file_xiStheory.create_dataset(
        'xiS4_1h_cs',
        data=hs.xiS4_1h_cs,
    )


    file_xiStheory.create_dataset(
        'xiS0_2h_ss',
        data=hs.xiS0_2h_ss,
    )
    file_xiStheory.create_dataset(
        'xiS2_2h_ss',
        data=hs.xiS2_2h_ss,
    )
    file_xiStheory.create_dataset(
        'xiS4_2h_ss',
        data=hs.xiS4_2h_ss,
    )


    file_xiStheory.create_dataset(
        'xiS0_1h_ss',
        data=hs.xiS0_1h_ss,
    )
    file_xiStheory.create_dataset(
        'xiS2_1h_ss',
        data=hs.xiS2_1h_ss,
    )
    file_xiStheory.create_dataset(
        'xiS4_1h_ss',
        data=hs.xiS4_1h_ss,
    )

    file_xiStheory.attrs['n_g'] = hs.n_g
    file_xiStheory.attrs['n_c'] = hs.n_c
    file_xiStheory.attrs['n_s'] = hs.n_s
    file_xiStheory.attrs['f_c'] = hs.f_c
    file_xiStheory.attrs['f_s'] = hs.f_s


file_xiR.close()
file_vm.close()

