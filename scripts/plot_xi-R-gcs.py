import numpy as np 
import h5py
import sys
import argparse
from pathlib import Path
import yaml
from scipy.interpolate import InterpolatedUnivariateSpline as ius

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{physics}'
params = {'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in'}
plt.rcParams.update(params)



from freyja.utils.config import ConfigHOD
from pysam import VariantFile



parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path", 
    type=Path,
    default="config_GLAMx100.yaml",
)
p = parser.parse_args()
with open(p.config_path, "r") as f:
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



file_xiR = h5py.File(
    snakemake.input[0],
    'r',
)

fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2,)

ax0 = plt.subplot(gs[0, 0])
ax0.set_xscale('log')

pairs = 'gg'
_lst = []
for ibox in range(1, 100+1):
    group_xiR = file_xiR[f'box{ibox}']
    rr = group_xiR[f'r_bincentre_{pairs}'][...]
    _lst.append(group_xiR[f'xiR_{pairs}'][...])
xiR_all = np.array(_lst)
xiR_mean = np.mean(
    xiR_all,
    axis=0,
)
xiR_stddev = np.std(
    xiR_all,
    axis=0,
)
ax0.errorbar(
    rr,
    rr**2 * xiR_mean,
    yerr = rr**2 * xiR_stddev,
    lw=1.0,
)
ax0.set_aspect(1.0 / ax0.get_data_ratio())
ax0.set_xlabel(r"$r / (h^{-1}\mathrm{Mpc})$")
ax0.set_ylabel(r"$r^2 \xi^{\mathrm{R}}_{ " + pairs + "}$")
plt.savefig(
    snakemake.output[0],
    bbox_inches="tight",
    pad_inches=0.05,
)
file_xiR.close()

