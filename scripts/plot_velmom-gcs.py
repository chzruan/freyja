import numpy as np 
import h5py
import pandas as pd
import sys
import argparse
from pathlib import Path
import yaml
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from pycorr import TwoPointCorrelationFunction
from freyja.utils.config import ConfigHOD, ConfigMeasure

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



file_vm = h5py.File(
    snakemake.input[0],
    "r",
)
moment_to_plot = 'm10' # c20, c02, ..., c04
fig = plt.figure(figsize=(17, 5))
gs = gridspec.GridSpec(1, 3,)
ax0 = plt.subplot(gs[0])
ax0.set_xscale("log")
ax1 = plt.subplot(gs[1])
ax1.set_xscale("log")
ax2 = plt.subplot(gs[2])
ax2.set_xscale("log")

_dict_axes = {
    'cc': ax0,
    'cs': ax1,
    'ss': ax2,
}
for pairs in ['cc', 'cs', 'ss']:
    _lst = []
    for ibox in range(1, 100+1):
        group_vm = file_vm[f'box{ibox}']
        rr = group_vm[f"r_velmom_{pairs}"][...]
        _lst.append(
            np.vstack((
                group_vm[f"m10_{pairs}"][...],
                group_vm[f"c20_{pairs}"][...],
                group_vm[f"c02_{pairs}"][...],
                group_vm[f"c12_{pairs}"][...],
                group_vm[f"c30_{pairs}"][...],
                group_vm[f"c40_{pairs}"][...],
                group_vm[f"c04_{pairs}"][...],
                group_vm[f"c22_{pairs}"][...],
            )).T
        )
    vm_all = np.array(_lst)
    vm_mean = np.mean(
        vm_all,
        axis=0,
    )
    vm_stddev = np.std(
        vm_all,
        axis=0,
    )
    ax = _dict_axes[pairs]
    ax.errorbar(
        rr,
        vm_mean[:, 1],
        yerr=vm_stddev[:, 1],
        lw=0.5,
        elinewidth=0.7,
        capsize=2,
        marker='o',
        markersize=2,
    )
    ax.set_xlabel(r'$r / (h^{-1}\mathrm{Mpc})$', fontsize=16,)
    ax.set_ylabel(r'$' + moment_to_plot + '$-' + pairs, fontsize=16,)
    ax.set_aspect(1.0 / ax.get_data_ratio())
plt.savefig(
    snakemake.output[0],
    bbox_inches="tight",
    pad_inches=0.05,
)

file_vm.close()

