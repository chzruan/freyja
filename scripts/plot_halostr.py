import numpy as np 
import h5py
import pandas as pd
import sys
import argparse
from pathlib import Path
import yaml

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

from pycorr import TwoPointCorrelationFunction
from freyja.utils.config import ConfigHOD, ConfigMeasure
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




file_data = h5py.File(
    snakemake.input.data,
    'r',
)
file_theory = h5py.File(
    snakemake.input.theory,
    'r',
)
f_s = file_theory.attrs['f_s']
f_c = file_theory.attrs['f_c']
s_max = 60.0


fig = plt.figure(figsize=(10, 4))
gs = gridspec.GridSpec(
    1, 2,
    wspace=0.23,
    width_ratios=[1, 1],
)
ax00 = plt.subplot(gs[0, 0])
ax01 = plt.subplot(gs[0, 1])



pairs = 'gg'
_lst = []
for ibox in range(1, 100+1):
    group_data = file_data[f'box{ibox}']
    mask = (group_data[f's_bincentre_{pairs}'][...] < s_max)
    ss_sim = group_data[f's_bincentre_{pairs}'][mask]
    xx = np.vstack((
        group_data[f'xiS0_{pairs}'][mask],
        group_data[f'xiS2_{pairs}'][mask],
        group_data[f'xiS2_{pairs}'][mask],
    )).T 
    _lst.append(xx)
xiS024_sim_all = np.array(_lst)
xiS024_sim_mean = np.mean(
    xiS024_sim_all,
    axis=0,
)
xiS024_sim_stddev = np.std(
    xiS024_sim_all,
    axis=0,
)
ax00.errorbar(
    ss_sim,
    ss_sim**2 * xiS024_sim_mean[:, 0],
    yerr=ss_sim**2 * xiS024_sim_stddev[:, 0],
    lw=0,
    capsize=2,
    marker='s',
    markersize=2,
    elinewidth=0.5,
    color='k',
    label=r'$\mathrm{ ' + pairs + r',\ data}$',
)
ax01.errorbar(
    ss_sim,
    ss_sim**2 * xiS024_sim_mean[:, 1],
    yerr=ss_sim**2 * xiS024_sim_stddev[:, 1],
    lw=0,
    capsize=2,
    marker='s',
    markersize=2,
    elinewidth=0.5,
    color='k',
    label=r'$\mathrm{ ' + pairs + r',\ data}$',
)

mask = file_theory['s_bincentre'][...] < s_max
ss_theory = file_theory['s_bincentre'][mask]
ax00.plot(
    ss_theory,
    ss_theory**2 * file_theory[f'xiS0_{pairs}'][mask],
    lw=1.5,
    color='k',
    label=r'$\mathrm{ ' + pairs + r',\ theory}$',

)
ax01.plot(
    ss_theory,
    ss_theory**2 * file_theory[f'xiS2_{pairs}'][mask],
    lw=1.5,
    color='k',
    label=r'$\mathrm{ ' + pairs + r',\ theory}$',
)





pairs = 'cc'
_lst = []
for ibox in range(1, 100+1):
    group_data = file_data[f'box{ibox}']
    mask = (group_data[f's_bincentre_{pairs}'][...] < s_max)
    ss_sim = group_data[f's_bincentre_{pairs}'][mask]
    xx = np.vstack((
        group_data[f'xiS0_{pairs}'][mask],
        group_data[f'xiS2_{pairs}'][mask],
        group_data[f'xiS2_{pairs}'][mask],
    )).T 
    _lst.append(xx)
xiS024_sim_all = np.array(_lst)
xiS024_sim_mean = np.mean(
    xiS024_sim_all,
    axis=0,
)
xiS024_sim_stddev = np.std(
    xiS024_sim_all,
    axis=0,
)
ax00.errorbar(
    ss_sim,
    ss_sim**2 * xiS024_sim_mean[:, 0] * f_c**2,
    yerr=ss_sim**2 * xiS024_sim_stddev[:, 0] * f_c**2,
    lw=0,
    marker='o',
    markersize=1.5,
    elinewidth=0.0,
    color='r',
)
ax01.errorbar(
    ss_sim,
    ss_sim**2 * xiS024_sim_mean[:, 1] * f_c**2,
    yerr=ss_sim**2 * xiS024_sim_stddev[:, 1] * f_c**2,
    lw=0,
    marker='o',
    markersize=1.5,
    elinewidth=0.0,
    color='r',
)


mask = file_theory['s_bincentre'][...] < s_max
ss_theory = file_theory['s_bincentre'][mask]
ax00.plot(
    ss_theory,
    ss_theory**2 * file_theory[f'xiS0_2h_{pairs}'][mask] * f_c**2,
    lw=0.7,
    color='r',
    label=r'$\mathrm{ ' + pairs + r', \ 2h}$',
)
ax01.plot(
    ss_theory,
    ss_theory**2 * file_theory[f'xiS2_2h_{pairs}'][mask] * f_c**2,
    lw=0.7,
    color='r',
    label=r'$\mathrm{ ' + pairs + r', \ 2h}$',
)







pairs = 'cs'
_lst = []
for ibox in range(1, 100+1):
    group_data = file_data[f'box{ibox}']
    mask = (group_data[f's_bincentre_{pairs}'][...] < s_max)
    ss_sim = group_data[f's_bincentre_{pairs}'][mask]
    xx = np.vstack((
        group_data[f'xiS0_{pairs}'][mask],
        group_data[f'xiS2_{pairs}'][mask],
        group_data[f'xiS2_{pairs}'][mask],
    )).T 
    _lst.append(xx)
xiS024_sim_all = np.array(_lst)
xiS024_sim_mean = np.mean(
    xiS024_sim_all,
    axis=0,
)
xiS024_sim_stddev = np.std(
    xiS024_sim_all,
    axis=0,
)
ax00.errorbar(
    ss_sim,
    ss_sim**2 * xiS024_sim_mean[:, 0] * 2 * f_c * f_s,
    yerr=ss_sim**2 * xiS024_sim_stddev[:, 0] * 2 * f_c * f_s,
    lw=0,
    marker='o',
    markersize=1.5,
    elinewidth=0.0,
    color='mediumseagreen',
)
ax01.errorbar(
    ss_sim,
    ss_sim**2 * xiS024_sim_mean[:, 1] * 2 * f_c * f_s,
    yerr=ss_sim**2 * xiS024_sim_stddev[:, 1] * 2 * f_c * f_s,
    lw=0,
    marker='o',
    markersize=1.5,
    elinewidth=0.0,
    color='mediumseagreen',
)


mask = file_theory['s_bincentre'][...] < s_max
ss_theory = file_theory['s_bincentre'][mask]
# ax00.plot(
#     ss_theory,
#     ss_theory**2 * (file_theory[f'xiS0_1h_{pairs}'][mask] + file_theory[f'xiS0_2h_{pairs}'][mask]),
#     lw=1.0,
# )
# ax01.plot(
#     ss_theory,
#     ss_theory**2 * (file_theory[f'xiS2_1h_{pairs}'][mask] + file_theory[f'xiS2_2h_{pairs}'][mask]),
#     lw=1.0,
# )

ax00.plot(
    ss_theory,
    ss_theory**2 * file_theory[f'xiS0_1h_{pairs}'][mask] * 2 * f_c * f_s,
    lw=1.0, linestyle='--',
    color='mediumseagreen',
    label=r'$\mathrm{ ' + pairs + r', \ 1h}$',
)
ax01.plot(
    ss_theory,
    ss_theory**2 * file_theory[f'xiS2_1h_{pairs}'][mask] * 2 * f_c * f_s,
    lw=1.0, linestyle='--',
    color='mediumseagreen',
    label=r'$\mathrm{ ' + pairs + r', \ 1h}$',
)
ax00.plot(
    ss_theory,
    ss_theory**2 * file_theory[f'xiS0_2h_{pairs}'][mask] * 2 * f_c * f_s,
    lw=1.0,
    color='mediumseagreen',
    label=r'$\mathrm{ ' + pairs + r', \ 2h}$',
)
ax01.plot(
    ss_theory,
    ss_theory**2 * file_theory[f'xiS2_2h_{pairs}'][mask] * 2 * f_c * f_s,
    lw=1.0,
    color='mediumseagreen',
    label=r'$\mathrm{ ' + pairs + r', \ 2h}$',
)













pairs = 'ss'
_lst = []
for ibox in range(1, 100+1):
    group_data = file_data[f'box{ibox}']
    mask = (group_data[f's_bincentre_{pairs}'][...] < s_max)
    ss_sim = group_data[f's_bincentre_{pairs}'][mask]
    xx = np.vstack((
        group_data[f'xiS0_{pairs}'][mask],
        group_data[f'xiS2_{pairs}'][mask],
        group_data[f'xiS2_{pairs}'][mask],
    )).T 
    _lst.append(xx)
xiS024_sim_all = np.array(_lst)
xiS024_sim_mean = np.mean(
    xiS024_sim_all,
    axis=0,
)
xiS024_sim_stddev = np.std(
    xiS024_sim_all,
    axis=0,
)
ax00.errorbar(
    ss_sim,
    ss_sim**2 * xiS024_sim_mean[:, 0] * f_s**2,
    yerr=ss_sim**2 * xiS024_sim_stddev[:, 0] * f_s**2,
    lw=0,
    marker='o',
    markersize=1.5,
    elinewidth=0.0,
    color='mediumblue',
)
ax01.errorbar(
    ss_sim,
    ss_sim**2 * xiS024_sim_mean[:, 1] * f_s**2,
    yerr=ss_sim**2 * xiS024_sim_stddev[:, 1] * f_s**2,
    lw=0,
    marker='o',
    markersize=1.5,
    elinewidth=0.0,
    color='mediumblue',
)


mask = file_theory['s_bincentre'][...] < s_max
ss_theory = file_theory['s_bincentre'][mask]
# ax00.plot(
#     ss_theory,
#     ss_theory**2 * (file_theory[f'xiS0_1h_{pairs}'][mask] + file_theory[f'xiS0_2h_{pairs}'][mask]),
#     lw=1.0,
# )
# ax01.plot(
#     ss_theory,
#     ss_theory**2 * (file_theory[f'xiS2_1h_{pairs}'][mask] + file_theory[f'xiS2_2h_{pairs}'][mask]),
#     lw=1.0,
# )

ax00.plot(
    ss_theory,
    ss_theory**2 * file_theory[f'xiS0_1h_{pairs}'][mask] * f_s**2,
    lw=1.0, linestyle='--',
    color='mediumblue',
    label=r'$\mathrm{ ' + pairs + r', \ 1h}$',
)
ax01.plot(
    ss_theory,
    ss_theory**2 * file_theory[f'xiS2_1h_{pairs}'][mask] * f_s**2,
    lw=1.0, linestyle='--',
    color='mediumblue',
    label=r'$\mathrm{ ' + pairs + r', \ 1h}$',
)
ax00.plot(
    ss_theory,
    ss_theory**2 * file_theory[f'xiS0_2h_{pairs}'][mask] * f_s**2,
    lw=1.0,
    color='mediumblue',
    label=r'$\mathrm{ ' + pairs + r', \ 2h}$',
)
ax01.plot(
    ss_theory,
    ss_theory**2 * file_theory[f'xiS2_2h_{pairs}'][mask] * f_s**2,
    lw=1.0,
    color='mediumblue',
    label=r'$\mathrm{ ' + pairs + r', \ 2h}$',
)











ax00.set_xlim([0, s_max])
ax01.set_xlim([0, s_max])

ax00.set_xlabel(
    r'$s / (h^{-1}\mathrm{Mpc})$',
    fontsize=16,
)
ax01.set_xlabel(
    r'$s / (h^{-1}\mathrm{Mpc})$',
    fontsize=16,
)
ax00.set_ylabel(
    r'$s^2 \xi^{\mathrm{S}}_0(s)$',
    fontsize=16,
)
ax01.set_ylabel(
    r'$s^2 \xi^{\mathrm{S}}_2(s)$',
    fontsize=16,
)
ax01.legend(
    frameon=False, 
    ncols=2,
    fontsize=12,
)

plt.savefig(
    snakemake.output.pdf,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.savefig(
    snakemake.output.png,
    dpi=400,
    bbox_inches="tight",
    pad_inches=0.05,
)


file_data.close()
file_theory.close()
