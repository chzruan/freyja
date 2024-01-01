import numpy as np
import sys
import os 
current_dir = os.path.dirname(os.path.abspath(__file__))
from scipy import signal

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
from cycler import cycler
custom_cycler = (cycler(color=['mediumblue', 'orange', 'green', 'red', 'purple', 'mediumseagreen', 'magenta', 'black']))
# ax0.set_prop_cycle(custom_cycler)
# ax0.xaxis.set_ticklabels([])
from astropy.cosmology import Planck15
redshift = 0.25
kms_to_Mpch = (1 + redshift) / (100 * Planck15.efunc(redshift))
from freyja.streaming import streaming_model
stsm = streaming_model(model='stsm')
gsm = streaming_model(model='gsm')

r_xiR, xiR, xiR_err = np.loadtxt(
    f'{current_dir}/example_data/xi-R-cc_LOWZ-13.5-13.7_cos0_z0.25.dat',
    unpack=True,
)
xiR = signal.savgol_filter(
    xiR,
    window_length=11, 
    polyorder=3, 
    mode="nearest",
)
velmom_all = np.loadtxt(f'{current_dir}/example_data/velmom_LOWZ-13.5-13.7_cc_cos0_z0.25.csv')
ss, xiS0, xiS2, xiS4, xiS0_err, xiS2_err, xiS4_err  = np.loadtxt(
    f'{current_dir}/example_data/xi-S024-cc_LOWZ-13.5-13.7_cos0_z0.25.dat',
    unpack=True,
)

s_output_binedge = np.geomspace(1.0, 60, 30)
mu_output_binedge = np.linspace(0, 1, 256)
s_stsm, xiS0_stsm, xiS2_stsm, xiS4_stsm = stsm(
    r_velmom=velmom_all[:, 0], 
    m10=velmom_all[:, 1],
    c20=velmom_all[:, 2], 
    c02=velmom_all[:, 3],
    c12=velmom_all[:, 4], 
    c30=velmom_all[:, 5],
    c40=velmom_all[:, 6], 
    c04=velmom_all[:, 7], 
    c22=velmom_all[:, 8],
    r_xiR=r_xiR, 
    xiR=xiR,
    s_output_binedge=s_output_binedge,
    mu_output_binedge=mu_output_binedge,
    kms_to_Mpch=kms_to_Mpch,
    return_multipoles=True,
)
s_gsm, xiS0_gsm, xiS2_gsm, xiS4_gsm = gsm(
    r_velmom=velmom_all[:, 0], 
    m10=velmom_all[:, 1],
    c20=velmom_all[:, 2], 
    c02=velmom_all[:, 3],
    c12=velmom_all[:, 4], 
    c30=velmom_all[:, 5],
    c40=velmom_all[:, 6], 
    c04=velmom_all[:, 7], 
    c22=velmom_all[:, 8],
    r_xiR=r_xiR, 
    xiR=xiR,
    s_output_binedge=s_output_binedge,
    mu_output_binedge=mu_output_binedge,
    kms_to_Mpch=kms_to_Mpch,
    return_multipoles=True,
)


fig = plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(1, 1,)
ax0 = plt.subplot(gs[0])
ax0.set_xscale("log")
pp = ax0.errorbar(
    ss,
    ss**2 * xiS0,
    yerr=ss**2 * xiS0_err,
    lw=0,
    elinewidth=0.7,
    marker='o',
    markersize=2,
)
ax0.plot(
    s_stsm,
    s_stsm**2 * xiS0_stsm,
    lw=1.0,
    color=pp[0].get_color(),
)
ax0.plot(
    s_stsm,
    s_stsm**2 * xiS0_gsm,
    lw=1.0,
    color=pp[0].get_color(),
    linestyle='--',
)

pp = ax0.errorbar(
    ss,
    ss**2 * xiS2,
    yerr=ss**2 * xiS2_err,
    lw=0,
    elinewidth=0.7,
    marker='o',
    markersize=2,
)
ax0.plot(
    s_stsm,
    s_stsm**2 * xiS2_stsm,
    lw=1.0,
    color=pp[0].get_color(),
)
ax0.plot(
    s_gsm,
    s_gsm**2 * xiS2_gsm,
    lw=1.0,
    color=pp[0].get_color(),
    linestyle='--',
)


plt.savefig(
    f"./plots/streaming_wrapper.pdf",
    bbox_inches="tight",
    pad_inches=0.05,
)


