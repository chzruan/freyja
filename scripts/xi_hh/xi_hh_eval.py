import numpy as np
import sys
import h5py
import argparse
from pathlib import Path

from freyja.cosma.xi_hh import load_cosmology_wrapper, load_xihh_data, load_ximm_data
from freyja.emulators import HMFEmulator, HaloBiasEmulator, HaloBetaEmulator


import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams.update({"font.size": 12})
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{physics}"
params = {
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
}
plt.rcParams.update(params)

imodel = 1

# 1. Load full xi_hh data from simulations
rr, logM, xi_hh, xi_sem = load_xihh_data(imodel)
xi_mm = load_ximm_data(rr, imodel)

# 2. Load Emulators
emulator_hmf = HMFEmulator()
emulator_bias = HaloBiasEmulator()
emulator_xihhratio = HaloBetaEmulator()
