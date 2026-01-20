import numpy as np
import sys
import h5py
import argparse
from pathlib import Path
from freyja.emulators.xi_R_hh_diffM import HaloBetaEmulator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

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
from cycler import cycler

custom_cycler = cycler(
    color=[
        "mediumblue",
        "orange",
        "green",
        "red",
        "purple",
        "mediumseagreen",
        "magenta",
        "black",
    ]
)
# ax0.set_prop_cycle(custom_cycler)
# ax0.xaxis.set_ticklabels([])

halo_beta_emulator = HaloBetaEmulator()
imodel = 1
halo_beta_emulator.compare_model_prediction(
    imodel=imodel,
    label="train",
)
