import numpy as np
import sys
import h5py
import argparse
from pathlib import Path

from freyja.emulators.halo_linear_bias import HaloLinearBiasEmulator

HP = {
    # Data Constraints
    "logM_cut_max": 13.9,  # Maximum log mass
    "logM_cut_min": 12.4,
    "r_cut_max": 75.0,  # Maximum scale in fitting halo bias in Mpc/h
    "r_cut_min": 35.0,  # Minimum scale
    "n_models": 59,  # Number of simulation models in the training set
    "loss_epsilon": 1e-6,  # Stability floor for variance
}

if __name__ == "__main__":
    akane = HaloLinearBiasEmulator()
    for imodel in range(30, 40):
        akane.compare_model_prediction(imodel)
