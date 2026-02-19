import numpy as np
import sys
import h5py
import argparse
from pathlib import Path

from freyja.emulators.xi_mm import MatterXiEmulator


if __name__ == "__main__":
    arisa = MatterXiEmulator()
    for imodel in range(60, 65):
        arisa.compare(imodel, save_plot=True)
    for imodel in range(20, 25):
        arisa.compare(imodel, save_plot=True)
