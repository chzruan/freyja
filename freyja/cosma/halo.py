import numpy as np
import sys
import h5py
import argparse
from pathlib import Path


def load_BDM_halo_catalog(CatalogFullPath):
    x, y, z, vx, vy, vz, Mtot = np.loadtxt(
        CatalogFullPath, unpack=True, skiprows=8, usecols=(0, 1, 2, 3, 4, 5, 7)
    )

    return x, y, z, vx, vy, vz, Mtot


def load_catalogue(
    imodel: int = 1,
    ibox: int = 1,
    gravity: str = "LCDM",
    snap_num: int = 137,
):
    r"""
    Load halo position, velocity, and mass data from catalog files.

    Parameters
    ----------
    imodel : int (0-64), optional
        Model identifier. If 0, uses DESI_MGx100 simulations.
        If > 0, uses mg_glam DurMun_hmfemu simulations. Default is 1.
    ibox : int, optional
        Box/run identifier for the simulation. Default is 1.
    gravity : str, optional
        Gravitational theory model. Options include "LCDM" and "fRn1"
        models. Default is "LCDM".
    snap_num : int, optional
        Snapshot number to load. Default is 137 (z = 0.25).

    Returns
    -------
    halo_pos : ndarray
        Halo positions with shape (N_halo, 3) containing [x, y, z] coordinates.
    halo_vel : ndarray
        Halo velocities with shape (N_halo, 3) containing [vx, vy, vz] components.
    halo_mass : ndarray
        Halo masses with shape (N_halo,).

    Notes
    -----
    - For imodel == 0 with LCDM gravity, catalogs are loaded from GR subdirectory.
    - For imodel == 0 with other gravity models, catalogs are loaded from the
      respective gravity model subdirectory.
    - For imodel > 0, catalogs are loaded from the DEGRACE-pilot simulation suite.
    - Catalog files are expected in .DAT format with naming convention
      CatshortV.{snap_num}.{ibox}.DAT.
    """

    if imodel == 0:
        if gravity == "LCDM":
            root_path = Path(
                f"/cosma8/data/dp203/dc-ruan1/DESI_MGx100/data/GR/Run{ibox}/CATALOGS/"
            )
        elif gravity == "fRn1":
            root_path = Path(
                f"/cosma8/data/dp203/dc-ruan1/mg_glam/F5n1_L1024Np2048Ng4096/Run{ibox}/CATALOGS/"
            )
        else:
            root_path = Path(
                f"/cosma8/data/dp203/dc-ruan1/DESI_MGx100/data/{gravity}/Run{ibox}/CATALOGS/"
            )
    else:
        root_path = Path(
            f"/cosma8/data/dp203/dc-ruan1/mg_glam/DurMun_hmfemu_{gravity}_wide_sample_first_64_model{imodel}_L1024Np2048Ng4096/Run{ibox}/CATALOGS/"
        )
    x, y, z, vx, vy, vz, halo_mass = load_BDM_halo_catalog(
        root_path / f"CatshortV.{str(snap_num).zfill(4)}.{str(ibox).zfill(4)}.DAT"
    )
    halo_pos = np.vstack((x, y, z)).T  # shape = (N_halo, 3)
    halo_vel = np.vstack((vx, vy, vz)).T
    return halo_pos, halo_vel, halo_mass
