"""
Utilities for loading halo-halo correlation function measurements and computing the corresponding matter correlation function for DEGRACE cosmological models 1-64. Provides helpers to fetch cosmological parameters, read halo pair-count statistics from HDF5 outputs, and transform averaged matter power spectra into configuration-space correlations using a Hankel transform.

"""

import h5py
import numpy as np
from pathlib import Path
from pyglam.durmun.dataload import CosmologyDurMun
from freyja.utils.pk_to_xi import compute_xi_from_Pk

# --- Default Constants & Paths ---
GRAVITY = "LCDM"
DATAFLAG = "wide_sample_first_64"
REDSHIFT = 0.25
SNAPNUM = 137


def load_cosmology_wrapper(imodel):
    """
    Loads cosmology parameters [Om0, h, S8, ns] for a given model ID (1-64) using pyglam.

    Parameters:
        imodel (int): Model ID (1-64).
    """
    cosmo = CosmologyDurMun.from_run(gravity=GRAVITY, dataflag=DATAFLAG, imodel=imodel)
    return np.array([cosmo.Om0, cosmo.h, cosmo.S8, cosmo.ns])


def load_xihh_data(imodel, gravity=GRAVITY, dataflag=DATAFLAG, redshift=REDSHIFT):
    """
    Loads xi_hh (halo-halo autocorrelation) and r_bins from HDF5.

    Returns:
        r_all (np.ndarray): Radial bins.
        logM_bins (np.ndarray): Mass bins.
        xi_hh (np.ndarray): Correlation function, shape (N_M, N_M, N_r).
        xi_sem (np.ndarray): Standard error of the mean, shape (N_M, N_M, N_r).
    """
    file_path = (
        Path("/cosma8/data/dp203/dc-ruan1/proj_emulator_RSD/work1/DMx64/data/")
        / f"xiR_hh-diffM_{gravity}_{dataflag}_z{redshift:.2f}_model{imodel}.hdf5"
    )

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with h5py.File(file_path, "r") as f:
        # Extract Mass Bins
        logM1_keys = sorted([k for k in f.keys() if k.startswith("logM1_")])
        logM_bins = np.array([float(k.split("_")[1]) for k in logM1_keys])

        # Extract Radial Bins (from the first entry)
        first_key = logM1_keys[0]
        first_sub_key = sorted(f[first_key].keys())[0]
        r_all = f[first_key][first_sub_key]["box1"]["r_bincentres"][:]

        # Initialize Arrays
        N_M = len(logM_bins)
        N_r = len(r_all)
        xi_hh = np.zeros((N_M, N_M, N_r))
        xi_sem = np.zeros((N_M, N_M, N_r))

        # Fill Data
        for i, k1 in enumerate(logM1_keys):
            k2_keys = sorted([k for k in f[k1].keys() if k.startswith("logM2_")])
            for j, k2 in enumerate(k2_keys):
                box_data = []
                for b in range(1, 6):
                    key = f"box{b}"
                    if key in f[k1][k2]:
                        box_data.append(f[k1][k2][key]["xi"][:])

                if box_data:
                    # Calculate Mean and SEM across boxes
                    mean_xi = np.mean(box_data, axis=0)
                    xi_hh[i, j, :] = mean_xi
                    xi_hh[j, i, :] = mean_xi  # Enforce symmetry

                    sem = np.std(box_data, axis=0) / np.sqrt(len(box_data))
                    xi_sem[i, j, :] = sem
                    xi_sem[j, i, :] = sem
                else:
                    xi_hh[i, j, :] = np.nan
                    xi_sem[i, j, :] = np.nan

    return r_all, logM_bins, xi_hh, xi_sem


def load_pkmm_data(
    imodel=1,
    gravity=GRAVITY,
    dataflag=DATAFLAG,
    snapnum=SNAPNUM,
    k_max=12.0,
    return_mean=True,
):
    """
    Loads Matter Power Spectrum (P(k)) for a given model (imodel = 1..64), computes mean P(k) across boxes, and returns k and P_mean arrays.
    If return_mean is False, returns list of P(k) arrays for each box instead of the mean.
    """

    pk_collection = []
    for ibox in range(1, 6):
        file_path = (
            Path("/cosma8/data/dp203/dc-ruan1/mg_glam/")
            / f"DurMun_hmfemu_{gravity}_{dataflag}_model{imodel}_L1024Np2048Ng4096"
            / f"Run{ibox}"
            / f"PowerDM.log.{str(snapnum).zfill(4)}.{str(ibox).zfill(4)}.dat"
        )

        if not file_path.exists():
            print(f"Warning: Power spectrum file missing: {file_path}")
            continue

        k, P = np.loadtxt(
            file_path,
            unpack=True,
            skiprows=3,
            usecols=(1, 3),
        )
        # Filter high-k noise if necessary
        mask = k < k_max
        pk_collection.append(P[mask])
        k_masked = k[mask]

    if not pk_collection:
        raise FileNotFoundError(f"No PowerDM files found for model {imodel}")
    if not return_mean:
        return k_masked, np.array(pk_collection)
    else:
        P_mean = np.mean(pk_collection, axis=0)
        return k_masked, P_mean


def load_linear_pkmm_data(
    imodel=1,
    gravity=GRAVITY,
    dataflag=DATAFLAG,
    k_max=12.0,
):
    """
    Loads Linear Matter Power Spectrum (P(k)) for a given model (imodel = 1..64), calculated from CAMB and used in generating the initial conditions of the simulations.
    """

    file_path = Path(
        f"/cosma8/data/dp203/dc-ruan1/mg_glam/DurMun_hmfemu_{gravity}_{dataflag}_model{imodel}_L1024Np2048Ng4096/PkTable.dat"
    )  # This file contains the linear P(k) used for initial conditions, calculated at z_init=100 and linearly extrapolated to redshift zero.
    k, P = np.loadtxt(
        file_path,
        unpack=True,
        skiprows=5,
    )
    # Filter high-k noise if necessary
    mask = k < k_max
    P = P[mask]
    k_masked = k[mask]

    return k_masked, P


def load_ximm_data(
    r_output, imodel=1, gravity=GRAVITY, dataflag=DATAFLAG, snapnum=SNAPNUM
):
    """
    Loads Matter Power Spectrum (P(k)), computes mean P(k),
    and converts to Correlation Function xi_mm using Hankel transform.
    """
    pk_collection = []

    for ibox in range(1, 6):
        file_path = (
            Path("/cosma8/data/dp203/dc-ruan1/mg_glam/")
            / f"DurMun_hmfemu_{gravity}_{dataflag}_model{imodel}_L1024Np2048Ng4096"
            / f"Run{ibox}"
            / f"PowerDM.log.{str(snapnum).zfill(4)}.{str(ibox).zfill(4)}.dat"
        )

        if not file_path.exists():
            print(f"Warning: Power spectrum file missing: {file_path}")
            continue

        k, P = np.loadtxt(
            file_path,
            unpack=True,
            skiprows=3,
            usecols=(1, 3),
        )
        # Filter high-k noise if necessary
        mask = k < 12.0
        pk_collection.append(P[mask])
        k_masked = k[mask]

    if not pk_collection:
        raise FileNotFoundError(f"No PowerDM files found for model {imodel}")

    P_mean = np.mean(pk_collection, axis=0)

    # Compute xi_mm from P_mean
    xi_mm = compute_xi_from_Pk(
        k_input=k_masked, P_input=P_mean, r_output=r_output, smooth_xi=False
    )
    return xi_mm


def load_xihh_fiducial_data(gravity="GR", redshift=0.25, N_boxes=int(100)):
    """
    Loads xi_hh (halo-halo autocorrelation) and r_bins for the fiducial Planck cosmology.

    Returns:
        r_all (np.ndarray): Radial bins.
        logM_bins (np.ndarray): Mass bins.
        xi_hh (np.ndarray): Correlation function, shape (N_M, N_M, N_r).
        xi_sem (np.ndarray): Standard error of the mean, shape (N_M, N_M, N_r).
    """
    file_path = Path(
        f"/cosma8/data/dp203/dc-ruan1/DESI_MGx100/data/xiR_hh-diffM_{gravity}_z{redshift:.2f}.hdf5"
    )

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with h5py.File(file_path, "r") as f:
        r_bins = f["r_bincentres"][:]
        logM_bins = f["log10M_bincentres"][:]

        _lst = []
        for ibox in range(1, N_boxes + 1):
            _lst.append(f[f"box{ibox}/xiR_hh_diffM"][...])
        xi_all_boxes = np.array(_lst)  # Shape (N_boxes, N_M, N_M, N_r)
        N_M = len(logM_bins)
        N_r = len(r_bins)
        xi_hh_mean = np.mean(xi_all_boxes, axis=0)
        xi_hh_sem = np.std(xi_all_boxes, axis=0) / np.sqrt(N_boxes)

    return r_bins, logM_bins, xi_hh_mean, xi_hh_sem


def load_pkmm_fiducial_data(
    gravity="GR",
    redshift=0.25,
    snapnum=137,
    N_boxes=int(100),
    return_mean=True,
    k_max=12.0,
):
    """
    Loads Matter Power Spectrum (P(k))
    if return_mean is True, computes mean P(k), else returns list of P(k) arrays.

    returns (return_mean=True):
        k (np.ndarray): Wavenumber array.
        P_mean (np.ndarray): Mean power spectrum.
    returns (return_mean=False):
        k (np.ndarray): Wavenumber array.
        pk_collection (list of np.ndarray): List of P(k) arrays for each box.
    """
    pk_collection = []
    for ibox in range(1, N_boxes + 1):
        file_path = (
            Path(f"/cosma8/data/dp203/dc-ruan1/DESI_MGx100/data/{gravity}/")
            / f"Run{ibox}"
            / f"PowerDM.log.{str(snapnum).zfill(4)}.{str(ibox).zfill(4)}.dat"
        )

        if not file_path.exists():
            print(f"Warning: Power spectrum file missing: {file_path}")
            continue

        k, P = np.loadtxt(
            file_path,
            unpack=True,
            skiprows=3,
            usecols=(1, 3),
        )
        # Filter high-k noise if necessary
        mask = k < k_max
        pk_collection.append(P[mask])
        k_masked = k[mask]

    if not pk_collection:
        raise FileNotFoundError(f"No PowerDM files found for snapnum {snapnum}")

    if not return_mean:
        return k_masked, pk_collection
    else:
        P_mean = np.mean(pk_collection, axis=0)
        return k_masked, P_mean


def load_ximm_fiducial_data(
    r_output, gravity="GR", redshift=0.25, snapnum=137, N_boxes=int(100)
):
    """
    Loads Matter Power Spectrum (P(k)), computes mean P(k),
    and converts to Correlation Function xi_mm using Hankel transform.
    """
    pk_collection = []
    for ibox in range(1, N_boxes + 1):
        file_path = (
            Path(f"/cosma8/data/dp203/dc-ruan1/DESI_MGx100/data/{gravity}/")
            / f"Run{ibox}"
            / f"PowerDM.log.{str(snapnum).zfill(4)}.{str(ibox).zfill(4)}.dat"
        )

        if not file_path.exists():
            print(f"Warning: Power spectrum file missing: {file_path}")
            continue

        k, P = np.loadtxt(
            file_path,
            unpack=True,
            skiprows=3,
            usecols=(1, 3),
        )
        # Filter high-k noise if necessary
        mask = k < 12.0
        pk_collection.append(P[mask])
        k_masked = k[mask]

    if not pk_collection:
        raise FileNotFoundError(f"No PowerDM files found for snapnum {snapnum}")

    P_mean = np.mean(pk_collection, axis=0)

    # Compute xi_mm from P_mean
    xi_mm = compute_xi_from_Pk(
        k_input=k_masked, P_input=P_mean, r_output=r_output, smooth_xi=False
    )
    return xi_mm
