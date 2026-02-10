import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.optimize import curve_fit

from colossus.cosmology import cosmology as cosmology_colossus
from colossus.lss import peaks

# --- Constants ---
# Tinker et al. (2010) fixed parameters for Delta=200m
TINKER_FIXED_PARAMS = {"a": 0.132, "b": 1.5, "c": 2.4}
DELTA_C = 1.686


def configure_plotting() -> None:
    """Configures Matplotlib settings based on user preferences."""
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 12})
    try:
        matplotlib.rcParams["text.usetex"] = True
        matplotlib.rcParams["text.latex.preamble"] = "\n".join(
            [r"\usepackage{amsmath}", r"\usepackage{physics}"]
        )
        params = {
            "xtick.top": True,
            "ytick.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
        plt.rcParams.update(params)
    except Exception:
        print("LaTeX not found, continuing without it.")


def get_peak_height(
    mass: np.ndarray,
    redshift: float = 0.25,
    cosmo_params: Optional[Tuple[float, float, float, float]] = None,
) -> np.ndarray:
    """
    Calculates peak height (nu) for given halo masses.

    Parameters
    ----------
    mass : np.ndarray
        Halo mass in M_sun/h.
    redshift : float
        Redshift of the snapshot.
    cosmo_params : tuple, optional
        (Om0, h, S8, ns). If None, uses Planck18.
    """
    if cosmo_params is not None:
        Om0, h, S8, ns = cosmo_params
        my_cosmo = {
            "flat": True,
            "H0": h * 100.0,
            "Om0": Om0,
            "Ob0": 0.043,  # Fixed baryon density from original script
            "sigma8": S8,
            "ns": ns,
        }
        cosmo = cosmology_colossus.setCosmology("my_cosmo", **my_cosmo)
    else:
        cosmo = cosmology_colossus.setCosmology("planck18")

    R = peaks.lagrangianR(mass)
    sigma = cosmo.sigma(R, z=redshift)
    return DELTA_C / sigma


def tinker10_bias(
    nu: np.ndarray, A: float, a: float, B: float, b: float, C: float, c: float
) -> np.ndarray:
    """Tinker et al. (2010) bias equation."""
    # Note: strictly usually depends on Delta, but often fit as constant
    # y = np.log10(200)

    term1 = 1.0 - A * (nu**a) / (nu**a + DELTA_C**a)
    term2 = B * (nu**b)
    term3 = C * (nu**c)
    return term1 + term2 + term3


def fit_tinker_halo_bias(
    measured_mass: np.ndarray,
    measured_bias: np.ndarray,
    target_mass: np.ndarray,
    redshift: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits Tinker parameters to measured data and extrapolates to target masses.

    Returns
    -------
    bias_extrapolated : np.ndarray
        Bias values at the target masses.
    popt : np.ndarray
        Fitted parameters [A, B, C].
    """
    # Combine masses to calculate Nu for the whole range at once
    all_mass = np.concatenate([measured_mass, target_mass])
    nu_all = get_peak_height(all_mass, redshift=redshift)

    # Split back into measured and target
    n_meas = len(measured_mass)
    nu_meas = nu_all[:n_meas]
    nu_target = nu_all[n_meas:]

    # Wrapper to freeze shape parameters (a, b, c) and fit normalizations (A, B, C)
    def constrained_tinker(nu, A, B, C):
        return tinker10_bias(
            nu,
            A,
            TINKER_FIXED_PARAMS["a"],
            B,
            TINKER_FIXED_PARAMS["b"],
            C,
            TINKER_FIXED_PARAMS["c"],
        )

    # Initial guess for A, B, C (from Tinker 2010 Table 2 approx)
    p0 = [1.0, 0.183, 0.265]

    popt, _ = curve_fit(constrained_tinker, nu_meas, measured_bias, p0=p0, maxfev=5000)

    print(f"Fitted Parameters: A={popt[0]:.3f}, B={popt[1]:.3f}, C={popt[2]:.3f}")

    bias_extrapolated = constrained_tinker(nu_target, *popt)
    return bias_extrapolated, popt


def load_data(filepath: Path) -> Tuple[Dict[str, Any], np.ndarray]:
    """Loads bias data and simulation parameters from HDF5."""
    data_bundle = {}

    with h5py.File(filepath, "r") as f:
        # Load simulation data groups
        for key in f:
            data_bundle[key] = {}
            for subkey in f[key]:
                if "R_bias" not in subkey:
                    data_bundle[key][subkey] = f[key][subkey][:]

        # Load attributes and binning
        logM_fit_min = f.attrs["logM_fit_min"]
        logM_fit_max = f.attrs["logM_fit_max"]
        logM_bins = data_bundle["N100"]["logM_bins"]

    mask_reliable = (logM_bins >= logM_fit_min) & (logM_bins <= logM_fit_max)
    return data_bundle, mask_reliable


def plot_results(
    data_bundle: Dict[str, Any],
    mask_reliable: np.ndarray,
    logM_meas: np.ndarray,
    fit_params: np.ndarray,
    output_path: Path,
) -> None:
    """Generates and saves the bias comparison plot."""
    logM_bins = data_bundle["N100"]["logM_bins"]
    A, B, C = fit_params

    # Calculate Nu for the plotting lines
    nu_plot = get_peak_height(10**logM_bins)

    # Calculate Tinker curve using fitted params and fixed shape params
    tinker_curve = tinker10_bias(
        nu_plot,
        A,
        TINKER_FIXED_PARAMS["a"],
        B,
        TINKER_FIXED_PARAMS["b"],
        C,
        TINKER_FIXED_PARAMS["c"],
    )

    # Mask for extrapolation vs fit region
    mask_extrap = logM_bins < logM_meas[-1] * 0.999

    # --- Plotting Configuration (Preserved) ---
    fig = plt.figure(figsize=(4.5, 4.5))
    gs = gridspec.GridSpec(1, 1)
    ax0 = plt.subplot(gs[0])

    # 1. N100 Simulation Data
    ax0.plot(
        logM_bins,
        np.sqrt(np.diag(data_bundle["N100"]["B12_sim"])),
        label=r"$\mathrm{True}\ (N_{\mathrm{box}}=100)$",
        color="r",
        lw=0,
        marker=".",
        markersize=6,
        zorder=99,
    )

    # 2. N5 Simulation Data (Reliable region)
    ax0.plot(
        logM_meas,
        np.sqrt(np.diag(data_bundle["N5"]["B12_sim"]))[mask_reliable],
        label=r"$\mathrm{True}\ (N_{\mathrm{box}}=5)$",
        color="purple",
        lw=0,
        marker="*",
        markersize=8,
    )

    # 3. Tinker Extrapolation (Dashed)
    ax0.plot(
        logM_bins[~mask_extrap],
        tinker_curve[~mask_extrap],
        label=r"$\mathrm{Tinker\ Extrapolation}$",
        color="k",
        lw=1.0,
        linestyle="--",
    )

    # 4. Tinker Fit (Solid)
    ax0.plot(
        logM_bins[mask_reliable],
        tinker_curve[mask_reliable],
        color="k",
        lw=1.6,
    )

    # 5. Fit Limit Line
    ax0.axvline(
        logM_meas[-1],
        color="gray",
        linestyle="--",
        lw=0.96,
        label=r"$\mathrm{Fit\ Limit}$",
    )

    ax0.set_xlabel(r"$\log_{10} [M / (h^{-1} M_\odot)]$", fontsize=15)
    ax0.set_ylabel(r"$b(M)$", fontsize=15)
    ax0.legend(frameon=False, fontsize=11.2)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
    print(f"Figure saved to {output_path}")


def main():
    # Setup
    configure_plotting()
    data_path = Path("./data/xi_hh_fiducial_bias_data_z0.25.hdf5")

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    # 1. Load Data
    data_bundle, mask_reliable = load_data(data_path)

    # 2. Prepare Data for Fitting
    # reliable low-mass measurements
    logM_meas = data_bundle["N5"]["logM_bins"][mask_reliable]
    M_meas = 10**logM_meas
    bias_meas = np.sqrt(np.diag(data_bundle["N5"]["B12_sim"]))[mask_reliable]

    # The high-mass target to extrapolate to
    logM_target = np.arange(logM_meas[-1] + 0.1, 15.0, 0.1)
    M_target = 10**logM_target

    # 3. Fit Model
    _, params = fit_tinker_halo_bias(
        M_meas,
        bias_meas,
        M_target,
    )

    # 4. Plot Results
    output_file = Path("./figs/bias_tinker_ext.pdf")
    plot_results(data_bundle, mask_reliable, logM_meas, params, output_file)


if __name__ == "__main__":
    main()
