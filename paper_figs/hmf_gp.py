r"""
Short script to test and visualize the performance of the HMF Gaussian Process Emulator.
Saves a plot comparing simulated and emulated cumulative HMFs.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import gridspec
from cycler import cycler

# Local module imports
from freyja.emulators import HMFEmulator

# ----------------------------------------------------------------------
# 0. CONFIGURATION & PATHS
# ----------------------------------------------------------------------
REDSHIFT = 0.25
N_MODELS = 64
N_PARAMS = 4  # [Om0, h, S8, ns]

# Resolve paths relative to this script
MODULE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = MODULE_DIR / f"data/cHMF_reformat_LCDM_wide64_z{REDSHIFT:.2f}.hdf5"
CHMF_SAVE_PATH = Path(f"figs/cHMF_comparison_z{REDSHIFT:.2f}.pdf")
CHMF_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Plotting Style
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.size": 12,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{physics}",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)
CUSTOM_CYCLER = cycler(
    color=[
        "black",
        "orange",
        "mediumblue",
        "mediumseagreen",
        "purple",
        "red",
        "green",
        "firebrick",
    ]
)

# ----------------------------------------------------------------------
# 1. LOAD DATA & RUN EMULATOR
# ----------------------------------------------------------------------
yumi = HMFEmulator(z=REDSHIFT)

with h5py.File(DATA_PATH, "r") as f:
    # Setup mass bins based on first model
    mask_mass = f["model1"]["log10M_binleftedges_sim"][...] > 12.5
    log10M_edges = f["model1"]["log10M_binleftedges_sim"][mask_mass]

    # Pre-allocate arrays
    X_test = np.zeros((N_MODELS, N_PARAMS))
    cHMF_sim = np.zeros((N_MODELS, len(log10M_edges)))
    cHMF_err_sim = np.zeros((N_MODELS, len(log10M_edges)))
    cHMF_emu = np.zeros((N_MODELS, len(log10M_edges)))

    # Data extraction loop (models 57-62 as per range)
    for i in range(57, 63):
        idx = i - 1
        grp = f[f"model{i}"]

        # Load params
        X_test[idx] = [
            grp.attrs["Om0"],
            grp.attrs["h"],
            grp.attrs["S8"],
            grp.attrs["ns"],
        ]

        cHMF_sim[idx] = grp["cHMF_sim"][mask_mass]
        cHMF_err_sim[idx] = grp["cHMF_err_sim"][mask_mass]
        cHMF_emu[idx] = yumi.cumulative_hmf(
            cosmo_params=X_test[idx], log10M_binleftedges=log10M_edges
        )


# Calculations
cHMF_reldiff = (cHMF_emu / cHMF_sim - 1.0) * 100.0
cHMF_reldiff_err = (cHMF_err_sim / cHMF_sim) * 100.0


# ----------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------
def cHMF_plot():
    fig = plt.figure(figsize=(4, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1], hspace=0)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)

    for ax in [ax0, ax1]:
        ax.set_xscale("log")
        ax.set_prop_cycle(CUSTOM_CYCLER)

    ax0.set_yscale("log")

    # Masks for visual distinction (Emulator vs Extrapolation)
    mask_main = log10M_edges <= 14.8
    mask_tail = log10M_edges > 14.7

    # Plotting Loop
    for i in range(57, 63):  # the test set models
        idx = i - 1
        label_sim = r"$\mathrm{Simulation}$" if i == 57 else None
        label_emu = r"$\mathrm{Emulator}$" if i == 57 else None
        label_fit = r"$\mathrm{Fitting}$" if i == 57 else None

        # Top Panel: Absolute cHMF
        eb = ax0.errorbar(
            10**log10M_edges,
            cHMF_sim[idx],
            yerr=cHMF_err_sim[idx],
            fmt=".",
            ms=5,
            elinewidth=0.7,
            label=label_sim,
        )
        color = eb[0].get_color()

        ax0.plot(
            10 ** log10M_edges[mask_main],
            cHMF_emu[idx, mask_main],
            color=color,
            lw=1.0,
            label=label_emu,
        )
        ax0.plot(
            10 ** log10M_edges[mask_tail],
            cHMF_emu[idx, mask_tail],
            color=color,
            lw=0.9,
            ls="--",
            label=label_fit,
        )

        # Bottom Panel: Residuals
        ax1.errorbar(
            10**log10M_edges,
            cHMF_reldiff[idx],
            yerr=cHMF_reldiff_err[idx],
            fmt=".",
            ms=4,
            elinewidth=0.7,
            color=color,
        )

    # Aesthetics
    ax1.fill_between(
        10**log10M_edges, -1, 1, color="gray", edgecolor="white", alpha=0.3, zorder=-1
    )
    ax1.axhline(0.0, color="k", ls="-", lw=0.7)

    ax0.set_xlim(10**12.8, 10 ** log10M_edges.max())
    ax1.set_xlim(10**12.8, 10 ** log10M_edges.max())
    ax0.set_ylim(9e-10, 2e-3)
    ax1.set_ylim(-2.2, 2.2)

    ax1.set_xlabel(r"$M_{\mathrm{vir}} \, [M_\odot/h]$", fontsize=16)
    ax0.set_ylabel(r"$n(>M) \, [(\mathrm{Mpc} / h)^{-3}]$", fontsize=13)
    ax1.set_ylabel(r"$\varepsilon (\%)$", fontsize=16)

    ax0.legend(loc="lower left", frameon=False, fontsize=12)

    plt.savefig(CHMF_SAVE_PATH, bbox_inches="tight", pad_inches=0.05)
    print(f"Plot saved to: {CHMF_SAVE_PATH}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    cHMF_plot()
