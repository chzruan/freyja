import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import simpson
from scipy.interpolate import interp1d

# Domain specific imports
from freyja.cosma.xi_hh import load_cosmology_wrapper, load_xihh_data, load_ximm_data
from freyja.emulators import HMFEmulator, HaloBiasEmulator, HaloBetaEmulator

# --- Configuration ---
warnings.filterwarnings(
    "ignore", message="use backend='jax' if desired", module="mcfit"
)

# Plotting style
matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{physics}",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)

# --- Helper Functions ---


def integrate_over_mass(xi_matrix, weights, norm):
    """
    Integrates a 3D correlation matrix (M1, M2, r) over mass bins M1 and M2.
    """
    # Integrate over M2 (axis 1) -> (N_M, N_r)
    integrated_M2 = simpson(xi_matrix * weights[:, :, np.newaxis], axis=1)
    # Integrate over M1 (axis 0) -> (N_r)
    integrated_total = simpson(integrated_M2, axis=0)
    return integrated_total / norm


def smooth_stitch(r, arr_small_scale, arr_large_scale, r_pivot, width=2.0):
    """
    Stitches two correlation function arrays using a tanh transition.
    """
    w = 0.5 * (1 + np.tanh((r - r_pivot) / width))
    return (1 - w) * arr_small_scale + w * arr_large_scale


# --- Main Execution ---

imodel = 60
r_stitch = 30.0  # Pivot scale [Mpc/h]

# 1. Load Data and Cosmology
cosmo = load_cosmology_wrapper(imodel)
rr, logM, xi_hh_diffM, xi_hh_diffM_sem = load_xihh_data(imodel)
# Load Linear Matter Correlation for Large Scale (LS) model (assumes same rr)
xi_mm_lin = load_ximm_data(rr, imodel)

print(f"Mass Range: {logM[0]:.2f} - {logM[-1]:.2f}")

# 2. Initialize Emulators
emulator_hmf = HMFEmulator()
emulator_bias = HaloBiasEmulator()
emulator_xihhratio = HaloBetaEmulator()

# 3. Generate Predictions
dndlog10M = emulator_hmf.get_dndlog10M(cosmo_params=cosmo, log10M_bincentres=logM)

# Non-linear emulator prediction
rr_NL, beta_diffM = emulator_xihhratio.predict_matrix(cosmo, logM)
xi_mm_NL = load_ximm_data(rr_NL, imodel)
xi_hh_diffM_emulated_raw = beta_diffM * xi_mm_NL[np.newaxis, np.newaxis, :]

# Large-scale (Linear) prediction:
# bias_matrix is ALREADY shape (N_M, N_M)
bias_matrix, err_matrix = emulator_bias.predict_matrix(cosmo, logM)

# FIX: Do not use np.outer here. bias_matrix is already 2D.
xi_hh_LS_matrix = bias_matrix[..., np.newaxis] * xi_mm_lin[np.newaxis, np.newaxis, :]
# 4. Compute Weights and Normalization
weights_matrix = np.outer(dndlog10M, dndlog10M)
nh2 = simpson(simpson(weights_matrix, axis=1), axis=0)

# 5. Compute Integrated Correlation Functions
# A. Simulation Data (Total)
xi_hh_sim = integrate_over_mass(xi_hh_diffM, weights_matrix, nh2)

# B. Emulated Small Scales (interpolated to simulation 'rr')
xi_hh_emu_on_NL = integrate_over_mass(xi_hh_diffM_emulated_raw, weights_matrix, nh2)
f_interp = interp1d(rr_NL, xi_hh_emu_on_NL, kind="cubic", fill_value="extrapolate")
xi_hh_emu = f_interp(rr)

# C. Linear Scale Model
xi_hh_lin = integrate_over_mass(xi_hh_LS_matrix, weights_matrix, nh2)

# 6. Perform the Stitch
xi_stitched = smooth_stitch(
    rr,
    arr_small_scale=xi_hh_sim,
    arr_large_scale=xi_hh_lin,
    r_pivot=r_stitch,
    width=2.0,
)

# --- Plotting with Residuals ---

fig = plt.figure(figsize=(6, 8))  # Increased height for two panels
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

# Top Panel: Correlation Functions
ax0 = plt.subplot(gs[0])
fac = rr**2
fac_NL = rr_NL**2

ax0.plot(rr, fac * xi_hh_sim, "k.", ms=5, label=r"Simulation ($\xi_{hh}$)")
ax0.plot(rr_NL, fac_NL * xi_hh_emu_on_NL, "g--", lw=1.5, label="Emulator (Non-Linear)")
ax0.plot(
    rr, fac * xi_hh_lin, color="mediumblue", ls=":", lw=1.5, label="Linear Bias Model"
)
ax0.plot(
    rr,
    fac * xi_stitched,
    color="r",
    lw=2.0,
    alpha=0.6,
    label="Stitched Model",
)

ax0.set_xscale("log")
ax0.set_ylabel(r"$r^2 \xi_{hh}(r) \ [h^{-2} \mathrm{Mpc}^2]$")
ax0.legend(frameon=False, fontsize=15)
ax0.set_xticklabels([])  # Hide x-ticks for top panel

# Bottom Panel: Residuals
ax1 = plt.subplot(gs[1])

# Calculate Fractional Residuals (Stitched vs Simulation)
# (Model - Data) / Data
residuals = (xi_stitched - xi_hh_sim) / xi_hh_sim

ax1.plot(rr, residuals, color="r", lw=1.5)
ax1.axhline(0.0, color="k", lw=1.0, ls="--")

# Styling
ax1.set_xscale("log")
ax1.set_xlabel(r"$r \ [h^{-1} \mathrm{Mpc}]$")
ax1.set_ylabel(r"$\Delta \xi / \xi_{\mathrm{sim}}$")
ax1.set_ylim(-0.1, 0.1)  # Adjust this range based on your accuracy requirements

# Shared axis limits
ax0.set_xlim(rr[0], rr[-1])
ax1.set_xlim(rr[0], rr[-1])

output_filename = f"xi_hh_stitched_resid_{imodel}.pdf"
plt.savefig(output_filename, bbox_inches="tight", pad_inches=0.05)
print(f"Saved plot to {output_filename}")
