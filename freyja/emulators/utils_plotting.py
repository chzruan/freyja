import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

# --- Matplotlib Styling ---
# Ensure Agg backend is used to avoid GUI errors on clusters
matplotlib.use("Agg")

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams.update({"font.size": 12})
matplotlib.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{physics}"
)
params = {
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
}
plt.rcParams.update(params)


def plot_validation_results(out_path, r, logM, true, pred, sem, imodel, label, metrics):
    """
    Generates and saves a validation plot comparing Emulator vs Simulation.

    Parameters:
        out_path (Path or str): Destination path for the plot.
        r (array): Radial bins.
        logM (array): Mass bins.
        true (array): True Beta values (N_M, N_M, N_r).
        pred (array): Predicted Beta values (N_M, N_M, N_r).
        sem (array): Standard Error of Mean (N_M, N_M, N_r).
        imodel (int): Model ID.
        label (str): Plot label (e.g., 'Test', 'Train').
        metrics (dict): Dictionary containing 'mean_chi2', etc.
    """
    fig = plt.figure(figsize=(14, 10))

    # Select 4 representative pairs to plot:
    # 1. Low Mass Auto-correlation
    # 2. Mid Mass Auto-correlation
    # 3. High Mass Auto-correlation
    # 4. Cross-correlation (Lowest x Highest)
    n = len(logM)
    idxs = [
        (3, 3),
        (n // 2, n // 2),
        (n - 4, n - 4),
        (3, n - 4),
    ]

    # Create subplots
    axes = fig.subplots(2, 2).flatten()

    fig.suptitle(
        f"Model {imodel} ({label}) | Mean $\\chi^2/dof$: {metrics['mean_chi2']:.2f}",
        fontsize=16,
    )

    for k, (i, j) in enumerate(idxs):
        ax = axes[k]

        # Extract data for this specific pair
        y_true = true[i, j, :]
        y_pred = pred[i, j, :]
        y_err = sem[i, j, :]

        m1, m2 = logM[i], logM[j]

        # Plot Simulation Data
        ax.errorbar(
            r,
            y_true,
            yerr=y_err,
            fmt="o",
            color="k",
            ms=3,
            alpha=0.6,
            label="Simulation",
        )

        # Plot Emulator Prediction
        ax.plot(r, y_pred, "r-", lw=2, label="Emulator")

        # Styling
        ax.set_title(f"$\\log M_1={m1:.2f}, \\log M_2={m2:.2f}$")
        ax.set_xscale("log")
        ax.set_ylabel(r"$\\beta(r)$")

        # Add legend only to the first plot
        if k == 0:
            ax.legend()

        # Calculate and display local Chi2 for this specific panel
        # Avoid division by zero with a small epsilon
        chi2_local = np.mean(((y_pred - y_true) / (y_err + 1e-9)) ** 2)

        ax.text(
            0.05,
            0.05,
            f"$\\chi^2_{{local}}={chi2_local:.2f}$",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Plot saved to {out_path}")
