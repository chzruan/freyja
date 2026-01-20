import jax
import jax.numpy as jnp
import numpy as np
from tinygp import GaussianProcess, kernels
from tinygp.helpers import dataclass
from pathlib import Path
import optax
from ..cosma.xi_hh import load_cosmology_wrapper, load_xihh_data, load_ximm_data

# --- GP Utilities ---
HP = {
    # Data Constraints
    "logM_cut_max": 13.8,  # Maximum log mass
    "logM_cut_min": 12.4,
    "r_cut_max": 75.0,  # Maximum scale in fitting halo bias in Mpc/h
    "r_cut_min": 35.0,  # Minimum scale
    "n_models": 59,  # Number of simulation models in the training set
    "loss_epsilon": 1e-6,  # Stability floor for variance
}


@dataclass
class RBFKernel(kernels.Kernel):
    """
    Radial Basis Function (RBF) kernel with anisotropic length-scales.
    """

    log_amp: jnp.ndarray
    log_scale: jnp.ndarray

    def evaluate(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        amp = jnp.exp(self.log_amp)
        ell = jnp.exp(self.log_scale)
        r = (X1 - X2) / ell
        r2 = jnp.dot(r, r)
        return amp * jnp.exp(-0.5 * r2)


def build_gp(params, X, diag_noise):
    """Construct a tinygp Gaussian Process object with heteroscedastic noise."""
    kernel = RBFKernel(
        log_amp=params["log_amp"],
        log_scale=params["log_scale"],
    )
    # diag_noise accounts for the variance of the measured bias values
    return GaussianProcess(kernel, X, diag=diag_noise)


# --- Main Emulator Class ---


class HaloBiasEmulator:
    """
    Gaussian Process Emulator for the scalar Halo Bias B12(M1, M2).

    Predicts the weighted average of xi_hh(r)/xi_mm(r) using a GP
    mapping (Cosmology, u, v) -> Bias.
    """

    def __init__(self, saved_path=None, hp=HP):
        self.hp = hp
        self.params = None
        self.X_train = None
        self.Y_train = None
        self.Y_err_train = None  # Variance of the bias measurements

        # Default scalers
        self.x_mean = None
        self.x_std = None

        if saved_path:
            self.load(saved_path)

    def _calculate_weighted_bias(self, imodel, r_mask, mask_M, logM_bins):
        """
        Loads data and computes the scalar bias B12 by inverse-variance weighting
        over the radial bins.
        """
        try:
            # Load Data
            cosmo = load_cosmology_wrapper(imodel)
            _, _, xi_hh, xi_sem = load_xihh_data(imodel)
            r_all, _, _, _ = load_xihh_data(1)
            xi_mm = load_ximm_data(r_all, imodel)

            # Slicing
            xi_hh_cut = xi_hh[mask_M][:, mask_M, :][:, :, r_mask]
            xi_sem_cut = xi_sem[mask_M][:, mask_M, :][:, :, r_mask]
            xi_mm_cut = xi_mm[r_mask]

            # Calculate Scale-dependent Bias beta(r)
            beta = xi_hh_cut / xi_mm_cut[None, None, :]
            beta_sem = xi_sem_cut / xi_mm_cut[None, None, :]

            # --- Weighted Average Logic ---
            # Weight w_r = 1 / sigma(r)^2
            # Add small epsilon to avoid div-by-zero
            ivar = 1.0 / (beta_sem**2 + self.hp.get("loss_epsilon", 1e-6))

            # Weighted Sum: B = sum(beta * w) / sum(w)
            sum_w = np.sum(ivar, axis=2)
            sum_bw = np.sum(beta * ivar, axis=2)

            bias_val = sum_bw / sum_w  # Shape (N_M, N_M)

            # Propagate Error: Var(B) = 1 / sum(w)
            bias_var = 1.0 / sum_w

            # Flatten into training lists
            inputs, targets, variances = [], [], []
            N_M = len(logM_bins)

            for i in range(N_M):
                for j in range(i, N_M):
                    # Symmetric inputs u, v
                    u = (logM_bins[i] + logM_bins[j]) / 2.0
                    v = (logM_bins[i] - logM_bins[j]) / 2.0

                    inputs.append(np.concatenate([cosmo, [u, v]]))
                    targets.append(bias_val[i, j])
                    variances.append(bias_var[i, j])

            return inputs, targets, variances

        except Exception as e:
            print(f"Skipping Model {imodel} during GP prep: {e}")
            return [], [], []

    def prepare_data(self):
        """Iterates over all models to build the full Training Set (X, Y, Y_err)."""
        print("Processing simulation data for GP...")
        all_X, all_Y, all_Var = [], [], []

        r_all, logM_all, _, _ = load_xihh_data(imodel=1)
        mask_r = (r_all >= self.hp["r_cut_min"]) & (r_all <= self.hp["r_cut_max"])
        mask_M = (logM_all <= self.hp["logM_cut_max"]) & (logM_all >= 0)
        logM_bins = logM_all[mask_M]

        for imodel in range(1, self.hp["n_models"] + 1):
            x, y, v = self._calculate_weighted_bias(imodel, mask_r, mask_M, logM_bins)
            all_X.extend(x)
            all_Y.extend(y)
            all_Var.extend(v)

        # Convert to numpy first
        X_raw = np.array(all_X)
        Y_raw = np.array(all_Y)
        Var_raw = np.array(all_Var)

        # --- Normalization ---
        # 1. Inputs (X): Standard Score (Mean 0, Std 1)
        self.x_mean = np.mean(X_raw, axis=0)
        self.x_std = np.std(X_raw, axis=0) + 1e-10
        self.X_train = jnp.array((X_raw - self.x_mean) / self.x_std)

        # 2. Targets (Y): Standard Score
        # This is CRITICAL for Zero-Mean GPs.
        self.y_mean = np.mean(Y_raw)
        self.y_std = np.std(Y_raw) + 1e-10

        # Normalize Y: (y - mu) / sigma
        self.Y_train = jnp.array((Y_raw - self.y_mean) / self.y_std)

        # Normalize Variance: Var(cy) = c^2 * Var(y) -> divide by y_std^2
        self.Y_err_train = jnp.array(Var_raw / (self.y_std**2))

        print(f"Data Prepared. N_samples: {len(self.Y_train)}")
        print(f"Target Mean: {self.y_mean:.4f}, Std: {self.y_std:.4f}")

    def train(self, learning_rate=0.1, n_steps=500):
        """
        Optimizes GP Hyperparameters using Optax (Adam).

        Parameters
        ----------
        learning_rate : float
            The learning rate for the Adam optimizer.
        n_steps : int
            Number of optimization steps.
        """
        if self.X_train is None:
            self.prepare_data()

        # Initialize parameters as a dictionary (Pytree)
        dim = self.X_train.shape[1]
        params = {"log_amp": jnp.zeros(()), "log_scale": jnp.zeros(dim)}  # log(1.0) = 0

        # Define the Optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)

        # Define Loss Function
        def loss_fn(p):
            gp = build_gp(p, self.X_train, self.Y_err_train)
            return -gp.log_probability(self.Y_train)

        # Define Single Optimization Step (JIT compiled)
        @jax.jit
        def step(params, opt_state):
            loss_val, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state, loss_val

        print(f"Training GP with Optax (Adam, lr={learning_rate}, steps={n_steps})...")

        # Training Loop
        for i in range(n_steps):
            params, opt_state, loss_val = step(params, opt_state)

            if i % 100 == 0:
                print(f"Step {i:4d} | Loss: {loss_val:.4f}")

        # Store final parameters
        self.params = params
        print(f"Final Loss: {loss_val:.4f}")
        print(f"Optimized Params: {self.params}")

    def predict(self, cosmo_params, u, v):
        """
        Predict Bias for a specific configuration.
        """
        if self.params is None:
            raise ValueError("GP not trained or loaded.")

        # Prepare Input
        raw_in = np.concatenate([cosmo_params, [u, v]])
        norm_in = (raw_in - self.x_mean) / self.x_std
        X_test = jnp.array(norm_in).reshape(1, -1)

        # Condition GP
        # Note: We use the normalized Y and Y_err here
        gp = build_gp(self.params, self.X_train, self.Y_err_train)
        cond = gp.condition(self.Y_train, X_test)

        mu_norm = cond.gp.mean[0]
        var_norm = cond.gp.variance[0]

        # --- Un-normalize ---
        # Bias = (pred * y_std) + y_mean
        bias_pred = (float(mu_norm) * self.y_std) + self.y_mean

        # Std = sqrt(var_norm) * y_std
        bias_std = float(jnp.sqrt(var_norm)) * self.y_std

        return bias_pred, bias_std

    def save(self, path):
        """Saves the GP state."""
        if self.params is None:
            print("Nothing to save.")
            return

        np.savez(
            path,
            # Parameters
            log_amp=self.params["log_amp"],
            log_scale=self.params["log_scale"],
            # Training Data
            X_train=self.X_train,
            Y_train=self.Y_train,
            Y_err_train=self.Y_err_train,
            # Scalers
            x_mean=self.x_mean,
            x_std=self.x_std,
            y_mean=self.y_mean,  # <--- NEW
            y_std=self.y_std,  # <--- NEW
        )
        print(f"GP Emulator saved to {path}")

    def load(self, path):
        """Loads a saved GP state."""
        data = np.load(path)
        self.params = {
            "log_amp": jnp.array(data["log_amp"]),
            "log_scale": jnp.array(data["log_scale"]),
        }
        self.X_train = jnp.array(data["X_train"])
        self.Y_train = jnp.array(data["Y_train"])
        self.Y_err_train = jnp.array(data["Y_err_train"])

        self.x_mean = data["x_mean"]
        self.x_std = data["x_std"]
        # Handle backward compatibility if you have old files without y stats
        self.y_mean = data.get("y_mean", 0.0)
        self.y_std = data.get("y_std", 1.0)

        print(f"GP Emulator loaded from {path}")

    def compare_model_prediction(self, imodel, label="Test", save_plot=True):
        """
        Evaluates the GP emulator against a specific simulation model.

        Parameters
        ----------
        imodel : int
            The index of the simulation model to test (1 to n_models).
        label : str
            Label for the plot title and filename.
        save_plot : bool
            Whether to save a comparison plot.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics (chi2, mse, fractional error).
        """
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

        if self.params is None:
            raise ValueError("GP Emulator not trained or loaded.")

        print(f"--- Evaluating GP Emulator on Model {imodel} [{label}] ---")

        # --- 1. Load Truth Data ---
        try:
            cosmo = load_cosmology_wrapper(imodel)
            r_all, logM_all, xi_hh, xi_sem = load_xihh_data(imodel)
            xi_mm = load_ximm_data(r_all, imodel)
        except Exception as e:
            print(f"Error loading validation model {imodel}: {e}")
            return None

        # Masks
        mask_r = (r_all >= self.hp["r_cut_min"]) & (r_all <= self.hp["r_cut_max"])
        mask_M = (logM_all <= self.hp["logM_cut_max"]) & (logM_all >= 0)

        logM_cut = logM_all[mask_M]
        N_M = len(logM_cut)

        # Slicing
        xi_hh_cut = xi_hh[mask_M][:, mask_M, :][:, :, mask_r]
        xi_sem_cut = xi_sem[mask_M][:, mask_M, :][:, :, mask_r]
        xi_mm_cut = xi_mm[mask_r]

        # Calculate True Bias (Weighted Average)
        beta = xi_hh_cut / xi_mm_cut[None, None, :]
        beta_sem = xi_sem_cut / xi_mm_cut[None, None, :]

        # Weights w = 1/sigma^2
        ivar = 1.0 / (beta_sem**2 + self.hp.get("loss_epsilon", 1e-9))

        # Weighted Mean: Sum(val * w) / Sum(w)
        sum_ivar = np.sum(ivar, axis=2)
        bias_true = np.sum(beta * ivar, axis=2) / sum_ivar

        # Uncertainty on the scalar bias: sqrt(1 / Sum(w))
        bias_sigma_true = np.sqrt(1.0 / sum_ivar)

        # --- 2. Generate Predictions ---
        bias_pred = np.zeros_like(bias_true)
        bias_sigma_pred = np.zeros_like(bias_true)

        for i in range(N_M):
            for j in range(i, N_M):
                # Calculate coordinates
                u = (logM_cut[i] + logM_cut[j]) / 2.0
                v = (logM_cut[i] - logM_cut[j]) / 2.0

                # Predict
                mu, std = self.predict(cosmo, u, v)

                # Fill symmetric matrix
                bias_pred[i, j] = mu
                bias_pred[j, i] = mu
                bias_sigma_pred[i, j] = std
                bias_sigma_pred[j, i] = std

        # --- 3. Compute Metrics ---
        # Chi2 using the simulation uncertainty (or GP uncertainty if preferred, usually sim)
        # We use the simulation uncertainty here to see how well we fit the "data"
        chi2_map = ((bias_pred - bias_true) ** 2) / (
            bias_sigma_true**2 + self.hp.get("loss_epsilon", 1e-6)
        )

        metrics = {
            "mean_chi2": np.mean(chi2_map),
            "mse": np.mean((bias_pred - bias_true) ** 2),
            "mean_frac_error": np.mean(
                np.abs(
                    (bias_pred - bias_true)
                    / (bias_true + self.hp.get("loss_epsilon", 1e-6))
                )
            ),
        }

        print(f"Metrics: Chi2/dof = {metrics['mean_chi2']:.4f}")
        print(f"Metrics: MSE      = {metrics['mse']:.2e}")
        print(f"Metrics: Mean %Err= {metrics['mean_frac_error']*100:.2f}%")

        # --- 4. Plotting (Scatter) ---
        if save_plot:
            save_path = (
                Path(self.hp.get("output_dir", "."))
                / f"gp_validation_model_{imodel}_{label}.png"
            )

            fig, ax = plt.subplots(figsize=(7, 6))

            # Scatter plot with error bars
            # Flatten arrays for plotting
            y_true = bias_true[np.triu_indices(N_M)]
            y_pred = bias_pred[np.triu_indices(N_M)]
            y_err = bias_sigma_true[np.triu_indices(N_M)]
            gp_err = bias_sigma_pred[np.triu_indices(N_M)]

            ax.errorbar(
                y_true,
                y_pred,
                xerr=y_err,
                yerr=gp_err,
                fmt="o",
                alpha=0.7,
                label="Predictions",
            )

            # Identity line
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "k--",
                alpha=0.5,
                label="Perfect Fit",
            )

            ax.set_xlabel(r"True Bias $B_{12}$ (Sim)")
            ax.set_ylabel(r"Emulated Bias $B_{12}$ (GP)")
            ax.set_title(
                f"Model {imodel} Validation: {label}\n$chi^2/dof={metrics['mean_chi2']:.2f}$"
            )
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.6)

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            print(f"Validation plot saved to {save_path}")

        return metrics
