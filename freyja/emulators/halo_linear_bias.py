import jax
import jax.numpy as jnp
import numpy as np
from tinygp import GaussianProcess, kernels
from tinygp.helpers import dataclass
from pathlib import Path
import optax
from typing import Tuple, Optional, Dict, Any
from scipy.optimize import curve_fit

# Colossus imports for Tinker fit
from colossus.cosmology import cosmology as cosmology_colossus
from colossus.lss import peaks

from ..cosma.xi_hh import load_cosmology_wrapper, load_xihh_data, load_ximm_data

MODULE_DIR = Path(__file__).parent


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

# --- Tinker Fit Utilities ---
TINKER_FIXED_PARAMS = {"a": 0.132, "b": 1.5, "c": 2.4}
DELTA_C = 1.686


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
        # Note: Ob0 is fixed to 0.043 in the user snippet
        my_cosmo = {
            "flat": True,
            "H0": h * 100.0,
            "Om0": Om0,
            "Ob0": 0.043,
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
    term1 = 1.0 - A * (nu**a) / (nu**a + DELTA_C**a)
    term2 = B * (nu**b)
    term3 = C * (nu**c)
    return term1 + term2 + term3


def fit_tinker_halo_bias(
    measured_mass: np.ndarray,
    measured_bias: np.ndarray,
    target_mass: np.ndarray,
    redshift: float = 0.25,
    cosmo_params: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits Tinker parameters to measured data and extrapolates to target masses.
    """
    # Combine masses to calculate Nu for the whole range at once
    all_mass = np.concatenate([measured_mass, target_mass])
    nu_all = get_peak_height(all_mass, redshift=redshift, cosmo_params=cosmo_params)

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

    # Handle cases where optimization might fail (e.g., small data range)
    try:
        popt, _ = curve_fit(
            constrained_tinker, nu_meas, measured_bias, p0=p0, maxfev=5000
        )
    except Exception as e:
        print(f"Tinker fit failed ({e}), returning default parameters.")
        popt = p0

    bias_extrapolated = constrained_tinker(nu_target, *popt)
    return bias_extrapolated, popt


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
    return GaussianProcess(kernel, X, diag=diag_noise)


# --- Main Emulator Class ---


class HaloLinearBiasEmulator:
    """
    Gaussian Process Emulator for the Linear Halo Bias b(M).

    Predicts the scalar bias b(M) = sqrt(xi_hh(r|M,M) / xi_mm(r)) using a GP
    mapping (Cosmology, logM) -> b(M).
    """

    def __init__(
        self,
        saved_path=MODULE_DIR / "checkpoints/linear_bias_gp_z0.25.npz",
        hp=HP,
        redshift=0.25,
    ):
        """
        Halo Linear Bias Emulator.

        Parameters
        ----------
        saved_path : str or Path, optional
            Path to load a pre-trained GP state. The default emulator is at redshift 0.25. If None, initializes an empty emulator for training.
        hp : dict
            Hyperparameters for data processing and training.
        redshift : float
            Redshift for the emulator (used in Tinker fit and peak height calculation).
        """
        self.hp = hp
        self.redshift = redshift
        self.params = None
        self.X_train = None
        self.Y_train = None
        self.Y_err_train = None  # Variance of the bias measurements

        # Default scalers
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        if saved_path is not None:
            # Check if path exists to avoid errors on fresh init
            if Path(saved_path).exists():
                self.load(saved_path)

    def _calculate_linear_bias(self, imodel, r_mask, mask_M, logM_bins):
        """
        Loads data and computes the linear bias b(M) by looking at the
        auto-correlation xi_hh(M, M) and taking the inverse-variance weighted
        average over radial bins.
        """
        try:
            # Load Data
            cosmo = load_cosmology_wrapper(imodel)
            _, _, xi_hh, xi_sem = load_xihh_data(imodel)
            r_all, _, _, _ = load_xihh_data(1)
            xi_mm = load_ximm_data(r_all, imodel)

            # 1. Extract Diagonals: xi_hh(M, M, r)
            # xi_hh shape is (N_M_all, N_M_all, N_r_all)
            # Diagonal gives (N_r_all, N_M_all) -> Transpose to (N_M_all, N_r_all)
            xi_hh_diag = np.diagonal(xi_hh, axis1=0, axis2=1).T
            xi_sem_diag = np.diagonal(xi_sem, axis1=0, axis2=1).T

            # 2. Apply Masks
            # mask_M filters mass bins, r_mask filters radial bins
            xi_hh_cut = xi_hh_diag[mask_M, :][:, r_mask]
            xi_sem_cut = xi_sem_diag[mask_M, :][:, r_mask]
            xi_mm_cut = xi_mm[r_mask]

            # 3. Calculate Ratio Square: R(r) = xi_hh(r) / xi_mm(r) approx b^2(r)
            ratio_sq = xi_hh_cut / xi_mm_cut[None, :]

            # Error on ratio (linear approx): sigma_R = sigma_xi_hh / xi_mm
            ratio_sq_err = xi_sem_cut / xi_mm_cut[None, :]

            # 4. Weighted Average over Radius
            # Weight w_r = 1 / Var(R)
            ivar = 1.0 / (ratio_sq_err**2 + self.hp.get("loss_epsilon", 1e-6))

            sum_w = np.sum(ivar, axis=1)
            sum_Rw = np.sum(ratio_sq * ivar, axis=1)

            b_squared = sum_Rw / sum_w
            b_squared_var = 1.0 / sum_w

            # 5. Convert to Linear Bias: b = sqrt(b^2)
            # Propagate Error: Var(b) = Var(b^2) / (4 * b^2)
            # Clip b^2 to small positive to avoid sqrt domain errors
            b_squared = np.maximum(b_squared, 1e-6)

            bias_val = np.sqrt(b_squared)
            bias_var = b_squared_var / (4.0 * b_squared)

            # 6. Format for GP
            inputs, targets, variances = [], [], []

            for i in range(len(logM_bins)):
                # Input: [Cosmo..., logM]
                inputs.append(np.concatenate([cosmo, [logM_bins[i]]]))
                targets.append(bias_val[i])
                variances.append(bias_var[i])

            return inputs, targets, variances

        except Exception as e:
            print(f"Skipping Model {imodel} during GP prep: {e}")
            return [], [], []

    def prepare_data(self):
        """Iterates over all models to build the full Training Set (X, Y, Y_err)."""
        print("Processing simulation data for Linear Bias GP...")
        all_X, all_Y, all_Var = [], [], []

        r_all, logM_all, _, _ = load_xihh_data(imodel=1)
        mask_r = (r_all >= self.hp["r_cut_min"]) & (r_all <= self.hp["r_cut_max"])
        mask_M = (logM_all <= self.hp["logM_cut_max"]) & (logM_all >= 0)
        logM_bins = logM_all[mask_M]

        for imodel in range(1, self.hp["n_models"] + 1):
            x, y, v = self._calculate_linear_bias(imodel, mask_r, mask_M, logM_bins)
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
        self.y_mean = np.mean(Y_raw)
        self.y_std = np.std(Y_raw) + 1e-10

        self.Y_train = jnp.array((Y_raw - self.y_mean) / self.y_std)
        self.Y_err_train = jnp.array(Var_raw / (self.y_std**2))

        print(f"Data Prepared. N_samples: {len(self.Y_train)}")
        print(f"Target Mean: {self.y_mean:.4f}, Std: {self.y_std:.4f}")

    def train(self, learning_rate=0.1, n_steps=500):
        """
        Optimizes GP Hyperparameters using Optax (Adam).
        """
        if self.X_train is None:
            self.prepare_data()

        # Initialize parameters as a dictionary (Pytree)
        dim = self.X_train.shape[1]
        params = {"log_amp": jnp.zeros(()), "log_scale": jnp.zeros(dim)}

        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)

        def loss_fn(p):
            gp = build_gp(p, self.X_train, self.Y_err_train)
            return -gp.log_probability(self.Y_train)

        @jax.jit
        def step(params, opt_state):
            loss_val, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state, loss_val

        print(f"Training GP with Optax (Adam, lr={learning_rate}, steps={n_steps})...")

        for i in range(n_steps):
            params, opt_state, loss_val = step(params, opt_state)
            if i % 100 == 0:
                print(f"Step {i:4d} | Loss: {loss_val:.4f}")

        self.params = params
        print(f"Final Loss: {loss_val:.4f}")
        print(f"Optimized Params: {self.params}")

    def predict_one(self, cosmo_params, logM):
        """
        Predict Bias for a single mass point.
        """
        if self.params is None:
            raise ValueError("GP not trained or loaded.")

        # Prepare Input
        raw_in = np.concatenate([cosmo_params, [logM]])
        norm_in = (raw_in - self.x_mean) / self.x_std
        X_test = jnp.array(norm_in).reshape(1, -1)

        gp = build_gp(self.params, self.X_train, self.Y_err_train)
        cond = gp.condition(self.Y_train, X_test)

        mu_norm = cond.gp.mean[0]
        var_norm = cond.gp.variance[0]

        bias_pred = (float(mu_norm) * self.y_std) + self.y_mean
        bias_std = float(jnp.sqrt(var_norm)) * self.y_std

        return bias_pred, bias_std

    def predict_noext(self, cosmo_params, logM_bins):
        """
        Evaluate the emulator for a 1D array of mass bins (Interpolation only).

        Returns
        -------
        b_pred : np.ndarray
        b_err : np.ndarray
        """
        if self.params is None:
            raise ValueError("GP not trained or loaded.")

        # Construct batch input
        n_points = len(logM_bins)
        cosmo_batch = np.tile(cosmo_params, (n_points, 1))
        raw_in = np.column_stack([cosmo_batch, logM_bins])

        norm_in = (raw_in - self.x_mean) / self.x_std
        X_test = jnp.array(norm_in)

        gp = build_gp(self.params, self.X_train, self.Y_err_train)
        cond = gp.condition(self.Y_train, X_test)

        mu_norm = cond.gp.mean
        var_norm = cond.gp.variance

        bias_pred = (np.array(mu_norm) * self.y_std) + self.y_mean
        bias_std = np.array(jnp.sqrt(var_norm)) * self.y_std

        return bias_pred, bias_std

    def predict(self, cosmo_params, logM_bins):
        """
        Predicts bias using a Tinker et al. (2010) fit to the GP predictions.
        Useful for extrapolating or enforcing physical shape constraints in the low-mass range.

        Parameters
        ----------
        cosmo_params : np.ndarray
            Cosmological parameters [Om0, h, S8, ns, ...].
        logM_bins : np.ndarray
            Target log10 mass bins for prediction.
        redshift : float
            Redshift for peak height calculation.

        Returns
        -------
        bias_pred : np.ndarray
            Predicted bias at logM_bins using the Tinker fit.
        params : np.ndarray
            Fitted Tinker parameters [A, B, C].
        """
        # 1. Define 'measured' range (where GP is trusted)
        # Current fixed values are for the fixed redshift 0.25 training set, can be made dynamic if needed
        logM_min = 12.5
        logM_max = 13.9
        logM_meas = np.linspace(logM_min, logM_max, 30)

        # 2. Get GP predictions for the measured range
        b_meas, _ = self.predict_noext(cosmo_params, logM_meas)

        # 3. Convert to physical mass
        mass_meas = 10**logM_meas
        mass_target = 10**logM_bins

        # 4. Fit Tinker parameters and evaluate
        # Unpack cosmo_params for colossus (expecting first 4 to be Om0, h, S8, ns)
        # Note: Always assume the standard parameter order used in this project.
        cosmo_tuple = tuple(cosmo_params[:4])

        b_ext, popt = fit_tinker_halo_bias(
            measured_mass=mass_meas,
            measured_bias=b_meas,
            target_mass=mass_target,
            redshift=self.redshift,
            cosmo_params=cosmo_tuple,
        )

        return b_ext, popt

    def save(self, path):
        """Saves the GP state."""
        if self.params is None:
            print("Nothing to save.")
            return

        np.savez(
            path,
            log_amp=self.params["log_amp"],
            log_scale=self.params["log_scale"],
            X_train=self.X_train,
            Y_train=self.Y_train,
            Y_err_train=self.Y_err_train,
            x_mean=self.x_mean,
            x_std=self.x_std,
            y_mean=self.y_mean,
            y_std=self.y_std,
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
        self.y_mean = data.get("y_mean", 0.0)
        self.y_std = data.get("y_std", 1.0)

        print(f"GP Emulator loaded from {path}")

    def compare_model_prediction(self, imodel, label="Test", save_plot=True):
        """
        Evaluates the GP emulator against a specific simulation model.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if self.params is None:
            raise ValueError("GP Emulator not trained or loaded.")

        print(f"--- Evaluating GP Emulator on Model {imodel} [{label}] ---")

        try:
            cosmo = load_cosmology_wrapper(imodel)
            r_all, logM_all, _, _ = load_xihh_data(imodel)
        except Exception as e:
            print(f"Error loading validation model {imodel}: {e}")
            return None

        # Masks
        mask_r = (r_all >= self.hp["r_cut_min"]) & (r_all <= self.hp["r_cut_max"])
        # mask_M = (logM_all <= self.hp["logM_cut_max"]) & (logM_all >= 0)
        mask_M = (logM_all <= 14.5) & (logM_all >= 0)

        # Calculate Truth (using same logic as training)
        logM_cut = logM_all[mask_M]

        _, b_true, b_var_true = self._calculate_linear_bias(
            imodel, mask_r, mask_M, logM_cut
        )
        b_true = np.array(b_true)
        b_err_true = np.sqrt(np.array(b_var_true))

        # Predict
        b_pred, _ = self.predict(cosmo, logM_cut)

        # Metrics
        chi2 = np.mean(((b_pred - b_true) / b_err_true) ** 2)
        mse = np.mean((b_pred - b_true) ** 2)
        frac_err = np.mean(np.abs((b_pred - b_true) / b_true))

        print(f"Metrics: Chi2/dof = {chi2:.4f}")
        print(f"Metrics: MSE      = {mse:.2e}")
        print(f"Metrics: Mean %Err= {frac_err*100:.2f}%")

        if save_plot:
            save_path = (
                Path(self.hp.get("output_dir", ".")) / f"gp_bias_val_model_{imodel}.pdf"
            )

            plt.figure(figsize=(6, 5))
            plt.errorbar(
                logM_cut, b_true, yerr=b_err_true, fmt="o", label="Sim Truth", alpha=0.7
            )
            plt.plot(logM_cut, b_pred, "r-", label="GP Prediction")

            plt.xlabel(r"$\log_{10} M$")
            plt.ylabel(r"Linear Bias $b(M)$")
            plt.title(f"Model {imodel}: Linear Bias Emulation")
            plt.legend()
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            print(f"Plot saved to {save_path}")

        return {"chi2": chi2, "mse": mse}
