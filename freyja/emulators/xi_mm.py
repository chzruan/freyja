from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from tinygp import GaussianProcess, kernels
from tinygp.helpers import dataclass
from pathlib import Path
import optax
from typing import Tuple, Optional, Dict, Any
from scipy.interpolate import InterpolatedUnivariateSpline

from ..cosma.xi_hh import load_cosmology_wrapper, load_ximm_data
from ..utils.pk_to_xi import compute_xi_from_Pk

MODULE_DIR = Path(__file__).parent


# --- GP Utilities ---
HP = {
    "redshift": 0.25,
    "n_models": 59,
    "r_min": 0.1,
    "r_max": 120.0,
    "n_r_bins": 180,
    "loss_epsilon": 1e-6,
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

    def evaluate_diag(self, X: jnp.ndarray) -> jnp.ndarray:
        """Return the diagonal of the kernel matrix."""
        return jnp.exp(self.log_amp)


def build_gp(params, X, diag_noise):
    """Construct a tinygp Gaussian Process object with heteroscedastic noise."""
    kernel = RBFKernel(
        log_amp=params["log_amp"],
        log_scale=params["log_scale"],
    )
    return GaussianProcess(kernel, X, diag=diag_noise)


class MatterXiEmulator:
    """
    Gaussian Process Emulator for the matter correlation function r^2 * xi_mm(r).
    """

    def __init__(
        self,
        checkpoint_path=MODULE_DIR / "checkpoints" / "matter_xi_gp_z0.25.npz",
        hp=HP,
    ):
        self.hp = hp
        self.redshift = hp.get("redshift", 0.25)
        self.params = None
        self.X_train = None
        self.Y_train = None
        self.Y_err_train = None
        self.X_val = None
        self.Y_val = None
        self.Y_err_val = None

        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        self.r_grid = np.geomspace(
            self.hp["r_min"], self.hp["r_max"], self.hp["n_r_bins"]
        )

        if checkpoint_path is not None:
            if Path(checkpoint_path).exists():
                self.load(checkpoint_path)

    def get_pk_linear_sigma8(
        self,
        sigma8_target,
        redshift,
        h=0.675,
        ombh2=0.022,
        omch2=0.122,
        ns=0.965,
        kmax=20.0,
        npoints=2000,
    ):
        """
        Calculates linear Matter P(k) normalized to a specific sigma8 using CAMB.
        """
        import camb
        from camb import model

        dummy_As = 2e-9
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=h * 100, ombh2=ombh2, omch2=omch2)
        pars.InitPower.set_params(ns=ns, As=dummy_As)

        # Ensure z=0 is calculated for sigma8_0 normalization
        eval_redshifts = list({redshift, 0.0})
        pars.set_matter_power(redshifts=eval_redshifts, kmax=kmax)

        pars.NonLinear = model.NonLinear_none

        results = camb.get_results(pars)
        sigma8_fid = results.get_sigma8_0()

        rescaling_factor = (sigma8_target / sigma8_fid) ** 2
        new_As = dummy_As * rescaling_factor

        kh, z, pk_fid = results.get_matter_power_spectrum(
            minkh=1e-4, maxkh=kmax, npoints=npoints
        )
        pk_final = pk_fid * rescaling_factor

        # Select P(k) at the requested redshift
        z_index = np.where(np.isclose(z, redshift))[0][0]

        return kh, pk_final[z_index], new_As

    def get_linear_pk_mm(self, cosmo):
        Om0, h, S8, ns = cosmo
        sigma8 = S8 / np.sqrt(Om0 / 0.3)
        omega_b_h2 = 0.0224
        omega_c_h2 = Om0 * h**2 - omega_b_h2

        k_Lin, P_Lin, _ = self.get_pk_linear_sigma8(
            sigma8_target=sigma8,
            h=h,
            ombh2=omega_b_h2,
            omch2=omega_c_h2,
            ns=ns,
            kmax=20.0,
            npoints=1024,
            redshift=self.redshift,
        )
        return k_Lin, P_Lin

    def _prepare_data(self, validation_split=0.2, random_seed=42):
        """
        Vectorized data preparation with train-validation split.
        """
        all_inputs, all_targets = [], []
        log_r_vec = np.log10(self.r_grid)[:, None]
        n_bins = len(self.r_grid)

        for imodel in range(1, self.hp["n_models"] + 1):
            cosmo = load_cosmology_wrapper(imodel)
            xi_mm = load_ximm_data(r_output=self.r_grid, imodel=imodel)

            k_Lin, P_Lin = self.get_linear_pk_mm(cosmo)
            xi_lin = compute_xi_from_Pk(
                k_input=k_Lin, P_input=P_Lin, r_output=self.r_grid, smooth_xi=False
            )

            # Target is xi / xi_lin
            target = xi_mm / xi_lin

            cosmo_tiled = np.tile(cosmo, (n_bins, 1))
            model_inputs = np.hstack([cosmo_tiled, log_r_vec])
            all_inputs.append(model_inputs)
            all_targets.append(target[:, None])

        inputs = np.vstack(all_inputs)
        targets = np.vstack(all_targets)

        # Shuffle and split data
        n_samples = inputs.shape[0]
        indices = np.arange(n_samples)
        rng = np.random.default_rng(random_seed)
        rng.shuffle(indices)

        split_idx = int(n_samples * (1 - validation_split))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        X_train_raw = inputs[train_indices]
        X_val_raw = inputs[val_indices]
        Y_train_raw = targets[train_indices]
        Y_val_raw = targets[val_indices]

        self.x_mean = np.mean(X_train_raw, axis=0)
        self.x_std = np.std(X_train_raw, axis=0) + 1e-10
        self.X_train = jnp.array((X_train_raw - self.x_mean) / self.x_std)
        self.X_val = jnp.array((X_val_raw - self.x_mean) / self.x_std)

        self.y_mean = np.mean(Y_train_raw)
        self.y_std = np.std(Y_train_raw) + 1e-10
        self.Y_train = jnp.array(((Y_train_raw - self.y_mean) / self.y_std).flatten())
        self.Y_val = jnp.array(((Y_val_raw - self.y_mean) / self.y_std).flatten())

        # For now, assume uniform noise
        self.Y_err_train = jnp.ones_like(self.Y_train) * self.hp["loss_epsilon"]
        self.Y_err_val = jnp.ones_like(self.Y_val) * self.hp["loss_epsilon"]

    def train(
        self,
        learning_rate=0.010,
        n_steps=2200,
        patience=10,
        min_delta=1e-5,
        validation_split=0.20,
    ):
        if self.X_train is None:
            self._prepare_data(validation_split=validation_split)

        dim = self.X_train.shape[1]
        self.params = {"log_amp": jnp.zeros(()), "log_scale": jnp.zeros(dim)}

        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(self.params)

        def loss_fn(p, X, Y, Y_err):
            gp = build_gp(p, X, Y_err)
            return -gp.log_probability(Y)

        # Jit-compiled training step
        @jax.jit
        def train_step(params, opt_state, X, Y, Y_err):
            loss_val, grads = jax.value_and_grad(loss_fn)(params, X, Y, Y_err)
            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state, loss_val

        # Jit-compiled validation loss calculation
        val_loss_fn = jax.jit(loss_fn)

        print(f"Training GP with Optax (Adam, lr={learning_rate}, steps={n_steps})...")

        best_val_loss = float("inf")
        patience_counter = 0

        for i in range(n_steps):
            self.params, opt_state, loss_val = train_step(
                self.params, opt_state, self.X_train, self.Y_train, self.Y_err_train
            )

            if i > 0 and i % 20 == 0:
                val_loss = val_loss_fn(
                    self.params, self.X_val, self.Y_val, self.Y_err_val
                )
                print(f"Step {i:4d} | Loss: {loss_val:.4f} | Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(
                        f"Early stopping at step {i}. No improvement in validation loss for {patience * 20} steps."
                    )
                    break

        final_loss = loss_fn(self.params, self.X_train, self.Y_train, self.Y_err_train)
        print(f"Final Loss: {final_loss:.4f}")
        print(f"Optimized Params: {self.params}")

    def _predict_raw(self, cosmo, r_array):
        if self.params is None:
            raise ValueError("GP not trained or loaded.")

        r_array = np.atleast_1d(r_array)
        log10r = np.log10(r_array)

        cosmo_tile = np.tile(cosmo, (len(log10r), 1))
        raw_in = np.column_stack([cosmo_tile, log10r])
        norm_in = (raw_in - self.x_mean) / self.x_std
        X_test = jnp.array(norm_in)

        gp = build_gp(self.params, self.X_train, self.Y_err_train)
        cond = gp.condition(self.Y_train, X_test)

        mu_norm = cond.gp.mean
        var_norm = cond.gp.variance

        r2_xi_pred = (np.array(mu_norm) * self.y_std) + self.y_mean
        r2_xi_std = np.array(jnp.sqrt(var_norm)) * self.y_std

        return r2_xi_pred, r2_xi_std

    def predict(self, cosmo, r_array):
        r_array = np.atleast_1d(r_array)
        xiratio_pred, xiratio_std = self._predict_raw(cosmo, r_array)

        mask_r = r_array >= 59.0
        # Force prediction to 1 at large r where xi_mm ~ xi_lin
        xiratio_pred[mask_r] = 1.0
        xiratio_std[mask_r] = 0.0

        k_Lin, P_Lin = self.get_linear_pk_mm(cosmo)
        xi_lin = compute_xi_from_Pk(
            k_input=k_Lin, P_input=P_Lin, r_output=r_array, smooth_xi=False
        )

        return xiratio_pred * xi_lin, xiratio_std * xi_lin

    def save(self, path):
        if self.params is None:
            print("Nothing to save.")
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
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
            hp=self.hp,
        )
        print(f"GP Emulator saved to {path}")

    def load(self, path):
        data = np.load(path, allow_pickle=True)
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

        self.hp = data["hp"].item()

        print(f"GP Emulator loaded from {path}")

    def compare(self, imodel, save_plot=True):
        if self.params is None:
            raise ValueError("GP not trained or loaded.")

        cosmo = load_cosmology_wrapper(imodel)

        # Use a standard r range for comparison
        # r_true = np.geomspace(self.hp["r_min"], self.hp["r_max"], 100)
        r_true = np.geomspace(0.1, 105.1, 100)
        xi_true = load_ximm_data(r_output=r_true, imodel=imodel)

        k_Lin, P_Lin = self.get_linear_pk_mm(cosmo)
        xi_lin = compute_xi_from_Pk(
            k_input=k_Lin, P_input=P_Lin, r_output=r_true, smooth_xi=False
        )

        # 2. Predict
        xi_pred, xi_std = self.predict(cosmo, r_true)

        # 3. Metrics
        mse = np.mean((xi_pred - xi_true) ** 2)
        # Avoid division by zero where xi crosses zero
        mask = np.abs(xi_true) > 1e-4
        mean_frac_error = np.mean(
            np.abs((xi_pred[mask] - xi_true[mask]) / xi_true[mask])
        )

        print(f"--- Xi Evaluation for Model {imodel} ---")
        print(f"MSE: {mse:.2e}")
        print(f"Mean Frac Error: {mean_frac_error*100:.2f}%")

        if save_plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(
                2, 1, sharex=True, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}
            )

            # Top panel: r^2 * xi(r) to flatten dynamic range
            ax[0].plot(r_true, r_true**2 * xi_true, "k-", label="Truth (Sim)")
            ax[0].plot(r_true, r_true**2 * xi_pred, "r--", label="Emulator (GP)")
            ax[0].fill_between(
                r_true,
                r_true**2 * (xi_pred - xi_std),
                r_true**2 * (xi_pred + xi_std),
                color="red",
                alpha=0.2,
            )
            ax[0].set_ylabel(r"$r^2 \xi(r)$")
            ax[0].set_xscale("log")
            ax[0].legend()
            ax[0].set_title(f"Xi Emulator Validation: Model {imodel}")

            # Bottom panel: Fractional Error
            ax[1].plot(r_true[mask], (xi_pred[mask] / xi_true[mask]) - 1, "r-")
            ax[1].axhline(0, color="k", linestyle=":")
            ax[1].set_ylabel(r"$\Delta \xi / \xi$")
            ax[1].set_ylim([-0.1, 0.1])
            ax[1].set_xlabel(r"$r$ [$h^{-1}$Mpc]")

            out_path = f"compare_xi_model_gp_{imodel}.pdf"
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            print(f"Plot saved to {out_path}")

        return {"mse": mse, "mean_frac_error": mean_frac_error}
