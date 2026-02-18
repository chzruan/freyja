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

from ..cosma.xi_hh import load_cosmology_wrapper, load_pkmm_data

MODULE_DIR = Path(__file__).parent


# --- GP Utilities ---
HP = {
    "redshift": 0.25,
    "n_models": 59,
    "k_min": 0.008,
    "k_max": 20.0,
    "n_k_bins": 50,
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


class MatterAlphaEmulator:
    """
    Gaussian Process Emulator for the matter power spectrum boost factor alpha(k).

    This class emulates the ratio alpha(k) = P_NL(k) / P_L(k).
    """

    def __init__(
        self,
        saved_path=MODULE_DIR / "checkpoints" / "matter_alpha_gp_z0.25.npz",
        hp=HP,
    ):
        self.hp = hp
        self.redshift = hp.get("redshift", 0.25)
        self.params = None
        self.X_train = None
        self.Y_train = None
        self.Y_err_train = None  # Placeholder, not used for now

        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        self.k_grid = np.logspace(
            np.log10(self.hp["k_min"]), np.log10(self.hp["k_max"]), self.hp["n_k_bins"]
        )

        if saved_path is not None:
            if Path(saved_path).exists():
                self.load(saved_path)

    def get_pk_linear_sigma8(
        self,
        sigma8_target,
        h=0.675,
        ombh2=0.022,
        omch2=0.122,
        ns=0.965,
        kmax=10.0,
        npoints=2000,
        redshift=0.0,
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
        eval_redshifts = sorted(list({redshift, 0.0}))
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

    def get_linear_pk_mm(self, cosmo_params):
        Om0, h, S8, ns = cosmo_params
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

    def _prepare_data(self):
        """
        Vectorized data preparation.
        """
        all_inputs, all_targets = [], []
        log_k_vec = np.log10(self.k_grid)[:, None]
        n_bins = len(self.k_grid)

        for imodel in range(1, self.hp["n_models"] + 1):
            cosmo = load_cosmology_wrapper(imodel)
            k_NL, Pk_NL = load_pkmm_data(imodel, return_mean=True)
            k_L, Pk_L = self.get_linear_pk_mm(cosmo)

            spl_NL = InterpolatedUnivariateSpline(np.log10(k_NL), np.log10(Pk_NL), k=3)
            spl_L = InterpolatedUnivariateSpline(np.log10(k_L), np.log10(Pk_L), k=3)

            log_ratio = spl_NL(np.log10(self.k_grid)) - spl_L(np.log10(self.k_grid))
            pk_ratio = 10**log_ratio

            cosmo_tiled = np.tile(cosmo, (n_bins, 1))
            model_inputs = np.hstack([cosmo_tiled, log_k_vec])
            all_inputs.append(model_inputs)
            all_targets.append(pk_ratio[:, None])

        inputs = np.vstack(all_inputs)
        targets = np.vstack(all_targets)

        self.x_mean = np.mean(inputs, axis=0)
        self.x_std = np.std(inputs, axis=0) + 1e-10
        self.X_train = jnp.array((inputs - self.x_mean) / self.x_std)

        self.y_mean = np.mean(targets)
        self.y_std = np.std(targets) + 1e-10
        self.Y_train = jnp.array(((targets - self.y_mean) / self.y_std).flatten())

        # For now, assume uniform noise
        self.Y_err_train = jnp.ones_like(self.Y_train) * self.hp["loss_epsilon"]

    def train(self, learning_rate=0.01, n_steps=500):
        if self.X_train is None:
            self._prepare_data()

        dim = self.X_train.shape[1]
        self.params = {"log_amp": jnp.zeros(()), "log_scale": jnp.zeros(dim)}

        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(self.params)

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
            self.params, opt_state, loss_val = step(self.params, opt_state)
            if i % 100 == 0:
                print(f"Step {i:4d} | Loss: {loss_val:.4f}")

        print(f"Final Loss: {loss_val:.4f}")
        print(f"Optimized Params: {self.params}")

    def _predict_raw(self, cosmo_params, k_array):
        if self.params is None:
            raise ValueError("GP not trained or loaded.")

        k_array = np.atleast_1d(k_array)
        log10k = np.log10(k_array)

        cosmo_tile = np.tile(cosmo_params, (len(log10k), 1))
        raw_in = np.column_stack([cosmo_tile, log10k])
        norm_in = (raw_in - self.x_mean) / self.x_std
        X_test = jnp.array(norm_in)

        gp = build_gp(self.params, self.X_train, self.Y_err_train)
        cond = gp.condition(self.Y_train, X_test)

        mu_norm = cond.gp.mean
        var_norm = cond.gp.variance

        alpha_pred = (np.array(mu_norm) * self.y_std) + self.y_mean
        alpha_std = np.array(jnp.sqrt(var_norm)) * self.y_std

        return alpha_pred, alpha_std

    def predict(self, cosmo_params, k_array):
        # Get raw predictions for the input k_array first
        alpha_pred, alpha_std = self._predict_raw(cosmo_params, k_array)

        # Asymptotic value
        k_asymp_range = np.linspace(0.04, 0.09, 5)
        alpha_asymp_pred, alpha_asymp_std = self._predict_raw(
            cosmo_params, k_asymp_range
        )
        alpha_asymptotic = np.mean(alpha_asymp_pred)
        alpha_asymptotic_std = np.mean(alpha_asymp_std)

        # Replace low-k values
        low_k_mask = k_array < 0.09
        alpha_pred[low_k_mask] = alpha_asymptotic
        alpha_std[low_k_mask] = alpha_asymptotic_std

        return alpha_pred, alpha_std

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
        k_NL, Pk_NL = load_pkmm_data(imodel, return_mean=True)
        k_L, Pk_L = self.get_linear_pk_mm(cosmo)  # Uses internal method

        # Interpolate truth
        _kk = np.geomspace(self.hp["k_min"], self.hp["k_max"], 100)
        spl_NL = InterpolatedUnivariateSpline(np.log10(k_NL), np.log10(Pk_NL), k=3)
        spl_L = InterpolatedUnivariateSpline(np.log10(k_L), np.log10(Pk_L), k=3)
        alpha_true = 10 ** spl_NL(np.log10(_kk)) / 10 ** spl_L(np.log10(_kk))

        alpha_pred, alpha_std = self.predict(cosmo, _kk)

        # Asymptotic value
        k_asymp_range = np.linspace(0.04, 0.09, 5)
        alpha_asymp_pred, _ = self.predict(cosmo, k_asymp_range)
        alpha_asymptotic = np.mean(alpha_asymp_pred)

        # Find threshold k
        try:
            k_threshold_idx = np.where(
                np.abs(alpha_pred - alpha_asymptotic) / alpha_asymptotic > 0.01
            )[0][0]
            k_threshold = _kk[k_threshold_idx]
            print(
                f"k threshold where alpha varies > 1% from asymptotic: {k_threshold:.4f} h/Mpc"
            )
        except IndexError:
            print(
                "Alpha does not vary more than 1% from asymptotic in the plotted range."
            )

        mse = np.mean((alpha_pred - alpha_true) ** 2)
        mean_frac_error = np.mean(np.abs((alpha_pred - alpha_true) / alpha_true))

        print(f"--- Model {imodel} ---")
        print(f"MSE: {mse:.2e}")
        print(f"Mean Frac Error: {mean_frac_error*100:.2f}%")

        if save_plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=(8, 6),
                gridspec_kw={"height_ratios": [3, 1]},
            )
            ax[0].plot(
                _kk, alpha_true, "k", marker=".", markersize=4, lw=0, label="Truth"
            )
            ax[0].plot(_kk, alpha_pred, "r-", label="Emulator")
            ax[0].fill_between(
                _kk,
                alpha_pred - alpha_std,
                alpha_pred + alpha_std,
                color="red",
                alpha=0.2,
            )
            ax[0].axhline(
                alpha_asymptotic,
                color="b",
                linestyle="--",
                label=f"Asymptotic alpha ({alpha_asymptotic:.3f})",
            )
            ax[0].set_ylabel(r"$\alpha(k)$")
            ax[0].set_xscale("log")
            ax[0].legend()
            ax[1].plot(_kk, (alpha_pred / alpha_true) - 1, "r-")
            ax[1].axhline(0, color="k", linestyle=":")
            ax[1].set_ylabel("Frac. Err")
            ax[1].set_ylim([-0.11, 0.11])
            ax[1].set_xlabel(r"$k$ [$h$ Mpc$^{-1}$]")
            plt.tight_layout()
            plt.savefig(f"compare_alpha_model_gp_{imodel}.pdf")
            plt.close()

        return {"mse": mse, "mean_frac_error": mean_frac_error}
