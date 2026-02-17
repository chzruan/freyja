import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, random_split
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from scipy.interpolate import InterpolatedUnivariateSpline

# Local imports
from ..cosma.xi_hh import load_cosmology_wrapper, load_pkmm_data

MODULE_DIR = Path(__file__).parent


class MatterAlphaNet(pl.LightningModule):
    """
    Neural Network for predicting the non-linear boost ratio:
    alpha(k) = P_NL(k) / P_Lin(k)
    """

    def __init__(self, input_dim, output_dim, scalers, hp):
        super().__init__()
        self.save_hyperparameters(hp)
        self.scalers = scalers

        layers = []
        in_dim = input_dim
        for h_dim in self.hparams["hidden_layers"]:
            layers.append(nn.Linear(in_dim, h_dim))
            # Dynamic activation selection
            act = (
                nn.GELU()
                if self.hparams.get("activation", "GELU") == "GELU"
                else nn.ReLU()
            )
            layers.append(act)
            layers.append(nn.Dropout(self.hparams["dropout"]))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = torch.nn.functional.mse_loss(y_pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        mse = torch.nn.functional.mse_loss(y_pred, y)
        self.log("val_mse", mse, prog_bar=True)
        return mse

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )


class MatterAlphaEmulator:
    def __init__(
        self, checkpoint_path=MODULE_DIR / "checkpoints/matter_alpha_z0.25.pt", hp=None
    ):
        """
        Emulator for the ratio alpha(k) = P_nonlinear(k) / P_linear(k).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hp = hp or {
            "hidden_layers": [64, 64],
            "activation": "GELU",
            "dropout": 0.00,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 128,
            "max_epochs": 3000,
            "patience": 100,
            "n_models": 55,
            "k_min": 0.005,
            "k_max": 15.0,
            "n_k_bins": 100,
        }

        self.scalers = {}
        self.k_grid = np.logspace(
            np.log10(self.hp["k_min"]), np.log10(self.hp["k_max"]), self.hp["n_k_bins"]
        )
        self.checkpoint_path = Path(checkpoint_path)

        # Attempt load if exists
        if self.checkpoint_path.exists():
            try:
                self.load(self.checkpoint_path)
            except Exception as e:
                print(
                    f"Warning: Failed to load emulator from {self.checkpoint_path}: {e}"
                )

    def get_pk_linear_sigma8(
        self,
        sigma8_target,
        h=0.675,
        ombh2=0.022,
        omch2=0.122,
        ns=0.965,
        kmax=10.0,
        npoints=2000,
    ):
        """
        Calculates linear Matter P(k) normalized to a specific sigma8 using CAMB.
        Note: Contains lazy imports to speed up module loading.
        """
        # LAZY IMPORT
        import camb
        from camb import model

        # 1. Setup CAMB with a dummy As (amplitude)
        dummy_As = 2e-9
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=h * 100, ombh2=ombh2, omch2=omch2)
        pars.InitPower.set_params(ns=ns, As=dummy_As)
        pars.set_matter_power(redshifts=[0.0], kmax=kmax)
        pars.NonLinear = model.NonLinear_none

        # 2. Run CAMB once to get the "fiducial" sigma8
        results = camb.get_results(pars)
        sigma8_fid = results.get_sigma8_0()

        # 3. Calculate the rescaling factor (P(k) ~ As ~ sigma8^2)
        rescaling_factor = (sigma8_target / sigma8_fid) ** 2
        new_As = dummy_As * rescaling_factor

        kh, z, pk_fid = results.get_matter_power_spectrum(
            minkh=1e-4, maxkh=kmax, npoints=npoints
        )
        pk_final = pk_fid * rescaling_factor

        return kh, pk_final[0], new_As

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
            npoints=2000,
        )
        return k_Lin, P_Lin

    def _prepare_data(self):
        """
        Vectorized data preparation.
        """
        all_inputs, all_targets = [], []

        # Pre-compute log10(k) column vector
        # Shape: (n_bins, 1)
        log_k_vec = np.log10(self.k_grid)[:, None]
        n_bins = len(self.k_grid)

        for imodel in range(1, self.hp["n_models"] + 1):
            # 1. Load Data
            cosmo = load_cosmology_wrapper(imodel)
            k_NL, Pk_NL = load_pkmm_data(imodel, return_mean=True)
            k_L, Pk_L = self.get_linear_pk_mm(cosmo)

            # 2. Interpolate to common k-grid
            # We interpolate in log-log space for stability
            spl_NL = InterpolatedUnivariateSpline(np.log10(k_NL), np.log10(Pk_NL), k=3)
            spl_L = InterpolatedUnivariateSpline(np.log10(k_L), np.log10(Pk_L), k=3)

            # Calculate ratio: alpha = P_NL / P_L
            log_ratio = spl_NL(np.log10(self.k_grid)) - spl_L(np.log10(self.k_grid))
            pk_ratio = 10**log_ratio

            # 3. Vectorize Inputs: [Cosmo_Tile, log10(k)]
            cosmo_tiled = np.tile(cosmo, (n_bins, 1))
            model_inputs = np.hstack([cosmo_tiled, log_k_vec])
            all_inputs.append(model_inputs)

            # 4. Vectorize Targets
            all_targets.append(pk_ratio[:, None])

        # Stack all
        inputs = np.vstack(all_inputs)
        targets = np.vstack(all_targets)

        # Scalers
        self.scalers = {
            "in_mean": inputs.mean(axis=0),
            "in_std": inputs.std(axis=0) + 1e-10,
            "tgt_mean": targets.mean(axis=0),
            "tgt_std": targets.std(axis=0) + 1e-10,
        }

        X = (inputs - self.scalers["in_mean"]) / self.scalers["in_std"]
        Y = (targets - self.scalers["tgt_mean"]) / self.scalers["tgt_std"]

        return torch.tensor(X, dtype=torch.float32), torch.tensor(
            Y, dtype=torch.float32
        )

    def train(self):
        X, Y = self._prepare_data()
        dataset = TensorDataset(X, Y)

        n_val = int(0.15 * len(dataset))
        train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])

        train_loader = DataLoader(
            train_set, batch_size=self.hp["batch_size"], shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_set, batch_size=self.hp["batch_size"], num_workers=4
        )

        self.model = MatterAlphaNet(X.shape[1], Y.shape[1], self.scalers, self.hp)

        callbacks = [
            ModelCheckpoint(monitor="val_mse", save_top_k=1, mode="min"),
            EarlyStopping(
                monitor="val_mse", patience=self.hp.get("patience", 50), mode="min"
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        trainer = pl.Trainer(
            max_epochs=self.hp["max_epochs"],
            accelerator="auto",
            devices=1,
            callbacks=callbacks,
            enable_progress_bar=True,
        )

        trainer.fit(self.model, train_loader, val_loader)

        print(f"Loading best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        self.model = MatterAlphaNet.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            input_dim=X.shape[1],
            output_dim=Y.shape[1],
            scalers=self.scalers,
            hp=self.hp,
        )
        self.model.to(self.device)
        self.save()

    def predict(self, cosmo_params, k_array):
        """Predict alpha(k) for a given cosmology and k values."""
        self.model.eval()
        k_array = np.atleast_1d(k_array)

        # Prepare Input: Tile cosmology and stack with log10(k)
        cosmo_tile = np.tile(cosmo_params, (len(k_array), 1))
        # Ensure k_array is (N, 1) for stacking
        raw_in = np.column_stack([cosmo_tile, np.log10(k_array)])

        norm_in = (raw_in - self.scalers["in_mean"]) / self.scalers["in_std"]
        t_in = torch.tensor(norm_in, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            norm_out = self.model(t_in).cpu().numpy()

        # Denormalize
        pred = (norm_out * self.scalers["tgt_std"]) + self.scalers["tgt_mean"]

        # Flatten to (N,) to match input shape
        return pred.flatten()

    def save(self):
        state = {
            "state_dict": self.model.state_dict(),
            "scalers": self.scalers,
            "hp": self.hp,
            "k_grid": self.k_grid,
        }
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, self.checkpoint_path)
        print(f"Emulator saved to {self.checkpoint_path}")

    def load(self, path):
        # weights_only=False required for custom dicts with numpy arrays
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.scalers = checkpoint["scalers"]
        self.hp = checkpoint["hp"]
        self.k_grid = checkpoint["k_grid"]

        self.model = MatterAlphaNet(
            len(self.scalers["in_mean"]),
            len(self.scalers["tgt_mean"]),
            self.scalers,
            self.hp,
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded emulator from {path}")

    def compare(self, imodel, save_plot=True):
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model not loaded.")

        cosmo = load_cosmology_wrapper(imodel)
        k_NL, Pk_NL = load_pkmm_data(imodel, return_mean=True)
        k_L, Pk_L = self.get_linear_pk_mm(cosmo)  # Uses internal method

        # Interpolate truth
        _kk = np.geomspace(1e-2, 14.0, 100)
        spl_NL = InterpolatedUnivariateSpline(np.log10(k_NL), np.log10(Pk_NL), k=3)
        spl_L = InterpolatedUnivariateSpline(np.log10(k_L), np.log10(Pk_L), k=3)
        alpha_true = 10 ** spl_NL(np.log10(_kk)) / 10 ** spl_L(np.log10(_kk))

        alpha_pred = self.predict(cosmo, _kk)  # already flattened

        mse = np.mean((alpha_pred - alpha_true) ** 2)
        mean_frac_error = np.mean(np.abs((alpha_pred - alpha_true) / alpha_true))

        print(f"--- Model {imodel} ---")
        print(f"MSE: {mse:.2e}")
        print(f"Mean Frac Error: {mean_frac_error*100:.2f}%")

        if save_plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(
                2, 1, sharex=True, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}
            )
            ax[0].plot(_kk, alpha_true, "k-", label="Truth")
            ax[0].plot(_kk, alpha_pred, "r--", label="Emulator")
            ax[0].set_ylabel(r"$\alpha(k)$")
            ax[0].set_xscale("log")
            ax[0].legend()
            ax[1].plot(_kk, (alpha_pred / alpha_true) - 1, "r-")
            ax[1].axhline(0, color="k", linestyle=":")
            ax[1].set_ylabel("Frac. Err")
            ax[1].set_ylim([-0.11, 0.11])

            plt.tight_layout()
            plt.savefig(f"compare_alpha_model_{imodel}.pdf")
            plt.close()

        return {"mse": mse, "mean_frac_error": mean_frac_error}


class MatterPkEmulator(MatterAlphaEmulator):
    def __init__(
        self,
        checkpoint_path=MODULE_DIR / "checkpoints/matter_alpha_z0.25.pt",
    ):
        super().__init__(checkpoint_path=checkpoint_path)

    def predict(self, cosmo_params, k_array):
        """
        Predict P_nonlinear(k) = alpha_pred(k) * P_linear_CAMB(k)
        Warning: This calls CAMB and is slower than pure emulation.
        """
        # 1. Get Alpha from NN (fast)
        alpha_pred = super().predict(cosmo_params, k_array)

        # 2. Get Linear Pk from CAMB (slow)
        k_Lin, P_Lin = self.get_linear_pk_mm(cosmo_params)

        # 3. Interpolate Linear Pk to k_array
        spl_L = InterpolatedUnivariateSpline(np.log10(k_Lin), np.log10(P_Lin), k=3)
        P_L_interp = 10 ** spl_L(np.log10(k_array))

        return alpha_pred * P_L_interp

    def compare(self, imodel, save_plot=True):
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model not loaded.")

        cosmo = load_cosmology_wrapper(imodel)
        k_NL, Pk_NL = load_pkmm_data(imodel, return_mean=True)
        mask_k = k_NL <= 12.0
        k_NL = k_NL[mask_k]
        Pk_NL = Pk_NL[mask_k]

        Pk_pred = self.predict(cosmo, k_NL)

        mse = np.mean((Pk_pred - Pk_NL) ** 2)
        mean_frac_error = np.mean(np.abs((Pk_pred - Pk_NL) / Pk_NL))

        print(f"--- Model {imodel} ---")
        print(f"MSE: {mse:.2e}")
        print(f"Mean Frac Error: {mean_frac_error*100:.2f}%")

        if save_plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(
                2, 1, sharex=True, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}
            )
            ax[0].plot(k_NL, k_NL * Pk_NL, "k-", label="Truth")
            ax[0].plot(k_NL, k_NL * Pk_pred, "r--", label="Emulator")
            ax[0].set_ylabel(r"$k P(k)$")
            ax[0].set_xscale("log")
            ax[0].legend()
            ax[1].plot(k_NL, (Pk_pred / Pk_NL) - 1, "r-")
            ax[1].axhline(0, color="k", linestyle=":")
            ax[1].set_ylabel("Frac. Err")
            ax[1].set_ylim([-0.11, 0.11])

            plt.tight_layout()
            plt.savefig(f"compare_Pk_mm_model_{imodel}.pdf")
            plt.close()

        return {"mse": mse, "mean_frac_error": mean_frac_error}
