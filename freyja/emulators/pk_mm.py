import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.interpolate import InterpolatedUnivariateSpline
import camb
from camb import model, initialpower
from ..cosma.xi_hh import load_cosmology_wrapper, load_pkmm_data

MODULE_DIR = Path(__file__).parent


class MatterAlphaNet(pl.LightningModule):
    def __init__(self, input_dim, output_dim, scalers, hp):
        super().__init__()
        self.save_hyperparameters(hp)
        self.scalers = scalers

        layers = []
        in_dim = input_dim
        for h_dim in self.hparams["hidden_layers"]:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(
                nn.GELU() if self.hparams["activation"] == "GELU" else nn.ReLU()
            )
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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer


class MatterAlphaEmulator:
    def __init__(
        self, checkpoint_path=MODULE_DIR / "checkpoints/matter_alpha_z0.25.pt", hp=None
    ):
        """
        Emulator for the ratio alpha(k) = P_nonlinear(k) / P_linear(k).

        Parameters:
        - checkpoint_path: Path to save/load the emulator state.
        - hp: Hyperparameters dictionary for the neural network and training.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        try:
            self.load(checkpoint_path)
        except Exception as e:
            print(f"Failed to load emulator from {checkpoint_path}: {e}")
        self.hp = hp or {
            "hidden_layers": [64, 64],
            "activation": "GELU",
            "dropout": 0.01,
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
        self.model = None
        self.scalers = {}
        self.k_grid = np.logspace(
            np.log10(self.hp["k_min"]), np.log10(self.hp["k_max"]), self.hp["n_k_bins"]
        )
        # Setup CAMB parameters for linear P(k) computation
        self.camb_params = camb.CAMBparams()

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
        Calculates linear Matter P(k) normalized to a specific sigma8.
        """

        # 1. Setup CAMB with a dummy As (amplitude)
        # We use As=2e-9 as a reasonable starting guess
        dummy_As = 2e-9

        pars = camb.CAMBparams()
        pars.set_cosmology(H0=h * 100, ombh2=ombh2, omch2=omch2)
        pars.InitPower.set_params(ns=ns, As=dummy_As)
        pars.set_matter_power(redshifts=[0.0], kmax=kmax)
        pars.NonLinear = model.NonLinear_none

        # 2. Run CAMB once to get the "fiducial" sigma8
        results = camb.get_results(pars)
        sigma8_fid = results.get_sigma8_0()  # Returns sigma8 at z=0

        # 3. Calculate the rescaling factor
        # Since P(k) ~ As ~ sigma8^2:
        # New_As = Old_As * (Target_sigma8 / Old_sigma8)^2
        rescaling_factor = (sigma8_target / sigma8_fid) ** 2
        new_As = dummy_As * rescaling_factor

        # --- OPTION A: Fast Method (Linear Theory Only) ---
        # If you ONLY need linear P(k), you can just multiply the array.
        kh, z, pk_fid = results.get_matter_power_spectrum(
            minkh=1e-4, maxkh=kmax, npoints=npoints
        )
        pk_final = pk_fid * rescaling_factor
        return kh, pk_final[0], new_As

    def load_linear_pkmm_data(self, imodel):
        cosmo = load_cosmology_wrapper(imodel)
        Om0, h, S8, ns = cosmo
        sigma8 = S8 / np.sqrt(Om0 / 0.3)
        omega_b_h2 = 0.0224
        omega_c_h2 = Om0 * h**2 - omega_b_h2
        k_Lin, P_Lin, final_As = self.get_pk_linear_sigma8(
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
        all_inputs, all_targets = [], []

        for imodel in range(1, self.hp["n_models"] + 1):
            cosmo = load_cosmology_wrapper(imodel)
            k_NL, Pk_NL = load_pkmm_data(imodel, return_mean=True)
            k_L, Pk_L = self.load_linear_pkmm_data(imodel)

            # Interpolation to common k-grid
            spl_NL = InterpolatedUnivariateSpline(np.log10(k_NL), np.log10(Pk_NL), k=3)
            spl_L = InterpolatedUnivariateSpline(np.log10(k_L), np.log10(Pk_L), k=3)

            pk_ratio = 10 ** spl_NL(np.log10(self.k_grid)) / 10 ** spl_L(
                np.log10(self.k_grid)
            )

            # Structure: [Om0, h, S8, ns, log10k]
            for i, k_val in enumerate(self.k_grid):
                all_inputs.append(np.concatenate([cosmo, [np.log10(k_val)]]))
                all_targets.append([pk_ratio[i]])

        inputs = np.array(all_inputs)
        targets = np.array(all_targets)

        # Standard Scalers
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
        """
        Train the Neural Network emulator with early stopping and LR monitoring.
        """
        # 1. Prepare data and scalers
        X, Y = self._prepare_data()
        dataset = TensorDataset(X, Y)

        # 2. Split data (85% train, 15% validation)
        n_val = int(0.15 * len(dataset))
        train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])

        train_loader = DataLoader(
            train_set, batch_size=self.hp["batch_size"], shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_set, batch_size=self.hp["batch_size"], num_workers=4
        )

        # 3. Initialize the network
        self.model = MatterAlphaNet(X.shape[1], Y.shape[1], self.scalers, self.hp)

        # 4. Set up Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor="val_mse", save_top_k=1, mode="min"
        )
        early_stop_callback = EarlyStopping(
            monitor="val_mse", patience=self.hp.get("patience", 50), mode="min"
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        # 5. Initialize Trainer
        trainer = pl.Trainer(
            max_epochs=self.hp["max_epochs"],
            accelerator="auto",
            devices=1,
            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        )

        # 6. Run Training
        trainer.fit(self.model, train_loader, val_loader)

        # 7. Load best model and save final state
        print(f"Loading best checkpoint from: {checkpoint_callback.best_model_path}")
        self.model = MatterAlphaNet.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            map_location=self.device,
            input_dim=X.shape[1],
            output_dim=Y.shape[1],
            scalers=self.scalers,
            hp=self.hp,
        )

        self.save()

    def predict(self, cosmo_params, k_array):
        """Predict alpha(k) for a given cosmology and k values."""
        self.model.eval()
        k_array = np.atleast_1d(k_array)
        cosmo_tile = np.tile(cosmo_params, (len(k_array), 1))
        raw_in = np.column_stack([cosmo_tile, np.log10(k_array)])

        norm_in = (raw_in - self.scalers["in_mean"]) / self.scalers["in_std"]
        t_in = torch.tensor(norm_in, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            norm_out = self.model(t_in).cpu().numpy()

        return (norm_out * self.scalers["tgt_std"]) + self.scalers["tgt_mean"]

    def save(self):
        """
        Save the emulator state, including the model weights, scalers, and hyperparameters.
        """
        state = {
            "state_dict": self.model.state_dict(),
            "scalers": self.scalers,
            "hp": self.hp,
            "k_grid": self.k_grid,
        }
        torch.save(state, self.checkpoint_path)
        print(f"Emulator saved to {self.checkpoint_path}")

    def load(self, path):
        """
        Load a saved emulator state from a file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.scalers = checkpoint["scalers"]
        self.hp = checkpoint["hp"]
        self.k_grid = checkpoint["k_grid"]

        input_dim = len(self.scalers["in_mean"])
        output_dim = len(self.scalers["tgt_mean"])

        self.model = MatterAlphaNet(
            input_dim,
            output_dim,
            self.scalers,
            self.hp,
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print(f"Successfully loaded emulator from {path}")

    def compare(self, imodel, save_plot=True):
        """
        Compare the emulator prediction with the truth for a specific simulation model.
        """
        if self.model is None:
            raise ValueError("Model not loaded.")

        # 1. Load Ground Truth Data
        cosmo = load_cosmology_wrapper(imodel)
        k_NL, Pk_NL = load_pkmm_data(imodel, return_mean=True)
        k_L, Pk_L = self.load_linear_pkmm_data(imodel)

        # Interpolate truth to the emulator's k_grid
        _kk = np.geomspace(1e-2, 14.0, 100)
        spl_NL = InterpolatedUnivariateSpline(np.log10(k_NL), np.log10(Pk_NL), k=3)
        spl_L = InterpolatedUnivariateSpline(np.log10(k_L), np.log10(Pk_L), k=3)
        alpha_true = 10 ** spl_NL(np.log10(_kk)) / 10 ** spl_L(np.log10(_kk))  #

        # 2. Get Emulator Prediction
        alpha_pred = self.predict(cosmo, _kk).flatten()

        # 3. Calculate Metrics
        mse = np.mean((alpha_pred - alpha_true) ** 2)
        mean_frac_error = np.mean(np.abs((alpha_pred - alpha_true) / alpha_true))

        metrics = {
            "mse": mse,
            "mean_frac_error": mean_frac_error,
        }

        print(f"--- Evaluation for Model {imodel} ---")
        print(f"MSE: {mse:.2e}")
        print(f"Mean Fractional Error: {mean_frac_error*100:.2f}%")

        # 4. Optional Plotting (adapting freyja's plotting style)
        if save_plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(
                2, 1, sharex=True, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}
            )

            ax[0].plot(_kk, alpha_true, "k-", label="Truth (Sim)")
            ax[0].plot(_kk, alpha_pred, "r--", label="Emulator")
            ax[0].set_ylabel(r"$\alpha(k) = P_{NL}/P_{Lin}$")
            ax[0].set_xscale("log")
            ax[0].legend()

            ax[1].plot(_kk, (alpha_pred / alpha_true) - 1, "r-")
            ax[1].axhline(0, color="k", linestyle=":")
            ax[1].set_ylabel("Fractional Error")
            ax[1].set_ylim([-0.11, 0.11])
            ax[1].set_xlabel(r"$k$ [$h$/Mpc]")

            plt.tight_layout()
            plt.savefig(f"compare_alpha_model_{imodel}.pdf")
            plt.close()

        return metrics
