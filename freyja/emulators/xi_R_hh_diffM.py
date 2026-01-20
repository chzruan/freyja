import torch
import numpy as np
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)

# Local imports
from ..cosma.xi_hh import load_cosmology_wrapper, load_xihh_data, load_ximm_data
from .xi_R_hh_diffM_dataset import HaloBetaDataset
from .xi_R_hh_diffM_network import BetaNet, HP
from .utils_plotting import plot_validation_results

# --- 1. Define Default Path ---
MODULE_DIR = Path(__file__).parent
DEFAULT_CKPT_PATH = MODULE_DIR / "checkpoints" / "halo_beta.pt"


class HaloBetaEmulator:
    def __init__(self, checkpoint_path=DEFAULT_CKPT_PATH, save_dir="."):
        self.save_dir = Path(save_dir)
        self.model = None
        self.scalers = {}
        self.r_bins = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if checkpoint_path is not None:
            ckpt = Path(checkpoint_path)
            if ckpt.exists():
                try:
                    self.load(ckpt)
                    print(f"Successfully loaded the emulator from: {ckpt}")
                except Exception as e:
                    print(f"Warning: Could not load the emulator at {ckpt}. Error: {e}")
            else:
                print(
                    f"Note: Default checkpoint not found at {ckpt}. Initialized empty emulator."
                )

    def _prepare_inputs(self, imodel, r_mask, mask_M, logM_bins):
        """
        Prepare masked halo-halo 2PCF xi_hh(r | logM1, logM2) inputs of cosmology-imodel for emulator training.

        Parameters
        ----------
        imodel : int
            1-64. Identifier of the cosmological model to load correlation data for.
        r_mask : ndarray of bool
            Mask selecting the radial bins (r) to retain.
        mask_M : ndarray of bool
            Mask selecting the halo-mass bins to retain.
        logM_bins : ndarray
            Logarithmic halo-mass bin centers corresponding to the masked data.
        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]
            - inputs: concatenated cosmology parameters with symmetric/antisymmetric
              mass combinations (u, v) for each halo-mass pair.
            - targets: beta = xi_hh / xi_mm values for the selected r bins.
            - weights: inverse-variance weights computed from beta uncertainties.
        """
        try:
            cosmo = load_cosmology_wrapper(imodel)
            _, _, xi_hh, xi_sem = load_xihh_data(imodel)
            # Retrieve full r array from IO to match mask
            r_all, _, _, _ = load_xihh_data(1)
            xi_mm = load_ximm_data(r_all, imodel)

            # Slicing
            xi_hh_cut = xi_hh[mask_M][:, mask_M, :][:, :, r_mask]
            xi_sem_cut = xi_sem[mask_M][:, mask_M, :][:, :, r_mask]
            xi_mm_cut = xi_mm[r_mask]

            beta = xi_hh_cut / xi_mm_cut[None, None, :]
            beta_sem = xi_sem_cut / xi_mm_cut[None, None, :]

            inputs, targets, weights = [], [], []
            N_M = len(logM_bins)

            for i in range(N_M):
                for j in range(i, N_M):
                    u = (logM_bins[i] + logM_bins[j]) / 2.0
                    v = (logM_bins[i] - logM_bins[j]) / 2.0

                    inputs.append(np.concatenate([cosmo, [u, v]]))
                    targets.append(beta[i, j, :])
                    weights.append(1.0 / (beta_sem[i, j, :] ** 2 + HP["loss_epsilon"]))

            return inputs, targets, weights
        except Exception as e:
            print(f"Skipping Model {imodel}: {e}")
            return [], [], []

    def load_data_and_prepare(self):
        print("Loading and processing training data...")
        all_inputs, all_targets, all_weights = [], [], []

        # Get R bins and Masks
        r_all, logM_all, _, _ = load_xihh_data(imodel=1)
        mask_r = (r_all >= HP["r_cut_min"]) & (r_all <= HP["r_cut_max"])
        mask_M = (logM_all <= HP["logM_cut_max"]) & (logM_all >= 0)

        self.r_bins = r_all[mask_r]
        logM_bins = logM_all[mask_M]

        print(f"Output vector size: {len(self.r_bins)}")

        for imodel in range(1, HP["n_models"] + 1):
            i, t, w = self._prepare_inputs(imodel, mask_r, mask_M, logM_bins)
            all_inputs.extend(i)
            all_targets.extend(t)
            all_weights.extend(w)

        # Convert to numpy
        inputs = np.array(all_inputs)
        targets = np.array(all_targets)
        weights = np.array(all_weights)

        # Normalization
        in_mean, in_std = np.mean(inputs, axis=0), np.std(inputs, axis=0) + 1e-10
        tgt_mean, tgt_std = np.mean(targets, axis=0), np.std(targets, axis=0) + 1e-10

        self.scalers = {
            "in_mean": in_mean,
            "in_std": in_std,
            "tgt_mean": tgt_mean,
            "tgt_std": tgt_std,
        }

        return (inputs - in_mean) / in_std, (targets - tgt_mean) / tgt_std, weights

    def train(self, model_name="beta_emulator.pt"):
        X, Y, W = self.load_data_and_prepare()
        dataset = HaloBetaDataset(X, Y, W)

        n_val = int(0.1 * len(dataset))
        train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])

        train_loader = DataLoader(
            train_set, batch_size=HP["batch_size"], shuffle=True, num_workers=4
        )
        val_loader = DataLoader(val_set, batch_size=HP["batch_size"], num_workers=4)

        self.model = BetaNet(
            X.shape[1],
            Y.shape[1],
            self.scalers,
            HP,
        )

        checkpoint = ModelCheckpoint(monitor="val_mse", save_top_k=1, mode="min")
        trainer = pl.Trainer(
            max_epochs=HP["max_epochs"],
            accelerator="auto",
            devices=1,
            callbacks=[
                checkpoint,
                EarlyStopping(monitor="val_mse", patience=HP["patience"]),
                LearningRateMonitor(),
            ],
        )

        trainer.fit(self.model, train_loader, val_loader)

        # Load best and save
        self.model = BetaNet.load_from_checkpoint(
            checkpoint.best_model_path, map_location=self.device, weights_only=False
        )
        self.save(model_name)

    def save(self, path):
        state = {
            "state_dict": self.model.state_dict(),
            "scalers": self.scalers,
            "r_bins": self.r_bins,
            "hp": HP,
        }
        torch.save(state, path)
        print(f"Emulator saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.scalers = checkpoint["scalers"]
        self.r_bins = checkpoint["r_bins"]

        input_dim = len(self.scalers["in_mean"])
        output_dim = len(self.scalers["tgt_mean"])

        self.model = BetaNet(
            input_dim,
            output_dim,
            self.scalers,
            checkpoint["hp"],
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, cosmo_params, u, v):
        if self.model is None:
            raise ValueError("Model not loaded")

        raw_in = np.concatenate([cosmo_params, [u, v]])
        norm_in = (raw_in - self.scalers["in_mean"]) / self.scalers["in_std"]
        t_in = torch.tensor(norm_in, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            norm_out = self.model(t_in).cpu().numpy().flatten()

        return (
            self.r_bins,
            (norm_out * self.scalers["tgt_std"]) + self.scalers["tgt_mean"],
        )

    def compare_model_prediction(self, imodel, label="Test", save_plot=True):
        if self.model is None:
            raise ValueError("Model not loaded.")
        print(f"--- Evaluating Model {imodel} [{label}] ---")

        # Load Truth (Repeating logic from preparation to ensure consistency)
        try:
            cosmo = load_cosmology_wrapper(imodel)
            r_all, logM_all, xi_hh, xi_sem = load_xihh_data(imodel)
            xi_mm = load_ximm_data(r_all, imodel)
        except Exception as e:
            print(f"Error loading model {imodel}: {e}")
            return None

        mask_r = (r_all >= HP["r_cut_min"]) & (r_all <= HP["r_cut_max"])
        mask_M = (logM_all <= HP["logM_cut_max"]) & (logM_all >= 0)

        r_cut = r_all[mask_r]
        logM_cut = logM_all[mask_M]
        N_M = len(logM_cut)

        xi_hh_cut = xi_hh[mask_M][:, mask_M, :][:, :, mask_r]
        xi_sem_cut = xi_sem[mask_M][:, mask_M, :][:, :, mask_r]
        xi_mm_cut = xi_mm[mask_r]

        beta_true = xi_hh_cut / xi_mm_cut[None, None, :]
        beta_sem = xi_sem_cut / xi_mm_cut[None, None, :]
        beta_pred = np.zeros_like(beta_true)

        for i in range(N_M):
            for j in range(i, N_M):
                u = (logM_cut[i] + logM_cut[j]) / 2.0
                v = (logM_cut[i] - logM_cut[j]) / 2.0
                _, pred = self.predict(cosmo, u, v)
                beta_pred[i, j, :] = pred
                beta_pred[j, i, :] = pred

        # Metrics
        chi2_map = ((beta_pred - beta_true) ** 2) / (beta_sem**2 + 1e-9)
        metrics = {
            "mean_chi2": np.mean(chi2_map),
            "mse": np.mean((beta_pred - beta_true) ** 2),
            "mean_frac_error": np.mean(
                np.abs((beta_pred - beta_true) / (beta_true + 1e-9))
            ),
        }
        print(f"Metrics: Chi2/dof={metrics['mean_chi2']:.2f}")

        if save_plot:
            out_path = self.save_dir / f"validation_model_{imodel}_{label}.pdf"
            plot_validation_results(
                out_path,
                r_cut,
                logM_cut,
                beta_true,
                beta_pred,
                beta_sem,
                imodel,
                label,
                metrics,
            )

        return metrics
