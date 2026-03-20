import torch
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)

# Local imports
from ..cosma.xi_hh import load_cosmology_wrapper, load_xihh_data
from ..cosma.xi_mm import load_ximm_data
from .xi_R_hh_diffM_dataset import HaloBetaDataset
from .xi_R_hh_diffM_network import BetaNet
from .xi_R_hh_diffM_network import HP as HyperParamsDefault
from .utils_plotting import plot_validation_results

# Import the Scalar Bias Emulator for normalization
from .halo_linear_bias import HaloLinearBiasEmulator
from .halo_mass_function import HMFEmulator
from .xi_mm import MatterXiEmulator

MODULE_DIR = Path(__file__).parent


class HaloBetaEmulator:
    def __init__(
        self,
        checkpoint_path=MODULE_DIR / "checkpoints/halo_beta_z0.25.pt",
        HP=HyperParamsDefault,
        bias_emulator_path=None,
        save_dir=".",
        redshift=0.25,
    ):
        """
        Emulator for the scale-dependent halo bias beta(r | logM1, logM2) = xi_hh / xi_mm.

        Parameters
        ----------
        HP : dict
            Hyperparameters and configuration for data processing and model training.
        checkpoint_path : str or Path
            Path to the trained emulator checkpoint file. If the file does not exist, the default emulator at redshift 0.25 will be used.
        bias_emulator_path : str or Path, optional
            Path to the linear bias emulator file.
        save_dir : str or Path, optional
            Directory to save outputs such as trained models and plots.
        redshift : float, optional
            Redshift at which the emulator operates. Available redshifts: [0.25,]. Default is 0.25.
        """
        self.redshift = redshift
        self.HP = HP
        self.save_dir = Path(save_dir)
        self.model = None
        self.scalers = {}
        self.r_bins = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Extrapolation state
        self.linear_bias_emulator = None
        self.bias_params_cache = {}  # Cache for power-law fits per cosmology
        self.logM_limit = self.HP.get("logM_cut_max", 13.8)
        self.extrap_window = 0.5  # Width of the fitting window in dex

        # Load Beta Neural Net
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            try:
                self.load(ckpt)
                print(f"Successfully loaded HaloBetaEmulator from: {ckpt}")
            except Exception as e:
                print(f"Warning: Could not load the emulator at {ckpt}. Error: {e}")

        # Load Scalar Bias Emulator (Lazy load or immediate)
        if bias_emulator_path is not None:
            self.load_bias_emulator(bias_emulator_path)
        else:
            # Try default location
            default_bias_path = MODULE_DIR / "checkpoints" / "halo_bias_gp.npz"
            if default_bias_path.exists():
                self.load_bias_emulator(default_bias_path)

    def _prepare_inputs(self, imodel):
        """
        Prepare masked halo-halo 2PCF xi_hh(r | logM1, logM2) inputs of cosmology-imodel for emulator training.

        Parameters
        ----------
        imodel : int
            1-64. Identifier of the cosmological model to load correlation data for.
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
            _, logM_all, xi_hh, xi_sem = load_xihh_data(imodel)
            # Retrieve full r array from IO to match mask
            r_all, _, _, _ = load_xihh_data(1)
            xi_mm = load_ximm_data(r_all, imodel)

            # Slicing
            mask_M = (logM_all <= self.HP["logM_cut_max"]) & (
                logM_all >= self.HP["logM_cut_min"]
            )  # Mask for mass bins
            mask_r = (r_all <= self.HP["r_cut_max"]) & (
                r_all >= self.HP["r_cut_min"]
            )  # Mask for radial bins
            logM_bins = logM_all[mask_M]
            xi_hh_cut = xi_hh[mask_M][:, mask_M, :][:, :, mask_r]
            xi_sem_cut = xi_sem[mask_M][:, mask_M, :][:, :, mask_r]
            xi_mm_cut = xi_mm[mask_r]

            # linear bias
            mask_bias = (r_all <= 75.0) & (r_all >= 35.0)
            ratio_sq = xi_hh[mask_M][:, mask_M, mask_bias] / xi_mm_cut[None, mask_bias]
            b1_sq = np.mean(ratio_sq, axis=2)

            beta = xi_hh_cut / xi_mm_cut[None, None, :] / b1_sq
            beta_sem = xi_sem_cut / xi_mm_cut[None, None, :] / b1_sq

            inputs, targets, weights = [], [], []
            N_M = len(logM_bins)

            for i in range(N_M):
                for j in range(i, N_M):
                    u = (logM_bins[i] + logM_bins[j]) / 2.0
                    v = (logM_bins[i] - logM_bins[j]) / 2.0

                    inputs.append(np.concatenate([cosmo, [u, v]]))
                    targets.append(beta[i, j, :])
                    weights.append(
                        1.0 / (beta_sem[i, j, :] ** 2 + self.HP["loss_epsilon"])
                    )

            return inputs, targets, weights
        except Exception as e:
            print(f"Skipping Model {imodel}: {e}")
            return [], [], []

    def load_data_and_prepare(self):
        print("Loading and processing training data...")
        all_inputs, all_targets, all_weights = [], [], []

        # Get R bins and Masks
        r_all, _, _, _ = load_xihh_data(imodel=1)
        mask_r = (r_all >= self.HP["r_cut_min"]) & (r_all <= self.HP["r_cut_max"])
        self.r_bins = r_all[mask_r]
        print(f"Output vector size: {len(self.r_bins)}")

        for imodel in range(1, self.HP["n_models"] + 1):
            i, t, w = self._prepare_inputs(imodel)
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
        # 1. Load and prepare data
        X, Y, W = self.load_data_and_prepare()
        dataset = HaloBetaDataset(X, Y, W)

        # 2. Split data
        n_val = int(0.15 * len(dataset))
        train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])

        train_loader = DataLoader(
            train_set, batch_size=self.HP["batch_size"], shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_set, batch_size=self.HP["batch_size"], num_workers=4
        )

        # 3. Instantiate the model
        # We capture the dimensions here to reuse them later
        input_dim = X.shape[1]
        output_dim = Y.shape[1]
        print(f"{input_dim = }, {output_dim = }")

        self.model = BetaNet(
            input_dim,
            output_dim,
            self.scalers,
            self.HP,
        )

        # 4. Set up trainer
        checkpoint = ModelCheckpoint(monitor="val_mse", save_top_k=1, mode="min")
        trainer = pl.Trainer(
            max_epochs=self.HP["max_epochs"],
            accelerator="auto",
            devices=1,
            callbacks=[
                checkpoint,
                EarlyStopping(monitor="val_mse", patience=self.HP["patience"]),
                LearningRateMonitor(),
            ],
        )

        # 5. Train
        trainer.fit(self.model, train_loader, val_loader)

        # 6. Load best model
        print(f"Loading best checkpoint from: {checkpoint.best_model_path}")

        self.model = BetaNet.load_from_checkpoint(
            checkpoint.best_model_path,
            map_location=self.device,
            weights_only=False,
            # Explicitly pass the arguments required by BetaNet.__init__
            input_dim=input_dim,
            output_dim=output_dim,
            scalers=self.scalers,
            HP=self.HP,
        )

        # 7. Save the final emulator state
        self.save(model_name)

    def save(self, path):
        state = {
            "state_dict": self.model.state_dict(),
            "scalers": self.scalers,
            "r_bins": self.r_bins,
            "hp": self.HP,
        }
        torch.save(state, path)
        print(f"Emulator saved to {path}")

    def load_bias_emulator(self, path):
        """Loads the scalar HaloLinearBiasEmulator used for high-mass extrapolation."""
        try:
            self.linear_bias_emulator = HaloLinearBiasEmulator(saved_path=path)
            print(f"Loaded auxiliary HaloLinearBiasEmulator from {path}")
        except Exception as e:
            print(f"Warning: Failed to load HaloLinearBiasEmulator: {e}")

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

    def predict_uv(self, cosmo_params, u, v):
        """
        Raw prediction using the Neural Network (Interpolation only).
        Expects u, v within training bounds.
        """
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

    def predict_from_masses(self, cosmo_params, logM1, logM2):
        """
        Primary prediction method for physical usage.
        Handles coordinate sorting, interpolation, and extrapolation automatically.

        Parameters
        ----------
        cosmo_params : array-like
            Cosmology [Om0, h, S8, ns]
        logM1, logM2 : float
            Log10 halo masses.

        Returns
        -------
        r_bins : np.ndarray
        beta : np.ndarray
            Predicted scale-dependent bias.
        """
        if self.linear_bias_emulator is None and (
            logM1 > self.logM_limit or logM2 > self.logM_limit
        ):
            print(
                "Warning: Extrapolation requested but HaloLinearBiasEmulator not loaded. Clamping inputs."
            )
            # Fallback: Clamp and predict without renormalization (inaccurate amplitude)
            return self.predict(
                cosmo_params,
                min((logM1 + logM2) / 2, self.logM_limit),
                -abs(logM1 - logM2) / 2,
            )

        # 1. Coordinate transformation & Sorting
        # Network trained on v <= 0 (i.e., m2 >= m1 in sorted sense)
        m_a, m_b = min(logM1, logM2), max(logM1, logM2)
        u_target = (m_a + m_b) / 2.0
        v_target = (m_a - m_b) / 2.0  # v <= 0

        return self.predict_uv(cosmo_params, u_target, v_target)

    def predict_noextrapolation(self, cosmo_params, logM_bins):
        """
        Fast evaluation of the emulator for a 2D grid of mass bins.

        Parameters
        ----------
        cosmo_params : np.ndarray
            1D array of cosmological parameters.
        logM_bins : np.ndarray
            1D array of logarithmic halo-mass bin centers.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - r_bins: Shape (N_r_bins). The radial bins associated with the predictions.
            - beta_matrix: 3D array of shape (N_M, N_M, N_r_bins) containing the predicted beta values beta(r | logM, logM). The matrix is symmetric in the first two dimensions.
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        N_M = len(logM_bins)

        # 1. Generate indices for the upper triangle (including diagonal)
        # This reduces compute by ~50% and respects the symmetry of the problem.
        idx_i, idx_j = np.triu_indices(N_M)

        m1 = logM_bins[idx_i]
        m2 = logM_bins[idx_j]

        # 2. Compute symmetric (u) and antisymmetric (v) coordinates
        u = (m1 + m2) / 2.0
        # Ensure v is negative to match the training domain (j >= i implies m2 >= m1 for sorted bins)
        v = -np.abs(m1 - m2) / 2.0

        # 3. Construct batch input: [cosmo_params, u, v]
        n_pairs = len(u)
        cosmo_batch = np.tile(cosmo_params, (n_pairs, 1))
        raw_in = np.column_stack([cosmo_batch, u, v])

        # 4. Normalize inputs
        norm_in = (raw_in - self.scalers["in_mean"]) / self.scalers["in_std"]

        # 5. Batch Prediction
        t_in = torch.tensor(norm_in, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            norm_out = self.model(t_in).cpu().numpy()

        # 6. Denormalize outputs
        pred_out = (norm_out * self.scalers["tgt_std"]) + self.scalers["tgt_mean"]

        # 7. Fill the symmetric output matrix
        beta_matrix = np.zeros((N_M, N_M, len(self.r_bins)))
        beta_matrix[idx_i, idx_j] = pred_out
        beta_matrix[idx_j, idx_i] = pred_out

        return self.r_bins, beta_matrix

    def predict(self, cosmo_params, logM_bins):
        mask_extrap = logM_bins > self.logM_limit
        
        # Optimization for no-extrapolation case
        if not np.any(mask_extrap):
            return self.predict_noextrapolation(cosmo_params, logM_bins)

        idx_noextrap = np.where(~mask_extrap)[0]
        idx_extrap = np.where(mask_extrap)[0]
        
        logM_noextrap = logM_bins[idx_noextrap]

        # predict_noextrapolation can be called with an empty array.
        # It will return r_bins and an empty beta_matrix.
        r_bins, beta_noextrap_submatrix = self.predict_noextrapolation(
            cosmo_params, logM_noextrap
        )
        
        n_bins = len(logM_bins)
        n_r = len(r_bins)
        beta_full = np.zeros((n_bins, n_bins, n_r))

        if len(logM_noextrap) > 0:
            beta_full[np.ix_(idx_noextrap, idx_noextrap)] = beta_noextrap_submatrix

        if len(idx_extrap) > 0:
            logM_extrap = logM_bins[idx_extrap]
            
            linear_bias_extrap = self.linear_bias_emulator.predict(cosmo_params, logM_extrap)
            linear_bias_limit = self.linear_bias_emulator.predict(cosmo_params, np.array([self.logM_limit]))[0]
            
            _, beta_limit_submatrix = self.predict_noextrapolation(cosmo_params, np.array([self.logM_limit]))
            beta_limit = beta_limit_submatrix[0, 0, :]

            # This is beta_norm(M) ~ b(M)/b(limit) * beta_norm(limit)
            ratio_b = linear_bias_extrap / linear_bias_limit
            beta_extrap_M_r = np.outer(ratio_b, beta_limit) # Shape (N_extrap, N_r)
            
            # (m, M) block - assuming independent of m
            if len(idx_noextrap) > 0:
                block_m_M = np.tile(beta_extrap_M_r, (len(idx_noextrap), 1, 1))
                beta_full[np.ix_(idx_noextrap, idx_extrap)] = block_m_M
                
                # (M, m) block
                # np.transpose swaps axes. We need to swap 0 and 1.
                beta_full[np.ix_(idx_extrap, idx_noextrap)] = np.transpose(block_m_M, (1, 0, 2))
            
            # (M, M') block
            b_MM_outer = np.outer(linear_bias_extrap, linear_bias_extrap)
            ratio_b_sq = b_MM_outer / (linear_bias_limit**2)
            beta_extrap_M_M = ratio_b_sq[:, :, np.newaxis] * beta_limit[np.newaxis, np.newaxis, :]
            beta_full[np.ix_(idx_extrap, idx_extrap)] = beta_extrap_M_M

        return r_bins, beta_full

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

        mask_r = (r_all >= self.HP["r_cut_min"]) & (r_all <= self.HP["r_cut_max"])
        mask_M = (logM_all <= self.HP["logM_cut_max"]) & (logM_all >= 0)

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
                _, pred = self.predict_uv(cosmo, u, v)
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


class HaloXiDiffMCalculator(HaloBetaEmulator, MatterXiEmulator):
    """
    Calculator for xi_hh(r | M1, M2) by stitching two approximations:
    - Small scales: xi_hh = beta(r|M1,M2) * xi_mm(r) from emulators.
    - Large scales: xi_hh = b(M1) * b(M2) * xi_mm(r) from linear bias.
    """

    def __init__(self, redshift=0.25, **kwargs):
        """
        Initializes the calculator by setting up the necessary emulators.

        Parameters
        ----------
        redshift : float, optional
            The operational redshift for all emulators. Default is 0.25.
        **kwargs : dict, optional
            Additional keyword arguments passed to the HaloBetaEmulator,
            such as `checkpoint_path`, `bias_emulator_path`, etc.
        """
        # Initialize the beta emulator, which internally handles the linear bias emulator
        HaloBetaEmulator.__init__(self, redshift=redshift, **kwargs)

        # Initialize the matter correlation function emulator
        MatterXiEmulator.__init__(self, redshift=redshift)

        # Define the stitching scale between small-scale and large-scale regimes
        self.r_stitch = 35.0

        print(
            f"HaloXiDiffMCalculator initialized. Stitching scale r_stitch = {self.r_stitch:.1f} Mpc/h"
        )

    def calculate_M12(self, cosmo_params, logM1, logM2, r):
        """
        Predicts the halo-halo correlation function xi_hh(r | M1, M2).

        Parameters
        ----------
        cosmo_params : np.ndarray
            Cosmological parameters for the emulators.
        logM1, logM2 : float
            Log10 of the two halo masses.
        r : np.ndarray
            Array of radial scales (r) in Mpc/h to compute xi_hh at.

        Returns
        -------
        np.ndarray
            The predicted halo-halo correlation function xi_hh(r).
        """
        # 1. Predict beta(r) on its native r-bins from the HaloBetaEmulator
        r_beta_native, beta_native = self.predict_from_masses(
            cosmo_params, logM1, logM2
        )

        # 2. Interpolate beta to the requested r-grid
        beta_interp = np.interp(
            r, r_beta_native, beta_native, left=beta_native[0], right=beta_native[-1]
        )

        # 3. Predict the matter correlation function xi_mm(r)
        xi_mm = MatterXiEmulator.predict(self, cosmo_params, r=r)

        # 4. Compute the small-scale prediction: xi_hh = beta * xi_mm
        xi_hh_small_scales = beta_interp * xi_mm

        # 5. Compute the large-scale prediction: xi_hh = b1 * b2 * xi_mm
        b1 = self.get_linear_bias(cosmo_params, logM1)
        b2 = self.get_linear_bias(cosmo_params, logM2)
        xi_hh_large_scales = b1 * b2 * xi_mm

        # 6. Smoothly stitch the two regimes using a sigmoid function
        transition_width = self.HP.get("stitching_width", 2.0)  # in Mpc/h
        stitching_weights = 1.0 / (
            1.0 + np.exp(-(r - self.r_stitch) / transition_width)
        )

        xi_hh_final = (
            1.0 - stitching_weights
        ) * xi_hh_small_scales + stitching_weights * xi_hh_large_scales

        return xi_hh_final

    def calculate_mass_matrix(self, cosmo_params, logM_bins, r):
        """
        Calculates the full halo-halo correlation function matrix xi_hh(r | M_i, M_j)
        for a given set of mass bins.

        Parameters
        ----------
        cosmo_params : np.ndarray
            Cosmological parameters.
        logM_bins : np.ndarray
            1D array of log10 halo mass bin centers.
        r : np.ndarray
            Array of radial scales (r) in Mpc/h.

        Returns
        -------
        np.ndarray
            A 3D array of shape (N_M, N_M, N_r) containing the predicted
            xi_hh(r | M_i, M_j).
        """
        # 1. Get beta matrix from the emulator.
        # predict_noextrapolation returns beta on the emulator's native r-bins.
        r_beta_native, beta_matrix_native = self.predict_noextrapolation(
            cosmo_params, logM_bins
        )

        # 2. Interpolate the beta matrix to the desired r grid.
        from scipy.interpolate import interp1d

        f_interp = interp1d(
            r_beta_native, beta_matrix_native, axis=-1, fill_value="extrapolate"
        )
        beta_matrix = f_interp(r)

        # 3. Get xi_mm(r) for all scales
        xi_mm = MatterXiEmulator.predict(self, cosmo_params, r)

        # 4. Calculate xi_hh on small scales
        xi_hh_small_scales = beta_matrix * xi_mm[None, None, :]

        # 5. Get linear biases for large scales
        biases = np.array(
            [self.get_linear_bias(cosmo_params, logM) for logM in logM_bins]
        )
        b_ij = np.outer(biases, biases)

        # 6. Calculate xi_hh on large scales
        xi_hh_large_scales = b_ij[:, :, None] * xi_mm[None, None, :]

        # 7. Stitch the results
        transition_width = self.HP.get("stitching_width", 2.0)
        weights = 1.0 / (1.0 + np.exp(-(r - self.r_stitch) / transition_width))

        xi_hh_final = (1.0 - weights[None, None, :]) * xi_hh_small_scales + weights[
            None, None, :
        ] * xi_hh_large_scales

        return xi_hh_final
