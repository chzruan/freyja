from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from ..cosma.velocity_moment_hh import load_velocity_moment_hh_transformed
from .xi_R_hh_diffM_dataset import HaloBetaDataset
from .xi_R_hh_diffM_network import HP as HyperParamsDefault
from .xi_R_hh_diffM_network import BetaNet

logger = logging.getLogger(__name__)

MODULE_DIR = Path(__file__).parent

TEST_IMODELS = [44, 53, 60, 61, 62, 63, 64]
ALL_IMODELS = list(range(1, 65))

DEFAULT_TARGET_KEY = "m10_transformed"
DEFAULT_FEATURE_NAMES = ("Om0", "h", "S8", "ns", "logM1", "logM2", "logr")

# Transformation anchor powers used by load_velocity_moment_hh_transformed:
# y = asinh(raw / m10_linear**power)
TARGET_M10_LINEAR_POWERS = {
    "m10_transformed": 1,
    "c20_transformed": 2,
    "c02_transformed": 2,
    "c30_transformed": 3,
    "c12_transformed": 3,
    "c40_transformed": 4,
    "c22_transformed": 4,
    "c04_transformed": 4,
}


class HaloVelocityMomentEmulator:
    """Emulator for transformed halo pairwise velocity moments.

    This class mirrors the overall design of :class:`HaloBetaEmulator` in
    ``freyja.emulators.xi_R_hh_diffM`` while using the transformed halo velocity
    moment data loaded via
    :func:`freyja.cosma.velocity_moment_hh.load_velocity_moment_hh_transformed`.

    The current target is ``m10_transformed`` (scalar per sample), with support
    for other ``*_transformed`` targets through the same pipeline.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = MODULE_DIR
        / "checkpoints/halo_velocity_moment_m10_transformed.pt",
        HP: dict[str, Any] | None = None,
        save_dir: str | Path = ".",
        target_key: str = DEFAULT_TARGET_KEY,
        logM_cut: float = 14.0,
        redshift: float = 0.25,
    ) -> None:
        self.HP: dict[str, Any] = dict(HyperParamsDefault if HP is None else HP)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: BetaNet | None = None
        self.scalers: dict[str, np.ndarray] = {}
        self.r_bins: np.ndarray | None = None

        self.target_key = str(target_key)
        self.logM_cut = float(logM_cut)
        self.redshift = float(redshift)

        self.feature_names = list(DEFAULT_FEATURE_NAMES)
        self.cosmo_param_names = list(DEFAULT_FEATURE_NAMES[:4])
        self.target_name = self.target_key
        self.target_raw_key = self._infer_raw_target_key(self.target_key)
        self.target_err_key = (
            None if self.target_raw_key is None else f"{self.target_raw_key}_err"
        )

        # Dataset / split state (normalized + raw).
        self.prepared = False
        self.sample_imodel_ids: np.ndarray | None = None
        self.sample_split_labels: np.ndarray | None = None

        self.X_all_raw: np.ndarray | None = None
        self.y_all_raw: np.ndarray | None = None
        self.W_all: np.ndarray | None = None

        self.X_train_raw: np.ndarray | None = None
        self.y_train_raw: np.ndarray | None = None
        self.W_train: np.ndarray | None = None
        self.X_val_raw: np.ndarray | None = None
        self.y_val_raw: np.ndarray | None = None
        self.W_val: np.ndarray | None = None
        self.X_test_raw: np.ndarray | None = None
        self.y_test_raw: np.ndarray | None = None
        self.W_test: np.ndarray | None = None

        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.X_val: np.ndarray | None = None
        self.y_val: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_test: np.ndarray | None = None

        self.train_imodels: list[int] = []
        self.val_imodels: list[int] = []
        self.test_imodels: list[int] = []
        self.data_prep_config: dict[str, Any] = {}
        self.data_summary: dict[str, Any] = {}
        self.best_checkpoint_path: str | None = None

        ckpt = None if checkpoint_path is None else Path(checkpoint_path)
        if ckpt is not None and ckpt.exists():
            try:
                self._load_inplace(ckpt)
                logger.info("Successfully loaded HaloVelocityMomentEmulator from %s", ckpt)
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning("Could not load emulator at %s: %s", ckpt, exc)

    @staticmethod
    def _infer_raw_target_key(target_key: str) -> str | None:
        if target_key.endswith("_transformed"):
            return target_key[: -len("_transformed")]
        return None

    def _estimate_target_sigma_transformed(self, data: dict[str, Any]) -> np.ndarray | None:
        """Approximate SEM in transformed space from raw SEM via local Jacobian.

        For targets produced as ``asinh(raw / m10_linear**p)``, the propagated
        uncertainty is

        ``sigma_y = sigma_raw / (|m10_linear**p| * sqrt(1 + (raw/anchor)^2))``.
        """
        power = TARGET_M10_LINEAR_POWERS.get(self.target_key)
        if power is None:
            return None
        if self.target_raw_key is None or self.target_err_key is None:
            return None
        if self.target_raw_key not in data or self.target_err_key not in data:
            return None
        if "m10_linear" not in data:
            return None

        raw = np.asarray(data[self.target_raw_key], dtype=float)
        raw_err = np.asarray(data[self.target_err_key], dtype=float)
        m10_linear = np.asarray(data["m10_linear"], dtype=float)
        anchor = m10_linear**power

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            z = raw / anchor
            deriv_abs = 1.0 / (np.abs(anchor) * np.sqrt(1.0 + z**2))
            sigma_y = raw_err * deriv_abs

        sigma_y = np.asarray(sigma_y, dtype=float)
        sigma_y[~np.isfinite(sigma_y)] = np.nan
        return sigma_y

    def _prepare_inputs_for_imodel(
        self,
        imodel: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Build row-wise samples for one cosmology model.

        Returns
        -------
        X_raw : np.ndarray
            Shape ``(N_rows, 7)`` with columns
            ``[Om0, h, S8, ns, logM1, logM2, logr]``.
        y_raw : np.ndarray
            Shape ``(N_rows, 1)`` for the selected transformed target.
        weights : np.ndarray
            Shape ``(N_rows, 1)`` (inverse variance style, with safe fallback).
        meta : dict
            Counts and filtering diagnostics for this imodel.
        """
        data = load_velocity_moment_hh_transformed(
            imodel,
            redshift=self.redshift,
            logM_cut=self.logM_cut,
        )
        if self.target_key not in data:
            available = sorted(k for k in data.keys() if k.endswith("_transformed"))
            raise KeyError(
                f"Target '{self.target_key}' not found for imodel={imodel}. "
                f"Available transformed targets: {available}"
            )

        cosmo_params = np.asarray(data["cosmo_params"], dtype=float)
        if cosmo_params.shape != (4,):
            raise ValueError(
                f"Expected cosmo_params shape (4,) for imodel={imodel}, got {cosmo_params.shape}"
            )

        logM_bins = np.asarray(data["logM_bins"], dtype=float)
        r_vm = np.asarray(data["r_vm"], dtype=float)
        if np.any(r_vm <= 0.0):
            raise ValueError(f"Encountered non-positive r_vm bins for imodel={imodel}.")
        logr_bins = np.log10(r_vm)

        target_grid = np.asarray(data[self.target_key], dtype=float)
        if target_grid.ndim != 3:
            raise ValueError(
                f"Expected target grid ndim=3 for imodel={imodel}, got {target_grid.ndim}"
            )

        n_m = logM_bins.size
        n_r = r_vm.size
        if target_grid.shape != (n_m, n_m, n_r):
            raise ValueError(
                f"Target grid shape mismatch for imodel={imodel}: "
                f"expected ({n_m}, {n_m}, {n_r}), got {target_grid.shape}"
            )

        if self.r_bins is None:
            self.r_bins = r_vm.copy()
        elif not np.allclose(self.r_bins, r_vm, rtol=1e-10, atol=0.0, equal_nan=True):
            raise ValueError(
                f"Inconsistent r_vm bins for imodel={imodel}; expected shared radial grid."
            )

        # Mirror HaloBetaEmulator's symmetry-aware preparation: use unique mass pairs only.
        idx_i, idx_j = np.triu_indices(n_m)
        n_pairs = idx_i.size
        n_rows_total = int(n_pairs * n_r)

        target_pairs = target_grid[idx_i, idx_j, :]  # (n_pairs, n_r)
        y_raw = target_pairs.reshape(-1, 1)

        logM1_pairs = logM_bins[idx_i]
        logM2_pairs = logM_bins[idx_j]
        logM1 = np.repeat(logM1_pairs, n_r)
        logM2 = np.repeat(logM2_pairs, n_r)
        logr = np.tile(logr_bins, n_pairs)
        cosmo_batch = np.repeat(cosmo_params[None, :], n_rows_total, axis=0)

        X_raw = np.column_stack([cosmo_batch, logM1, logM2, logr]).astype(float, copy=False)

        sigma_y_grid = self._estimate_target_sigma_transformed(data)
        if sigma_y_grid is not None:
            sigma_y = np.asarray(sigma_y_grid[idx_i, idx_j, :], dtype=float).reshape(-1, 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                weights = 1.0 / (sigma_y**2 + float(self.HP.get("loss_epsilon", 1.0e-6)))
            bad_w = ~np.isfinite(weights) | (weights <= 0.0)
            if np.any(bad_w):
                weights = weights.copy()
                weights[bad_w] = 1.0
        else:
            weights = np.ones_like(y_raw, dtype=float)

        finite_mask = (
            np.all(np.isfinite(X_raw), axis=1)
            & np.isfinite(y_raw[:, 0])
            & np.isfinite(weights[:, 0])
            & (weights[:, 0] > 0.0)
        )

        meta = {
            "imodel": int(imodel),
            "rows_before_filter": int(n_rows_total),
            "rows_after_filter": int(finite_mask.sum()),
            "rows_dropped": int(n_rows_total - finite_mask.sum()),
            "n_mass_bins": int(n_m),
            "n_r_bins": int(n_r),
        }

        if not np.any(finite_mask):
            raise ValueError(
                f"All rows were filtered out for imodel={imodel} and target={self.target_key}."
            )

        return X_raw[finite_mask], y_raw[finite_mask], weights[finite_mask], meta

    @staticmethod
    def _split_imodels(
        imodels: Sequence[int],
        *,
        test_imodels: Sequence[int],
        val_fraction: float,
        seed: int,
    ) -> tuple[list[int], list[int], list[int]]:
        if not imodels:
            raise ValueError("No imodels were provided.")

        imodels_unique = sorted({int(m) for m in imodels})
        test_set = {int(m) for m in test_imodels}
        test = [m for m in imodels_unique if m in test_set]
        dev = [m for m in imodels_unique if m not in test_set]

        if len(dev) == 0:
            raise ValueError(
                "No train/val imodels remain after removing TEST_IMODELS. "
                "Provide additional imodels outside the fixed test set."
            )

        if not (0.0 <= float(val_fraction) < 1.0):
            raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}.")

        rng = np.random.default_rng(int(seed))
        dev_arr = np.asarray(dev, dtype=int)
        perm = rng.permutation(dev_arr)

        if len(dev_arr) == 1 or float(val_fraction) == 0.0:
            n_val = 0
        else:
            n_val = int(np.floor(float(val_fraction) * len(dev_arr)))
            n_val = max(1, n_val)
            n_val = min(n_val, len(dev_arr) - 1)

        val = sorted(perm[:n_val].tolist())
        train = sorted(perm[n_val:].tolist())
        return train, val, sorted(test)

    @staticmethod
    def _fit_scalers(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        if X.ndim != 2 or y.ndim != 2:
            raise ValueError("Expected 2D arrays for X and y when fitting scalers.")
        in_mean = np.mean(X, axis=0)
        in_std = np.std(X, axis=0) + 1.0e-10
        tgt_mean = np.mean(y, axis=0)
        tgt_std = np.std(y, axis=0) + 1.0e-10
        return {
            "in_mean": in_mean.astype(float),
            "in_std": in_std.astype(float),
            "tgt_mean": tgt_mean.astype(float),
            "tgt_std": tgt_std.astype(float),
        }

    @staticmethod
    def _normalize(X: np.ndarray, y: np.ndarray, scalers: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        Xn = (X - scalers["in_mean"]) / scalers["in_std"]
        yn = (y - scalers["tgt_mean"]) / scalers["tgt_std"]
        return Xn.astype(float), yn.astype(float)

    def _require_prepared_data(self) -> None:
        if not self.prepared:
            raise RuntimeError(
                "Data are not prepared. Call prepare_data(...) before training/comparing."
            )

    def prepare_data(
        self,
        imodels: Sequence[int] = ALL_IMODELS,
        *,
        test_imodels: Sequence[int] = TEST_IMODELS,
        val_fraction: float = 0.1,
        seed: int = 1234,
    ) -> dict[str, Any]:
        """Load transformed velocity-moment data and build train/val/test splits.

        Parameters
        ----------
        imodels
            Iterable of cosmology model IDs to load.
        test_imodels
            Held-out TEST set IDs. Rows from these imodels are never used for
            train/val.
        val_fraction
            Fraction of non-test imodels assigned to validation.
        seed
            RNG seed for the train/val imodel split.

        Returns
        -------
        dict
            Summary of split sizes and filtering statistics.
        """
        train_imodels, val_imodels, test_imodels_final = self._split_imodels(
            imodels,
            test_imodels=test_imodels,
            val_fraction=val_fraction,
            seed=seed,
        )

        logger.info(
            "Preparing halo velocity moment data for target=%s (train=%d imodels, val=%d, test=%d)",
            self.target_key,
            len(train_imodels),
            len(val_imodels),
            len(test_imodels_final),
        )

        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        w_parts: list[np.ndarray] = []
        sample_imodel_parts: list[np.ndarray] = []
        sample_split_parts: list[np.ndarray] = []
        per_model_meta: list[dict[str, Any]] = []

        split_map = {m: "train" for m in train_imodels}
        split_map.update({m: "val" for m in val_imodels})
        split_map.update({m: "test" for m in test_imodels_final})

        for imodel in sorted({int(m) for m in imodels}):
            split = split_map.get(int(imodel))
            if split is None:
                # Should never happen because split_map is built from the same imodel list.
                continue
            try:
                Xi, yi, wi, meta = self._prepare_inputs_for_imodel(int(imodel))
            except Exception as exc:
                raise RuntimeError(f"Failed to prepare imodel={imodel}: {exc}") from exc

            X_parts.append(Xi)
            y_parts.append(yi)
            w_parts.append(wi)
            sample_imodel_parts.append(np.full(len(Xi), int(imodel), dtype=int))
            sample_split_parts.append(np.full(len(Xi), split, dtype="<U5"))
            meta["split"] = split
            per_model_meta.append(meta)

            logger.info(
                "Prepared imodel=%d [%s]: %d -> %d rows",
                imodel,
                split,
                meta["rows_before_filter"],
                meta["rows_after_filter"],
            )

        if not X_parts:
            raise RuntimeError("No data were loaded. Cannot prepare dataset.")

        X_all_raw = np.concatenate(X_parts, axis=0)
        y_all_raw = np.concatenate(y_parts, axis=0)
        W_all = np.concatenate(w_parts, axis=0)
        sample_imodel_ids = np.concatenate(sample_imodel_parts, axis=0)
        sample_split_labels = np.concatenate(sample_split_parts, axis=0)

        mask_train = sample_split_labels == "train"
        mask_val = sample_split_labels == "val"
        mask_test = sample_split_labels == "test"

        if not np.any(mask_train):
            raise RuntimeError("No training rows available after filtering.")
        if not np.any(mask_val):
            logger.warning(
                "No validation rows available (val_fraction=%s, dev imodel count=%d). "
                "Training will proceed without meaningful validation if train() is called.",
                val_fraction,
                len(train_imodels) + len(val_imodels),
            )
        if not np.any(mask_test):
            logger.warning("No test rows were produced for the requested imodel list.")

        # Fit scalers on train split only.
        scalers = self._fit_scalers(X_all_raw[mask_train], y_all_raw[mask_train])

        X_train, y_train = self._normalize(X_all_raw[mask_train], y_all_raw[mask_train], scalers)
        X_val, y_val = self._normalize(X_all_raw[mask_val], y_all_raw[mask_val], scalers)
        X_test, y_test = self._normalize(X_all_raw[mask_test], y_all_raw[mask_test], scalers)

        self.scalers = scalers
        self.sample_imodel_ids = sample_imodel_ids
        self.sample_split_labels = sample_split_labels

        self.X_all_raw = X_all_raw
        self.y_all_raw = y_all_raw
        self.W_all = W_all

        self.X_train_raw = X_all_raw[mask_train]
        self.y_train_raw = y_all_raw[mask_train]
        self.W_train = W_all[mask_train]
        self.X_val_raw = X_all_raw[mask_val]
        self.y_val_raw = y_all_raw[mask_val]
        self.W_val = W_all[mask_val]
        self.X_test_raw = X_all_raw[mask_test]
        self.y_test_raw = y_all_raw[mask_test]
        self.W_test = W_all[mask_test]

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        self.train_imodels = train_imodels
        self.val_imodels = val_imodels
        self.test_imodels = test_imodels_final
        self.data_prep_config = {
            "imodels": sorted({int(m) for m in imodels}),
            "test_imodels": list(map(int, test_imodels_final)),
            "val_fraction": float(val_fraction),
            "seed": int(seed),
            "target_key": self.target_key,
            "logM_cut": float(self.logM_cut),
            "redshift": float(self.redshift),
            "feature_names": list(self.feature_names),
        }

        def _split_summary(mask: np.ndarray, split_name: str) -> dict[str, Any]:
            Xs = X_all_raw[mask]
            ys = y_all_raw[mask]
            Ws = W_all[mask]
            mids = sample_imodel_ids[mask]
            return {
                "name": split_name,
                "n_rows": int(mask.sum()),
                "n_imodels": int(len(np.unique(mids))),
                "imodels": sorted(np.unique(mids).astype(int).tolist()),
                "x_all_finite_fraction": float(np.all(np.isfinite(Xs), axis=1).mean())
                if len(Xs)
                else 0.0,
                "y_finite_fraction": float(np.isfinite(ys[:, 0]).mean()) if len(ys) else 0.0,
                "weights_finite_fraction": float(np.isfinite(Ws[:, 0]).mean()) if len(Ws) else 0.0,
            }

        self.data_summary = {
            "target_key": self.target_key,
            "feature_names": list(self.feature_names),
            "n_features": int(X_all_raw.shape[1]),
            "n_targets": int(y_all_raw.shape[1]),
            "rows_total": int(len(X_all_raw)),
            "train": _split_summary(mask_train, "train"),
            "val": _split_summary(mask_val, "val"),
            "test": _split_summary(mask_test, "test"),
            "per_imodel": per_model_meta,
        }

        self.prepared = True
        logger.info(
            "Prepared dataset: total=%d rows (train=%d, val=%d, test=%d)",
            self.data_summary["rows_total"],
            self.data_summary["train"]["n_rows"],
            self.data_summary["val"]["n_rows"],
            self.data_summary["test"]["n_rows"],
        )
        return self.data_summary

    def train(
        self,
        model_name: str | Path = "halo_velocity_moment_m10_transformed.pt",
        *,
        epochs: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
        patience: int | None = None,
        seed: int | None = None,
        device: str | None = None,
        num_workers: int = 4,
    ) -> None:
        """Train the emulator on prepared data using the HaloBetaEmulator workflow."""
        self._require_prepared_data()

        if self.X_train is None or self.y_train is None or self.W_train is None:
            raise RuntimeError("Training arrays are missing. Call prepare_data(...) first.")

        if self.X_val is None or self.y_val is None or self.W_val is None or len(self.X_val) == 0:
            raise RuntimeError(
                "Validation split is empty. Increase the number of non-test imodels or set a non-zero val_fraction."
            )

        hp = dict(self.HP)
        if epochs is not None:
            hp["max_epochs"] = int(epochs)
        if batch_size is not None:
            hp["batch_size"] = int(batch_size)
        if lr is not None:
            hp["learning_rate"] = float(lr)
        if patience is not None:
            hp["patience"] = int(patience)

        if seed is not None:
            pl.seed_everything(int(seed), workers=True)

        train_dataset = HaloBetaDataset(self.X_train, self.y_train, self.W_train)
        val_dataset = HaloBetaDataset(self.X_val, self.y_val, self.W_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(hp["batch_size"]),
            shuffle=True,
            num_workers=int(num_workers),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(hp["batch_size"]),
            num_workers=int(num_workers),
        )

        input_dim = int(self.X_train.shape[1])
        output_dim = int(self.y_train.shape[1])
        logger.info("Training HaloVelocityMomentEmulator with input_dim=%d, output_dim=%d", input_dim, output_dim)

        self.model = BetaNet(input_dim, output_dim, self.scalers, hp)

        checkpoint_cb = ModelCheckpoint(monitor="val_mse", save_top_k=1, mode="min")
        callbacks = [
            checkpoint_cb,
            EarlyStopping(monitor="val_mse", patience=int(hp["patience"])),
            LearningRateMonitor(),
        ]

        accelerator = "auto"
        devices = 1
        if device is not None:
            dev = str(device).strip().lower()
            if dev in {"cpu"}:
                accelerator, devices = "cpu", 1
            elif dev in {"gpu", "cuda"}:
                accelerator, devices = "gpu", 1
            elif dev in {"auto"}:
                accelerator, devices = "auto", 1
            else:
                raise ValueError(
                    f"Unsupported device='{device}'. Expected one of: auto, cpu, gpu, cuda."
                )

        trainer = pl.Trainer(
            max_epochs=int(hp["max_epochs"]),
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
        )

        trainer.fit(self.model, train_loader, val_loader)

        if not checkpoint_cb.best_model_path:
            raise RuntimeError("Training finished but no best checkpoint was recorded by ModelCheckpoint.")

        self.best_checkpoint_path = checkpoint_cb.best_model_path
        logger.info("Loading best checkpoint from %s", self.best_checkpoint_path)

        self.model = BetaNet.load_from_checkpoint(
            self.best_checkpoint_path,
            map_location=self.device,
            weights_only=False,
            input_dim=input_dim,
            output_dim=output_dim,
            scalers=self.scalers,
            HP=hp,
        )
        self.model.to(self.device)
        self.model.eval()
        self.HP = hp

        # Match HaloBetaEmulator behavior: save final emulator state after training.
        self.save(model_name)

    def save(self, path: str | Path) -> None:
        """Save model weights, scalers, and metadata (HaloBetaEmulator-style format)."""
        if self.model is None:
            raise RuntimeError("Model is not trained/loaded. Nothing to save.")

        out_path = Path(path)
        if out_path.parent != Path():
            out_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "state_dict": self.model.state_dict(),
            "scalers": self.scalers,
            "r_bins": self.r_bins,
            "hp": self.HP,
            # Extra metadata (kept additive so the core HaloBetaEmulator keys still exist)
            "target_key": self.target_key,
            "target_raw_key": self.target_raw_key,
            "target_err_key": self.target_err_key,
            "feature_names": list(self.feature_names),
            "cosmo_param_names": list(self.cosmo_param_names),
            "logM_cut": float(self.logM_cut),
            "redshift": float(self.redshift),
            "train_imodels": list(self.train_imodels),
            "val_imodels": list(self.val_imodels),
            "test_imodels": list(self.test_imodels),
            "data_prep_config": dict(self.data_prep_config),
            "data_summary": dict(self.data_summary),
        }
        torch.save(state, out_path)
        logger.info("HaloVelocityMomentEmulator saved to %s", out_path)

    def _load_inplace(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if "scalers" not in checkpoint or "state_dict" not in checkpoint:
            raise ValueError(
                f"Invalid checkpoint '{path}': missing required keys 'scalers'/'state_dict'."
            )

        self.scalers = checkpoint["scalers"]
        self.r_bins = checkpoint.get("r_bins", None)
        self.HP = dict(checkpoint.get("hp", self.HP))

        self.target_key = str(checkpoint.get("target_key", self.target_key))
        self.target_raw_key = checkpoint.get("target_raw_key", self._infer_raw_target_key(self.target_key))
        self.target_err_key = checkpoint.get(
            "target_err_key",
            None if self.target_raw_key is None else f"{self.target_raw_key}_err",
        )
        self.feature_names = list(checkpoint.get("feature_names", self.feature_names))
        self.cosmo_param_names = list(
            checkpoint.get("cosmo_param_names", self.cosmo_param_names)
        )
        self.logM_cut = float(checkpoint.get("logM_cut", self.logM_cut))
        self.redshift = float(checkpoint.get("redshift", self.redshift))
        self.train_imodels = list(map(int, checkpoint.get("train_imodels", [])))
        self.val_imodels = list(map(int, checkpoint.get("val_imodels", [])))
        self.test_imodels = list(map(int, checkpoint.get("test_imodels", [])))
        self.data_prep_config = dict(checkpoint.get("data_prep_config", {}))
        self.data_summary = dict(checkpoint.get("data_summary", {}))

        in_mean = np.asarray(self.scalers["in_mean"], dtype=float)
        tgt_mean = np.asarray(self.scalers["tgt_mean"], dtype=float)
        input_dim = int(in_mean.shape[0])
        output_dim = int(np.atleast_1d(tgt_mean).shape[0])

        self.model = BetaNet(
            input_dim=input_dim,
            output_dim=output_dim,
            scalers=self.scalers,
            hp=self.HP,
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load(cls, path: str | Path) -> "HaloVelocityMomentEmulator":
        """Reconstruct an emulator from disk."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        hp = dict(checkpoint.get("hp", HyperParamsDefault))
        obj = cls(
            checkpoint_path=None,
            HP=hp,
            target_key=str(checkpoint.get("target_key", DEFAULT_TARGET_KEY)),
            logM_cut=float(checkpoint.get("logM_cut", 14.0)),
            redshift=float(checkpoint.get("redshift", 0.25)),
        )
        obj._load_inplace(path)
        return obj

    def _broadcast_predict_inputs(
        self,
        cosmo_params: Any,
        logM1: Any,
        logM2: Any,
        logr: Any,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        cosmo = np.asarray(cosmo_params, dtype=float)
        if cosmo.ndim == 1:
            if cosmo.shape[0] != 4:
                raise ValueError(
                    f"cosmo_params must have length 4, got shape {cosmo.shape}."
                )
            cosmo_leading_shape: tuple[int, ...] = ()
        else:
            if cosmo.shape[-1] != 4:
                raise ValueError(
                    f"cosmo_params last dimension must be 4, got shape {cosmo.shape}."
                )
            cosmo_leading_shape = tuple(cosmo.shape[:-1])

        m1 = np.asarray(logM1, dtype=float)
        m2 = np.asarray(logM2, dtype=float)
        lr = np.asarray(logr, dtype=float)
        try:
            broadcast_shape = np.broadcast_shapes(
                m1.shape, m2.shape, lr.shape, cosmo_leading_shape
            )
        except ValueError as exc:
            raise ValueError(
                "Inputs could not be broadcast together: "
                f"cosmo leading shape {cosmo_leading_shape}, "
                f"logM1 {m1.shape}, logM2 {m2.shape}, logr {lr.shape}"
            ) from exc

        try:
            m1_b, m2_b, lr_b = (
                np.broadcast_to(m1, broadcast_shape),
                np.broadcast_to(m2, broadcast_shape),
                np.broadcast_to(lr, broadcast_shape),
            )
        except ValueError as exc:
            raise ValueError(
                f"logM1/logM2/logr could not be broadcast to common shape {broadcast_shape}."
            ) from exc

        n_rows = int(np.prod(broadcast_shape, dtype=int)) if broadcast_shape else 1

        if cosmo.ndim == 1:
            cosmo_flat = np.repeat(cosmo.reshape(1, 4), n_rows, axis=0)
        else:
            target_shape = broadcast_shape + (4,)
            try:
                cosmo_b = np.broadcast_to(cosmo, target_shape)
            except ValueError as exc:
                raise ValueError(
                    f"cosmo_params shape {cosmo.shape} is not broadcastable to {target_shape}."
                ) from exc
            cosmo_flat = np.asarray(cosmo_b, dtype=float).reshape(n_rows, 4)

        X_raw = np.column_stack(
            [
                cosmo_flat,
                m1_b.reshape(-1),
                m2_b.reshape(-1),
                lr_b.reshape(-1),
            ]
        ).astype(float, copy=False)
        return X_raw, broadcast_shape

    def predict(
        self,
        cosmo_params: Any,
        logM1: Any,
        logM2: Any,
        logr: Any,
        *,
        return_numpy: bool = True,
    ) -> np.ndarray | torch.Tensor | float:
        """Predict transformed halo velocity moment values.

        Parameters
        ----------
        cosmo_params
            Cosmology parameters ``[Om0, h, S8, ns]`` or an array broadcastable
            to ``(..., 4)``.
        logM1, logM2, logr
            Scalars or array-like inputs broadcast together.
        return_numpy
            If ``True`` (default), return a NumPy array (or float for scalar
            input). Otherwise return a CPU ``torch.Tensor``.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded/trained.")
        if not self.scalers:
            raise RuntimeError("Scalers are missing; cannot run prediction.")

        X_raw, out_shape = self._broadcast_predict_inputs(cosmo_params, logM1, logM2, logr)

        if not np.all(np.isfinite(X_raw)):
            bad = np.argwhere(~np.isfinite(X_raw))
            first_bad = tuple(int(v) for v in bad[0]) if bad.size else None
            raise ValueError(f"Non-finite prediction inputs encountered at index {first_bad}.")

        X_norm = (X_raw - self.scalers["in_mean"]) / self.scalers["in_std"]
        t_in = torch.tensor(X_norm, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            y_norm = self.model(t_in).detach().cpu().numpy()
        y = y_norm * self.scalers["tgt_std"] + self.scalers["tgt_mean"]
        y = np.asarray(y, dtype=float).reshape(-1)
        y = y.reshape(out_shape if out_shape else ())

        if return_numpy:
            return float(y) if y.shape == () else y
        return torch.as_tensor(y, dtype=torch.float32)

    @staticmethod
    def _metrics_from_truth_pred(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(mask):
            raise ValueError("No finite truth/pred pairs available to compute metrics.")
        yt = y_true[mask]
        yp = y_pred[mask]
        resid = yp - yt
        eps = 1.0e-9
        metrics = {
            "n": float(len(yt)),
            "rmse": float(np.sqrt(np.mean(resid**2))),
            "mae": float(np.mean(np.abs(resid))),
            "mean_residual": float(np.mean(resid)),
            "std_residual": float(np.std(resid)),
            "mean_frac_error": float(np.mean(np.abs(resid) / (np.abs(yt) + eps))),
            "median_frac_error": float(np.median(np.abs(resid) / (np.abs(yt) + eps))),
            "max_abs_error": float(np.max(np.abs(resid))),
        }
        return metrics

    def _get_split_arrays(self, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        split_l = split.lower()
        if split_l == "train":
            X, y, w = self.X_train_raw, self.y_train_raw, self.W_train
        elif split_l == "val":
            X, y, w = self.X_val_raw, self.y_val_raw, self.W_val
        elif split_l == "test":
            X, y, w = self.X_test_raw, self.y_test_raw, self.W_test
        else:
            raise ValueError(f"Unknown split='{split}'. Expected one of: train, val, test.")
        if X is None or y is None or w is None:
            raise RuntimeError(
                f"Split '{split}' is unavailable. Call prepare_data(...) before compare()."
            )
        return X, y, w

    def compare(
        self,
        splits: str | Sequence[str] = ("val", "test"),
        *,
        include_train: bool = False,
        outdir: str | Path | None = None,
        prefix: str | None = None,
        max_scatter_points: int = 50000,
        random_seed: int = 1234,
    ) -> dict[str, dict[str, float]]:
        """Compare predictions vs truth on prepared splits with metrics and plots.

        Parameters
        ----------
        splits
            One split name or a sequence of split names. Typical use:
            ``("val", "test")``.
        include_train
            If True, append ``train`` to the requested splits.
        outdir
            Optional directory for diagnostic figures.
        prefix
            Optional filename prefix for plots.
        max_scatter_points
            Maximum points plotted in scatter panels (subsampled for speed).
        random_seed
            RNG seed for plot subsampling.
        """
        self._require_prepared_data()
        if self.model is None:
            raise RuntimeError("Model not loaded/trained.")

        if isinstance(splits, str):
            split_list = [splits]
        else:
            split_list = [str(s) for s in splits]
        if include_train and "train" not in [s.lower() for s in split_list]:
            split_list = list(split_list) + ["train"]

        split_list = [s.lower() for s in split_list]
        # Preserve order while removing duplicates.
        seen: set[str] = set()
        split_list = [s for s in split_list if not (s in seen or seen.add(s))]

        rng = np.random.default_rng(int(random_seed))
        plot_dir: Path | None = None
        if outdir is not None:
            plot_dir = Path(outdir)
            plot_dir.mkdir(parents=True, exist_ok=True)

        metrics_out: dict[str, dict[str, float]] = {}

        for split in split_list:
            X_raw, y_true, _ = self._get_split_arrays(split)
            if len(X_raw) == 0:
                logger.warning("Skipping compare for split '%s': no rows.", split)
                metrics_out[split] = {
                    "n": 0.0,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "mean_residual": np.nan,
                    "std_residual": np.nan,
                    "mean_frac_error": np.nan,
                    "median_frac_error": np.nan,
                    "max_abs_error": np.nan,
                }
                continue

            y_pred = self.predict(
                X_raw[:, :4],
                X_raw[:, 4],
                X_raw[:, 5],
                X_raw[:, 6],
                return_numpy=True,
            )
            y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1, 1)
            metrics = self._metrics_from_truth_pred(y_true[:, 0], y_pred_arr[:, 0])
            metrics_out[split] = metrics
            logger.info(
                "[%s] n=%d RMSE=%.5e MAE=%.5e mean|frac|=%.5e",
                split,
                int(metrics["n"]),
                metrics["rmse"],
                metrics["mae"],
                metrics["mean_frac_error"],
            )

            if plot_dir is None:
                continue

            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except Exception as exc:  # pragma: no cover - optional runtime dep behavior
                logger.warning("Matplotlib unavailable; skipping plots for split '%s': %s", split, exc)
                continue

            y_t = y_true[:, 0]
            y_p = y_pred_arr[:, 0]
            logr = X_raw[:, 6]
            logM1 = X_raw[:, 4]
            logM2 = X_raw[:, 5]
            resid = y_p - y_t
            frac = np.abs(resid) / (np.abs(y_t) + 1.0e-9)
            finite = (
                np.isfinite(y_t)
                & np.isfinite(y_p)
                & np.isfinite(resid)
                & np.isfinite(logr)
                & np.isfinite(logM1)
                & np.isfinite(logM2)
            )
            if not np.any(finite):
                continue

            idx = np.flatnonzero(finite)
            if len(idx) > int(max_scatter_points):
                idx = np.sort(rng.choice(idx, size=int(max_scatter_points), replace=False))

            y_t_s = y_t[idx]
            y_p_s = y_p[idx]
            resid_s = resid[idx]
            logr_s = logr[idx]
            mean_mass_s = 0.5 * (logM1[idx] + logM2[idx])
            frac_s = frac[idx]

            fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
            ax = axes[0, 0]
            ax.scatter(y_t_s, y_p_s, s=4, alpha=0.35, linewidths=0)
            lim_lo = float(np.nanmin([np.nanmin(y_t_s), np.nanmin(y_p_s)]))
            lim_hi = float(np.nanmax([np.nanmax(y_t_s), np.nanmax(y_p_s)]))
            if np.isfinite(lim_lo) and np.isfinite(lim_hi) and lim_hi > lim_lo:
                ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1)
                ax.set_xlim(lim_lo, lim_hi)
                ax.set_ylim(lim_lo, lim_hi)
            ax.set_xlabel(f"True {self.target_key}")
            ax.set_ylabel(f"Pred {self.target_key}")
            ax.set_title(f"{split}: true vs pred")
            ax.grid(alpha=0.25, linewidth=0.5)

            ax = axes[0, 1]
            ax.scatter(logr_s, resid_s, s=4, alpha=0.35, linewidths=0)
            ax.axhline(0.0, color="k", ls="--", lw=1)
            ax.set_xlabel("log10(r)")
            ax.set_ylabel("Residual (pred - true)")
            ax.set_title(f"{split}: residual vs logr")
            ax.grid(alpha=0.25, linewidth=0.5)

            ax = axes[1, 0]
            ax.scatter(mean_mass_s, resid_s, s=4, alpha=0.35, linewidths=0)
            ax.axhline(0.0, color="k", ls="--", lw=1)
            ax.set_xlabel("0.5*(logM1 + logM2)")
            ax.set_ylabel("Residual (pred - true)")
            ax.set_title(f"{split}: residual vs mean mass")
            ax.grid(alpha=0.25, linewidth=0.5)

            ax = axes[1, 1]
            ax.hist(resid_s, bins=80, alpha=0.85)
            ax.set_xlabel("Residual (pred - true)")
            ax.set_ylabel("Count")
            ax.set_title(
                f"{split}: RMSE={metrics['rmse']:.3e}, MAE={metrics['mae']:.3e}, "
                f"mean|frac|={metrics['mean_frac_error']:.3e}"
            )
            ax.grid(alpha=0.25, linewidth=0.5)

            base = "halo_velocity_moment_compare" if not prefix else str(prefix)
            fig_path = plot_dir / f"{base}_{self.target_key}_{split}.png"
            fig.savefig(fig_path, dpi=180, bbox_inches="tight")
            plt.close(fig)

            # Secondary diagnostic: fractional error vs logr.
            fig2, ax2 = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
            ax2.scatter(logr_s, frac_s, s=4, alpha=0.35, linewidths=0)
            ax2.set_xlabel("log10(r)")
            ax2.set_ylabel("|pred - true| / (|true| + 1e-9)")
            ax2.set_title(f"{split}: fractional error vs logr")
            ax2.grid(alpha=0.25, linewidth=0.5)
            fig2_path = plot_dir / f"{base}_{self.target_key}_{split}_frac_vs_logr.png"
            fig2.savefig(fig2_path, dpi=180, bbox_inches="tight")
            plt.close(fig2)

        return metrics_out


__all__ = [
    "ALL_IMODELS",
    "TEST_IMODELS",
    "HaloVelocityMomentEmulator",
]


if __name__ == "__main__":  # pragma: no cover - manual demo helper
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    emu = HaloVelocityMomentEmulator(checkpoint_path=None)
    emu.prepare_data(ALL_IMODELS, seed=1234, val_fraction=0.1)
    emu.train(
        model_name=MODULE_DIR / "checkpoints/halo_velocity_moment_m10_transformed_demo.pt",
        epochs=5,
        batch_size=1024,
        patience=3,
    )
    metrics = emu.compare(("test",), outdir=MODULE_DIR / "data/halo_vm_compare_demo")
    logger.info("Test metrics: %s", metrics.get("test"))

    save_path = MODULE_DIR / "checkpoints/halo_velocity_moment_m10_transformed_demo_roundtrip.pt"
    emu.save(save_path)
    emu2 = HaloVelocityMomentEmulator.load(save_path)

    # Round-trip prediction sanity check using the first available test row.
    if emu.X_test_raw is not None and len(emu.X_test_raw) > 0:
        row = emu.X_test_raw[0]
        pred = emu2.predict(row[:4], row[4], row[5], row[6])
        logger.info("Round-trip single prediction (%s): %.6e", emu2.target_key, float(pred))
