"""PyTorch Lightning emulator for halo/galaxy velocity moments.

This module provides:

1. ``VelocityMomentDataModule`` for loading halo-halo velocity-moment data from
   ``load_velocity_moment_hh_transformed`` and constructing train/val/test
   splits at the *cosmology-model* level.
2. ``VelocityMomentEmulator``: a symmetry-aware MLP (permutation symmetry in
   the halo masses) that predicts the full radial vector of a chosen velocity
   moment (default: physical-space ``m10``).
3. Lightweight scaler utilities (no sklearn dependency) and checkpoint /
   inference helpers.

Notes
-----
- The underlying loader returns both raw and transformed moment arrays. By
  default this module trains on raw ``m10`` so ``predict()`` returns physical
  units after inverse scaling. If you prefer the transformed target, set
  ``target_key="m10_transformed"`` in ``VelocityMomentDataConfig``.
- Mass-input representation is configurable. Default is raw mass (``"mass"``),
  but the loader naturally provides ``log10M`` bins and this module can use
  either.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    # Preferred relative import when used inside the freyja package.
    from ..cosma.velocity_moment_hh import load_velocity_moment_hh_transformed

    _LOADER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import environment dependent
    load_velocity_moment_hh_transformed = None  # type: ignore[assignment]
    _LOADER_IMPORT_ERROR = exc


ArrayLike = Union[np.ndarray, Sequence[float], float]


def _ensure_2d_float32(x: np.ndarray) -> np.ndarray:
    """Return a contiguous ``float32`` 2D array."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("Expected a 2D array.")
    return np.ascontiguousarray(x.astype(np.float32, copy=False))


@dataclass
class VelocityMomentSplitConfig:
    """Controls train/validation/test partitioning at the cosmology-model level."""

    n_models_total: int = 64
    test_model_ids: Tuple[int, ...] = (44, 53, 60, 61, 62, 63, 64)
    val_fraction: float = 0.2
    random_seed: int = 1234

    @classmethod
    def from_any(
        cls, cfg: Optional[Union["VelocityMomentSplitConfig", Mapping[str, Any]]]
    ) -> "VelocityMomentSplitConfig":
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg
        if isinstance(cfg, Mapping):
            return cls(**dict(cfg))
        raise TypeError("data_split_config must be None, a dict-like object, or VelocityMomentSplitConfig.")


@dataclass
class VelocityMomentDataConfig:
    """Data-building and preprocessing options."""

    target_key: str = "m10"
    # ``mass_input_space`` defines the raw mass representation in X_raw.
    # - "mass": X stores M (physical); model applies log() internally.
    # - "log10": X stores log10(M); model uses values directly as log masses.
    # - "ln": X stores ln(M); model uses values directly as log masses.
    mass_input_space: str = "mass"
    use_abs_mass_difference: bool = True
    use_upper_triangle_only: bool = True
    include_diagonal: bool = True
    logM_cut: Optional[float] = 14.0
    gravity: str = "LCDM"
    redshift: float = 0.25
    # Extra kwargs passed to the custom loader if needed.
    loader_kwargs: Optional[Dict[str, Any]] = None

    @classmethod
    def from_any(
        cls, cfg: Optional[Union["VelocityMomentDataConfig", Mapping[str, Any]]]
    ) -> "VelocityMomentDataConfig":
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg
        if isinstance(cfg, Mapping):
            return cls(**dict(cfg))
        raise TypeError("data_config must be None, a dict-like object, or VelocityMomentDataConfig.")


@dataclass
class VelocityMomentModelConfig:
    """Neural network and optimization hyperparameters."""

    hidden_dim: int = 128
    n_hidden_layers: int = 2
    activation: str = "gelu"  # "gelu" or "silu"
    dropout: float = 0.0
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    plateau_factor: float = 0.5
    plateau_patience: int = 8
    plateau_min_lr: float = 1.0e-6
    use_abs_mass_difference: bool = True
    mass_input_space: str = "mass"
    # Lightning logs:
    log_train_step: bool = False

    @classmethod
    def from_any(
        cls, cfg: Optional[Union["VelocityMomentModelConfig", Mapping[str, Any]]]
    ) -> "VelocityMomentModelConfig":
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg
        if isinstance(cfg, Mapping):
            return cls(**dict(cfg))
        raise TypeError("model_config must be None, a dict-like object, or VelocityMomentModelConfig.")


class StandardScalerND:
    """Simple feature-wise standardization utility for 2D arrays.

    This replaces ``sklearn.preprocessing.StandardScaler`` so the module remains
    self-contained (numpy/torch/lightning only).
    """

    def __init__(self, eps: float = 1.0e-12) -> None:
        self.eps = float(eps)
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "StandardScalerND":
        x = _ensure_2d_float32(x)
        if x.shape[0] == 0:
            raise ValueError("Cannot fit scaler on an empty array.")
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std = np.where(np.isfinite(std) & (std > self.eps), std, 1.0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        self.mean_ = mean.astype(np.float32, copy=False)
        self.scale_ = std.astype(np.float32, copy=False)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fit before calling transform().")
        x = _ensure_2d_float32(x)
        return (x - self.mean_[None, :]) / self.scale_[None, :]

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fit before calling inverse_transform().")
        x = _ensure_2d_float32(x)
        return x * self.scale_[None, :] + self.mean_[None, :]

    def state_dict(self) -> Dict[str, np.ndarray]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fit yet.")
        return {"mean": self.mean_.copy(), "scale": self.scale_.copy()}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, np.ndarray]) -> "StandardScalerND":
        obj = cls()
        obj.mean_ = np.asarray(state["mean"], dtype=np.float32).copy()
        obj.scale_ = np.asarray(state["scale"], dtype=np.float32).copy()
        return obj


def _activation_module(name: str) -> nn.Module:
    key = str(name).lower()
    if key == "gelu":
        return nn.GELU()
    if key == "silu":
        return nn.SiLU()
    raise ValueError("Unsupported activation '{}'. Use 'gelu' or 'silu'.".format(name))


def _build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    n_hidden_layers: int,
    activation: str,
    dropout: float = 0.0,
) -> nn.Sequential:
    if n_hidden_layers < 1:
        raise ValueError("n_hidden_layers must be >= 1.")
    layers: List[nn.Module] = []
    last_dim = int(in_dim)
    for _ in range(int(n_hidden_layers)):
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(_activation_module(activation))
        if dropout and dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, int(out_dim)))
    return nn.Sequential(*layers)


def _as_numpy_1d(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr[None]
    if arr.ndim != 1:
        raise ValueError("{} must be a scalar or 1D array.".format(name))
    return arr


def _symmetry_transform_numpy(
    x_raw: np.ndarray,
    n_cosmo_params: int,
    mass_input_space: str = "mass",
    use_abs_mass_difference: bool = True,
) -> np.ndarray:
    """Convert raw inputs ``[cosmo..., M1, M2]`` to symmetry-aware features."""
    x_raw = _ensure_2d_float32(x_raw)
    if x_raw.shape[1] != n_cosmo_params + 2:
        raise ValueError(
            "Expected {} raw input columns ({} cosmo + M1 + M2), got {}.".format(
                n_cosmo_params + 2, n_cosmo_params, x_raw.shape[1]
            )
        )

    cosmo = x_raw[:, :n_cosmo_params]
    m1 = x_raw[:, n_cosmo_params]
    m2 = x_raw[:, n_cosmo_params + 1]

    mass_mode = str(mass_input_space).lower()
    if mass_mode == "mass":
        if np.any(m1 <= 0.0) or np.any(m2 <= 0.0):
            raise ValueError("Mass inputs must be positive when mass_input_space='mass'.")
        logm1 = np.log(m1)
        logm2 = np.log(m2)
    elif mass_mode in ("log10", "ln"):
        logm1 = m1
        logm2 = m2
    else:
        raise ValueError("Unsupported mass_input_space '{}': use 'mass', 'log10', or 'ln'.".format(mass_input_space))

    u = 0.5 * (logm1 + logm2)
    v = 0.5 * (logm1 - logm2)
    if use_abs_mass_difference:
        v = np.abs(v)

    out = np.concatenate([cosmo, u[:, None], v[:, None]], axis=1)
    return np.ascontiguousarray(out.astype(np.float32, copy=False))


def _mass_pair_indices(
    n_mass_bins: int,
    *,
    use_upper_triangle_only: bool,
    include_diagonal: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return flattened (i, j) indices for the selected mass-pair rows."""
    if int(n_mass_bins) <= 0:
        raise ValueError("n_mass_bins must be positive.")

    if use_upper_triangle_only:
        k = 0 if include_diagonal else 1
        i_idx, j_idx = np.triu_indices(int(n_mass_bins), k=k)
        return i_idx.astype(np.int64, copy=False), j_idx.astype(np.int64, copy=False)

    ii, jj = np.indices((int(n_mass_bins), int(n_mass_bins)), dtype=np.int64)
    i_idx = ii.reshape(-1)
    j_idx = jj.reshape(-1)
    if not include_diagonal:
        keep = i_idx != j_idx
        i_idx = i_idx[keep]
        j_idx = j_idx[keep]
    return i_idx.astype(np.int64, copy=False), j_idx.astype(np.int64, copy=False)


class VelocityMomentTensorDataset(Dataset):
    """Torch dataset storing raw inputs and normalized targets."""

    def __init__(self, x_raw: np.ndarray, y_scaled: np.ndarray) -> None:
        self.x_raw = torch.as_tensor(_ensure_2d_float32(x_raw), dtype=torch.float32)
        self.y_scaled = torch.as_tensor(_ensure_2d_float32(y_scaled), dtype=torch.float32)
        if self.x_raw.shape[0] != self.y_scaled.shape[0]:
            raise ValueError("x_raw and y_scaled must have the same number of rows.")

    def __len__(self) -> int:
        return int(self.x_raw.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x_raw[idx], self.y_scaled[idx]


def build_velocity_moment_arrays_from_loader(
    model_ids: Optional[Sequence[int]] = None,
    data_config: Optional[Union[VelocityMomentDataConfig, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build flattened row-wise arrays from the custom loader.

    Returns a dictionary with:
    - ``X_raw``: shape ``(N_samples, N_cosmo + 2)``
    - ``y``: shape ``(N_samples, N_r)``
    - ``row_model_ids``: shape ``(N_samples,)``
    - ``r_bins``, ``target_key``, ``n_cosmo_params`` and other metadata

    This is useful outside Lightning as a lightweight preprocessing entry point.
    """

    cfg = VelocityMomentDataConfig.from_any(data_config)
    if load_velocity_moment_hh_transformed is None:
        raise ImportError(
            "Could not import load_velocity_moment_hh_transformed. "
            "Original error: {}".format(_LOADER_IMPORT_ERROR)
        )

    ids = list(model_ids) if model_ids is not None else list(range(1, 64 + 1))
    rows_x: List[np.ndarray] = []
    rows_y: List[np.ndarray] = []
    rows_model_id: List[np.ndarray] = []
    rows_pair_idx: List[np.ndarray] = []

    r_bins_ref: Optional[np.ndarray] = None
    n_cosmo_params_ref: Optional[int] = None
    pair_index_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    loader_kwargs = dict(cfg.loader_kwargs or {})
    for imodel in ids:
        payload = load_velocity_moment_hh_transformed(
            int(imodel),
            gravity=cfg.gravity,
            redshift=cfg.redshift,
            logM_cut=cfg.logM_cut,
            **loader_kwargs
        )
        cosmo_params = np.asarray(payload["cosmo_params"], dtype=np.float32).reshape(-1)
        logm_bins = np.asarray(payload["logM_bins"], dtype=np.float32).reshape(-1)
        r_bins = np.asarray(payload["r_vm"], dtype=np.float32).reshape(-1)

        if cfg.target_key not in payload:
            raise KeyError(
                "target_key='{}' not found in loader payload keys.".format(cfg.target_key)
            )
        y_cube = np.asarray(payload[cfg.target_key], dtype=np.float32)
        if y_cube.ndim != 3:
            raise ValueError(
                "Expected target cube with shape (N_M, N_M, N_r); got {}.".format(y_cube.shape)
            )

        n_m1, n_m2, n_r = y_cube.shape
        if n_m1 != len(logm_bins) or n_m2 != len(logm_bins):
            raise ValueError(
                "Target cube mass dimensions {} do not match logM bins {}.".format(y_cube.shape[:2], len(logm_bins))
            )
        if n_r != len(r_bins):
            raise ValueError(
                "Target cube radial dimension {} does not match r_vm length {}.".format(n_r, len(r_bins))
            )

        if r_bins_ref is None:
            r_bins_ref = r_bins.copy()
        elif not np.allclose(r_bins, r_bins_ref, rtol=1e-6, atol=0.0):
            raise ValueError("r_vm bins differ between models; fixed-output emulator requires a common radial grid.")

        if n_cosmo_params_ref is None:
            n_cosmo_params_ref = int(cosmo_params.shape[0])
        elif int(cosmo_params.shape[0]) != n_cosmo_params_ref:
            raise ValueError("cosmo_params dimension changed across models.")

        if str(cfg.mass_input_space).lower() == "mass":
            mass_values = np.power(10.0, logm_bins.astype(np.float64)).astype(np.float32)
        elif str(cfg.mass_input_space).lower() == "log10":
            mass_values = logm_bins.astype(np.float32, copy=True)
        elif str(cfg.mass_input_space).lower() == "ln":
            mass_values = (np.log(10.0) * logm_bins.astype(np.float64)).astype(np.float32)
        else:
            raise ValueError(
                "Unsupported mass_input_space '{}': use 'mass', 'log10', or 'ln'.".format(cfg.mass_input_space)
            )

        # Flatten (mass-pair) rows in bulk; each row target is the full radial vector.
        if n_m1 != n_m2:
            raise ValueError("Expected a square mass-pair grid, got ({}, {}).".format(n_m1, n_m2))
        if n_m1 not in pair_index_cache:
            pair_index_cache[n_m1] = _mass_pair_indices(
                n_m1,
                use_upper_triangle_only=cfg.use_upper_triangle_only,
                include_diagonal=cfg.include_diagonal,
            )
        i_idx, j_idx = pair_index_cache[n_m1]

        y_rows = np.asarray(y_cube[i_idx, j_idx, :], dtype=np.float32)
        n_pairs = int(y_rows.shape[0])
        if n_pairs == 0:
            continue

        x_rows = np.empty((n_pairs, int(cosmo_params.shape[0]) + 2), dtype=np.float32)
        x_rows[:, :-2] = cosmo_params.astype(np.float32, copy=False)[None, :]
        x_rows[:, -2] = mass_values[i_idx]
        x_rows[:, -1] = mass_values[j_idx]

        keep_rows = np.all(np.isfinite(y_rows), axis=1) & np.all(np.isfinite(x_rows), axis=1)
        if not np.any(keep_rows):
            continue

        rows_x.append(np.ascontiguousarray(x_rows[keep_rows], dtype=np.float32))
        rows_y.append(np.ascontiguousarray(y_rows[keep_rows], dtype=np.float32))
        rows_model_id.append(np.full(int(np.count_nonzero(keep_rows)), int(imodel), dtype=np.int64))
        rows_pair_idx.append(
            np.ascontiguousarray(
                np.stack([i_idx[keep_rows], j_idx[keep_rows]], axis=1),
                dtype=np.int64,
            )
        )

    if not rows_x:
        raise RuntimeError("No valid rows were produced from the loader. Check target_key/masks/logM_cut.")

    x_raw = np.concatenate(rows_x, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(rows_y, axis=0).astype(np.float32, copy=False)
    row_model_ids = np.concatenate(rows_model_id, axis=0).astype(np.int64, copy=False)
    row_pair_idx = np.concatenate(rows_pair_idx, axis=0).astype(np.int64, copy=False)

    if r_bins_ref is None or n_cosmo_params_ref is None:
        raise RuntimeError("Internal error: missing reference metadata after data construction.")

    return {
        "X_raw": x_raw,
        "y": y,
        "row_model_ids": row_model_ids,
        "row_pair_idx": row_pair_idx,
        "r_bins": r_bins_ref.astype(np.float32, copy=False),
        "target_key": cfg.target_key,
        "n_cosmo_params": int(n_cosmo_params_ref),
        "mass_input_space": cfg.mass_input_space,
        "data_config": asdict(cfg),
        "feature_names_raw": tuple(
            ["cosmo_{}".format(i) for i in range(int(n_cosmo_params_ref))] + ["M1", "M2"]
        ),
    }


class VelocityMomentDataModule(pl.LightningDataModule):
    """Lightning DataModule for halo velocity-moment emulation.

    The split is performed at the cosmology-model level (not row level):
    - Test models are fixed: [44, 53, 60, 61, 62, 63, 64]
    - Remaining models are randomly split 80/20 into train/validation

    Scaling is fit only on the training rows and then applied to all splits.
    """

    def __init__(
        self,
        batch_size: int = 256,
        data_split_config: Optional[Union[VelocityMomentSplitConfig, Mapping[str, Any]]] = None,
        data_config: Optional[Union[VelocityMomentDataConfig, Mapping[str, Any]]] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = int(batch_size)
        self.data_split_config = VelocityMomentSplitConfig.from_any(data_split_config)
        self.data_config = VelocityMomentDataConfig.from_any(data_config)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)

        self._built = False
        self._all_arrays: Optional[Dict[str, Any]] = None
        self._split_row_indices: Dict[str, np.ndarray] = {}
        self._split_model_ids: Dict[str, List[int]] = {}

        self.input_scaler: Optional[StandardScalerND] = None
        self.target_scaler: Optional[StandardScalerND] = None

        self._train_dataset: Optional[VelocityMomentTensorDataset] = None
        self._val_dataset: Optional[VelocityMomentTensorDataset] = None
        self._test_dataset: Optional[VelocityMomentTensorDataset] = None

        # Metadata populated after ``setup``.
        self.n_cosmo_params: Optional[int] = None
        self.n_raw_inputs: Optional[int] = None
        self.n_sym_features: Optional[int] = None
        self.n_outputs: Optional[int] = None
        self.target_key: Optional[str] = None
        self.r_bins: Optional[np.ndarray] = None
        self.feature_names_raw: Optional[Tuple[str, ...]] = None

    def prepare_data(self) -> None:
        # No-op: data is loaded in ``setup`` via the custom package loader.
        return None

    def _build_splits(self, all_row_model_ids: np.ndarray) -> None:
        cfg = self.data_split_config
        all_model_ids = list(range(1, int(cfg.n_models_total) + 1))
        present_model_ids = sorted(int(x) for x in np.unique(np.asarray(all_row_model_ids, dtype=np.int64)))
        unexpected = [m for m in present_model_ids if m not in all_model_ids]
        if unexpected:
            raise ValueError("Found row model ids outside the valid range: {}".format(unexpected))

        fixed_test = sorted(set(int(x) for x in cfg.test_model_ids))
        missing_test = [m for m in fixed_test if m not in all_model_ids]
        if missing_test:
            raise ValueError("Test model ids are outside the valid range: {}".format(missing_test))
        missing_test_rows = [m for m in fixed_test if m not in present_model_ids]
        if missing_test_rows:
            raise RuntimeError(
                "Some fixed test models produced no rows after filtering: {}.".format(missing_test_rows)
            )

        remaining = [m for m in present_model_ids if m not in fixed_test]
        if not remaining:
            raise ValueError("No train/val models remain after removing the test set.")

        rng = np.random.default_rng(int(cfg.random_seed))
        remaining_shuffled = list(remaining)
        rng.shuffle(remaining_shuffled)

        n_val_models = int(round(float(cfg.val_fraction) * len(remaining_shuffled)))
        n_val_models = max(1, min(len(remaining_shuffled) - 1, n_val_models))
        val_model_ids = sorted(remaining_shuffled[:n_val_models])
        train_model_ids = sorted(remaining_shuffled[n_val_models:])

        row_model_ids = np.asarray(all_row_model_ids, dtype=np.int64)
        train_mask = np.isin(row_model_ids, np.asarray(train_model_ids, dtype=np.int64))
        val_mask = np.isin(row_model_ids, np.asarray(val_model_ids, dtype=np.int64))
        test_mask = np.isin(row_model_ids, np.asarray(fixed_test, dtype=np.int64))

        if not np.any(train_mask):
            raise RuntimeError("No training rows found after model-level split.")
        if not np.any(val_mask):
            raise RuntimeError("No validation rows found after model-level split.")
        if not np.any(test_mask):
            raise RuntimeError("No test rows found after model-level split.")

        self._split_row_indices = {
            "train": np.where(train_mask)[0].astype(np.int64),
            "val": np.where(val_mask)[0].astype(np.int64),
            "test": np.where(test_mask)[0].astype(np.int64),
        }
        self._split_model_ids = {
            "train": train_model_ids,
            "val": val_model_ids,
            "test": fixed_test,
        }

    def setup(self, stage: Optional[str] = None) -> None:
        if self._built:
            return

        arrays = build_velocity_moment_arrays_from_loader(
            model_ids=list(range(1, int(self.data_split_config.n_models_total) + 1)),
            data_config=self.data_config,
        )
        self._all_arrays = arrays
        self._build_splits(arrays["row_model_ids"])

        x_raw = np.asarray(arrays["X_raw"], dtype=np.float32)
        y = np.asarray(arrays["y"], dtype=np.float32)
        n_cosmo = int(arrays["n_cosmo_params"])

        # Fit scalers on symmetry features and targets using the training rows only.
        train_rows = self._split_row_indices["train"]
        x_train_sym = _symmetry_transform_numpy(
            x_raw[train_rows],
            n_cosmo_params=n_cosmo,
            mass_input_space=self.data_config.mass_input_space,
            use_abs_mass_difference=self.data_config.use_abs_mass_difference,
        )
        y_train = y[train_rows]

        self.input_scaler = StandardScalerND().fit(x_train_sym)
        self.target_scaler = StandardScalerND().fit(y_train)
        y_scaled_all = self.target_scaler.transform(y)

        self._train_dataset = VelocityMomentTensorDataset(x_raw[train_rows], y_scaled_all[train_rows])
        val_rows = self._split_row_indices["val"]
        test_rows = self._split_row_indices["test"]
        self._val_dataset = VelocityMomentTensorDataset(x_raw[val_rows], y_scaled_all[val_rows])
        self._test_dataset = VelocityMomentTensorDataset(x_raw[test_rows], y_scaled_all[test_rows])

        self.n_cosmo_params = n_cosmo
        self.n_raw_inputs = int(x_raw.shape[1])
        self.n_sym_features = int(x_train_sym.shape[1])
        self.n_outputs = int(y.shape[1])
        self.target_key = str(arrays["target_key"])
        self.r_bins = np.asarray(arrays["r_bins"], dtype=np.float32)
        self.feature_names_raw = tuple(arrays["feature_names_raw"])

        self._built = True

    def _make_loader(self, dataset: Optional[Dataset], shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting dataloaders.")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=bool(shuffle),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=bool(self.num_workers > 0),
            drop_last=self.drop_last if shuffle else False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self._test_dataset, shuffle=False)

    def dataset_split_summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary of the model- and row-level split."""
        if not self._built:
            raise RuntimeError("Call setup() before requesting a split summary.")
        return {
            "n_models_total": int(self.data_split_config.n_models_total),
            "target_key": self.target_key,
            "mass_input_space": self.data_config.mass_input_space,
            "use_abs_mass_difference": bool(self.data_config.use_abs_mass_difference),
            "batch_size": self.batch_size,
            "rows": {k: int(len(v)) for k, v in self._split_row_indices.items()},
            "model_ids": {k: list(v) for k, v in self._split_model_ids.items()},
            "n_cosmo_params": int(self.n_cosmo_params if self.n_cosmo_params is not None else -1),
            "n_outputs": int(self.n_outputs if self.n_outputs is not None else -1),
        }

    def save_split_summary_json(self, path: Union[str, Path]) -> Path:
        """Persist the split summary to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.dataset_split_summary(), f, indent=2, sort_keys=True)
        return path

    def get_preprocessor_state(self) -> Dict[str, np.ndarray]:
        """Return scalers/metadata needed by the model for training and inference."""
        if not self._built or self.input_scaler is None or self.target_scaler is None:
            raise RuntimeError("Call setup() before exporting preprocessor state.")
        if self.n_cosmo_params is None or self.r_bins is None:
            raise RuntimeError("DataModule metadata is not initialized.")
        x_state = self.input_scaler.state_dict()
        y_state = self.target_scaler.state_dict()
        return {
            "x_mean": x_state["mean"],
            "x_scale": x_state["scale"],
            "y_mean": y_state["mean"],
            "y_scale": y_state["scale"],
            "r_bins": np.asarray(self.r_bins, dtype=np.float32),
            "n_cosmo_params": np.asarray([self.n_cosmo_params], dtype=np.int64),
        }

    def save_scalers(self, path: Union[str, Path]) -> Path:
        """Save scaler parameters and metadata to ``.npz``."""
        state = self.get_preprocessor_state()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **state)
        return path

    @staticmethod
    def load_scalers(path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Load scaler parameters and metadata from ``.npz``."""
        with np.load(Path(path), allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    def attach_preprocessors_to_model(self, model: "VelocityMomentEmulator") -> None:
        """Convenience helper: copy fit scalers/metadata onto a model."""
        if not isinstance(model, VelocityMomentEmulator):
            raise TypeError("Expected a VelocityMomentEmulator instance.")
        if str(model.cfg.mass_input_space).lower() != str(self.data_config.mass_input_space).lower():
            raise ValueError(
                "Model/data mass_input_space mismatch: '{}' vs '{}'.".format(
                    model.cfg.mass_input_space, self.data_config.mass_input_space
                )
            )
        if bool(model.cfg.use_abs_mass_difference) != bool(self.data_config.use_abs_mass_difference):
            raise ValueError(
                "Model/data use_abs_mass_difference mismatch: {} vs {}.".format(
                    model.cfg.use_abs_mass_difference, self.data_config.use_abs_mass_difference
                )
            )
        model.attach_preprocessors_from_state(self.get_preprocessor_state())
        if self.target_key is not None:
            model.target_key = self.target_key


class VelocityMomentEmulator(pl.LightningModule):
    """Symmetry-aware MLP emulator for velocity moments.

    Raw input format is assumed to be:
    ``[cosmo_0, ..., cosmo_(D-1), M1, M2]``.

    The model applies a permutation-symmetry feature transform before the MLP:
    ``u = 0.5 * (log M1 + log M2)``, ``v = 0.5 * (log M1 - log M2)`` (optionally
    ``|v|``).

    Training targets are expected to be standardized already (the DataModule
    provides normalized y). The model stores input/target scaler parameters as
    buffers so they persist in Lightning checkpoints.
    """

    def __init__(
        self,
        n_cosmo_params: int,
        n_outputs: int,
        model_config: Optional[Union[VelocityMomentModelConfig, Mapping[str, Any]]] = None,
        target_key: str = "m10",
    ) -> None:
        super().__init__()
        cfg = VelocityMomentModelConfig.from_any(model_config)
        self.cfg = cfg

        self.n_cosmo_params = int(n_cosmo_params)
        self.n_outputs = int(n_outputs)
        self.n_raw_inputs = self.n_cosmo_params + 2
        self.n_sym_features = self.n_cosmo_params + 2
        self.target_key = str(target_key)

        self.trunk = _build_mlp(
            in_dim=self.n_sym_features,
            out_dim=self.n_outputs,
            hidden_dim=int(cfg.hidden_dim),
            n_hidden_layers=int(cfg.n_hidden_layers),
            activation=str(cfg.activation),
            dropout=float(cfg.dropout),
        )
        self.loss_fn = nn.MSELoss()

        # Scalers and metadata are attached after DataModule.setup().
        self.register_buffer("_x_mean", torch.zeros(self.n_sym_features, dtype=torch.float32))
        self.register_buffer("_x_scale", torch.ones(self.n_sym_features, dtype=torch.float32))
        self.register_buffer("_y_mean", torch.zeros(self.n_outputs, dtype=torch.float32))
        self.register_buffer("_y_scale", torch.ones(self.n_outputs, dtype=torch.float32))
        self.register_buffer("_r_bins", torch.zeros(self.n_outputs, dtype=torch.float32))
        self.register_buffer("_r_bins_available", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("_preprocessors_ready", torch.tensor(False, dtype=torch.bool))

        self.save_hyperparameters(
            {
                "n_cosmo_params": self.n_cosmo_params,
                "n_outputs": self.n_outputs,
                "target_key": self.target_key,
                "model_config": asdict(cfg),
            }
        )

    @property
    def preprocessors_ready(self) -> bool:
        return bool(self._preprocessors_ready.item())

    @property
    def r_bins(self) -> Optional[np.ndarray]:
        if not bool(self._r_bins_available.item()):
            return None
        return self._r_bins.detach().cpu().numpy().copy()

    def attach_preprocessors(
        self,
        input_scaler: StandardScalerND,
        target_scaler: StandardScalerND,
        r_bins: Optional[np.ndarray] = None,
    ) -> None:
        """Attach fitted scalers from the DataModule."""
        x_state = input_scaler.state_dict()
        y_state = target_scaler.state_dict()
        state: Dict[str, np.ndarray] = {
            "x_mean": x_state["mean"],
            "x_scale": x_state["scale"],
            "y_mean": y_state["mean"],
            "y_scale": y_state["scale"],
            "r_bins": np.asarray(r_bins, dtype=np.float32) if r_bins is not None else np.empty(0, dtype=np.float32),
            "n_cosmo_params": np.asarray([self.n_cosmo_params], dtype=np.int64),
        }
        self.attach_preprocessors_from_state(state)

    def attach_preprocessors_from_state(self, state: Mapping[str, np.ndarray]) -> None:
        """Attach scalers/metadata from a dict (e.g. DataModule export or .npz load)."""
        n_cosmo_arr = state.get("n_cosmo_params")
        if n_cosmo_arr is not None:
            n_cosmo_loaded = int(np.asarray(n_cosmo_arr, dtype=np.int64).reshape(-1)[0])
            if n_cosmo_loaded != self.n_cosmo_params:
                raise ValueError(
                    "n_cosmo_params mismatch in preprocessor state: expected {}, got {}.".format(
                        self.n_cosmo_params, n_cosmo_loaded
                    )
                )

        x_mean = torch.as_tensor(np.asarray(state["x_mean"], dtype=np.float32), dtype=torch.float32)
        x_scale = torch.as_tensor(np.asarray(state["x_scale"], dtype=np.float32), dtype=torch.float32)
        y_mean = torch.as_tensor(np.asarray(state["y_mean"], dtype=np.float32), dtype=torch.float32)
        y_scale = torch.as_tensor(np.asarray(state["y_scale"], dtype=np.float32), dtype=torch.float32)

        if x_mean.numel() != self.n_sym_features or x_scale.numel() != self.n_sym_features:
            raise ValueError(
                "Input scaler dimension mismatch: expected {}, got {} / {}.".format(
                    self.n_sym_features, x_mean.numel(), x_scale.numel()
                )
            )
        if y_mean.numel() != self.n_outputs or y_scale.numel() != self.n_outputs:
            raise ValueError(
                "Target scaler dimension mismatch: expected {}, got {} / {}.".format(
                    self.n_outputs, y_mean.numel(), y_scale.numel()
                )
            )

        with torch.no_grad():
            self._x_mean.copy_(x_mean.view_as(self._x_mean))
            self._x_scale.copy_(x_scale.view_as(self._x_scale))
            self._y_mean.copy_(y_mean.view_as(self._y_mean))
            self._y_scale.copy_(y_scale.view_as(self._y_scale))

            r_bins_arr = np.asarray(state.get("r_bins", np.empty(0, dtype=np.float32)), dtype=np.float32)
            if r_bins_arr.size == 0:
                self._r_bins.zero_()
                self._r_bins_available.fill_(False)
            else:
                if int(r_bins_arr.size) != self.n_outputs:
                    raise ValueError(
                        "r_bins length mismatch: expected {}, got {}.".format(self.n_outputs, r_bins_arr.size)
                    )
                self._r_bins.copy_(torch.as_tensor(r_bins_arr, dtype=torch.float32).view_as(self._r_bins))
                self._r_bins_available.fill_(True)
            self._preprocessors_ready.fill_(True)

    def _maybe_attach_from_datamodule(self) -> None:
        """Attach scalers from the trainer's datamodule if not already attached."""
        if self.preprocessors_ready:
            return
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        dm = getattr(trainer, "datamodule", None)
        if isinstance(dm, VelocityMomentDataModule):
            if not getattr(dm, "_built", False):
                dm.setup(stage="fit")
            dm.attach_preprocessors_to_model(self)

    def on_fit_start(self) -> None:
        self._maybe_attach_from_datamodule()

    def on_validation_start(self) -> None:
        self._maybe_attach_from_datamodule()

    def on_test_start(self) -> None:
        self._maybe_attach_from_datamodule()

    def _raw_to_symmetry_features_torch(self, x_raw: torch.Tensor) -> torch.Tensor:
        if x_raw.ndim == 1:
            x_raw = x_raw.unsqueeze(0)
        if x_raw.ndim != 2 or x_raw.size(-1) != self.n_raw_inputs:
            raise ValueError(
                "Expected x_raw with shape (N, {}), got {}.".format(self.n_raw_inputs, tuple(x_raw.shape))
            )

        cosmo = x_raw[:, : self.n_cosmo_params]
        m1 = x_raw[:, self.n_cosmo_params]
        m2 = x_raw[:, self.n_cosmo_params + 1]
        mass_mode = str(self.cfg.mass_input_space).lower()

        if mass_mode == "mass":
            if torch.any(m1 <= 0) or torch.any(m2 <= 0):
                raise ValueError("Mass inputs must be positive when mass_input_space='mass'.")
            logm1 = torch.log(m1)
            logm2 = torch.log(m2)
        elif mass_mode in ("log10", "ln"):
            logm1 = m1
            logm2 = m2
        else:
            raise ValueError("Unsupported mass_input_space '{}'.".format(self.cfg.mass_input_space))

        u = 0.5 * (logm1 + logm2)
        v = 0.5 * (logm1 - logm2)
        if self.cfg.use_abs_mass_difference:
            v = torch.abs(v)

        return torch.cat([cosmo, u.unsqueeze(-1), v.unsqueeze(-1)], dim=-1)

    def _scale_inputs(self, x_sym: torch.Tensor) -> torch.Tensor:
        if not self.preprocessors_ready:
            raise RuntimeError("Input scalers are not attached. Call datamodule.setup() and attach scalers before training/inference.")
        return (x_sym - self._x_mean.unsqueeze(0)) / self._x_scale.unsqueeze(0)

    def _inverse_scale_targets(self, y_scaled: torch.Tensor) -> torch.Tensor:
        if not self.preprocessors_ready:
            raise RuntimeError("Target scalers are not attached.")
        return y_scaled * self._y_scale.unsqueeze(0) + self._y_mean.unsqueeze(0)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Forward pass from raw inputs to normalized target predictions."""
        x_sym = self._raw_to_symmetry_features_torch(x_raw.float())
        x_scaled = self._scale_inputs(x_sym)
        return self.trunk(x_scaled)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x_raw, y_scaled = batch
        y_hat = self.forward(x_raw)
        loss = self.loss_fn(y_hat, y_scaled)
        self.log("train/loss", loss, on_step=self.cfg.log_train_step, on_epoch=True, prog_bar=True, batch_size=x_raw.size(0))
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x_raw, y_scaled = batch
        y_hat = self.forward(x_raw)
        loss = self.loss_fn(y_hat, y_scaled)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x_raw.size(0))
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x_raw, y_scaled = batch
        y_hat = self.forward(x_raw)
        loss = self.loss_fn(y_hat, y_scaled)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x_raw.size(0))
        return loss

    def predict_step(
        self,
        batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Sequence[torch.Tensor]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        if isinstance(batch, torch.Tensor):
            x_raw = batch
        elif isinstance(batch, (tuple, list)):
            if len(batch) == 0:
                raise ValueError("predict_step received an empty batch.")
            x_raw = batch[0]
        else:
            raise TypeError("Unsupported batch type for predict_step: {}.".format(type(batch).__name__))
        y_scaled = self.forward(x_raw)
        return self._inverse_scale_targets(y_scaled)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(self.cfg.plateau_factor),
            patience=int(self.cfg.plateau_patience),
            min_lr=float(self.cfg.plateau_min_lr),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _build_raw_inputs_for_predict(
        self,
        M1: ArrayLike,
        M2: ArrayLike,
        cosmo_params: Union[ArrayLike, np.ndarray],
    ) -> np.ndarray:
        m1 = _as_numpy_1d(M1, "M1")
        m2 = _as_numpy_1d(M2, "M2")
        if m1.shape != m2.shape:
            raise ValueError("M1 and M2 must have matching shapes.")

        cosmo = np.asarray(cosmo_params, dtype=np.float64)
        if cosmo.ndim == 1:
            if cosmo.shape[0] != self.n_cosmo_params:
                raise ValueError(
                    "cosmo_params has length {}, expected {}.".format(cosmo.shape[0], self.n_cosmo_params)
                )
            cosmo_2d = np.repeat(cosmo[None, :], repeats=m1.shape[0], axis=0)
        elif cosmo.ndim == 2:
            if cosmo.shape[1] != self.n_cosmo_params:
                raise ValueError(
                    "cosmo_params has shape {}, expected (N, {}).".format(cosmo.shape, self.n_cosmo_params)
                )
            if cosmo.shape[0] != m1.shape[0]:
                raise ValueError(
                    "If cosmo_params is 2D, its first dimension must match len(M1). Got {} and {}.".format(
                        cosmo.shape[0], m1.shape[0]
                    )
                )
            cosmo_2d = cosmo
        else:
            raise ValueError("cosmo_params must be shape ({},) or (N, {}).".format(self.n_cosmo_params, self.n_cosmo_params))

        x_raw = np.concatenate(
            [
                cosmo_2d.astype(np.float32, copy=False),
                m1[:, None].astype(np.float32, copy=False),
                m2[:, None].astype(np.float32, copy=False),
            ],
            axis=1,
        )
        return np.ascontiguousarray(x_raw, dtype=np.float32)

    @torch.no_grad()
    def predict(
        self,
        M1: ArrayLike,
        M2: ArrayLike,
        cosmo_params: Union[ArrayLike, np.ndarray],
        batch_size: Optional[int] = None,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Predict the target in original (de-standardized) target space.

        Parameters
        ----------
        M1, M2
            Raw mass inputs or log-mass inputs, depending on ``model_config.mass_input_space``.
            Scalars or 1D arrays are accepted.
        cosmo_params
            Either a single cosmology vector ``(D_cosmo,)`` or a batch ``(N, D_cosmo)``.
        batch_size
            Optional minibatch size for inference. If ``None``, run in one pass.
        return_numpy
            If True (default), return a numpy array of shape ``(N, N_velmom)``.

        Returns
        -------
        np.ndarray or torch.Tensor
            Predictions in the original target space used during training
            (e.g. physical ``m10`` if ``target_key='m10'``).
        """
        if not self.preprocessors_ready:
            raise RuntimeError("Scalers are not attached. Load a trained checkpoint or attach scalers before calling predict().")

        was_training = self.training
        self.eval()

        x_raw_np = self._build_raw_inputs_for_predict(M1=M1, M2=M2, cosmo_params=cosmo_params)
        x_raw = torch.as_tensor(x_raw_np, dtype=torch.float32, device=self.device)

        if batch_size is None or int(batch_size) <= 0:
            y_scaled = self.forward(x_raw)
            y_pred = self._inverse_scale_targets(y_scaled)
        else:
            chunks: List[torch.Tensor] = []
            bs = int(batch_size)
            for start in range(0, x_raw.size(0), bs):
                stop = min(start + bs, x_raw.size(0))
                y_scaled_chunk = self.forward(x_raw[start:stop])
                chunks.append(self._inverse_scale_targets(y_scaled_chunk))
            y_pred = torch.cat(chunks, dim=0)

        if was_training:
            self.train()

        if return_numpy:
            return y_pred.detach().cpu().numpy()
        return y_pred

    @classmethod
    def load_for_inference(
        cls,
        checkpoint_path: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = "cpu",
        **load_kwargs: Any
    ) -> "VelocityMomentEmulator":
        """Load a Lightning checkpoint and return an ``eval()`` model."""
        model = cls.load_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            map_location=map_location,
            **load_kwargs
        )
        model.eval()
        return model

    @classmethod
    def load_from_checkpoint_eval(
        cls,
        checkpoint_path: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = "cpu",
        **load_kwargs: Any,
    ) -> "VelocityMomentEmulator":
        """Alias for a checkpoint loader that immediately returns ``eval()`` mode."""
        return cls.load_for_inference(
            checkpoint_path=checkpoint_path,
            map_location=map_location,
            **load_kwargs
        )

    def save_checkpoint_with_trainer(self, trainer: pl.Trainer, path: Union[str, Path]) -> None:
        """Convenience wrapper around ``trainer.save_checkpoint``."""
        trainer.save_checkpoint(str(path))

    def describe_target_space(self) -> str:
        """Human-readable description of what ``predict()`` returns."""
        if self.target_key == "m10":
            return "physical m10 (after inverse standardization)"
        if self.target_key.endswith("_transformed"):
            return "{} (transformed target, after inverse standardization)".format(self.target_key)
        return "{} (after inverse standardization)".format(self.target_key)

    def _convert_log10m_to_model_mass_inputs(self, logm: np.ndarray) -> np.ndarray:
        """Convert loader ``log10M`` bin centers into the model's configured mass input space."""
        logm = np.asarray(logm, dtype=np.float32)
        mode = str(self.cfg.mass_input_space).lower()
        if mode == "mass":
            return np.power(10.0, logm.astype(np.float64)).astype(np.float32)
        if mode == "log10":
            return logm.astype(np.float32, copy=True)
        if mode == "ln":
            return (np.log(10.0) * logm.astype(np.float64)).astype(np.float32)
        raise ValueError("Unsupported mass_input_space '{}'.".format(self.cfg.mass_input_space))

    def compare(
        self,
        imodel: int,
        *,
        gravity: str = "LCDM",
        redshift: float = 0.25,
        logM_cut: Optional[float] = 14.0,
        pair_indices: Optional[Sequence[Tuple[int, int]]] = None,
        mass_pairs_log10: Optional[Sequence[Tuple[float, float]]] = None,
        max_pairs: int = 4,
        plot_path: Optional[Union[str, Path]] = None,
        dpi: int = 160,
        title: Optional[str] = None,
        show: bool = False,
    ) -> Dict[str, Any]:
        """Compare emulator predictions to loader data for one cosmology model and plot.

        Parameters
        ----------
        imodel
            Cosmology model index (1..64 in the current dataset).
        pair_indices
            Optional list of ``(i, j)`` indices into ``logM_bins``.
        mass_pairs_log10
            Optional list of mass pairs specified as ``(log10M1, log10M2)``.
            Nearest available bins are used.
        max_pairs
            Number of default pairs to plot if no explicit pairs are supplied.
        plot_path
            If provided, save the figure to this path.

        Returns
        -------
        dict
            Contains ``figure``, ``axes``, ``plot_path``, selected ``pair_indices``,
            and the plotted arrays for further inspection.
        """
        if load_velocity_moment_hh_transformed is None:
            raise ImportError(
                "Could not import load_velocity_moment_hh_transformed. "
                "Original error: {}".format(_LOADER_IMPORT_ERROR)
            )
        if not self.preprocessors_ready:
            raise RuntimeError("Scalers are not attached. Load a trained checkpoint or attach scalers before compare().")

        payload = load_velocity_moment_hh_transformed(
            int(imodel),
            gravity=gravity,
            redshift=redshift,
            logM_cut=logM_cut,
        )
        if self.target_key not in payload:
            raise KeyError(
                "Target '{}' is not available in loader payload keys.".format(self.target_key)
            )

        r = np.asarray(payload["r_vm"], dtype=np.float32).reshape(-1)
        logm_bins = np.asarray(payload["logM_bins"], dtype=np.float32).reshape(-1)
        cosmo_params = np.asarray(payload["cosmo_params"], dtype=np.float32).reshape(-1)
        y_cube = np.asarray(payload[self.target_key], dtype=np.float32)
        if y_cube.ndim != 3:
            raise ValueError(
                "Expected payload['{}'] with shape (N_M, N_M, N_r), got {}.".format(
                    self.target_key, y_cube.shape
                )
            )
        if y_cube.shape[0] != logm_bins.size or y_cube.shape[1] != logm_bins.size:
            raise ValueError("Mass dimensions of payload target do not match logM_bins.")
        if y_cube.shape[2] != r.size:
            raise ValueError("Radial dimension of payload target does not match r_vm.")
        if cosmo_params.size != self.n_cosmo_params:
            raise ValueError(
                "Loader cosmo_params length {} does not match emulator n_cosmo_params {}.".format(
                    cosmo_params.size, self.n_cosmo_params
                )
            )
        if self.r_bins is not None and len(self.r_bins) == len(r):
            if not np.allclose(self.r_bins, r, rtol=1e-5, atol=0.0):
                raise ValueError("Loaded r_vm grid does not match the emulator's stored output grid.")

        def _nearest_idx(values: np.ndarray, target: float) -> int:
            return int(np.argmin(np.abs(values - float(target))))

        def _default_pairs(n_mass: int, n_keep: int) -> List[Tuple[int, int]]:
            candidates = [
                (0, 0),
                (max(0, n_mass // 2), max(0, n_mass // 2)),
                (n_mass - 1, n_mass - 1),
                (0, n_mass - 1),
                (max(0, n_mass // 2), n_mass - 1),
            ]
            out: List[Tuple[int, int]] = []
            seen = set()
            for i, j in candidates:
                if i < 0 or j < 0 or i >= n_mass or j >= n_mass:
                    continue
                key = (int(i), int(j))
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
                if len(out) >= max(1, int(n_keep)):
                    break
            return out

        selected_pairs: List[Tuple[int, int]] = []
        if pair_indices is not None:
            for i, j in pair_indices:
                ii = int(i)
                jj = int(j)
                if ii < 0 or jj < 0 or ii >= logm_bins.size or jj >= logm_bins.size:
                    raise IndexError(
                        "pair_indices contains ({}, {}) outside valid range [0, {}).".format(
                            ii, jj, logm_bins.size
                        )
                    )
                selected_pairs.append((ii, jj))
        elif mass_pairs_log10 is not None:
            for m1, m2 in mass_pairs_log10:
                selected_pairs.append((_nearest_idx(logm_bins, m1), _nearest_idx(logm_bins, m2)))
        else:
            selected_pairs = _default_pairs(int(logm_bins.size), int(max_pairs))

        if not selected_pairs:
            raise ValueError("No mass pairs selected for compare().")

        model_mass_inputs = self._convert_log10m_to_model_mass_inputs(logm_bins)
        pair_i = np.asarray([p[0] for p in selected_pairs], dtype=np.int64)
        pair_j = np.asarray([p[1] for p in selected_pairs], dtype=np.int64)
        m1_inputs = model_mass_inputs[pair_i]
        m2_inputs = model_mass_inputs[pair_j]

        y_pred = np.asarray(
            self.predict(M1=m1_inputs, M2=m2_inputs, cosmo_params=cosmo_params, return_numpy=True),
            dtype=np.float32,
        )
        if y_pred.shape != (len(selected_pairs), r.size):
            raise ValueError(
                "Unexpected prediction shape {}; expected ({}, {}).".format(
                    y_pred.shape, len(selected_pairs), r.size
                )
            )

        y_true = np.asarray(y_cube[pair_i, pair_j, :], dtype=np.float32)
        y_err_key = "{}_err".format(self.target_key)
        y_err = None
        if y_err_key in payload:
            try:
                y_err = np.asarray(payload[y_err_key], dtype=np.float32)[pair_i, pair_j, :]
            except Exception:
                y_err = None

        try:
            import matplotlib

            if not show:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise ImportError(
                "compare() requires matplotlib for plotting. Install matplotlib in the active environment."
            ) from exc

        fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.2), constrained_layout=True)
        cmap = plt.get_cmap("tab10")

        plotted_pairs: List[Tuple[int, int]] = []
        for k, (i, j) in enumerate(selected_pairs):
            yp = np.asarray(y_pred[k], dtype=float)
            yt = np.asarray(y_true[k], dtype=float)
            mask = np.isfinite(yp) & np.isfinite(yt) & np.isfinite(r)
            if not np.any(mask):
                continue

            label = "({:.2f},{:.2f})".format(float(logm_bins[i]), float(logm_bins[j]))
            color = cmap(k % 10)
            ax.plot(r[mask], yp[mask], color=color, lw=1.8, label="pred " + label)
            ax.plot(r[mask], yt[mask], color=color, lw=0.0, marker="o", ms=3.2, alpha=0.9, label="data " + label)

            if y_err is not None:
                ye = np.asarray(y_err[k], dtype=float)
                if ye.shape == yt.shape:
                    m_band = mask & np.isfinite(ye)
                    if np.any(m_band):
                        lo = yt - ye
                        hi = yt + ye
                        band_mask = m_band & np.isfinite(lo) & np.isfinite(hi)
                        if np.any(band_mask):
                            ax.fill_between(
                                r[band_mask],
                                lo[band_mask],
                                hi[band_mask],
                                color=color,
                                alpha=0.12,
                                linewidth=0.0,
                            )
            plotted_pairs.append((i, j))

        ax.set_xscale("log")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.set_xlabel(r"$r\,[h^{-1}\mathrm{Mpc}]$")
        ax.set_ylabel(self.target_key)
        ax.set_title(
            title
            if title is not None
            else "Emulator vs data: imodel={} target={}".format(int(imodel), self.target_key)
        )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ncol = 2 if len(handles) > 4 else 1
            ax.legend(handles, labels, fontsize=8, ncol=ncol)

        saved_path: Optional[Path] = None
        if plot_path is not None:
            saved_path = Path(plot_path)
            saved_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(saved_path, dpi=int(dpi), bbox_inches="tight")

        if show:
            plt.show()

        return {
            "figure": fig,
            "axes": ax,
            "plot_path": None if saved_path is None else str(saved_path),
            "imodel": int(imodel),
            "target_key": self.target_key,
            "r": r,
            "logM_bins": logm_bins,
            "pair_indices": plotted_pairs,
            "y_pred": y_pred,
            "y_true": y_true,
            "y_err": y_err,
        }

    @classmethod
    def compare_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        imodel: int,
        *,
        map_location: Optional[Union[str, torch.device]] = "cpu",
        **compare_kwargs: Any
    ) -> Dict[str, Any]:
        """Load a checkpoint in eval mode and call ``compare(imodel, ...)``."""
        model = cls.load_for_inference(
            checkpoint_path=checkpoint_path,
            map_location=map_location,
        )
        return model.compare(imodel=int(imodel), **compare_kwargs)


def _example_usage() -> None:
    """Minimal example showing the expected workflow.

    This is intentionally not executed automatically because it may trigger a
    heavy data load in shared environments.
    """
    example = r'''
from pytorch_lightning import Trainer
from freyja.emulators.halo_velocity_moment import (
    VelocityMomentDataModule,
    VelocityMomentDataConfig,
    VelocityMomentSplitConfig,
    VelocityMomentEmulator,
    VelocityMomentModelConfig,
)

# Optional placeholder: if you want to inspect the raw loader payload yourself.
# from freyja.cosma.velocity_moment_hh import load_velocity_moment_hh_transformed
# payload = load_velocity_moment_hh_transformed(1)
# X, y, y_err = ...  # build custom arrays if needed

dm = VelocityMomentDataModule(
    batch_size=1024,
    data_split_config=VelocityMomentSplitConfig(
        val_fraction=0.2,
        random_seed=1234,
    ),
    data_config=VelocityMomentDataConfig(
        target_key="m10",         # or "m10_transformed"
        mass_input_space="mass",  # predict() then accepts physical masses
        use_upper_triangle_only=True,
    ),
)
dm.setup("fit")

model = VelocityMomentEmulator(
    n_cosmo_params=dm.n_cosmo_params,
    n_outputs=dm.n_outputs,
    model_config=VelocityMomentModelConfig(
        hidden_dim=128,
        n_hidden_layers=2,
        activation="gelu",
        lr=3e-4,
    ),
    target_key=dm.target_key or "m10",
)

# Attach scalers so the model can train (and later predict in de-standardized units).
dm.attach_preprocessors_to_model(model)

trainer = Trainer(
    max_epochs=50,
    accelerator="auto",
    devices="auto",
    log_every_n_steps=50,
)
trainer.fit(model, datamodule=dm)
trainer.test(model, datamodule=dm)

# Save checkpoint (includes model weights + scaler buffers)
model.save_checkpoint_with_trainer(trainer, "halo_vm.ckpt")

# Load for inference
infer_model = VelocityMomentEmulator.load_for_inference("halo_vm.ckpt")

# Predict a full radial m10 vector for one cosmology + one mass pair
cosmo = [0.30, 0.67, 0.80, 0.965]  # example only; match your loader convention
pred = infer_model.predict(
    M1=1.0e13,
    M2=3.0e13,
    cosmo_params=cosmo,
)
print(pred.shape)  # (1, N_r)
print(infer_model.describe_target_space())
'''
    print(example)


if __name__ == "__main__":
    _example_usage()
