"""Export halo velocity-moment scan data (emulator + truth) for later plotting.

This script uses a trained ``HaloVelocityMomentEmulator`` checkpoint (default
run dir: ``scripts/train/outputs/halo_vm_m10_beta_style``) plus
``load_velocity_moment_hh_transformed`` to export two scan modes:

1. Fixed cosmology (one imodel), fixed M1, varying M2
2. Fixed masses (M1, M2), varying cosmologies (default imodel=57..63)

For each scan it saves both emulator predictions and simulation truth for:
- the emulator target (e.g. ``m10_transformed``, ``c20_transformed``)
- the corresponding raw moment (e.g. ``m10``, ``c20``), reconstructed using
  ``m10_linear`` with the appropriate power

Outputs are saved as ``.npz`` files for plotting later.

Example
-------
python3 halo_vm_m10_data.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_on_path()

from freyja.cosma.velocity_moment_hh import (
    load_velocity_moment_hh_transformed,
)  # noqa: E402
from freyja.emulators.halo_velocity_moment import (
    HaloVelocityMomentEmulator,
    TARGET_M10_LINEAR_POWERS,
)  # noqa: E402


DEFAULT_RUN_DIR = Path(
    "/cosma8/data/dp203/dc-ruan1/freyja_codex/freyja/scripts/train/outputs/halo_vm_m10_beta_style"
)
DEFAULT_CHECKPOINT_NAME = "halo_velocity_moment_m10_transformed.pt"
DEFAULT_IMODELS_COSMO_SCAN = [
    57,
    58,
    59,
    56,
    61,
    62,
    63,
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export halo velocity-moment emulator-vs-truth scan data to NPZ files."
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Training run directory containing checkpoints/.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit .pt checkpoint path; overrides --run-dir resolution.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("data"),
        help="Directory where .npz outputs are written.",
    )

    # Fixed cosmology, varying M2
    p.add_argument("--imodel-fixed", type=int, default=58)
    p.add_argument("--m1-fixed", type=float, default=13.1)
    p.add_argument("--m2-min", type=float, default=12.5)
    p.add_argument("--m2-max", type=float, default=14.0)
    p.add_argument("--m2-num", type=int, default=16)

    # Fixed masses, varying cosmologies
    p.add_argument("--m1-cosmo-scan", type=float, default=13.1)
    p.add_argument("--m2-cosmo-scan", type=float, default=13.1)
    p.add_argument(
        "--imodels-cosmo-scan",
        type=int,
        nargs="*",
        default=DEFAULT_IMODELS_COSMO_SCAN,
        help="Cosmology imodels for the fixed-mass scan (default: 57..63).",
    )

    p.add_argument(
        "--target-key",
        type=str,
        default=None,
        help=(
            "Optional expected emulator target key (e.g. m10_transformed, c20_transformed). "
            "If provided, validates the loaded checkpoint."
        ),
    )
    p.add_argument(
        "--outfile-prefix",
        type=str,
        default=None,
        help=(
            "Prefix for output .npz filenames. Default is derived from checkpoint target, "
            "e.g. halo_vm_m10 or halo_vm_c20."
        ),
    )
    return p


def _resolve(path: Path) -> Path:
    p = path.expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def _resolve_checkpoint(run_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        ckpt = _resolve(explicit)
        if not ckpt.exists():
            raise FileNotFoundError(f"Explicit checkpoint not found: {ckpt}")
        return ckpt

    run_dir = _resolve(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    candidate = ckpt_dir / DEFAULT_CHECKPOINT_NAME
    if candidate.exists():
        return candidate.resolve()

    pts = sorted(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
    if pts:
        pts = sorted(pts, key=lambda p: p.stat().st_mtime, reverse=True)
        return pts[0].resolve()

    raise FileNotFoundError(
        f"Could not resolve emulator checkpoint in {ckpt_dir} (tried {candidate.name} and *.pt)."
    )


def _nearest_mass_index(logM_bins: np.ndarray, target: float) -> int:
    logM_bins = np.asarray(logM_bins, dtype=float)
    return int(np.argmin(np.abs(logM_bins - float(target))))


def _infer_raw_target_key(target_key: str) -> str:
    if not target_key.endswith("_transformed"):
        raise ValueError(
            f"Expected transformed target key ending with '_transformed', got {target_key!r}."
        )
    return target_key[: -len("_transformed")]


def _target_anchor_power(target_key: str) -> int:
    try:
        return int(TARGET_M10_LINEAR_POWERS[target_key])
    except KeyError as exc:
        known = ", ".join(sorted(TARGET_M10_LINEAR_POWERS))
        raise KeyError(
            f"Unsupported target_key={target_key!r} for reconstruction. Known: {known}"
        ) from exc


def _predict_pair_from_loader_data(
    emu: HaloVelocityMomentEmulator,
    *,
    data: dict[str, np.ndarray],
    logM1_req: float,
    logM2_req: float,
) -> tuple[dict[str, np.ndarray | float], tuple[int, int]]:
    """Predict and extract truth for one mass pair from transformed loader output."""
    target_key = str(emu.target_key)
    raw_key = _infer_raw_target_key(target_key)
    power = _target_anchor_power(target_key)
    if raw_key not in data:
        raise KeyError(f"Raw target {raw_key!r} not found in loader output.")
    if target_key not in data:
        raise KeyError(f"Transformed target {target_key!r} not found in loader output.")

    r = np.asarray(data["r_vm"], dtype=float)
    if np.any(r <= 0.0):
        raise ValueError("r_vm contains non-positive values.")
    logr = np.log10(r)

    logM_bins = np.asarray(data["logM_bins"], dtype=float)
    i = _nearest_mass_index(logM_bins, logM1_req)
    j = _nearest_mass_index(logM_bins, logM2_req)
    logM1_use = float(logM_bins[i])
    logM2_use = float(logM_bins[j])

    cosmo = np.asarray(data["cosmo_params"], dtype=float)
    m10_linear = np.asarray(data["m10_linear"][i, j, :], dtype=float)
    raw_true = np.asarray(data[raw_key][i, j, :], dtype=float)
    target_true_transformed = np.asarray(data[target_key][i, j, :], dtype=float)

    target_emu_transformed = np.asarray(
        emu.predict(
            cosmo,
            np.full_like(logr, logM1_use, dtype=float),
            np.full_like(logr, logM2_use, dtype=float),
            logr,
            return_numpy=True,
        ),
        dtype=float,
    )
    raw_emu = np.sinh(target_emu_transformed) * (m10_linear**power)

    return (
        {
            "r_bins": r,
            "logM1": logM1_use,
            "logM2": logM2_use,
            "m10_linear": m10_linear,
            "raw_true": raw_true,
            "raw_emu": raw_emu,
            "transformed_true": target_true_transformed,
            "transformed_emu": target_emu_transformed,
        },
        (i, j),
    )


def main() -> None:
    args = build_parser().parse_args()

    ckpt_path = _resolve_checkpoint(args.run_dir, args.checkpoint)
    outdir = _resolve(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    m2_num = int(args.m2_num)
    if m2_num < 1:
        raise ValueError(f"--m2-num must be >=1, got {m2_num}")
    m2_grid_req = np.linspace(float(args.m2_min), float(args.m2_max), m2_num)

    imodels_cosmo_scan = [int(m) for m in args.imodels_cosmo_scan]
    if len(imodels_cosmo_scan) == 0:
        raise ValueError("No --imodels-cosmo-scan provided.")

    print(f"Loading HaloVelocityMomentEmulator checkpoint: {ckpt_path}")
    emu = HaloVelocityMomentEmulator.load(ckpt_path)
    target_key = str(emu.target_key)
    raw_key = _infer_raw_target_key(target_key)
    target_power = _target_anchor_power(target_key)
    if args.target_key is not None and str(args.target_key) != target_key:
        raise RuntimeError(
            f"Checkpoint target_key={target_key!r} does not match --target-key={args.target_key!r}"
        )
    outfile_prefix = (
        str(args.outfile_prefix)
        if args.outfile_prefix is not None
        else f"halo_vm_{raw_key}"
    )
    print(
        f"Using target={target_key} (raw={raw_key}, m10_linear power={target_power}); "
        f"outfile_prefix={outfile_prefix}"
    )

    # ------------------------------------------------------------
    # 1) Fixed cosmology, fixed M1, varying M2
    # ------------------------------------------------------------
    imodel_fixed = int(args.imodel_fixed)
    m1_fixed_req = float(args.m1_fixed)
    data_fixed = load_velocity_moment_hh_transformed(
        imodel_fixed,
        redshift=float(emu.redshift),
        logM_cut=float(emu.logM_cut),
    )
    cosmo_fixed = np.asarray(data_fixed["cosmo_params"], dtype=float)
    logM_bins_fixed = np.asarray(data_fixed["logM_bins"], dtype=float)
    m1_fixed_idx = _nearest_mass_index(logM_bins_fixed, m1_fixed_req)
    m1_fixed_use = float(logM_bins_fixed[m1_fixed_idx])

    print(
        f"[1/2] Fixed cosmology scan: imodel={imodel_fixed}, logM1={m1_fixed_req:.3f} "
        f"(used {m1_fixed_use:.3f}), varying logM2 over "
        f"{m2_grid_req[0]:.3f}..{m2_grid_req[-1]:.3f} ({len(m2_grid_req)} points)"
    )

    r_ref: np.ndarray | None = None
    m2_grid_used: list[float] = []
    m2_index_used: list[int] = []
    raw_emu_list: list[np.ndarray] = []
    raw_true_list: list[np.ndarray] = []
    m10_lin_list: list[np.ndarray] = []
    t_emu_list: list[np.ndarray] = []
    t_true_list: list[np.ndarray] = []

    for m2_req in m2_grid_req:
        pair, (i, j) = _predict_pair_from_loader_data(
            emu,
            data=data_fixed,
            logM1_req=m1_fixed_req,
            logM2_req=float(m2_req),
        )
        r_here = np.asarray(pair["r_bins"], dtype=float)
        if r_ref is None:
            r_ref = r_here
        elif not np.allclose(r_ref, r_here, rtol=1e-10, atol=0.0, equal_nan=True):
            raise RuntimeError("Inconsistent r bins in fixed-imodel scan.")

        m2_grid_used.append(float(pair["logM2"]))
        m2_index_used.append(int(j))
        raw_emu_list.append(np.asarray(pair["raw_emu"], dtype=float))
        raw_true_list.append(np.asarray(pair["raw_true"], dtype=float))
        m10_lin_list.append(np.asarray(pair["m10_linear"], dtype=float))
        t_emu_list.append(np.asarray(pair["transformed_emu"], dtype=float))
        t_true_list.append(np.asarray(pair["transformed_true"], dtype=float))

    out_fixed_imodel = outdir / f"{outfile_prefix}_fixed_imodel_m2scan.npz"
    raw_emu_arr = np.stack(raw_emu_list, axis=0)
    raw_true_arr = np.stack(raw_true_list, axis=0)
    t_emu_arr = np.stack(t_emu_list, axis=0)
    t_true_arr = np.stack(t_true_list, axis=0)
    m10_lin_arr = np.stack(m10_lin_list, axis=0)
    fixed_imodel_payload = dict(
        mode="fixed_imodel_vary_m2",
        checkpoint_path=str(ckpt_path),
        target_key=np.array(target_key),
        raw_target_key=np.array(raw_key),
        target_power=np.array(target_power, dtype=int),
        imodel=np.array(imodel_fixed, dtype=int),
        cosmo_params=cosmo_fixed,
        r_bins=np.asarray(r_ref, dtype=float),
        logM_bins_truth=logM_bins_fixed,
        logM1_requested=np.array(m1_fixed_req, dtype=float),
        logM1=np.array(m1_fixed_use, dtype=float),
        logM2_grid_requested=np.asarray(m2_grid_req, dtype=float),
        logM2_grid=np.asarray(m2_grid_used, dtype=float),
        logM2_indices=np.asarray(m2_index_used, dtype=int),
        raw_emu=raw_emu_arr,
        raw_true=raw_true_arr,
        m10_linear=m10_lin_arr,
        transformed_emu=t_emu_arr,
        transformed_true=t_true_arr,
    )
    if raw_key == "m10":
        fixed_imodel_payload.update(
            m10_emu=raw_emu_arr,
            m10_true=raw_true_arr,
            m10_transformed_emu=t_emu_arr,
            m10_transformed_true=t_true_arr,
        )
    np.savez(
        out_fixed_imodel,
        **fixed_imodel_payload,
    )
    print(f"Saved: {out_fixed_imodel}")

    # ------------------------------------------------------------
    # 2) Fixed masses, varying cosmologies
    # ------------------------------------------------------------
    m1_cosmo_req = float(args.m1_cosmo_scan)
    m2_cosmo_req = float(args.m2_cosmo_scan)
    print(
        f"[2/2] Fixed masses scan: logM1={m1_cosmo_req:.3f}, logM2={m2_cosmo_req:.3f}, "
        f"varying imodels={imodels_cosmo_scan}"
    )

    r_ref2: np.ndarray | None = None
    cosmo_list: list[np.ndarray] = []
    logM1_used_list: list[float] = []
    logM2_used_list: list[float] = []
    logM1_idx_list: list[int] = []
    logM2_idx_list: list[int] = []
    raw_emu_cosmo: list[np.ndarray] = []
    raw_true_cosmo: list[np.ndarray] = []
    m10_lin_cosmo: list[np.ndarray] = []
    t_emu_cosmo: list[np.ndarray] = []
    t_true_cosmo: list[np.ndarray] = []
    logM_bins_ref: np.ndarray | None = None

    for idx, imodel in enumerate(imodels_cosmo_scan):
        data_i = load_velocity_moment_hh_transformed(
            int(imodel),
            redshift=float(emu.redshift),
            logM_cut=float(emu.logM_cut),
        )
        pair, (i, j) = _predict_pair_from_loader_data(
            emu,
            data=data_i,
            logM1_req=m1_cosmo_req,
            logM2_req=m2_cosmo_req,
        )

        r_here = np.asarray(pair["r_bins"], dtype=float)
        if r_ref2 is None:
            r_ref2 = r_here
        elif not np.allclose(r_ref2, r_here, rtol=1e-10, atol=0.0, equal_nan=True):
            raise RuntimeError("Inconsistent r bins in fixed-masses cosmology scan.")

        logM_bins_i = np.asarray(data_i["logM_bins"], dtype=float)
        if logM_bins_ref is None:
            logM_bins_ref = logM_bins_i
        elif not np.allclose(
            logM_bins_ref, logM_bins_i, rtol=0.0, atol=0.0, equal_nan=True
        ):
            # Keep going; save only the first reference grid and per-model snapped masses.
            pass

        cosmo_list.append(np.asarray(data_i["cosmo_params"], dtype=float))
        logM1_used_list.append(float(pair["logM1"]))
        logM2_used_list.append(float(pair["logM2"]))
        logM1_idx_list.append(int(i))
        logM2_idx_list.append(int(j))
        raw_emu_cosmo.append(np.asarray(pair["raw_emu"], dtype=float))
        raw_true_cosmo.append(np.asarray(pair["raw_true"], dtype=float))
        m10_lin_cosmo.append(np.asarray(pair["m10_linear"], dtype=float))
        t_emu_cosmo.append(np.asarray(pair["transformed_emu"], dtype=float))
        t_true_cosmo.append(np.asarray(pair["transformed_true"], dtype=float))

    out_fixed_masses = outdir / f"{outfile_prefix}_fixed_masses_cosmoscan.npz"
    raw_emu_cosmo_arr = np.stack(raw_emu_cosmo, axis=0)
    raw_true_cosmo_arr = np.stack(raw_true_cosmo, axis=0)
    t_emu_cosmo_arr = np.stack(t_emu_cosmo, axis=0)
    t_true_cosmo_arr = np.stack(t_true_cosmo, axis=0)
    m10_lin_cosmo_arr = np.stack(m10_lin_cosmo, axis=0)
    fixed_masses_payload = dict(
        mode="fixed_masses_vary_cosmology",
        checkpoint_path=str(ckpt_path),
        target_key=np.array(target_key),
        raw_target_key=np.array(raw_key),
        target_power=np.array(target_power, dtype=int),
        imodels=np.asarray(imodels_cosmo_scan, dtype=int),
        cosmo_params=np.stack(cosmo_list, axis=0),
        r_bins=np.asarray(r_ref2, dtype=float),
        logM_bins_truth=np.asarray(
            logM_bins_ref if logM_bins_ref is not None else np.array([]), dtype=float
        ),
        logM1_requested=np.array(m1_cosmo_req, dtype=float),
        logM2_requested=np.array(m2_cosmo_req, dtype=float),
        logM1=np.asarray(logM1_used_list, dtype=float),
        logM2=np.asarray(logM2_used_list, dtype=float),
        logM1_indices=np.asarray(logM1_idx_list, dtype=int),
        logM2_indices=np.asarray(logM2_idx_list, dtype=int),
        raw_emu=raw_emu_cosmo_arr,
        raw_true=raw_true_cosmo_arr,
        m10_linear=m10_lin_cosmo_arr,
        transformed_emu=t_emu_cosmo_arr,
        transformed_true=t_true_cosmo_arr,
    )
    if raw_key == "m10":
        fixed_masses_payload.update(
            m10_emu=raw_emu_cosmo_arr,
            m10_true=raw_true_cosmo_arr,
            m10_transformed_emu=t_emu_cosmo_arr,
            m10_transformed_true=t_true_cosmo_arr,
        )
    np.savez(
        out_fixed_masses,
        **fixed_masses_payload,
    )
    print(f"Saved: {out_fixed_masses}")


if __name__ == "__main__":
    main()
