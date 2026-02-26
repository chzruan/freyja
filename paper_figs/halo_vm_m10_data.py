"""Export halo velocity-moment m10 scan data (emulator + truth) for later plotting.

This script uses a trained ``HaloVelocityMomentEmulator`` checkpoint (default
run dir: ``scripts/train/outputs/halo_vm_m10_beta_style``) plus
``load_velocity_moment_hh_transformed`` to export two scan modes:

1. Fixed cosmology (one imodel), fixed M1, varying M2
2. Fixed masses (M1, M2), varying cosmologies (default imodel=57..63)

For each scan it saves both emulator predictions and simulation truth for:
- ``m10_transformed`` (direct emulator target)
- ``m10`` (physical units; emulator reconstructed using ``m10_linear``)

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
)  # noqa: E402


DEFAULT_RUN_DIR = Path(
    "/cosma8/data/dp203/dc-ruan1/freyja_codex/freyja/scripts/train/outputs/halo_vm_m10_beta_style"
)
DEFAULT_CHECKPOINT_NAME = "halo_velocity_moment_m10_transformed.pt"
DEFAULT_IMODELS_COSMO_SCAN = list(range(57, 64))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export halo velocity moment m10 emulator-vs-truth scan data to NPZ files."
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
    p.add_argument("--imodel-fixed", type=int, default=59)
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
        "--outfile-prefix",
        type=str,
        default="halo_vm_m10",
        help="Prefix for output .npz filenames.",
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


def _predict_pair_from_loader_data(
    emu: HaloVelocityMomentEmulator,
    *,
    data: dict[str, np.ndarray],
    logM1_req: float,
    logM2_req: float,
) -> tuple[dict[str, np.ndarray | float], tuple[int, int]]:
    """Predict and extract truth for one mass pair from transformed loader output."""
    if emu.target_key != "m10_transformed":
        raise RuntimeError(
            f"This export script expects a m10_transformed checkpoint, got target_key={emu.target_key!r}."
        )

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
    m10_true = np.asarray(data["m10"][i, j, :], dtype=float)
    m10_true_transformed = np.asarray(data["m10_transformed"][i, j, :], dtype=float)

    m10_emu_transformed = np.asarray(
        emu.predict(
            cosmo,
            np.full_like(logr, logM1_use, dtype=float),
            np.full_like(logr, logM2_use, dtype=float),
            logr,
            return_numpy=True,
        ),
        dtype=float,
    )
    m10_emu = np.sinh(m10_emu_transformed) * m10_linear

    return (
        {
            "r_bins": r,
            "logM1": logM1_use,
            "logM2": logM2_use,
            "m10_linear": m10_linear,
            "m10_true": m10_true,
            "m10_emu": m10_emu,
            "m10_transformed_true": m10_true_transformed,
            "m10_transformed_emu": m10_emu_transformed,
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
    if emu.target_key != "m10_transformed":
        raise RuntimeError(
            f"Expected a checkpoint trained on m10_transformed, got target_key={emu.target_key!r}"
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
    m10_emu_list: list[np.ndarray] = []
    m10_true_list: list[np.ndarray] = []
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
        m10_emu_list.append(np.asarray(pair["m10_emu"], dtype=float))
        m10_true_list.append(np.asarray(pair["m10_true"], dtype=float))
        m10_lin_list.append(np.asarray(pair["m10_linear"], dtype=float))
        t_emu_list.append(np.asarray(pair["m10_transformed_emu"], dtype=float))
        t_true_list.append(np.asarray(pair["m10_transformed_true"], dtype=float))

    out_fixed_imodel = outdir / f"{args.outfile_prefix}_fixed_imodel_m2scan.npz"
    np.savez(
        out_fixed_imodel,
        mode="fixed_imodel_vary_m2",
        checkpoint_path=str(ckpt_path),
        target_key=np.array(emu.target_key),
        imodel=np.array(imodel_fixed, dtype=int),
        cosmo_params=cosmo_fixed,
        r_bins=np.asarray(r_ref, dtype=float),
        logM_bins_truth=logM_bins_fixed,
        logM1_requested=np.array(m1_fixed_req, dtype=float),
        logM1=np.array(m1_fixed_use, dtype=float),
        logM2_grid_requested=np.asarray(m2_grid_req, dtype=float),
        logM2_grid=np.asarray(m2_grid_used, dtype=float),
        logM2_indices=np.asarray(m2_index_used, dtype=int),
        m10_emu=np.stack(m10_emu_list, axis=0),
        m10_true=np.stack(m10_true_list, axis=0),
        m10_linear=np.stack(m10_lin_list, axis=0),
        m10_transformed_emu=np.stack(t_emu_list, axis=0),
        m10_transformed_true=np.stack(t_true_list, axis=0),
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
    m10_emu_cosmo: list[np.ndarray] = []
    m10_true_cosmo: list[np.ndarray] = []
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
        m10_emu_cosmo.append(np.asarray(pair["m10_emu"], dtype=float))
        m10_true_cosmo.append(np.asarray(pair["m10_true"], dtype=float))
        m10_lin_cosmo.append(np.asarray(pair["m10_linear"], dtype=float))
        t_emu_cosmo.append(np.asarray(pair["m10_transformed_emu"], dtype=float))
        t_true_cosmo.append(np.asarray(pair["m10_transformed_true"], dtype=float))

    out_fixed_masses = outdir / f"{args.outfile_prefix}_fixed_masses_cosmoscan.npz"
    np.savez(
        out_fixed_masses,
        mode="fixed_masses_vary_cosmology",
        checkpoint_path=str(ckpt_path),
        target_key=np.array(emu.target_key),
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
        m10_emu=np.stack(m10_emu_cosmo, axis=0),
        m10_true=np.stack(m10_true_cosmo, axis=0),
        m10_linear=np.stack(m10_lin_cosmo, axis=0),
        m10_transformed_emu=np.stack(t_emu_cosmo, axis=0),
        m10_transformed_true=np.stack(t_true_cosmo, axis=0),
    )
    print(f"Saved: {out_fixed_masses}")


if __name__ == "__main__":
    main()
