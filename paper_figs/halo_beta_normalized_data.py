"""Export normalized scale-dependent halo bias data for later plotting.

This script uses:
- ``freyja.emulators.xi_R_hh_diffM.HaloBetaEmulator``
- ``freyja.emulators.halo_linear_bias.HaloLinearBiasEmulator``

to compute the normalized scale-dependent bias

    beta_norm(r | M1, M2) = beta(r | M1, M2) / [b(M1) b(M2)]

and save arrays for two scan modes:

1. Fixed cosmology (one ``imodel``), fixed ``M1``, varying ``M2``
2. Fixed ``M1`` and ``M2``, varying cosmologies (default ``imodel=57..63``)

The outputs are saved in ``.npz`` format for plotting later.

Example
-------
python3 paper_figs/halo_beta_normalized_data.py
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

from freyja.cosma.xi_hh import (  # noqa: E402
    load_cosmology_wrapper,
    load_xihh_data,
    load_ximm_data,
)
from freyja.emulators.halo_linear_bias import HaloLinearBiasEmulator  # noqa: E402
from freyja.emulators.xi_R_hh_diffM import HaloBetaEmulator  # noqa: E402


DEFAULT_IMODELS_COSMO_SCAN = list(range(57, 64))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Export normalized scale-dependent halo bias beta/[b(M1)b(M2)] "
            "for fixed-cosmology and fixed-mass scans."
        )
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("data"),
        help="Directory where .npz outputs are written.",
    )

    # Fixed cosmology, varying M2 scan
    p.add_argument("--imodel-fixed", type=int, default=60)
    p.add_argument("--m1-fixed", type=float, default=13.5, help="Fixed log10(M1).")
    p.add_argument("--m2-min", type=float, default=12.5)
    p.add_argument("--m2-max", type=float, default=14.0)
    p.add_argument("--m2-num", type=int, default=16)

    # Fixed masses, varying cosmologies scan
    p.add_argument("--m1-cosmo-scan", type=float, default=13.5)
    p.add_argument("--m2-cosmo-scan", type=float, default=13.5)
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
        default="halo_beta_normalized",
        help="Prefix for output .npz filenames.",
    )
    return p


def _predict_linear_bias_scalar(
    bias_emu: HaloLinearBiasEmulator,
    cosmo_params: np.ndarray,
    logM: float,
) -> float:
    b = bias_emu.predict(
        np.asarray(cosmo_params, dtype=float), np.array([logM], dtype=float)
    )
    b = np.asarray(b, dtype=float).reshape(-1)
    if b.size != 1 or not np.isfinite(b[0]):
        raise RuntimeError(f"Invalid linear-bias prediction for logM={logM}: {b}")
    return float(b[0])


def _predict_beta_norm(
    beta_emu: HaloBetaEmulator,
    bias_emu: HaloLinearBiasEmulator,
    cosmo_params: np.ndarray,
    logM1: float,
    logM2: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    r_bins, beta_pred = beta_emu.predict_from_masses(
        np.asarray(cosmo_params, dtype=float),
        float(logM1),
        float(logM2),
    )
    r_bins = np.asarray(r_bins, dtype=float)
    beta_pred = np.asarray(beta_pred, dtype=float)
    b1 = _predict_linear_bias_scalar(bias_emu, cosmo_params, float(logM1))
    b2 = _predict_linear_bias_scalar(bias_emu, cosmo_params, float(logM2))

    denom = b1 * b2
    if not np.isfinite(denom) or denom == 0.0:
        raise RuntimeError(
            f"Invalid normalization b(M1)b(M2) for logM1={logM1}, logM2={logM2}: {denom}"
        )
    beta_norm = beta_pred / denom
    return r_bins, beta_norm, b1, b2


def _masked_truth_beta_grid(
    beta_emu: HaloBetaEmulator,
    imodel: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load simulation beta(r|M1,M2) for truth comparisons.

    Important:
    ``HaloBetaEmulator.r_bins`` comes from the loaded checkpoint, while
    ``HaloBetaEmulator.HP`` may still reflect module defaults. To avoid
    accidental r-range truncation mismatches, we keep the full simulation r-grid
    here and interpolate truth onto the emulator grid later.
    """
    r_all, logM_all, xi_hh, _ = load_xihh_data(int(imodel))
    xi_mm = load_ximm_data(r_all, int(imodel))

    hp = beta_emu.HP
    mask_m = (logM_all >= hp["logM_cut_min"]) & (logM_all <= hp["logM_cut_max"])

    r_bins = np.asarray(r_all, dtype=float)
    logM_bins = np.asarray(logM_all[mask_m], dtype=float)
    xi_hh_cut = np.asarray(xi_hh[mask_m][:, mask_m, :], dtype=float)
    xi_mm_cut = np.asarray(xi_mm, dtype=float)

    beta_true = xi_hh_cut / xi_mm_cut[None, None, :]
    return r_bins, logM_bins, beta_true


def _nearest_mass_index(logM_bins: np.ndarray, target: float) -> int:
    logM_bins = np.asarray(logM_bins, dtype=float)
    return int(np.argmin(np.abs(logM_bins - float(target))))


def _predict_and_truth_beta_norm_for_pair(
    *,
    beta_emu: HaloBetaEmulator,
    bias_emu: HaloLinearBiasEmulator,
    cosmo_params: np.ndarray,
    r_bins_truth: np.ndarray,
    logM_bins_truth: np.ndarray,
    beta_true_grid: np.ndarray,
    logM1_req: float,
    logM2_req: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """Compute emulator + truth normalized beta for a mass pair (snapped to truth bins)."""
    i = _nearest_mass_index(logM_bins_truth, logM1_req)
    j = _nearest_mass_index(logM_bins_truth, logM2_req)
    logM1_use = float(logM_bins_truth[i])
    logM2_use = float(logM_bins_truth[j])

    r_bins_emu, beta_norm_emu, b1, b2 = _predict_beta_norm(
        beta_emu,
        bias_emu,
        np.asarray(cosmo_params, dtype=float),
        logM1_use,
        logM2_use,
    )
    r_bins_emu = np.asarray(r_bins_emu, dtype=float)
    beta_true = np.asarray(beta_true_grid[i, j, :], dtype=float)
    same_r_grid = (
        np.shape(r_bins_emu) == np.shape(r_bins_truth)
        and np.allclose(r_bins_emu, r_bins_truth, rtol=1e-10, atol=0.0, equal_nan=True)
    )
    if not same_r_grid:
        # HaloBetaEmulator.r_bins comes from its training checkpoint and can differ
        # from the current default HP cut settings. Interpolate truth onto the
        # emulator r-grid so comparisons use the same x-axis.
        rmin_truth = float(np.nanmin(r_bins_truth))
        rmax_truth = float(np.nanmax(r_bins_truth))
        if np.any(r_bins_emu < rmin_truth) or np.any(r_bins_emu > rmax_truth):
            raise RuntimeError(
                "Emulator r bins extend outside the truth masked r range; "
                f"emu=[{np.nanmin(r_bins_emu):.3g},{np.nanmax(r_bins_emu):.3g}], "
                f"truth=[{rmin_truth:.3g},{rmax_truth:.3g}]"
            )
        beta_true = np.interp(r_bins_emu, r_bins_truth, beta_true)

    beta_norm_true = beta_true / (b1 * b2)
    return r_bins_emu, beta_norm_emu, beta_norm_true, b1, b2, logM1_use, logM2_use


def main() -> None:
    args = build_parser().parse_args()

    outdir = args.outdir.expanduser()
    if not outdir.is_absolute():
        outdir = (Path.cwd() / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    m2_num = int(args.m2_num)
    if m2_num < 1:
        raise ValueError(f"--m2-num must be >= 1, got {m2_num}")

    m2_grid_req = np.linspace(float(args.m2_min), float(args.m2_max), m2_num)
    imodels_cosmo_scan = [int(m) for m in args.imodels_cosmo_scan]
    if len(imodels_cosmo_scan) == 0:
        raise ValueError("No --imodels-cosmo-scan provided.")

    print("Loading emulators...")
    bias_emu = HaloLinearBiasEmulator()
    beta_emu = HaloBetaEmulator()

    # Ensure HaloBetaEmulator has a linear-bias emulator if it needs extrapolation internals.
    if getattr(beta_emu, "linear_bias_emulator", None) is None:
        beta_emu.linear_bias_emulator = bias_emu

    if beta_emu.model is None:
        raise RuntimeError("Failed to load HaloBetaEmulator checkpoint/model.")
    if bias_emu.params is None:
        raise RuntimeError("Failed to load HaloLinearBiasEmulator checkpoint/model.")

    # ------------------------------------------------------------------
    # 1) Fixed cosmology (imodel), fixed M1, varying M2
    # ------------------------------------------------------------------
    imodel_fixed = int(args.imodel_fixed)
    m1_fixed = float(args.m1_fixed)
    cosmo_fixed = np.asarray(load_cosmology_wrapper(imodel_fixed), dtype=float)
    r_bins_truth_fixed, logM_bins_truth_fixed, beta_true_grid_fixed = _masked_truth_beta_grid(
        beta_emu, imodel_fixed
    )
    m1_fixed_used = float(logM_bins_truth_fixed[_nearest_mass_index(logM_bins_truth_fixed, m1_fixed)])

    beta_norm_emu_m2_list: list[np.ndarray] = []
    beta_norm_true_m2_list: list[np.ndarray] = []
    b2_list: list[float] = []
    b1_fixed_val: float | None = None
    r_bins_ref: np.ndarray | None = None
    m2_grid_used: list[float] = []

    print(
        f"[1/2] Fixed cosmology scan: imodel={imodel_fixed}, logM1={m1_fixed:.3f}, "
        f"varying logM2 over {m2_grid_req[0]:.3f}..{m2_grid_req[-1]:.3f} ({len(m2_grid_req)} points)"
    )
    for logM2_req in m2_grid_req:
        (
            r_bins,
            beta_norm_emu,
            beta_norm_true,
            b1,
            b2,
            _m1_use,
            m2_use,
        ) = _predict_and_truth_beta_norm_for_pair(
            beta_emu=beta_emu,
            bias_emu=bias_emu,
            cosmo_params=cosmo_fixed,
            r_bins_truth=r_bins_truth_fixed,
            logM_bins_truth=logM_bins_truth_fixed,
            beta_true_grid=beta_true_grid_fixed,
            logM1_req=m1_fixed,
            logM2_req=float(logM2_req),
        )
        if r_bins_ref is None:
            r_bins_ref = r_bins
            b1_fixed_val = b1
        elif not np.allclose(r_bins_ref, r_bins, rtol=1e-10, atol=0.0, equal_nan=True):
            raise RuntimeError("Inconsistent r bins across M2 scan predictions.")
        beta_norm_emu_m2_list.append(beta_norm_emu)
        beta_norm_true_m2_list.append(beta_norm_true)
        b2_list.append(b2)
        m2_grid_used.append(float(m2_use))

    beta_norm_emu_m2_scan = np.stack(beta_norm_emu_m2_list, axis=0)  # (N_M2, N_r)
    beta_norm_true_m2_scan = np.stack(beta_norm_true_m2_list, axis=0)  # (N_M2, N_r)

    out_fixed_cosmo = outdir / f"{args.outfile_prefix}_fixed_imodel_m2scan.npz"
    np.savez(
        out_fixed_cosmo,
        mode="fixed_imodel_vary_m2",
        imodel=np.array(imodel_fixed, dtype=int),
        cosmo_params=cosmo_fixed,
        r_bins=np.asarray(r_bins_ref, dtype=float),
        logM_bins_truth=np.asarray(logM_bins_truth_fixed, dtype=float),
        logM1_requested=np.array(m1_fixed, dtype=float),
        logM1=np.array(m1_fixed_used, dtype=float),
        logM2_grid_requested=np.asarray(m2_grid_req, dtype=float),
        logM2_grid=np.asarray(m2_grid_used, dtype=float),
        b1=np.array(float(b1_fixed_val), dtype=float),
        b2_grid=np.asarray(b2_list, dtype=float),
        beta_norm_emu=np.asarray(beta_norm_emu_m2_scan, dtype=float),
        beta_norm_true=np.asarray(beta_norm_true_m2_scan, dtype=float),
    )
    print(f"Saved: {out_fixed_cosmo}")

    # ------------------------------------------------------------------
    # 2) Fixed masses, varying cosmologies
    # ------------------------------------------------------------------
    m1_cosmo = float(args.m1_cosmo_scan)
    m2_cosmo = float(args.m2_cosmo_scan)

    # Use imodel=1 mass grid as canonical truth mass-bin grid (same across models in this dataset).
    _, logM_bins_truth_ref, _ = _masked_truth_beta_grid(beta_emu, 1)
    m1_cosmo_used = float(logM_bins_truth_ref[_nearest_mass_index(logM_bins_truth_ref, m1_cosmo)])
    m2_cosmo_used = float(logM_bins_truth_ref[_nearest_mass_index(logM_bins_truth_ref, m2_cosmo)])

    beta_norm_emu_cosmo_list: list[np.ndarray] = []
    beta_norm_true_cosmo_list: list[np.ndarray] = []
    cosmo_param_list: list[np.ndarray] = []
    b1_list: list[float] = []
    b2_list_cosmo: list[float] = []
    r_bins_ref2: np.ndarray | None = None

    print(
        f"[2/2] Fixed masses scan: logM1={m1_cosmo:.3f}, logM2={m2_cosmo:.3f}, "
        f"varying imodels={imodels_cosmo_scan}"
    )
    for imodel in imodels_cosmo_scan:
        cosmo = np.asarray(load_cosmology_wrapper(int(imodel)), dtype=float)
        r_bins_truth_i, logM_bins_truth_i, beta_true_grid_i = _masked_truth_beta_grid(
            beta_emu, int(imodel)
        )
        (
            r_bins,
            beta_norm_emu,
            beta_norm_true,
            b1,
            b2,
            _m1_use,
            _m2_use,
        ) = _predict_and_truth_beta_norm_for_pair(
            beta_emu=beta_emu,
            bias_emu=bias_emu,
            cosmo_params=cosmo,
            r_bins_truth=r_bins_truth_i,
            logM_bins_truth=logM_bins_truth_i,
            beta_true_grid=beta_true_grid_i,
            logM1_req=m1_cosmo,
            logM2_req=m2_cosmo,
        )
        if r_bins_ref2 is None:
            r_bins_ref2 = r_bins
        elif not np.allclose(r_bins_ref2, r_bins, rtol=1e-10, atol=0.0, equal_nan=True):
            raise RuntimeError("Inconsistent r bins across cosmology scan predictions.")
        beta_norm_emu_cosmo_list.append(beta_norm_emu)
        beta_norm_true_cosmo_list.append(beta_norm_true)
        cosmo_param_list.append(cosmo)
        b1_list.append(b1)
        b2_list_cosmo.append(b2)

    beta_norm_emu_cosmo_scan = np.stack(beta_norm_emu_cosmo_list, axis=0)  # (N_imodel, N_r)
    beta_norm_true_cosmo_scan = np.stack(beta_norm_true_cosmo_list, axis=0)  # (N_imodel, N_r)
    cosmo_param_arr = np.stack(cosmo_param_list, axis=0)  # (N_imodel, 4)

    out_fixed_masses = outdir / f"{args.outfile_prefix}_fixed_masses_cosmoscan.npz"
    np.savez(
        out_fixed_masses,
        mode="fixed_masses_vary_cosmology",
        imodels=np.asarray(imodels_cosmo_scan, dtype=int),
        cosmo_params=cosmo_param_arr,
        r_bins=np.asarray(r_bins_ref2, dtype=float),
        logM_bins_truth=np.asarray(logM_bins_truth_ref, dtype=float),
        logM1_requested=np.array(m1_cosmo, dtype=float),
        logM2_requested=np.array(m2_cosmo, dtype=float),
        logM1=np.array(m1_cosmo_used, dtype=float),
        logM2=np.array(m2_cosmo_used, dtype=float),
        b1=np.asarray(b1_list, dtype=float),
        b2=np.asarray(b2_list_cosmo, dtype=float),
        beta_norm_emu=np.asarray(beta_norm_emu_cosmo_scan, dtype=float),
        beta_norm_true=np.asarray(beta_norm_true_cosmo_scan, dtype=float),
    )
    print(f"Saved: {out_fixed_masses}")


if __name__ == "__main__":
    main()
