#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

# Avoid JAX trying to initialize unavailable CUDA plugins when importing
# freyja.cosma.velocity_moment_hh.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from freyja.cosma.xi_hh import (  # noqa: E402
    load_cosmology_wrapper,
    load_linear_pkmm_data,
    load_pkmm_data,
    load_xihh_data,
    load_ximm_data,
)
from freyja.cosma.velocity_moment_hh import (  # noqa: E402
    load_velocity_moment_hh,
    load_velocity_moment_hh_transformed,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data"
DEFAULT_GRAVITY = "LCDM"
DEFAULT_DATAFLAG = "wide_sample_first_64"
DEFAULT_REDSHIFT = 0.25
DEFAULT_SNAPNUM = 137
DEFAULT_LOGM_CUT = 14.0
DEFAULT_MODEL_MIN = 1
DEFAULT_MODEL_MAX = 64

XIHH_TEMPLATE = (
    "/cosma8/data/dp203/dc-ruan1/proj_emulator_RSD/work1/DMx64/data/"
    "xiR_hh-diffM_{gravity}_{dataflag}_z{redshift:.2f}_model{imodel}.hdf5"
)
PKMM_TEMPLATE = (
    "/cosma8/data/dp203/dc-ruan1/mg_glam/"
    "DurMun_hmfemu_{gravity}_{dataflag}_model{imodel}_L1024Np2048Ng4096/"
    "Run{ibox}/PowerDM.log.{snapnum:04d}.{ibox:04d}.dat"
)
PKMM_LINEAR_TEMPLATE = (
    "/cosma8/data/dp203/dc-ruan1/mg_glam/"
    "DurMun_hmfemu_{gravity}_{dataflag}_model{imodel}_L1024Np2048Ng4096/PkTable.dat"
)
VELMOM_TEMPLATE = (
    "/cosma8/data/dp203/dc-ruan1/proj_emulator_RSD/work1/DMx64/data/velmom/"
    "vm_hh-diffM_first64_{gravity}_model{imodel}_z{redshift:.2f}.hdf5"
)
FID_XIHH_TEMPLATE = (
    "/cosma8/data/dp203/dc-ruan1/DESI_MGx100/data/xiR_hh-diffM_GR_z{redshift:.2f}.hdf5"
)
FID_PKMM_TEMPLATE = (
    "/cosma8/data/dp203/dc-ruan1/DESI_MGx100/data/GR/"
    "Run{ibox}/PowerDM.log.{snapnum:04d}.{ibox:04d}.dat"
)

RAW_VELMOM_KEYS = (
    "m10",
    "c20",
    "c02",
    "c30",
    "c12",
    "c40",
    "c22",
    "c04",
    "m10_err",
    "c20_err",
    "c02_err",
    "c30_err",
    "c12_err",
    "c40_err",
    "c22_err",
    "c04_err",
)
TRANSFORMED_VELMOM_KEYS = (
    "m10_linear",
    "m10_transformed",
    "c20_transformed",
    "c02_transformed",
    "c30_transformed",
    "c12_transformed",
    "c40_transformed",
    "c22_transformed",
    "c04_transformed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export COSMA-only processed Freyja data into split HDF5 files."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gravity", type=str, default=DEFAULT_GRAVITY)
    parser.add_argument("--dataflag", type=str, default=DEFAULT_DATAFLAG)
    parser.add_argument("--redshift", type=float, default=DEFAULT_REDSHIFT)
    parser.add_argument("--snapnum", type=int, default=DEFAULT_SNAPNUM)
    parser.add_argument("--logM-cut", type=float, default=DEFAULT_LOGM_CUT)
    parser.add_argument("--model-min", type=int, default=DEFAULT_MODEL_MIN)
    parser.add_argument("--model-max", type=int, default=DEFAULT_MODEL_MAX)
    parser.add_argument("--models", nargs="+", type=int)
    parser.add_argument("--include-fiducial", action="store_true")
    parser.add_argument("--include-pk-boxes", action="store_true")
    parser.add_argument("--include-transformed-velmom", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_models(args: argparse.Namespace) -> list[int]:
    if args.models:
        models = sorted(set(int(m) for m in args.models))
    else:
        models = list(range(int(args.model_min), int(args.model_max) + 1))
    if args.include_fiducial and 0 not in models:
        models = [0] + models
    return models


def file_stem(args: argparse.Namespace) -> str:
    return f"{args.gravity}_{args.dataflag}_z{args.redshift:.2f}"


def ensure_output_path(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
        path.unlink()


def set_common_attrs(handle: h5py.File, args: argparse.Namespace, source: str) -> None:
    handle.attrs["gravity"] = args.gravity
    handle.attrs["dataflag"] = args.dataflag
    handle.attrs["redshift"] = args.redshift
    handle.attrs["snapnum"] = args.snapnum
    handle.attrs["source"] = source
    handle.attrs["excludes_halo_catalogs"] = True


def add_model_attrs(group: h5py.Group, imodel: int, n_boxes: int) -> None:
    group.attrs["imodel"] = imodel
    group.attrs["is_fiducial"] = imodel == 0
    group.attrs["n_boxes"] = n_boxes


def write_dataset(
    group: h5py.Group,
    name: str,
    data: np.ndarray,
    *,
    compress: bool = False,
) -> None:
    arr = np.asarray(data)
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    kwargs = {}
    if compress and arr.ndim >= 2:
        kwargs["compression"] = "gzip"
        kwargs["shuffle"] = True
    group.create_dataset(name, data=arr, **kwargs)


def create_output_handles(args: argparse.Namespace) -> dict[str, h5py.File]:
    stem = file_stem(args)
    paths = {
        "cosmo": args.output_dir / f"cosmo_params_{stem}.hdf5",
        "xihh": args.output_dir / f"xiR_hh_diffM_{stem}.hdf5",
        "pkmm": args.output_dir / f"pk_mm_nonlinear_{stem}.hdf5",
        "pkmm_linear": args.output_dir / f"pk_mm_linear_{stem}.hdf5",
        "ximm": args.output_dir / f"xi_mm_{stem}.hdf5",
        "velmom": args.output_dir / f"velmom_hh_{stem}.hdf5",
    }
    if args.include_transformed_velmom:
        paths["velmom_transformed"] = (
            args.output_dir / f"velmom_hh_transformed_{stem}.hdf5"
        )

    handles: dict[str, h5py.File] = {}
    for path in paths.values():
        ensure_output_path(path, args.overwrite)

    handles["cosmo"] = h5py.File(paths["cosmo"], "w")
    handles["xihh"] = h5py.File(paths["xihh"], "w")
    handles["pkmm"] = h5py.File(paths["pkmm"], "w")
    handles["pkmm_linear"] = h5py.File(paths["pkmm_linear"], "w")
    handles["ximm"] = h5py.File(paths["ximm"], "w")
    handles["velmom"] = h5py.File(paths["velmom"], "w")
    if args.include_transformed_velmom:
        handles["velmom_transformed"] = h5py.File(paths["velmom_transformed"], "w")

    set_common_attrs(handles["cosmo"], args, "freyja.cosma.xi_hh.load_cosmology_wrapper")
    set_common_attrs(handles["xihh"], args, "freyja.cosma.xi_hh.load_xihh_data")
    set_common_attrs(handles["pkmm"], args, "freyja.cosma.xi_hh.load_pkmm_data")
    set_common_attrs(
        handles["pkmm_linear"], args, "freyja.cosma.xi_hh.load_linear_pkmm_data"
    )
    set_common_attrs(handles["ximm"], args, "freyja.cosma.xi_hh.load_ximm_data")
    set_common_attrs(
        handles["velmom"], args, "freyja.cosma.velocity_moment_hh.load_velocity_moment_hh"
    )
    if args.include_transformed_velmom:
        set_common_attrs(
            handles["velmom_transformed"],
            args,
            "freyja.cosma.velocity_moment_hh.load_velocity_moment_hh_transformed",
        )

    handles["xihh"].attrs["source_template"] = XIHH_TEMPLATE
    handles["xihh"].attrs["fiducial_source_template"] = FID_XIHH_TEMPLATE
    handles["pkmm"].attrs["source_template"] = PKMM_TEMPLATE
    handles["pkmm"].attrs["fiducial_source_template"] = FID_PKMM_TEMPLATE
    handles["pkmm_linear"].attrs["source_template"] = PKMM_LINEAR_TEMPLATE
    handles["velmom"].attrs["source_template"] = VELMOM_TEMPLATE
    handles["velmom"].attrs["logM_cut"] = args.logM_cut
    if args.include_transformed_velmom:
        handles["velmom_transformed"].attrs["logM_cut"] = args.logM_cut

    return handles


def write_cosmo_group(handle: h5py.File, imodel: int, cosmo_params: np.ndarray) -> None:
    group = handle.create_group(f"model{imodel}")
    add_model_attrs(group, imodel, 100 if imodel == 0 else 5)
    write_dataset(group, "cosmo_params", cosmo_params)


def write_xihh_group(
    handle: h5py.File,
    imodel: int,
    r_all: np.ndarray,
    logM_bins: np.ndarray,
    xi_hh: np.ndarray,
    xi_sem: np.ndarray,
) -> None:
    group = handle.create_group(f"model{imodel}")
    add_model_attrs(group, imodel, 100 if imodel == 0 else 5)
    write_dataset(group, "r", r_all)
    write_dataset(group, "logM_bins", logM_bins)
    write_dataset(group, "xi_hh", xi_hh, compress=True)
    write_dataset(group, "xi_sem", xi_sem, compress=True)


def write_pk_group(
    handle: h5py.File,
    imodel: int,
    *,
    k: np.ndarray | None,
    pk_mean: np.ndarray | None,
    pk_boxes: np.ndarray | None = None,
    linear: bool = False,
) -> None:
    group = handle.create_group(f"model{imodel}")
    add_model_attrs(group, imodel, 100 if imodel == 0 else 5)
    if k is None or pk_mean is None:
        group.attrs["missing"] = True
        group.attrs["missing_reason"] = "No matching linear PkTable source is defined for model0."
        return
    name = "pkmm_linear" if linear else "pkmm_mean"
    write_dataset(group, "k", k)
    write_dataset(group, name, pk_mean)
    if pk_boxes is not None:
        write_dataset(group, "pkmm_boxes", pk_boxes, compress=True)


def write_ximm_group(
    handle: h5py.File, imodel: int, r_all: np.ndarray, ximm: np.ndarray
) -> None:
    group = handle.create_group(f"model{imodel}")
    add_model_attrs(group, imodel, 100 if imodel == 0 else 5)
    write_dataset(group, "r", r_all)
    write_dataset(group, "ximm", ximm)


def write_velmom_group(
    handle: h5py.File, imodel: int, data: dict[str, np.ndarray], keys: Iterable[str]
) -> None:
    group = handle.create_group(f"model{imodel}")
    add_model_attrs(group, imodel, 5)
    write_dataset(group, "r", data["r_vm"])
    write_dataset(group, "logM_bins", data["logM_bins"])
    for key in keys:
        write_dataset(group, key, data[key], compress=True)


def build_description(args: argparse.Namespace, models: list[int]) -> str:
    stem = file_stem(args)
    lines = [
        f"# COSMA processed data export: {stem}",
        "",
        "## Coverage",
        "",
        f"- Models: {models[0]}..{models[-1]}" if models else "- Models: none",
        f"- Number of models: {len(models)}",
        f"- Gravity: `{args.gravity}`",
        f"- Data flag: `{args.dataflag}`",
        f"- Redshift: `{args.redshift:.2f}`",
        f"- Snapnum: `{args.snapnum}`",
        f"- Halo velocity-moment logM cut: `{args.logM_cut:.2f}`",
        f"- Includes fiducial model0: `{args.include_fiducial}`",
        f"- Includes transformed velocity moments: `{args.include_transformed_velmom}`",
        f"- Includes nonlinear P(k) box stacks: `{args.include_pk_boxes}`",
        "- Halo catalogs are not included.",
        "",
        "## Output files",
        "",
        f"- `data/cosmo_params_{stem}.hdf5`: one `cosmo_params` vector per model.",
        f"- `data/xiR_hh_diffM_{stem}.hdf5`: `r`, `logM_bins`, `xi_hh`, `xi_sem`.",
        f"- `data/pk_mm_nonlinear_{stem}.hdf5`: nonlinear matter `k`, `pkmm_mean`"
        + (", `pkmm_boxes`." if args.include_pk_boxes else "."),
        f"- `data/pk_mm_linear_{stem}.hdf5`: linear matter `k`, `pkmm_linear`.",
        f"- `data/xi_mm_{stem}.hdf5`: `r`, `ximm` on the halo-correlation radial grid.",
        f"- `data/velmom_hh_{stem}.hdf5`: raw halo velocity moments and SEMs for models with COSMA velocity-moment files.",
    ]
    if args.include_transformed_velmom:
        lines.append(
            f"- `data/velmom_hh_transformed_{stem}.hdf5`: transformed halo velocity moments."
        )

    lines.extend(
        [
            "",
            "## COSMA sources",
            "",
            f"- Halo-halo correlation HDF5: `{XIHH_TEMPLATE}`",
            f"- Nonlinear matter power spectra: `{PKMM_TEMPLATE}`",
            f"- Linear matter power spectra: `{PKMM_LINEAR_TEMPLATE}`",
            f"- Halo velocity moments: `{VELMOM_TEMPLATE}`",
            f"- Fiducial halo-halo correlation: `{FID_XIHH_TEMPLATE}`",
            f"- Fiducial nonlinear matter power spectra: `{FID_PKMM_TEMPLATE}`",
            "",
            "## Notes",
            "",
            "- `xi_mm` is generated from the nonlinear box-averaged matter power spectrum on the same `r` grid used by `xi_hh`.",
            "- `xi_hh` and raw velocity-moment arrays are symmetric in halo-mass-pair indices after box averaging.",
            "- All floating-point datasets are written as `float32` with gzip compression for multidimensional arrays.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    models = resolve_models(args)
    if not models:
        raise ValueError("No models selected.")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    description_path = (
        args.output_dir / f"cosma_processed_description_{file_stem(args)}.md"
    )
    ensure_output_path(description_path, args.overwrite)

    handles = create_output_handles(args)
    try:
        for imodel in models:
            cosmo_params = load_cosmology_wrapper(imodel)
            r_all, logM_bins, xi_hh, xi_sem = load_xihh_data(
                imodel,
                gravity=args.gravity,
                dataflag=args.dataflag,
                redshift=args.redshift,
            )
            ximm = load_ximm_data(
                r_all,
                imodel=imodel,
                gravity=args.gravity,
                dataflag=args.dataflag,
                snapnum=args.snapnum,
            )
            k_pk, pk_mean = load_pkmm_data(
                imodel=imodel,
                gravity=args.gravity,
                dataflag=args.dataflag,
                snapnum=args.snapnum,
                return_mean=True,
            )
            pk_boxes = None
            if args.include_pk_boxes:
                _, pk_boxes = load_pkmm_data(
                    imodel=imodel,
                    gravity=args.gravity,
                    dataflag=args.dataflag,
                    snapnum=args.snapnum,
                    return_mean=False,
                )
            if imodel == 0:
                k_lin, pk_linear = None, None
            else:
                k_lin, pk_linear = load_linear_pkmm_data(
                    imodel=imodel,
                    gravity=args.gravity,
                    dataflag=args.dataflag,
                )

            write_cosmo_group(handles["cosmo"], imodel, cosmo_params)
            write_xihh_group(handles["xihh"], imodel, r_all, logM_bins, xi_hh, xi_sem)
            write_pk_group(handles["pkmm"], imodel, k=k_pk, pk_mean=pk_mean, pk_boxes=pk_boxes)
            write_pk_group(
                handles["pkmm_linear"],
                imodel,
                k=k_lin,
                pk_mean=pk_linear,
                linear=True,
            )
            write_ximm_group(handles["ximm"], imodel, r_all, ximm)

            if imodel != 0:
                velmom = load_velocity_moment_hh(
                    imodel,
                    gravity=args.gravity,
                    redshift=args.redshift,
                    logM_cut=args.logM_cut,
                )
                write_velmom_group(handles["velmom"], imodel, velmom, RAW_VELMOM_KEYS)

                if args.include_transformed_velmom:
                    velmom_transformed = load_velocity_moment_hh_transformed(
                        imodel,
                        gravity=args.gravity,
                        redshift=args.redshift,
                        logM_cut=args.logM_cut,
                    )
                    write_velmom_group(
                        handles["velmom_transformed"],
                        imodel,
                        velmom_transformed,
                        TRANSFORMED_VELMOM_KEYS,
                    )
    finally:
        for handle in handles.values():
            handle.close()

    description_path.write_text(build_description(args, models))
    print(f"Wrote split COSMA exports to {args.output_dir}")


if __name__ == "__main__":
    main()
