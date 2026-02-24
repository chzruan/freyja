import h5py
import numpy as np
from pathlib import Path
from .xi_hh import load_cosmology_wrapper

from freyja.emulators import HaloLinearBiasEmulator, MatterXiEmulator

emulator_linear_bias = HaloLinearBiasEmulator()
emulator_matter_xi = MatterXiEmulator()

import pyccl as ccl


def _bar_xi_from_xi(r: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Compute volume-averaged correlation function \bar{xi}(r) on a tabulated grid."""
    r = np.asarray(r, dtype=float)
    xi = np.asarray(xi, dtype=float)
    if r.ndim != 1 or xi.ndim != 1 or r.shape != xi.shape:
        raise ValueError("r and xi must be 1D arrays with matching shapes.")
    if np.any(r <= 0.0) or np.any(np.diff(r) <= 0.0):
        raise ValueError("r must be strictly increasing and positive.")

    y = xi * r**2
    dr = np.diff(r)
    cum = np.zeros_like(r)
    cum[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dr)
    return 3.0 * cum / (r**3)


def f_from_pyccl(a, Om0=0.3, Ob0=0.049, h=0.67, ns=0.965, sigma8=0.8, w0=-1.0, wa=0.0):
    """
    Return f(a)=dlnD/dlna from pyccl. a can be scalar or array.
    """
    cosmo = ccl.Cosmology(
        Omega_c=Om0 - Ob0, Omega_b=Ob0, h=h, n_s=ns, sigma8=sigma8, w0=w0, wa=wa
    )
    a = np.atleast_1d(a).astype(float)
    f = ccl.growth_rate(cosmo, a)  # f(a)
    return f[0] if f.size == 1 else f


def m10_anchor(
    cosmo_params,
    r: np.ndarray,
    *,
    xi_mm: np.ndarray,
    a: float,
    H: float,
    f: float,
    b1: float = 1.0,
    b2: float = 1.0,
    include_denominator: bool = False,
) -> dict:
    """
    Large-scale anchor for the first radial pairwise-velocity moment m_10(r).

    Uses (linear / large-scale) pair-conservation-based form:
      m10_mm(r) = -(2/3) * a * H * f * r * bar_xi_mm(r)              (xi << 1)

    For two halo bins (M1, M2) with linear bias b1, b2 and no velocity bias:
      m10_hh(r|M1,M2) ≈ -(2/3) * a * H * f * r * ((b1+b2)/2) * bar_xi_mm(r)

    Optionally, you can include a weakly-nonlinear denominator:
      / (1 + b1*b2*xi_mm(r))
    which matches the common "1+xi" pair-weighting structure.

    Parameters
    ----------
    cosmo_params : any
        Retained for API compatibility (not used directly).
    r : (N,) array
        Separation in (Mpc/h) or your chosen length unit.
    xi_mm : (N,) array
        Matter correlation function evaluated on the same ``r`` grid.
    a : float
        Scale factor.
    H : float
        Hubble rate H(a). Units must be consistent with your velocity units.
        If you want velocities in (km/s) and r in (Mpc/h), supply H in (km/s)/(Mpc/h).
    f : float
        Linear growth rate f(a) = d ln D / d ln a.
    b1, b2 : float
        Linear bias factors for the two halo bins. Defaults to matter (1,1).
    include_denominator : bool
        If True, returns m10 / (1 + b1*b2*xi_mm). On very large scales this is ~1.

    Returns
    -------
    out : dict
        {
          "xi_mm": xi_mm(r),
          "bar_xi_mm": bar_xi_mm(r),
          "m10_mm": matter anchor (b1=b2=1),
          "m10_hh": halo-bin anchor (b1,b2),
        }
    """
    _ = cosmo_params  # kept for API compatibility
    r = np.asarray(r, dtype=float)
    xi_mm = np.asarray(xi_mm, dtype=float)
    if xi_mm.shape != r.shape:
        raise ValueError("xi_mm must have the same shape as r.")

    bar_xi = _bar_xi_from_xi(r, xi_mm)

    pref = -(2.0 / 3.0) * a * H * f
    m10_mm = pref * r * bar_xi
    m10_hh = pref * r * (0.5 * (b1 + b2)) * bar_xi

    if include_denominator:
        denom_mm = 1.0 + xi_mm
        denom_hh = 1.0 + (b1 * b2) * xi_mm
        m10_mm = m10_mm / denom_mm
        m10_hh = m10_hh / denom_hh

    return {
        "xi_mm": xi_mm,
        "bar_xi_mm": bar_xi,
        "m10_mm": m10_mm,
        "m10_hh": m10_hh,
    }


# --- Default Constants & Paths ---
GRAVITY = "LCDM"
DATAFLAG = "wide_sample_first_64"
REDSHIFT = 0.25
SNAPNUM = 137
MOMENT_FIELDS = ("m10", "c20", "c02", "c30", "c12", "c40", "c22", "c04")


def load_velocity_moment_hh(imodel, gravity=GRAVITY, redshift=REDSHIFT, logM_cut=14.0):
    """Load the halo-halo velocity moment data for a given cosmological model and redshift.

    Parameters
    ----------
    imodel : int
        Index of the cosmological model to load.
    gravity : str, optional
        Gravity model to use, by default "LCDM".
    redshift : float, optional
        Redshift of the data to load, by default 0.25.
    logM_cut : float or None, optional
        Only load halo mass bins with ``logM <= logM_cut``. If ``None``, no mass
        cut is applied. Default is ``14.0``.

    Returns
    -------
    dict
        Dictionary containing box-averaged halo-halo velocity moments on a
        regular mass-pair grid. Moment arrays have shape ``(N_M, N_M, N_r)`` and
        are averaged over available ``box_*`` groups. ``*_err`` entries are the
        standard error of the mean (SEM) across boxes.
    """
    # Load cosmology information
    cosmo_params = load_cosmology_wrapper(imodel)

    # Construct the file path based on the provided parameters
    base_path = Path(
        "/cosma8/data/dp203/dc-ruan1/proj_emulator_RSD/work1/DMx64/data/velmom/"
    )
    file_name = f"vm_hh-diffM_first64_{gravity}_model{imodel}_z{redshift:.2f}.hdf5"
    file_path = base_path / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    def _mass_from_group(name, prefix):
        if not name.startswith(prefix + "_"):
            raise ValueError(f"Unexpected group name '{name}' (expected '{prefix}_*').")
        return float(name.split("_", 1)[1])

    def _box_sort_key(name):
        return int(name.split("_", 1)[1])

    # Load the data from the HDF5 file
    with h5py.File(file_path, "r") as f:
        box_keys = sorted(
            [k for k in f.keys() if k.startswith("box_")], key=_box_sort_key
        )
        if not box_keys:
            raise RuntimeError(f"No 'box_*' groups found in {file_path}")

        if "log10M_bincentres" in f:
            logM_bins_all = np.asarray(f["log10M_bincentres"][:], dtype=float)
        else:
            # Fallback: infer from the first box.
            masses = set()
            for k1 in f[box_keys[0]].keys():
                if not k1.startswith("logM1_"):
                    continue
                masses.add(round(_mass_from_group(k1, "logM1"), 2))
                for k2 in f[box_keys[0]][k1].keys():
                    if k2.startswith("logM2_"):
                        masses.add(round(_mass_from_group(k2, "logM2"), 2))
            logM_bins_all = np.array(sorted(masses), dtype=float)

        if logM_cut is None:
            logM_bins = logM_bins_all
        else:
            logM_bins = logM_bins_all[logM_bins_all <= float(logM_cut)]
        if logM_bins.size == 0:
            raise ValueError(
                f"No halo mass bins satisfy logM <= {logM_cut} in file {file_path}."
            )

        if "rbins_centres" in f:
            r_vm = np.asarray(f["rbins_centres"][:], dtype=float)
        else:
            # Fallback: grab from the first available pair in the first box.
            first_box = f[box_keys[0]]
            first_k1 = sorted(k for k in first_box.keys() if k.startswith("logM1_"))[0]
            first_k2 = sorted(
                k for k in first_box[first_k1].keys() if k.startswith("logM2_")
            )[0]
            r_vm = np.asarray(first_box[first_k1][first_k2]["r_velmom"][:], dtype=float)

        n_m = len(logM_bins)
        n_r = len(r_vm)
        mass_to_idx = {f"{m:.2f}": i for i, m in enumerate(logM_bins)}

        stacks = {
            field: [[[] for _ in range(n_m)] for _ in range(n_m)]
            for field in MOMENT_FIELDS
        }
        counts = np.zeros((n_m, n_m), dtype=int)

        for box_key in box_keys:
            gp_box = f[box_key]
            for k1 in sorted(gp_box.keys()):
                if not k1.startswith("logM1_"):
                    continue
                m1 = _mass_from_group(k1, "logM1")
                if (logM_cut is not None) and (m1 > float(logM_cut)):
                    continue
                i = mass_to_idx.get(f"{m1:.2f}")
                if i is None:
                    raise RuntimeError(
                        f"Mass {m1:.2f} from {k1} not found in log10M_bincentres."
                    )

                gp1 = gp_box[k1]
                for k2 in sorted(gp1.keys()):
                    if not k2.startswith("logM2_"):
                        continue
                    m2 = _mass_from_group(k2, "logM2")
                    if (logM_cut is not None) and (m2 > float(logM_cut)):
                        continue
                    j = mass_to_idx.get(f"{m2:.2f}")
                    if j is None:
                        raise RuntimeError(
                            f"Mass {m2:.2f} from {k2} not found in log10M_bincentres."
                        )

                    gp2 = gp1[k2]
                    r_here = np.asarray(gp2["r_velmom"][:], dtype=float)
                    if not np.allclose(
                        r_here, r_vm, rtol=1e-10, atol=0.0, equal_nan=True
                    ):
                        raise RuntimeError(
                            f"Inconsistent r_velmom bins for {box_key}/{k1}/{k2}."
                        )

                    counts[i, j] += 1
                    for field in MOMENT_FIELDS:
                        if field not in gp2:
                            raise KeyError(
                                f"Missing field '{field}' in {box_key}/{k1}/{k2}"
                            )
                        arr = np.asarray(gp2[field][:], dtype=float)
                        if arr.shape != (n_r,):
                            raise RuntimeError(
                                f"Unexpected shape for {field} in {box_key}/{k1}/{k2}: {arr.shape}"
                            )
                        stacks[field][i][j].append(arr)

        # Aggregate over boxes and place data on a symmetric (M1, M2) grid.
        means = {
            field: np.full((n_m, n_m, n_r), np.nan, dtype=float)
            for field in MOMENT_FIELDS
        }
        errs = {
            field: np.full((n_m, n_m, n_r), np.nan, dtype=float)
            for field in MOMENT_FIELDS
        }

        for i in range(n_m):
            for j in range(n_m):
                n_ij = counts[i, j]
                if n_ij == 0:
                    continue
                for field in MOMENT_FIELDS:
                    stack = np.stack(stacks[field][i][j], axis=0)
                    mean = np.mean(stack, axis=0)
                    sem = np.std(stack, axis=0, ddof=0) / np.sqrt(stack.shape[0])
                    means[field][i, j, :] = mean
                    errs[field][i, j, :] = sem
                    means[field][j, i, :] = mean
                    errs[field][j, i, :] = sem

        velocity_moment_data = {
            "r_vm": r_vm,
            "logM_bins": logM_bins,
            "m10": means["m10"],
            "c20": means["c20"],
            "c02": means["c02"],
            "c30": means["c30"],
            "c12": means["c12"],
            "c40": means["c40"],
            "c22": means["c22"],
            "c04": means["c04"],
            "m10_err": errs["m10"],
            "c20_err": errs["c20"],
            "c02_err": errs["c02"],
            "c30_err": errs["c30"],
            "c12_err": errs["c12"],
            "c40_err": errs["c40"],
            "c22_err": errs["c22"],
            "c04_err": errs["c04"],
            "cosmo_params": cosmo_params,
            "redshift": redshift,
            "logM_cut": logM_cut,
        }

    return velocity_moment_data


def load_velocity_moment_hh_transformed(
    imodel, gravity=GRAVITY, redshift=REDSHIFT, logM_cut=14.0
):
    """Load and transform the halo-halo velocity moment data for a given cosmological model and redshift.

    This function applies a transformation to the raw velocity moment data to
    convert it into a more physically interpretable form. The specific
    transformation applied is based on the cosmological parameters and the
    redshift of the data.

    Parameters
    ----------
    imodel : int
        Index of the cosmological model to load.
    gravity : str, optional
        Gravity model to use, by default "LCDM".
    redshift : float, optional
        Redshift of the data to load, by default 0.25.
    logM_cut : float or None, optional
        Only load halo mass bins with ``logM <= logM_cut``. If ``None``, no mass
        cut is applied. Default is ``14.0``.

    Returns
    -------
    dict
        Dictionary containing transformed halo-halo velocity moments on a
        regular mass-pair grid. Moment arrays have shape ``(N_M, N_M, N_r)`` and
        are averaged over available ``box_*`` groups. ``*_err`` entries are the
        standard error of the mean (SEM) across boxes.
    """
    raw_data = load_velocity_moment_hh(
        imodel, gravity=gravity, redshift=redshift, logM_cut=logM_cut
    )

    def _check_nan(name, arr, *, allow_nan=False):
        n_nan = int(np.isnan(np.asarray(arr)).sum())
        if (not allow_nan) and n_nan:
            raise ValueError(f"{name} contains {n_nan} NaN values.")
        return n_nan

    # Placeholder for actual transformation logic.
    # For demonstration purposes, we'll just copy the raw data.
    transformed_data = raw_data.copy()
    cosmo_params = raw_data["cosmo_params"]
    logM_bins = raw_data["logM_bins"]
    r_vm = raw_data["r_vm"]
    redshift = raw_data["redshift"]
    linear_bias = emulator_linear_bias.predict(cosmo_params, logM_bins)

    a = 1.0 / (1.0 + redshift)
    Om0, h, S8, ns = cosmo_params
    H = h * 100.0
    omega_b_h2 = 0.0224  # fixed in DEGRACE-1 simulations
    Ob0 = omega_b_h2 / h**2
    sigma8 = S8 / np.sqrt(Om0 / 0.3)
    f = f_from_pyccl(a, Om0=Om0, Ob0=Ob0, h=h, ns=ns, sigma8=sigma8, w0=-1.0, wa=0.0)
    xi_mm_linear = np.asarray(emulator_matter_xi.predict_linear(cosmo_params, r_vm), dtype=float)
    bar_xi_linear = _bar_xi_from_xi(r_vm, xi_mm_linear)
    pref = -(2.0 / 3.0) * a * H * f
    base = pref * r_vm * bar_xi_linear  # (N_r,)
    b1 = np.asarray(linear_bias, dtype=float)[:, None]
    b2 = np.asarray(linear_bias, dtype=float)[None, :]
    bias_mean = 0.5 * (b1 + b2)  # (N_M, N_M)
    denom = 1.0 + (b1 * b2)[:, :, None] * xi_mm_linear[None, None, :]
    m10_linear = base[None, None, :] * bias_mean[:, :, None] / denom

    transformed_data["m10_linear"] = m10_linear
    with np.errstate(divide="ignore", invalid="ignore"):
        transformed_data["m10_transformed"] = np.asinh(raw_data["m10"] / m10_linear)
        transformed_data["c20_transformed"] = np.asinh(raw_data["c20"] / m10_linear**2)
        transformed_data["c02_transformed"] = np.asinh(raw_data["c02"] / m10_linear**2)
        transformed_data["c30_transformed"] = np.asinh(raw_data["c30"] / m10_linear**3)
        transformed_data["c12_transformed"] = np.asinh(raw_data["c12"] / m10_linear**3)
        transformed_data["c40_transformed"] = np.asinh(raw_data["c40"] / m10_linear**4)
        transformed_data["c22_transformed"] = np.asinh(raw_data["c22"] / m10_linear**4)
        transformed_data["c04_transformed"] = np.asinh(raw_data["c04"] / m10_linear**4)

    moment_keys = ("m10", "c20", "c02", "c30", "c12", "c40", "c22", "c04")
    for key in moment_keys:
        # Raw arrays may legitimately contain NaNs for unavailable mass pairs.
        _check_nan(f"raw_{key}", raw_data[key], allow_nan=True)

    for raw_key, out_key in (
        ("m10", "m10_transformed"),
        ("c20", "c20_transformed"),
        ("c02", "c02_transformed"),
        ("c30", "c30_transformed"),
        ("c12", "c12_transformed"),
        ("c40", "c40_transformed"),
        ("c22", "c22_transformed"),
        ("c04", "c04_transformed"),
    ):
        out_arr = np.asarray(transformed_data[out_key], dtype=float)
        _check_nan(out_key, out_arr, allow_nan=True)
        # Be permissive: invalid divisions / anchors can create NaN/Inf on a subset
        # of bins. Keep them as NaN so downstream loaders can mask/drop them.
        bad = ~np.isfinite(out_arr)
        if np.any(bad):
            out_arr = out_arr.copy()
            out_arr[bad] = np.nan
            transformed_data[out_key] = out_arr
    return transformed_data
