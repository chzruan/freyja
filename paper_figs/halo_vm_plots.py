"""Plot halo velocity-moment scan data from saved NPZ files.

Reads the outputs from ``paper_figs/halo_vm_m10_data.py`` and makes two
separate PDF figures:

1. Fixed cosmology (fixed M1, varying M2)
2. Fixed masses (varying cosmology)

Each figure contains:
- upper panel: emulator prediction (line) and truth data (markers)
- lower panel: relative difference in percent, 100 * (emu / true - 1)

The default plotted quantity is the raw moment corresponding to the saved
emulator target (e.g. ``m10``, ``c20``, ``c02``). You can switch to the
transformed target with ``--quantity transformed`` (or pass the explicit target
name, e.g. ``c20_transformed``).

Example
-------
python3 halo_vm_m10_plots.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams.update({"font.size": 12})
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{physics}"
params = {
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
}
plt.rcParams.update(params)
from cycler import cycler

custom_cycler = cycler(
    color=[
        "black",
        "lightseagreen",
        "orange",
        "red",
        "darkviolet",
        "mediumblue",
        "green",
    ]
)


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_on_path()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot halo velocity-moment emulator-vs-truth scans from NPZ files."
    )
    p.add_argument(
        "--datadir",
        type=Path,
        default=Path("data"),
        help="Directory containing NPZ outputs from halo_vm_m10_data.py.",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="halo_vm_m10",
        help="Filename prefix used by halo_vm_m10_data.py.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("figs"),
        help="Directory for output PDF figures.",
    )
    p.add_argument(
        "--quantity",
        type=str,
        default="raw",
        help=(
            "Quantity to plot. Accepts 'raw' (default), 'transformed', or explicit "
            "names like 'm10', 'm10_transformed', 'c20', 'c20_transformed'."
        ),
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI (PDF remains mostly vector).",
    )
    p.add_argument(
        "--legend-ncol",
        type=int,
        default=2,
        help="Legend columns for the cosmology-scan figure.",
    )
    p.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Optional y-axis minimum for top panels.",
    )
    p.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Optional y-axis maximum for top panels.",
    )
    p.add_argument(
        "--reldiff-ylim-pct",
        type=float,
        default=11.2,
        help="Symmetric y-limit (percent) for relative-difference subpanels.",
    )
    p.add_argument(
        "--reldiff-mask-min-abs-true",
        type=float,
        default=1.0e-6,
        help="Mask threshold |truth| > threshold for relative-difference panel.",
    )
    return p


def _resolve(path: Path) -> Path:
    p = path.expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def _apply_plot_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = {
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "font.size": 12,
    }
    plt.rcParams.update(params)


def _set_y_limits(ax, ymin: float | None, ymax: float | None) -> None:
    if ymin is None and ymax is None:
        return
    lo, hi = ax.get_ylim()
    if ymin is not None:
        lo = float(ymin)
    if ymax is not None:
        hi = float(ymax)
    ax.set_ylim(lo, hi)


def _relative_diff_percent(
    y_emu: np.ndarray, y_true: np.ndarray, min_abs_true: float
) -> np.ndarray:
    y_emu = np.asarray(y_emu, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    out = np.full_like(y_true, np.nan, dtype=float)
    m = (
        np.isfinite(y_emu)
        & np.isfinite(y_true)
        & (np.abs(y_true) > float(min_abs_true))
    )
    out[m] = 100.0 * ((y_emu[m] / y_true[m]) - 1.0)
    return out


def _payload_scalar_str(payload: dict[str, np.ndarray], key: str) -> str | None:
    if key not in payload:
        return None
    v = payload[key]
    arr = np.asarray(v)
    if arr.ndim == 0:
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return None


def _infer_target_names(payload: dict[str, np.ndarray]) -> tuple[str, str]:
    target_key = _payload_scalar_str(payload, "target_key")
    raw_key = _payload_scalar_str(payload, "raw_target_key")

    if target_key is None:
        if "m10_transformed_emu" in payload and "m10_transformed_true" in payload:
            target_key = "m10_transformed"
        elif "transformed_emu" in payload and "transformed_true" in payload:
            raise ValueError(
                "NPZ payload has transformed arrays but no target_key metadata."
            )
    if raw_key is None:
        if target_key is not None and target_key.endswith("_transformed"):
            raw_key = target_key[: -len("_transformed")]
        elif "m10_emu" in payload and "m10_true" in payload:
            raw_key = "m10"
        elif "raw_emu" in payload and "raw_true" in payload:
            raise ValueError(
                "NPZ payload has raw arrays but no raw_target_key metadata."
            )

    if raw_key is None or target_key is None:
        raise ValueError(
            "Could not infer raw/transformed target names from NPZ payload. "
            "Expected target_key/raw_target_key metadata or legacy m10_* keys."
        )
    return raw_key, target_key


def _raw_quantity_label(raw_key: str) -> str:
    if raw_key == "m10":
        return rf"$m_{{10}}(r | M_1, M_2) / (\mathrm{{km}}\,\mathrm{{s}}^{{-1}})$"
    prefix = "".join(ch for ch in raw_key if ch.isalpha()) or raw_key[0]
    digits = "".join(ch for ch in raw_key if ch.isdigit())
    if digits:
        return rf"${prefix}_{{{digits}}}(r | M_1, M_2)$"
    return raw_key


def _transformed_quantity_label(raw_key: str) -> str:
    if raw_key == "m10":
        return r"$m_{10,\mathrm{transformed}}$"
    prefix = "".join(ch for ch in raw_key if ch.isalpha()) or raw_key[0]
    digits = "".join(ch for ch in raw_key if ch.isdigit())
    if digits:
        return rf"${prefix}_{{{digits}}}^{{(\mathrm{{transformed}})}}$"
    return f"{raw_key}_transformed"


def _quantity_keys(
    payload: dict[str, np.ndarray], quantity: str
) -> tuple[str, str, str, str]:
    raw_key, target_key = _infer_target_names(payload)
    q = str(quantity).strip()

    if q in {"", "auto", "raw", raw_key}:
        if "raw_emu" in payload and "raw_true" in payload:
            return ("raw_emu", "raw_true", _raw_quantity_label(raw_key), raw_key)
        legacy_emu = f"{raw_key}_emu"
        legacy_true = f"{raw_key}_true"
        if legacy_emu in payload and legacy_true in payload:
            return (legacy_emu, legacy_true, _raw_quantity_label(raw_key), raw_key)
        raise KeyError(f"Raw quantity arrays for {raw_key!r} not found in NPZ payload.")

    if q in {"transformed", target_key}:
        if "transformed_emu" in payload and "transformed_true" in payload:
            return (
                "transformed_emu",
                "transformed_true",
                _transformed_quantity_label(raw_key),
                target_key,
            )
        legacy_emu = f"{target_key}_emu"
        legacy_true = f"{target_key}_true"
        if legacy_emu in payload and legacy_true in payload:
            return (
                legacy_emu,
                legacy_true,
                _transformed_quantity_label(raw_key),
                target_key,
            )
        raise KeyError(
            f"Transformed quantity arrays for {target_key!r} not found in NPZ payload."
        )

    raise ValueError(
        f"Unsupported --quantity={quantity!r}. Expected raw/transformed aliases or one of "
        f"{raw_key!r}, {target_key!r}."
    )


def plot_fixed_imodel_m2scan(
    payload: dict[str, np.ndarray],
    *,
    quantity: str,
    out_path: Path,
    dpi: int = 300,
    ymin: float | None = None,
    ymax: float | None = None,
    reldiff_ylim_pct: float = 10.0,
    reldiff_mask_min_abs_true: float = 1.0e-6,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.cm import ScalarMappable

    k_emu, k_true, y_label, _ = _quantity_keys(payload, quantity)
    r = np.asarray(payload["r_bins"], dtype=float)
    y_emu = np.asarray(payload[k_emu], dtype=float)
    y_true = np.asarray(payload[k_true], dtype=float)
    m2_grid = np.asarray(payload["logM2_grid"], dtype=float)
    imodel = int(np.asarray(payload["imodel"]).item())
    m1_use = float(np.asarray(payload["logM1"]).item())
    m1_req = float(np.asarray(payload.get("logM1_requested", payload["logM1"])).item())

    if y_emu.shape != y_true.shape or y_emu.shape != (len(m2_grid), len(r)):
        raise ValueError(
            f"Shape mismatch in fixed-imodel M2 scan: y_emu {y_emu.shape}, "
            f"y_true {y_true.shape}, m2_grid {m2_grid.shape}, r {r.shape}"
        )

    fig, ax = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7.0, 7.0),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0},
        # constrained_layout=True,
    )

    cmap = plt.get_cmap("rainbow")
    norm = mcolors.Normalize(vmin=float(np.min(m2_grid)), vmax=float(np.max(m2_grid)))
    colors = [cmap(norm(float(m2))) for m2 in m2_grid]

    for i in range(len(m2_grid)):
        ax[0].plot(r, y_emu[i], color=colors[i], lw=1.5)
        ax[0].plot(
            r, y_true[i], color=colors[i], lw=0.0, marker="o", ms=2.5, alpha=0.85
        )

        rel = _relative_diff_percent(y_emu[i], y_true[i], reldiff_mask_min_abs_true)
        m = np.isfinite(rel)
        if np.any(m):
            ax[1].plot(r[m], rel[m], color=colors[i], lw=1.2)

    ax[0].set_xlim([1, 100])
    ax[1].set_xlim([1, 100])
    ax[0].set_xscale("log")
    ax[1].set_xscale("log")
    ax[0].set_ylabel(y_label, fontsize=17)
    ax[1].set_ylabel(r"$\varepsilon (\%)$", fontsize=20)
    ax[1].set_xlabel(r"$r\,[h^{-1}\mathrm{Mpc}]$", fontsize=20)

    title = rf"Fixed cosmology (imodel={imodel}), fixed $\log M_1={m1_use:.2f}$"
    if not np.isclose(m1_use, m1_req):
        title += rf" (req: {m1_req:.2f})"
    # ax[0].set_title(title + " | lines: emulator, markers: data")
    # ax[0].grid(alpha=0.25, linewidth=0.5)
    # ax[1].grid(alpha=0.25, linewidth=0.5)
    ax[1].axhline(0.0, color="gray", ls="--", lw=0.7)

    _set_y_limits(ax[0], ymin, ymax)
    ylim = abs(float(reldiff_ylim_pct))
    ax[1].set_ylim([-ylim, ylim])

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label(r"$\log M_2$")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_fixed_masses_cosmoscan(
    payload: dict[str, np.ndarray],
    *,
    quantity: str,
    out_path: Path,
    dpi: int = 300,
    legend_ncol: int = 2,
    ymin: float | None = None,
    ymax: float | None = None,
    reldiff_ylim_pct: float = 10.0,
    reldiff_mask_min_abs_true: float = 1.0e-6,
) -> None:
    import matplotlib.pyplot as plt

    k_emu, k_true, y_label, _ = _quantity_keys(payload, quantity)
    r = np.asarray(payload["r_bins"], dtype=float)
    y_emu = np.asarray(payload[k_emu], dtype=float)
    y_true = np.asarray(payload[k_true], dtype=float)
    imodels = np.asarray(payload["imodels"], dtype=int)
    m1_use = np.asarray(payload["logM1"], dtype=float)
    m2_use = np.asarray(payload["logM2"], dtype=float)
    m1_req = float(np.asarray(payload.get("logM1_requested", m1_use[0])).item())
    m2_req = float(np.asarray(payload.get("logM2_requested", m2_use[0])).item())

    if y_emu.shape != y_true.shape or y_emu.shape != (len(imodels), len(r)):
        raise ValueError(
            f"Shape mismatch in fixed-masses cosmology scan: y_emu {y_emu.shape}, "
            f"y_true {y_true.shape}, imodels {imodels.shape}, r {r.shape}"
        )

    fig, ax = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7.0, 7.0),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0},
        # constrained_layout=True,
    )
    for a in ax:
        a.set_prop_cycle(custom_cycler)

    for i, imodel in enumerate(imodels):
        pp = ax[0].plot(r, y_emu[i], lw=1.5, label=f"imodel={int(imodel)}")
        ax[0].plot(
            r,
            y_true[i],
            color=pp[0].get_color(),
            lw=0.0,
            marker="o",
            ms=2.5,
            alpha=0.85,
        )

        rel = _relative_diff_percent(y_emu[i], y_true[i], reldiff_mask_min_abs_true)
        m = np.isfinite(rel)
        if np.any(m):
            ax[1].plot(r[m], rel[m], color=pp[0].get_color(), lw=1.2)

    ax[0].set_xlim([1, 100])
    ax[1].set_xlim([1, 100])
    ax[0].set_xscale("log")
    ax[1].set_xscale("log")
    ax[0].set_ylabel(y_label, fontsize=17)
    ax[1].set_ylabel(r"$\varepsilon (\%)$", fontsize=20)
    ax[1].set_xlabel(r"$r / (h^{-1}\mathrm{Mpc})$", fontsize=20)

    # Use first snapped mass values if all same; otherwise show requested values.
    title = (
        rf"Fixed masses scan: $\log M_1={float(m1_use[0]):.2f}$, $\log M_2={float(m2_use[0]):.2f}$"
        if np.allclose(m1_use, m1_use[0]) and np.allclose(m2_use, m2_use[0])
        else rf"Fixed masses scan (requested $\log M_1={m1_req:.2f}$, $\log M_2={m2_req:.2f}$)"
    )
    # ax[0].set_title(title + " | lines: emulator, markers: data")
    # ax[0].grid(alpha=0.25, linewidth=0.5)
    # ax[1].grid(alpha=0.25, linewidth=0.5)
    ax[1].axhline(0.0, color="gray", ls="--", lw=0.7)

    _set_y_limits(ax[0], ymin, ymax)
    ylim = abs(float(reldiff_ylim_pct))
    ax[1].set_ylim([-ylim, ylim])
    # ax[0].legend(fontsize=9, ncol=max(1, int(legend_ncol)), loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    _apply_plot_style()

    datadir = _resolve(args.datadir)
    outdir = _resolve(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prefix = str(args.prefix)
    file_fixed_imodel = datadir / f"{prefix}_fixed_imodel_m2scan.npz"
    file_fixed_masses = datadir / f"{prefix}_fixed_masses_cosmoscan.npz"

    payload_fixed_imodel = _load_npz(file_fixed_imodel)
    payload_fixed_masses = _load_npz(file_fixed_masses)

    qty = str(args.quantity)
    _, _, _, qty_slug = _quantity_keys(payload_fixed_imodel, qty)
    out_fixed_imodel_pdf = outdir / f"{prefix}_{qty_slug}_fixed_imodel_m2scan.pdf"
    out_fixed_masses_pdf = outdir / f"{prefix}_{qty_slug}_fixed_masses_cosmoscan.pdf"

    plot_fixed_imodel_m2scan(
        payload_fixed_imodel,
        quantity=qty,
        out_path=out_fixed_imodel_pdf,
        dpi=int(args.dpi),
        ymin=args.ymin,
        ymax=args.ymax,
        reldiff_ylim_pct=float(args.reldiff_ylim_pct),
        reldiff_mask_min_abs_true=float(args.reldiff_mask_min_abs_true),
    )
    print(f"Saved plot: {out_fixed_imodel_pdf}")

    plot_fixed_masses_cosmoscan(
        payload_fixed_masses,
        quantity=qty,
        out_path=out_fixed_masses_pdf,
        dpi=int(args.dpi),
        legend_ncol=int(args.legend_ncol),
        ymin=args.ymin,
        ymax=args.ymax,
        reldiff_ylim_pct=float(args.reldiff_ylim_pct),
        reldiff_mask_min_abs_true=float(args.reldiff_mask_min_abs_true),
    )
    print(f"Saved plot: {out_fixed_masses_pdf}")


if __name__ == "__main__":
    main()
