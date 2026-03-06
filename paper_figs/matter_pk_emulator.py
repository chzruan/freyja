"""Compare matter P(k) emulator predictions vs data for test cosmologies.

This script mirrors the plotting style of ``MatterAlphaEmulator.compare()`` in
``freyja.emulators.pk_mm`` (top panel: spectra, bottom panel: fractional error),
but overlays multiple held-out test models (default ``imodel=60..64``) in a
single figure with distinct colors and saves a PDF.

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from cycler import cycler

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

from freyja.cosma.xi_hh import load_cosmology_wrapper, load_pkmm_data  # noqa: E402
from freyja.emulators.pk_mm import MatterPkEmulator  # noqa: E402


DEFAULT_IMODELS = range(57, 64)
DEFAULT_OUTPDF = Path("./figs/matter_pk_emulator_test.pdf")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Overlay matter power spectrum emulator predictions vs simulation data "
            "for test cosmologies in one PDF figure."
        )
    )
    p.add_argument(
        "--imodels",
        type=int,
        nargs="*",
        default=DEFAULT_IMODELS,
        help="Model IDs to compare (default: 57..63).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPDF,
        help="Output PDF path.",
    )
    p.add_argument(
        "--frac-err-ylim",
        type=float,
        default=2.8,
        help="Symmetric y-limit (percent) for fractional error panel.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=500,
        help="Figure DPI (mainly affects rasterized artists, PDF remains vector where possible).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    imodels = [int(m) for m in args.imodels]
    if len(imodels) == 0:
        raise ValueError("No imodels provided.")

    out_pdf = args.out.expanduser()
    if not out_pdf.is_absolute():
        out_pdf = (Path.cwd() / out_pdf).resolve()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    matplotlib.use("Agg")
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

    emulator = MatterPkEmulator()

    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[2.5, 1],
        wspace=0,
        hspace=0,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax = [ax0, ax1]
    ax[0].xaxis.set_ticklabels([])

    for a in ax:
        a.set_prop_cycle(custom_cycler)

    for i, imodel in enumerate(imodels):
        cosmo = load_cosmology_wrapper(imodel)
        k_nl, pk_nl_all = load_pkmm_data(imodel, return_mean=False)
        pk_nl = pk_nl_all.mean(axis=0)
        pk_nl_sem = pk_nl_all.std(axis=0) / np.sqrt(pk_nl_all.shape[0])
        k_nl = k_nl[::3]
        pk_nl = pk_nl[::3]
        pk_nl_sem = pk_nl_sem[::3]
        pk_pred = emulator.predict(cosmo, k_nl)

        mse = float(np.mean((pk_pred - pk_nl) ** 2))
        mean_frac_error = float(np.mean(np.abs((pk_pred - pk_nl) / pk_nl)))
        rmse = float(np.sqrt(np.mean((pk_pred - pk_nl) ** 2)))

        # Top panel: k * P(k), following MatterPkEmulator.compare() style.

        if i == 0:
            pp = ax[0].errorbar(
                k_nl,
                k_nl * pk_nl,
                yerr=k_nl * pk_nl_sem,
                marker="o",
                markersize=3.0,
                lw=0.0,
                elinewidth=0.7,
                alpha=0.95,
                label=r"$\mathrm{Data}$",
            )
            ax[0].plot(
                k_nl,
                k_nl * pk_pred,
                color=pp[0].get_color(),
                lw=1.6,
                alpha=0.95,
                label=r"$\mathrm{Emulator}$",
            )
        else:
            pp = ax[0].errorbar(
                k_nl,
                k_nl * pk_nl,
                yerr=k_nl * pk_nl_sem,
                marker="o",
                markersize=3.0,
                lw=0.0,
                elinewidth=0.7,
                alpha=0.95,
            )
            ax[0].plot(
                k_nl,
                k_nl * pk_pred,
                color=pp[0].get_color(),
                lw=1.6,
                alpha=0.95,
            )

        # Bottom panel: fractional error.
        frac_err = (pk_nl / pk_pred) - 1.0
        frac_err *= 100.0
        frac_err_sem = (pk_nl_sem / pk_pred) * 100.0
        ax[1].errorbar(
            k_nl,
            frac_err,
            yerr=frac_err_sem,
            color=pp[0].get_color(),
            lw=0,
            elinewidth=0.7,
            marker=".",
            markersize=5.0,
        )

    ax[0].set_xscale("log")
    ax[1].set_xscale("log")

    ax[0].set_xlim([k_nl.min(), 6.0])
    ax[1].set_xlim([k_nl.min(), 6.0])

    ax[0].set_ylabel(r"$k \times P(k)$", fontsize=20)
    ax[0].legend(fontsize=18, loc="best", frameon=False)

    ax[1].axhline(
        0.0,
        color="gray",
        linestyle="--",
        lw=0.7,
    )
    ax[1].axhline(
        -1.0,
        color="gray",
        linestyle="--",
        lw=0.7,
    )
    ax[1].axhline(
        1.0,
        color="gray",
        linestyle="--",
        lw=0.7,
    )
    ax[1].set_ylabel(r"$\varepsilon (\%)$", fontsize=20)
    ax[0].set_ylim([0, 1550])
    ylim = abs(float(args.frac_err_ylim))
    ax[1].set_ylim([-ylim, ylim])
    ax[1].set_xlabel(r"$k / (h \, \mathrm{Mpc}^{-1})$", fontsize=20)

    fig.savefig(out_pdf, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.028)
    plt.close(fig)

    print(f"Saved figure: {out_pdf}")


if __name__ == "__main__":
    main()
