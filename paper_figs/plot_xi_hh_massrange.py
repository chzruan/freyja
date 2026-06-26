#!/usr/bin/env python3
r"""Paper-quality ξ_hh(r | >M, >M) data-vs-emulator figure (reproducible).

Reads the self-contained ``xi_hh_massrange_data.npz`` (produced once by
``build_xi_hh_data.py``) and draws the figure. Depends only on numpy +
matplotlib — no haloemu / halocat / simulation caches needed, so the figure
is reproducible anywhere.

Style follows
``freyja_codex/agent/freyja/paper_figs/bias_tinker_extrapolation.py``
(usetex + physics, ticks-in on all four sides, frameon-free legends). Shown:
the cumulative-threshold halo auto-correlation for haloes above the labelled
mass threshold, for four interior design cosmologies labelled by imodel.
Points: measured (mean ± SEM over the boxes). Solid: emulator
ξ_hh = D b(>M)² ξ_mm. Dashed: linear factorization b̄² ξ_mm (D = 1).

Run::

    python3 plot_xi_hh_massrange.py
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"font.size": 12})
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath}\usepackage{physics}")
matplotlib.rcParams.update({"xtick.top": True, "ytick.right": True,
                            "xtick.direction": "in", "ytick.direction": "in"})
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
#: curated, harmonious, colour-blind-friendly palette (deep blue, teal,
#: gold, rose) — one hue per cosmology, ordered by S_8
PALETTE = ["#1d4e89", "lightseagreen", "#edae49", "#d1495b"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.path.join(HERE,
                                                   "xi_hh_massrange_data.npz"))
    p.add_argument("--out", default=os.path.join(
        HERE, "xi_hh_massrange_data_vs_emulator.pdf"))
    args = p.parse_args(argv)

    d = np.load(args.data, allow_pickle=True)
    gravity = str(d["gravity"])
    redshift = float(d["redshift"])
    thr = float(d["threshold"])
    n_box = int(d["n_box"])
    S8, bbar = d["S8"], d["b"]
    imodels = d["imodels"]
    r, xi_d, sem_d = d["r"], d["xi_d"], d["sem_d"]
    xi_e, xi_lin = d["xi_e"], d["xi_lin"]
    n = len(S8)

    r_hi = float(np.max(r))

    fig = plt.figure(figsize=(5.0, 6.3))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.0, 1.35], hspace=0.0)
    ax = plt.subplot(gs[0])
    axr = plt.subplot(gs[1], sharex=ax)

    for i in range(n):
        c = PALETTE[i % len(PALETTE)]
        ri = r[i]
        ax.plot(ri, ri ** 2 * xi_lin[i], "--", color=c, lw=0.9, alpha=0.9,
                zorder=3)
        ax.plot(ri, ri ** 2 * xi_e[i], "-", color=c, lw=1.5, zorder=4)
        ax.errorbar(ri, ri ** 2 * xi_d[i], yerr=ri ** 2 * sem_d[i],
                    fmt="o", ms=3.6, color=c, mfc=c, mec="k", mew=0.35,
                    lw=0, elinewidth=0.7, capsize=0, zorder=5)
        ok = np.isfinite(xi_e[i])
        axr.plot(ri[ok], 100.0 * (xi_e[i][ok] / xi_d[i][ok] - 1.0), "-",
                 color=c, lw=1.5)
        axr.plot(ri[ok], 100.0 * (xi_lin[i][ok] / xi_d[i][ok] - 1.0), "--",
                 color=c, lw=0.9, alpha=0.9)

    ax.set_xscale("log")
    ax.set_ylabel(r"$r^2\,\xi_{hh}(r) / "
                  r"(h^{-1}\,\mathrm{Mpc})^2$", fontsize=14)
    # ax.text(0.045, 0.055,
    #         r"$\log_{10}[M/(h^{-1}M_\odot)] > %.2f$" % thr + "\n"
    #         r"$\mathrm{%s},\ z=%.2f$" % (gravity, redshift),
    #         transform=ax.transAxes, fontsize=10.5, va="bottom", ha="left")

    # legend 1: cosmologies (colour = one design model, imodel + its S_8)
    cosmo_handles = [Line2D([], [], color=PALETTE[i % len(PALETTE)], lw=2.2,
                            label=r"$\mathrm{imodel}=%d\ (S_8=%.3f)$"
                            % (int(imodels[i]), float(S8[i])))
                     for i in range(n)]
    leg1 = ax.legend(handles=cosmo_handles, frameon=False, fontsize=9.0,
                     loc="upper left", bbox_to_anchor=(0.015, 0.995),
                     handlelength=1.6, labelspacing=0.3)
    ax.add_artist(leg1)
    # legend 2: line-style / marker encoding (neutral grey)
    style_handles = [
        Line2D([], [], color="0.25", lw=1.5, label=r"$\mathrm{emulator}$"),
        Line2D([], [], color="0.25", lw=0.9, ls="--",
               label=r"$b^2 \xi_{\mathrm{mm}}\ (\mathrm{linear})$"),
        Line2D([], [], color="0.25", lw=0, marker="o", ms=4,
               mec="k", mew=0.35,
               label=r"$\mathrm{data}\ (N_{\mathrm{box}}=%d)$" % n_box)]
    ax.legend(handles=style_handles, frameon=False, fontsize=9.0,
              loc="upper right", handlelength=1.8, labelspacing=0.3)
    ax.tick_params(labelbottom=False)
    ymax = max((r[i] ** 2 * xi_d[i]).max() for i in range(n))
    ax.set_ylim(top=1.26 * ymax)
    # ax.grid(True, ls=":", alpha=0.35)

    axr.axhline(0, color="k", lw=0.8)
    axr.axhspan(-3, 3, color="0.88", alpha=0.7)
    axr.axvspan(60.0, r_hi * 1.05, color="lightsteelblue", alpha=0.22)
    axr.text(78, 6.6, r"$\mathrm{noise\text{-}dominated}$", fontsize=8,
             ha="center", va="center", color="0.4")
    axr.set_xscale("log")
    axr.set_ylim(-9.5, 9.5)
    axr.set_xlim(r.min() * 0.92, r_hi * 1.05)
    axr.set_xlabel(r"$r / (h^{-1}\,\mathrm{Mpc})$", fontsize=14)
    axr.set_ylabel(r"$\mathrm{emu}/\mathrm{data}-1\ [\%]$", fontsize=11.5)
    # axr.grid(True, ls=":", alpha=0.35)

    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.028)
    print(f"[plot] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
