#!/usr/bin/env python3
r"""Build the self-contained data file for the ξ_hh(r | >M, >M) figure.

Runs the haloemu emulators and loads the measured cumulative-threshold
correlation cube once, then writes everything the plot needs to a single
``xi_hh_massrange_data.npz`` so the figure can be regenerated with no
dependence on haloemu / halocat / the simulation caches.

This is the *heavy* step (needs the cosemu environment + COSMA data). Run
it once::

    micromamba run -n cosemu python3 build_xi_hh_data.py

then use ``plot_xi_hh_massrange.py`` (numpy + matplotlib only) to draw.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # noqa: E402

import numpy as np  # noqa: E402

MASS_THR = 12.5          # requested threshold (snapped to the measured grid)
HERE = os.path.dirname(os.path.abspath(__file__))


def load_threshold_xi(gravity, redshift, imodel, iboxes, want_thr):
    """Measured cumulative ξ_hh(r | >M, >M): mean ± SEM over boxes, from the
    xi_AB_thresh cube. Pure load on the pinned disk grid (never re-measures).
    Returns (r, thr_used, xi, sem)."""
    from haloemu.core.grids import discover_xi_AB_grid, aligned_r_edges
    from halocat import XiABLoader

    thr_disk, r_edges = discover_xi_AB_grid(gravity, redshift, imodel,
                                            iboxes[0])
    ti = int(np.argmin(np.abs(thr_disk - want_thr)))
    stack, r = [], None
    with aligned_r_edges(r_edges):
        loader = XiABLoader(thresholds=thr_disk)
        for ib in iboxes:
            if not loader.exists(gravity, redshift, imodel, ib):
                continue
            rec = loader.get(gravity, redshift, imodel, ib)
            r = rec.r if r is None else r
            stack.append(rec.xi_AB[:, ti, ti])
    a = np.array(stack)
    n = a.shape[0]
    return (r, float(thr_disk[ti]), np.nanmean(a, axis=0),
            np.nanstd(a, axis=0, ddof=1) / np.sqrt(n))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gravity", default="LCDM")
    p.add_argument("--redshift", type=float, default=0.25)
    p.add_argument("--imodels", default="55,56,57,58")
    p.add_argument("--iboxes", default="1-5")
    p.add_argument("--r-lo", type=float, default=2.0)
    p.add_argument("--r-hi", type=float, default=102.0)
    p.add_argument("--out", default=os.path.join(HERE,
                                                 "xi_hh_massrange_data.npz"))
    args = p.parse_args(argv)

    from haloemu.core.util import parse_int_list
    from haloemu.registry import Registry
    from halocat.cosmology import get_cosmology

    iboxes = parse_int_list(args.iboxes)
    imodels = parse_int_list(args.imodels)

    reg = Registry()
    deps = {n: reg.load(n, args.gravity, args.redshift)
            for n in ("b_cum", "xi_mm", "xi_hh_smallr")}
    tk = deps["b_cum"].theta_keys
    r_mm = np.asarray(deps["xi_mm"].coord)

    rows = []
    thr_used = None
    for im in imodels:
        cosmo = get_cosmology(args.gravity, im)
        theta = np.array([cosmo[k] for k in tk])
        r_meas, thr, xi_d, sem_d = load_threshold_xi(
            args.gravity, args.redshift, im, iboxes, MASS_THR)
        thr_used = thr
        sel = ((r_meas >= args.r_lo) & (r_meas <= args.r_hi)
               & (r_meas >= r_mm[0]) & (r_meas <= r_mm[-1]))
        r_sel = r_meas[sel]
        ximm = np.interp(np.log(r_sel), np.log(r_mm),
                         r_mm ** 2 * deps["xi_mm"].predict(theta)[0]) / r_sel ** 2
        be = float(np.ravel(deps["b_cum"].predict_mass(theta, thr))[0])
        D = deps["xi_hh_smallr"].interp_D(theta, np.array([thr]),
                                          np.array([thr]), r_sel)[0, 0]
        xi_lin = be * be * ximm           # linear factorization (D = 1)
        xi_e = D * xi_lin                  # trained small-r departure
        sub60 = r_sel < 60.0
        rms60 = float(np.sqrt(np.nanmean(
            (xi_e[sub60] / xi_d[sel][sub60] - 1.0) ** 2)))
        rows.append({"imodel": im, "S8": float(cosmo["S_8"]), "b": be,
                     "r": r_sel, "xi_d": xi_d[sel], "sem_d": sem_d[sel],
                     "xi_e": xi_e, "xi_lin": xi_lin})
        print(f"[build] imodel {im:2d}  S8={cosmo['S_8']:.3f}  b(>{thr})="
              f"{be:.3f}  emu/data-1 RMS(r<60)={100 * rms60:.2f}%")

    rows.sort(key=lambda d: d["imodel"])           # plotting order (by imodel)
    np.savez_compressed(
        args.out,
        gravity=args.gravity, redshift=args.redshift,
        threshold=float(thr_used), mass_thr_requested=MASS_THR,
        imodels=np.array([d["imodel"] for d in rows]),
        iboxes=np.array(iboxes), n_box=len(iboxes),
        S8=np.array([d["S8"] for d in rows]),
        b=np.array([d["b"] for d in rows]),
        r=np.stack([d["r"] for d in rows]),
        xi_d=np.stack([d["xi_d"] for d in rows]),
        sem_d=np.stack([d["sem_d"] for d in rows]),
        xi_e=np.stack([d["xi_e"] for d in rows]),
        xi_lin=np.stack([d["xi_lin"] for d in rows]),
        provenance=("xi_hh(r|>M,>M) cumulative-threshold auto-correlation; "
                    "emulator = D*b(>M)^2*xi_mm (haloemu b_cum/xi_hh_smallr/"
                    "xi_mm); data = xi_AB_thresh cube, mean+-SEM over boxes"),
    )
    print(f"[build] wrote {args.out}  ({len(rows)} cosmologies)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
