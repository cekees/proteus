#!/usr/bin/env python3
"""
Plot full c-field colormaps: analytical (left) and FCT (right) side-by-side.
Reads ls_CCS_so.h5 + henry_reference.npz from the current directory.
"""
from __future__ import annotations

import argparse
import re
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def latest_step(h5_path, varname="u"):
    pat = re.compile(rf"^{re.escape(varname)}_t(\d+)$")
    with h5py.File(h5_path, "r") as f:
        steps = [int(m.group(1)) for k in f.keys()
                 if (m := pat.match(k)) is not None]
    if not steps:
        raise RuntimeError(f"no {varname}_t<N> in {h5_path}")
    return max(steps)


def read_h5(h5_path, step, varname="u"):
    with h5py.File(h5_path, "r") as f:
        c = f[f"{varname}_t{step}"][:]
        nodes = f["nodesSpatial_Domain0"][:]
        elems = f["elementsSpatial_Domain0"][:]
    return c, nodes, elems


def reorder_analytical(x_ana, y_ana, c_ana, nodes):
    if (np.allclose(nodes[:, 0], x_ana) and np.allclose(nodes[:, 1], y_ana)):
        return c_ana
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([x_ana, y_ana]))
    _, idx = tree.query(nodes[:, :2])
    return c_ana[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fct_h5", default="ls_CCS_so.h5")
    p.add_argument("--npz", default="henry_reference.npz")
    p.add_argument("--var", default="u")
    p.add_argument("--step", type=int, default=None,
                   help="time step (default: last)")
    p.add_argument("--levels", type=int, default=21)
    p.add_argument("--isos", type=float, nargs="+", default=[0.1, 0.5, 0.9],
                   help="isochlor levels to overlay (default 0.1 0.5 0.9)")
    p.add_argument("--out", default="c_field_analytical_vs_FCT.png")
    p.add_argument("--clip", action="store_true",
                   help="clip analytical c to [0,1] (suppress Fourier wiggles)")
    p.add_argument("--tag", default="right_dbc/ref_1")
    args = p.parse_args()

    ref = np.load(args.npz)
    a = float(ref["a"]); b = float(ref["b"])
    x_ana, y_ana, c_ana_raw = ref["x"], ref["y"], ref["c"]
    if args.clip:
        c_ana_raw = np.clip(c_ana_raw, 0.0, 1.0)

    step = args.step if args.step is not None else latest_step(args.fct_h5, args.var)
    c_fct, nodes, elems = read_h5(args.fct_h5, step, args.var)
    c_ana = reorder_analytical(x_ana, y_ana, c_ana_raw, nodes)

    x = nodes[:, 0]; y = nodes[:, 1]
    triang = mtri.Triangulation(x, y, elems)
    levels = np.linspace(0.0, 1.0, args.levels)

    print(f"step={step}  analytical c in [{c_ana.min():.3f},{c_ana.max():.3f}]  "
          f"FCT c in [{c_fct.min():.3f},{c_fct.max():.3f}]", file=sys.stderr)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.0),
                            gridspec_kw=dict(width_ratios=[1, 1.07]))

    cf0 = axs[0].tricontourf(triang, c_ana, levels=levels, cmap="viridis", extend="both")
    if args.isos:
        axs[0].tricontour(triang, c_ana, levels=args.isos,
                          colors="white", linewidths=1.0)
    axs[0].set_title(f"semi-analytical c   (a={a:.4g}, b={b:.4g})")
    axs[0].set_aspect("equal"); axs[0].set_xlabel("x"); axs[0].set_ylabel("y")

    cf1 = axs[1].tricontourf(triang, c_fct, levels=levels, cmap="viridis", extend="both")
    if args.isos:
        axs[1].tricontour(triang, c_fct, levels=args.isos,
                          colors="white", linewidths=1.0)
    axs[1].set_title(f"FCT c   (step {step})")
    axs[1].set_aspect("equal"); axs[1].set_xlabel("x")

    cbar = fig.colorbar(cf1, ax=axs.tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("c")

    fig.suptitle(f"Henry concentration field — {args.tag}   "
                 f"(white isolines: " + ", ".join(f"{int(round(100*l))}%" for l in args.isos) + ")")
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}", file=sys.stderr)

    err = c_fct - c_ana
    L2 = np.sqrt(np.mean(err ** 2))
    Linf = np.abs(err).max()
    print(f"  ||c_FCT - c_ana||_2  = {L2:.4e}", file=sys.stderr)
    print(f"  ||c_FCT - c_ana||_inf = {Linf:.4e}", file=sys.stderr)


if __name__ == "__main__":
    main()
