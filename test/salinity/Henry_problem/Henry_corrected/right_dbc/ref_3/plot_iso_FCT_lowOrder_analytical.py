#!/usr/bin/env python3
"""
Overlay the X% isochlor for analytical, FCT, and low-order runs.

Reads:
    ls_CCS_so.h5            (FCT run, this directory)
    low_order/ls_CCS_so.h5  (low-order run)
    henry_reference.npz     (semi-analytical reference)

Writes one PNG per requested level (default: 10, 50).
"""
from __future__ import annotations

import argparse
import os
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
        return c_ana, nodes[:, 0], nodes[:, 1]
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([x_ana, y_ana]))
    _, idx = tree.query(nodes[:, :2])
    return c_ana[idx], nodes[:, 0], nodes[:, 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fct_h5", default="ls_CCS_so.h5")
    p.add_argument("--lo_h5", default="low_order/ls_CCS_so.h5")
    p.add_argument("--npz", default="henry_reference.npz")
    p.add_argument("--var", default="u")
    p.add_argument("--levels", type=float, nargs="+", default=[0.1, 0.5])
    p.add_argument("--tag", default="right_dbc/ref_1")
    args = p.parse_args()

    # analytical
    ref = np.load(args.npz)
    a = float(ref["a"])
    b = float(ref["b"])
    x_ana, y_ana, c_ana = ref["x"], ref["y"], ref["c"]
    print(f"analytical: a={a}, b={b}, c in [{c_ana.min():.3f}, {c_ana.max():.3f}]",
          file=sys.stderr)

    # FCT
    fct_step = latest_step(args.fct_h5, args.var)
    c_fct, nodes_fct, elems_fct = read_h5(args.fct_h5, fct_step, args.var)
    print(f"FCT:        step {fct_step}, c in [{c_fct.min():.3f}, {c_fct.max():.3f}]",
          file=sys.stderr)

    # low-order
    lo_step = latest_step(args.lo_h5, args.var)
    c_lo, nodes_lo, elems_lo = read_h5(args.lo_h5, lo_step, args.var)
    print(f"low-order:  step {lo_step}, c in [{c_lo.min():.3f}, {c_lo.max():.3f}]",
          file=sys.stderr)

    # mesh sanity: FCT and low-order should share the same mesh
    if not (np.allclose(nodes_fct, nodes_lo) and np.array_equal(elems_fct, elems_lo)):
        print("WARNING: FCT and low-order meshes differ — using FCT mesh and "
              "remapping low-order via nearest-neighbor", file=sys.stderr)
        from scipy.spatial import cKDTree
        tree = cKDTree(nodes_lo[:, :2])
        _, idx = tree.query(nodes_fct[:, :2])
        c_lo = c_lo[idx]

    c_ana_n, x_n, y_n = reorder_analytical(x_ana, y_ana, c_ana, nodes_fct)

    triang = mtri.Triangulation(x_n, y_n, elems_fct)

    for lvl in args.levels:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.tricontour(triang, c_ana_n, levels=[lvl],
                      colors="C0", linewidths=2.4,
                      linestyles="-")
        ax.tricontour(triang, c_fct, levels=[lvl],
                      colors="C3", linewidths=2.0,
                      linestyles="--")
        ax.tricontour(triang, c_lo, levels=[lvl],
                      colors="C2", linewidths=2.0,
                      linestyles=":")

        from matplotlib.lines import Line2D
        legend = [
            Line2D([0], [0], color="C0", lw=2.4, ls="-",
                   label=f"Analytical (a={a:.4g}, b={b:.4g})"),
            Line2D([0], [0], color="C3", lw=2.0, ls="--",
                   label=f"FCT  (step {fct_step})"),
            Line2D([0], [0], color="C2", lw=2.0, ls=":",
                   label=f"Low-order  (step {lo_step})"),
        ]
        ax.legend(handles=legend, loc="lower right", framealpha=0.95)

        ax.set_xlim(x_n.min(), x_n.max())
        ax.set_ylim(y_n.min(), y_n.max())
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{int(round(100*lvl))}% isochlor: analytical vs FCT vs low-order  ({args.tag})")
        ax.grid(alpha=0.3)

        out = f"iso_{int(round(100*lvl))}_FCT_lowOrder_analytical.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
