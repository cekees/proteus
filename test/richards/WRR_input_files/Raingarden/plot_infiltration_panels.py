"""FCT raingarden (VGM, nnx=41): pressure-head fields at four output times.

Four panels a)-d) in the style of the reference infiltration figure: filled
contours of psi on the unstructured mesh, one shared colour scale so the panels
are directly comparable, and the fine-over-coarse material interface drawn on top.
"""
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


BASE = Path(__file__).resolve().parent
SCHEME = os.environ.get("SCHEME", "FCT")     # which run directory to draw
H5 = BASE / SCHEME / "Raingarden.h5"
XMF = BASE / SCHEME / "Raingarden.xmf"
OUT = BASE / "fct_nnx41_pressure_head_panels.png"
FIELD = "pressure_head"
CMAP = "jet"          # matches the reference figure; see note in the write-up
NLEV = 64
TARGET_TIMES = [0.1, 0.36, 0.5]   # days
NCOL = 3              # panels per row; None -> 1 column for a single time, else 2
# The IC is psi = -z, so the field spans [-5, 0.2] m while by t = 0.75 d three
# quarters of the domain sits in [-0.1, 0.2].  A linear scale over the full range
# paints every wetted node the same red; asinh keeps the full range on one shared
# scale but resolves the near-saturation band where the front actually lives.
LINEAR_WIDTH = 0.25   # m; set to None for a plain linear scale
CB_TICKS = [-5.0, -3.0, -2.0, -1.0, -0.5, -0.2, -0.1, 0.0, 0.2]


def load_times(xmf_path):
    root = ET.parse(xmf_path).getroot()
    out = {}
    for grid in root.findall('.//Grid[@GridType="Uniform"]'):
        el = grid.find("Time")
        if el is None:
            continue
        try:
            out[int(el.attrib["Name"])] = float(el.attrib["Value"])
        except (KeyError, ValueError):
            continue
    return out


def material_edges(nodes, elements, mat):
    """Interior edges separating two different element material tags."""
    owner = {}
    for e, t in enumerate(elements):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            owner.setdefault((a, b) if a < b else (b, a), []).append(e)
    return [(nodes[a, :2], nodes[b, :2]) for (a, b), o in owner.items()
            if len(o) == 2 and mat[o[0]] != mat[o[1]]]


def main():
    h5 = h5py.File(H5, "r")
    times = load_times(XMF)
    steps_all = sorted(int(k[len(FIELD) + 2:]) for k in h5 if k.startswith(f"{FIELD}_t"))
    steps = [min(steps_all, key=lambda s: abs(times.get(s, np.inf) - tt))
             for tt in TARGET_TIMES]

    nodes = h5["nodesSpatial_Domain0"][:]
    elements = h5["elementsSpatial_Domain0"][:]
    mat = h5["elementMaterialTypes_t0"][:]
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    iface = material_edges(nodes, elements, mat)

    fields = [h5[f"{FIELD}_t{s}"][:] for s in steps]
    vmin = min(float(u.min()) for u in fields)
    vmax = max(float(u.max()) for u in fields)
    if LINEAR_WIDTH is None:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        levels = np.linspace(vmin, vmax, NLEV + 1)
    else:
        norm = matplotlib.colors.AsinhNorm(linear_width=LINEAR_WIDTH,
                                           vmin=vmin, vmax=vmax)
        levels = np.asarray(norm.inverse(np.linspace(0.0, 1.0, NLEV + 1)))
        levels[0], levels[-1] = vmin, vmax

    x_lim = (float(nodes[:, 0].min()), float(nodes[:, 0].max()))
    z_lim = (float(nodes[:, 1].min()), float(nodes[:, 1].max()))
    panel_w = 2.7
    panel_h = panel_w * (z_lim[1] - z_lim[0]) / (x_lim[1] - x_lim[0])
    ncol = NCOL if NCOL else (1 if len(steps) == 1 else 2)
    nrow = int(np.ceil(len(steps) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(ncol * panel_w + 2.2, nrow * panel_h + 1.1),
        sharex=True, sharey=True, constrained_layout=True, squeeze=False)

    for k, (ax, s, u) in enumerate(zip(axes.ravel(), steps, fields)):
        cf = ax.tricontourf(tri, u, levels=levels, cmap=CMAP, norm=norm,
                            extend="neither")
        # kill the hairline seams between bands (ContourSet is a Collection on
        # matplotlib >= 3.8; .collections was removed in 3.10)
        try:
            cf.set_edgecolor("face")
        except AttributeError:
            for c in cf.collections:
                c.set_edgecolor("face")
        for p0, p1 in iface:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="0.25", lw=0.8, zorder=3)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*z_lim)
        ax.set_aspect("equal")
        ax.set_title(f"{'abcdefgh'[k]}) t = {times[s]:.3f} d", fontsize=11)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for ax in axes.ravel()[len(steps):]:   # unused cells in a ragged grid
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("x [m]")
    for ax in axes[:, 0]:
        ax.set_ylabel("z [m]")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.02,
                      ticks=[t for t in CB_TICKS if vmin <= t <= vmax])
    cb.ax.set_yticklabels([f"{t:g}" for t in CB_TICKS if vmin <= t <= vmax])
    cb.set_label(r"pressure head $\psi$ [m]"
                 + ("" if LINEAR_WIDTH is None else "  (asinh scale)"))
    # No figure title: the caption carries it.  The run is identified in the
    # stdout report below instead.
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(f"{SCHEME}: {nodes.shape[0]} nodes / {elements.shape[0]} triangles")
    print(f"wrote {OUT}")
    print(f"shared colour range: [{vmin:.4f}, {vmax:.4f}] m")
    for k, (s, u) in enumerate(zip(steps, fields)):
        print(f"  {'abcd'[k]}) step {s:4d}  t = {times[s]:.4f} d   "
              f"psi in [{u.min():.4f}, {u.max():.4f}]")
    h5.close()


if __name__ == "__main__":
    main()
