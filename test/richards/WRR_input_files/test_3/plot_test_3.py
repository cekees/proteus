#!/usr/bin/env python
"""test_3 figure -- every scheme against Szymkiewicz (2009) Fig. 6.

Reads <scheme>/re_vgm_sand_10m_1d.h5 written by run_test_3.sh and saves
test_3_schemes_vs_szymkiewicz.png next to this file, plus a depth-normalized L2
head error per scheme.

    ./run_test_3.sh          # runs the schemes, then calls this
    python plot_test_3.py    # re-draw from archives already on disk
"""

import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ARCHIVE = "re_vgm_sand_10m_1d.h5"
OUT = HERE / "test_3_schemes_vs_szymkiewicz.png"

FRAME = 1000                    # T = 0.125 d over nDTout=1000 outputs -> 3 h
SCHEMES = os.environ.get("SCHEMES", "stab_2 FCT").split()

STYLE = {
    "stab_0": ("tab:blue", "x--", "Galerkin (STAB=0)"),
    "stab_2": ("tab:green", "d-.", "Stabilized (STAB=2)"),
    "FCT": ("tab:red", "o-", "FCT (STAB=2 + limiter)"),
}


def profile(scheme, frame):
    """(node x, pressure head) for one scheme at one archive frame."""
    path = HERE / scheme / ARCHIVE
    if not path.exists():
        path = HERE / scheme / ARCHIVE.replace(".h5", "global.h5")
    with h5py.File(path, "r") as f:
        field = f"pressure_head_t{frame}"
        if field not in f:
            raise KeyError(f"{field} missing from {path}")
        psi = f[field][:]
        x = f["nodesSpatial_Domain0"][:, 0]
    order = np.argsort(x)
    return x[order], psi[order]


def depth_rms(z_ref, psi_model, psi_ref):
    """Depth-normalized L2 head error."""
    span = z_ref[-1] - z_ref[0]
    return np.sqrt(np.trapezoid((psi_model - psi_ref) ** 2, z_ref) / span)


# ------------------------------------------------------------- reference data
# The deck runs nn=11 on a 5 m column, i.e. dz = 50 cm.  Score against the
# paper's own dz = 50 cm curve (K_NEW, eq. 11 -- 11 points, one per node), not
# against the converged K_INT reference, which is a dz = 0.05 cm solution: on a
# 10-cell mesh the front carries ~0.4 m of pure spatial error either way, so the
# converged curve measures the grid, not the scheme.  It is drawn, unscored, to
# show where both are heading.
szym = np.loadtxt(HERE / "data_szymkiewicz_fig6_knew.csv", delimiter=",", comments="#")
SZYM_DEPTH, SZYM_PSI = szym[:, 0] / 100.0, szym[:, 1] / 100.0

szym_ref = np.loadtxt(HERE / "data_szymkiewicz_fig6_reference.csv",
                      delimiter=",", comments="#")
SZYM_REF_DEPTH, SZYM_REF_PSI = szym_ref[:, 0] / 100.0, szym_ref[:, 1] / 100.0


# ------------------------------------------------------------------- plotting
plt.rcParams.update({"font.size": 11, "axes.labelsize": 13, "legend.fontsize": 9})
fig, ax = plt.subplots(figsize=(6.0, 6.2))

report = []
for scheme in SCHEMES:
    color, fmt, label = STYLE[scheme]
    x, psi = profile(scheme, FRAME)
    depth = 5.0 - x
    ax.plot(psi, depth, fmt, color=color, label=label, markersize=5, linewidth=1.6)
    report.append((label, depth_rms(
        SZYM_DEPTH, np.interp(SZYM_DEPTH, depth[::-1], psi[::-1]), SZYM_PSI)))

ax.plot(SZYM_REF_PSI, SZYM_REF_DEPTH, "-", color="0.65", linewidth=1.8,
        label="Szymkiewicz $K_{INT}$ ref. (dz=0.05 cm)")
ax.plot(SZYM_PSI, SZYM_DEPTH, "s--", color="black", markersize=6,
        markerfacecolor="none", label="Szymkiewicz $K_{NEW}$ (dz=50 cm)")
ax.axvline(-7.5, color="0.6", lw=0.8, ls=":")
ax.axvline(-0.075, color="0.6", lw=0.8, ls=":")
ax.set_title("test_3 -- Szymkiewicz Fig. 6, t = 3 h  (dz = 50 cm)")
ax.set_xlabel("Pressure head $\\psi$ [m]")
ax.set_ylabel("Depth [m]")
ax.set_ylim(5.0, 0.0)
ax.grid(True, alpha=0.3)
ax.legend(loc="best", framealpha=0.95)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved:", OUT)

print(f"\n{'scheme':<30s}{'depth-rms psi [m]':>20s}")
print("-" * 50)
for label, rms in report:
    print(f"{label:<30s}{rms:>20.4e}")
