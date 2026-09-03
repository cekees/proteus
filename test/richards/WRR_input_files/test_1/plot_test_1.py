#!/usr/bin/env python
"""test_1 figure -- every scheme against Celia et al. (1990) Fig. 6(b).

Reads <scheme>/re_vgm_sand_10m_1d.h5 written by run_test_1.sh and saves
test_1_schemes_vs_celia.png next to this file, plus a depth-normalized L2 head
error for each scheme.

    ./run_test_1.sh          # runs the schemes, then calls this
    python plot_test_1.py    # re-draw from archives already on disk
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
OUT = HERE / "test_1_schemes_vs_celia.png"

FRAME = 1000                    # T = 1 d over nDTout=1000 outputs -> 24 h
SCHEMES = os.environ.get("SCHEMES", "stab_0 stab_2 FCT").split()

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
# The deck runs nn=41 on a 1 m column, i.e. dz = 2.5 cm -- exactly the grid of
# Celia's Fig. 6(b).  Score against the 85-point digitization of that figure,
# not against data_celia.csv, which is a 12-point hand trace of the same bundle.
# The three digitized dt's are indistinguishable, so the 2.4 min member --
# closest to this deck's dt = 1.44 min -- stands for the bundle.
celia = np.loadtxt(HERE / "data_celia_dt2.4min.csv", delimiter=",", comments="#")
CELIA_Z, CELIA_PSI = celia[:, 2] / 100.0, celia[:, 1] / 100.0
_order = np.argsort(CELIA_Z)
CELIA_Z, CELIA_PSI = CELIA_Z[_order], CELIA_PSI[_order]

trace = np.loadtxt(HERE / "data_celia.csv", delimiter=",")
TRACE_Z, TRACE_PSI = trace[:, 2] / 100.0, trace[:, 1] / 100.0


# ------------------------------------------------------------------- plotting
plt.rcParams.update({"font.size": 11, "axes.labelsize": 13, "legend.fontsize": 9})
fig, ax = plt.subplots(figsize=(6.0, 6.2))

report = []
for scheme in SCHEMES:
    color, fmt, label = STYLE[scheme]
    x, psi = profile(scheme, FRAME)
    ax.plot(psi, x, fmt, color=color, label=label, markersize=4, linewidth=1.6)
    report.append((label,
                   depth_rms(CELIA_Z, np.interp(CELIA_Z, x, psi), CELIA_PSI)))

ax.plot(TRACE_PSI, TRACE_Z, "s", color="0.65", markersize=6,
        markerfacecolor="none", label="Celia 12-pt hand trace")
ax.plot(CELIA_PSI, CELIA_Z, "-", color="black", linewidth=2.0,
        label="Celia (1990) Fig. 6(b), dz=2.5 cm")
ax.axvline(-10.0, color="0.6", lw=0.8, ls=":")
ax.set_title("test_1 -- Celia Fig. 6(b), t = 24 h  (dz = 2.5 cm)")
ax.set_xlabel("Pressure head $\\psi$ [m]")
ax.set_ylabel("Elevation z [m]")
ax.grid(True, alpha=0.3)
ax.legend(loc="best", framealpha=0.95)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved:", OUT)

print(f"\n{'scheme':<30s}{'depth-rms psi [m]':>20s}")
print("-" * 50)
for label, rms in report:
    print(f"{label:<30s}{rms:>20.4e}")
