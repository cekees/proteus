#!/usr/bin/env python
"""test_2 figure -- every scheme against HYDRUS-1D on the 20 m column.

Reads <scheme>/re_vgm_sand_10m_1d.h5 written by run_test_2.sh and saves
test_2_schemes_vs_hydrus.png next to this file, plus a depth-normalized L2 head
error per scheme and snapshot.

    ./run_test_2.sh          # runs the schemes, then calls this
    python plot_test_2.py    # re-draw from archives already on disk
"""

import csv
import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ARCHIVE = "re_vgm_sand_10m_1d.h5"
OUT = HERE / "test_2_schemes_vs_hydrus.png"

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
with (HERE / "data_hydrus.csv").open(newline="", encoding="utf-8-sig") as stream:
    rows = list(csv.reader(stream))
block = []
for row in rows[2:]:
    values = []
    for value in row[13:23]:
        try:
            values.append(float(value))
        except ValueError:
            values.append(np.nan)
    block.append(values)
block = np.asarray(block)

# (hours, archive frame, psi column, z column).  T = 2 d over 1000 outputs, so
# frame 104 is 4.99 h and frame 229 is 10.99 h -- 30 s shy of the HYDRUS
# snapshots, which is far below the plotted line width.
SNAPS = []
for hours, frame, pcol, zcol in ((5.0, 104, 2, 3), (11.0, 229, 4, 5), (48.0, 1000, 6, 7)):
    valid = np.isfinite(block[:, pcol]) & np.isfinite(block[:, zcol])
    z = block[valid, zcol]
    order = np.argsort(z)
    SNAPS.append((hours, frame, z[order], block[valid, pcol][order]))


# ------------------------------------------------------------------- plotting
plt.rcParams.update({"font.size": 11, "axes.labelsize": 13, "legend.fontsize": 9})
fig, ax = plt.subplots(figsize=(6.0, 6.2))

report = []
for hours, frame, z_ref, psi_ref in SNAPS:
    for scheme in SCHEMES:
        color, fmt, label = STYLE[scheme]
        x, psi = profile(scheme, frame)
        z = x - 20.0
        ax.plot(psi, z, fmt, color=color, markersize=3, linewidth=1.6,
                markevery=8, label=label if hours == 5.0 else None)
        report.append(("{0:g} h".format(hours), label,
                       depth_rms(z_ref, np.interp(z_ref, z, psi), psi_ref)))
    ax.plot(psi_ref, z_ref, "-", color="black", linewidth=2.0,
            label="HYDRUS-1D" if hours == 5.0 else None)
    # Label each snapshot at its own front, not at the top: all three share the
    # ponded top node and the labels would land on top of each other there.
    ax.annotate("{0:g} h".format(hours), xy=(0.0, z_ref[np.argmin(psi_ref)]),
                xytext=(-46, 10), textcoords="offset points", fontsize=11,
                bbox=dict(boxstyle="square,pad=0.25", facecolor="white",
                          edgecolor="0.5", alpha=0.9))

ax.set_title("test_2 -- HYDRUS-1D, t = 5 / 11 / 48 h")
ax.set_xlabel("Pressure head $\\psi$ [m]")
ax.set_ylabel("Elevation z [m]")
ax.set_ylim(-20.0, 0.0)
ax.grid(True, alpha=0.3)
ax.legend(loc="best", framealpha=0.95)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved:", OUT)

print(f"\n{'snapshot':<10s}{'scheme':<30s}{'depth-rms psi [m]':>20s}")
print("-" * 60)
for when, label, rms in report:
    print(f"{when:<10s}{label:<30s}{rms:>20.4e}")
