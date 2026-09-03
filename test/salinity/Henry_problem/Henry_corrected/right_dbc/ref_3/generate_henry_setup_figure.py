#!/usr/bin/env python3
"""
Generate a schematic figure for the corrected Henry setup.

The figure summarizes the geometry, initial condition, and boundary
conditions used in test/salinity/Henry_problem/Henry_corrected.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


LENGTH = 2.0
HEIGHT = 1.0
Q_IN = -6.6e-5
RHO_FW = 1000.0
RHO_SW = 1025.0


def add_hatched_boundary(ax, x0, y0, x1, y1, dx, dy, n_lines):
    """Draw short hatch marks along a boundary segment."""
    for i in range(n_lines + 1):
        s = i / float(n_lines)
        x = x0 + s * (x1 - x0)
        y = y0 + s * (y1 - y0)
        ax.plot([x, x + dx], [y, y + dy], color="black", lw=0.8)


def add_inflow_arrows(ax, x, y0, y1, dx, n_arrows):
    """Draw horizontal inflow arrows on the left boundary."""
    for i in range(n_arrows):
        y = y0 + (i + 0.5) * (y1 - y0) / n_arrows
        ax.add_patch(
            FancyArrowPatch(
                (x - dx, y),
                (x, y),
                arrowstyle="-|>",
                mutation_scale=10,
                lw=0.9,
                color="black",
            )
        )


fig, ax = plt.subplots(figsize=(11, 4.8))

x0, y0 = 0.0, 0.0
x1, y1 = LENGTH, HEIGHT

# Hydrostatic pressure-head shading
ny, nx = 300, 600
y = np.linspace(y0, y1, ny)
psi_ic = (RHO_SW / RHO_FW) * (HEIGHT - y)
psi_field = np.repeat(psi_ic[:, None], nx, axis=1)
ax.imshow(
    psi_field,
    extent=(x0, x1, y0, y1),
    origin="lower",
    cmap="Blues",
    alpha=0.28,
    aspect="auto",
    zorder=0,
)

# Domain box
ax.add_patch(
    Rectangle((x0, y0), LENGTH, HEIGHT, facecolor="white", edgecolor="black", lw=1.2)
)

# Impermeable top/bottom hatching
add_hatched_boundary(ax, x0, y1, x1, y1, -0.035, 0.035, 28)
add_hatched_boundary(ax, x0, y0, x1, y0, -0.035, -0.035, 28)

# Left inflow arrows
add_inflow_arrows(ax, x0, y0, y1, 0.18, 18)

# Dimension arrows
ax.annotate(
    "",
    xy=(x0, -0.10),
    xytext=(x1, -0.10),
    arrowprops=dict(arrowstyle="<->", lw=1.0, color="black"),
)
ax.text((x0 + x1) / 2.0, -0.145, "2.0 m", ha="center", va="top", fontsize=12)

ax.annotate(
    "",
    xy=(x1 + 0.10, y0),
    xytext=(x1 + 0.10, y1),
    arrowprops=dict(arrowstyle="<->", lw=1.0, color="black"),
)
ax.text(x1 + 0.13, (y0 + y1) / 2.0, "1.0 m", ha="center", va="center", fontsize=12)

# Left-side labels
ax.text(-0.62, 0.86, "Freshwater", ha="left", va="center", fontsize=13)
ax.text(-0.72, 0.56, rf"$\mathbf{{u}}\cdot\mathbf{{n}} = {Q_IN:.1e}\ \mathrm{{m/s}}$", fontsize=13)
ax.text(-0.36, 0.34, r"$c = 0$", fontsize=13)

# Right-side labels
ax.text(2.22, 0.86, "Saltwater", ha="left", va="center", fontsize=13)
ax.text(
    2.05,
    0.73,
    rf"$\psi = \frac{{\rho_s}}{{\rho_f}}(L_y-y)"
    rf"\quad \left(\rho_s/\rho_f = {RHO_SW/RHO_FW:.3f}\right)$",
    fontsize=12,
)
ax.text(2.05, 0.50, r"$\mathbf{u}\cdot\mathbf{n} > 0:$", fontsize=12)
ax.text(2.05, 0.40, r"$\mathbf{n}\cdot(\mathbf{A}\nabla c)=0$", fontsize=12)
ax.text(2.05, 0.23, r"$\mathbf{u}\cdot\mathbf{n} < 0:$", fontsize=12)
ax.text(
    2.05,
    0.11,
    r"$c=1$ in the advective upwind flux",
    fontsize=12,
)

# Top/bottom description
ax.text(0.42, 1.08, r"Top/Bottom: no-flow for transport diffusion", fontsize=11)
ax.text(0.48, -0.23, r"Initial condition: $c(x,0)=0$", fontsize=12)
ax.text(
    0.32,
    0.08,
    r"Hydrostatic pressure-head shading:"
    "\n"
    r"$\psi(x,y,0)=\dfrac{\rho_s}{\rho_f}(L_y-y)$",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.8),
)

# Clean layout
ax.set_xlim(-0.85, 3.15)
ax.set_ylim(-0.28, 1.18)
ax.axis("off")

outdir = Path(__file__).resolve().parent
png_path = outdir / "henry_corrected_setup.png"
pdf_path = outdir / "henry_corrected_setup.pdf"

fig.tight_layout()
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print(f"Wrote {png_path}")
print(f"Wrote {pdf_path}")
