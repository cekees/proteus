"""Three-panel evolution figure for the FCT run of the HYDRUS-1D sand column:

  a) pressure head psi(z)          b) hydraulic conductivity K(z)          c) moisture content theta(z)

psi is read from re_vgm_sand_10m_1d.h5; K and theta are the Mualem-van Genuchten
closures evaluated nodally from psi with the parameters of re_vgm_sand_10m_1d_p.py.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

here = Path(__file__).resolve().parent

# ---------------------------------------------------------------- parameters
# (mirrors re_vgm_sand_10m_1d_p.py)
T_HOURS = 48.0        # T = 48.0/24.0 d
DEPTH   = 20.0        # L[0]; HYDRUS z = x[0] - DEPTH

thetaS  = 0.43
thetaR  = 0.045
alpha   = 14.5        # 1/m
n_vg    = 2.68
m_vg    = 1.0 - 1.0/n_vg
Ks_mps  = 0.297/3600.0   # 0.297 m/h -> m/s  (= 7.128 m/d)

# snapshots to draw (hours).  T = 48 h, so 90 h is not available in this run.
TIMES  = [0.0, 5.0, 11.0, 48.0]
#same palette family as test_1/plot_celia_compare.py: plain saturated named colours
COLORS = ["black", "blue", "green", "red"]
STYLES = ["-", "-", "-", "-"]
LW     = 5.0

plt.rcParams.update({
    "font.size":       22,
    "axes.labelsize":  24,
    "axes.titlesize":  24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})


def vgm(psi):
    """Mualem-van Genuchten: effective saturation, moisture content, conductivity."""
    psi = np.asarray(psi, dtype=float)
    Se = np.where(psi < 0.0,
                  (1.0 + (alpha*np.abs(psi))**n_vg)**(-m_vg),
                  1.0)
    Se = np.clip(Se, 1.0e-12, 1.0)
    theta = thetaR + Se*(thetaS - thetaR)
    K = Ks_mps*np.sqrt(Se)*(1.0 - (1.0 - Se**(1.0/m_vg))**m_vg)**2
    return Se, theta, K


# --------------------------------------------------------------- proteus data
h5 = h5py.File(here/"re_vgm_sand_10m_1d.h5", "r")
ndtout = sum(1 for k in h5.keys() if k.startswith("pressure_head_t")) - 1
x = h5["nodesSpatial_Domain0"][:, 0]
order = np.argsort(x)
z = x[order] - DEPTH                       # 0 at the surface, -20 m at the bottom
print(f"{ndtout} output steps over {T_HOURS:g} h ({T_HOURS/ndtout*60:.2f} min apart), "
      f"{len(z)} nodes, dz = {abs(z[1]-z[0]):.3f} m")


def profile(t_hours):
    """psi(z) at t_hours, linearly interpolated between the bracketing outputs."""
    s = t_hours/T_HOURS*ndtout
    i0 = int(np.floor(s))
    i1 = min(i0 + 1, ndtout)
    w = s - i0
    p0 = h5[f"pressure_head_t{i0}"][:][order]
    p1 = h5[f"pressure_head_t{i1}"][:][order]
    return (1.0 - w)*p0 + w*p1


# ------------------------------------------------------------------- plotting
fig, axes = plt.subplots(1, 3, figsize=(19.0, 6.6), sharey=True)
ax_psi, ax_K, ax_th = axes

for t_hours, color, style in zip(TIMES, COLORS, STYLES):
    psi = profile(t_hours)
    _, theta, K = vgm(psi)
    label = f"t = {t_hours:g} h"
    for ax, val in ((ax_psi, psi), (ax_K, K), (ax_th, theta)):
        ax.plot(val, z, color=color, linestyle=style, linewidth=LW, alpha=1.0,
                label=label)

ax_psi.set_xlabel("pressure head [m]")
ax_psi.set_ylabel("elevation z [m]")
ax_psi.set_title("head at different time steps")

ax_K.set_xscale("log")
ax_K.set_xlabel("hydraulic conductivity [m/s]")
ax_K.set_title("K at different time steps")
#dry sand at psi = -20 m has K ~ 1e-20 m/s, so the axis has to span the full range
ax_K.set_xlim(1.0e-20, 1.0e-3)
ax_K.set_xticks([1e-20, 1e-16, 1e-12, 1e-8, 1e-4])

ax_th.set_xlabel("moisture content [-]")
ax_th.set_title(r"$\theta$ at different time steps")
ax_th.set_xlim(0.0, thetaS*1.02)

for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-DEPTH, 0.0)
    ax.tick_params(labelsize=16, width=1.2, length=6)
ax_psi.legend(fontsize=17, loc="lower left", handlelength=2.6,
              labelspacing=0.5, borderpad=0.7, framealpha=0.95)

fig.tight_layout()
out = here/"psi_K_theta_evolution.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved to: {out}")

# ------------------------------------------------------ front + storage table
print("\n t [h]   wet front z [m]   theta_top   K_top [m/s]   storage int(theta)dz [m]")
for t_hours in TIMES:
    psi = profile(t_hours)
    _, theta, K = vgm(psi)
    #deepest z of the wetted band that is contiguous with the surface (psi > -1 m)
    wet_down = (psi > -1.0)[::-1]           # index 0 = surface
    k = np.argmin(wet_down) if not wet_down.all() else len(wet_down)
    zf = z[::-1][k-1] if k else np.nan      # nan = surface itself is dry
    storage = np.trapezoid(theta, z)        # z ascends, so this is positive
    print(f"{t_hours:6.1f}  {zf:15.2f}  {theta[-1]:10.4f}  {K[-1]:12.3e}  {storage:22.4f}")
