import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP = 1000  # final output step
BASE = "/home/abarua4/proteus/test/richards/test_1"

# --- Celia benchmark from CSV (cm -> m) ---
csv = np.loadtxt(BASE + "/data_celia.csv", delimiter=",")
z_celia = csv[:, 2] / 100.0
psi_celia = csv[:, 1] / 100.0

# --- numerical runs that are present ---
runs = [
    (BASE + "/stab_0/re_vgm_sand_10m_1d.h5", "Galerkin",    dict(fmt="x--", color="blue")),
    (BASE + "/stab_2/re_vgm_sand_10m_1d.h5", "Lower Order", dict(fmt="d-",  color="tab:blue")),
    (BASE + "/FCT/re_vgm_sand_10m_1d.h5",    "FCT",         dict(fmt="o-",  color="red")),
]

plt.figure(figsize=(6, 5))
for path, label, sty in runs:
    try:
        f = h5py.File(path, "r")
    except (OSError, FileNotFoundError):
        print(f"skip (missing): {path}")
        continue
    steps = sorted(int(k.split("_t")[1]) for k in f.keys()
                   if k.startswith("pressure_head_t"))
    last = min(STEP, steps[-1])
    if last == 0:
        print(f"skip (diverged, only IC written): {label}")
        f.close()
        continue
    z = f["nodesSpatial_Domain0"][:][:, 0]
    psi = f["pressure_head_t%d" % last][:]
    order = np.argsort(z)
    plt.plot(psi[order], z[order], sty["fmt"], label=label, color=sty["color"])
    f.close()

plt.plot(psi_celia, z_celia, "s", label="Celia et al. (1990)", color="black")

plt.xlabel("Pressure Head [m]")
plt.ylabel("Depth [m]")
plt.title("1D Pressure Head vs Depth at T = 1 day")
plt.legend()
plt.grid(True)
plt.tight_layout()
out = "1d_pressure_profiles_with_celia_regen.png"
plt.savefig(out, dpi=150)
print("saved:", out)
