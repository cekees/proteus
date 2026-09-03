import os
import glob
import numpy as np
import h5py
import xml.etree.ElementTree as ET

# -----------------------------
# Analytical solution settings (smooth Gaussian pulse)
# -----------------------------
C0 = 1.0
v = 0.3
D = 0.007
x0 = 0.5
sigma0 = 0.08

# Pick datasets by index u_t<idx>
TIME_INDICES = [0, 100, 200]

# If empty, script auto-discovers tadr_level_*.h5 in this folder
H5_FILES = []

OUT_DIR = "plots_tadr_analytical"
N_ANALYTICAL_POINTS = 1200


def c_gaussian_ic(x, t, c0=C0, vel=v, diff=D, center=x0, sigma=sigma0):
    x = np.asarray(x)
    var_t = sigma**2 + 2.0*diff*t
    amp = c0*np.sqrt((sigma**2)/var_t)
    dx = x - (center + vel*t)
    return amp*np.exp(-0.5*(dx*dx)/var_t)


def get_time_from_h5(f, t_idx):
    mesh_key = f"Mesh_Spatial_Domain_{t_idx}"
    if mesh_key not in f:
        return None
    root = ET.fromstring(f[mesh_key][:])
    node = root.find("Time")
    if node is None:
        return None
    return float(node.attrib["Value"])


def available_u_indices(f):
    idx = []
    for k in f.keys():
        if k.startswith("u_t"):
            try:
                idx.append(int(k[3:]))
            except ValueError:
                pass
    return sorted(idx)


def infer_nnx(nodes):
    x = np.unique(np.round(nodes[:, 0], 14))
    return int(x.size)


def load_numeric_line(h5_path, t_idx):
    with h5py.File(h5_path, "r") as f:
        key = f"u_t{t_idx}"
        if key not in f:
            return None
        nodes = f["nodesSpatial_Domain0"][:]
        u = f[key][:]
        t_val = get_time_from_h5(f, t_idx)

    x = nodes[:, 0]
    order = np.argsort(x)
    x = x[order]
    u = u[order]
    nnx = infer_nnx(nodes)
    return x, u, t_val, nnx


def pick_h5_files():
    if H5_FILES:
        return [p for p in H5_FILES if os.path.exists(p)]
    files = sorted(glob.glob("tadr_level_*.h5"))
    return files


def main():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it in your environment and rerun "
            "`python Plot_TADR_analytical.py`."
        ) from e

    os.makedirs(OUT_DIR, exist_ok=True)

    h5_files = pick_h5_files()
    if not h5_files:
        raise FileNotFoundError(
            "No .h5 files found. Set H5_FILES or run in Step_analytical_solution directory."
        )

    # Keep only files that have at least one requested time index
    valid_files = []
    for p in h5_files:
        with h5py.File(p, "r") as f:
            idx_set = set(available_u_indices(f))
        if any(ti in idx_set for ti in TIME_INDICES):
            valid_files.append(p)

    if not valid_files:
        raise ValueError("None of the requested TIME_INDICES exist in discovered h5 files.")

    for t_idx in TIME_INDICES:
        curves = []
        t_plot = None

        for p in valid_files:
            loaded = load_numeric_line(p, t_idx)
            if loaded is None:
                continue
            x, u, t_val, nnx = loaded
            curves.append((p, x, u, nnx))
            if t_plot is None and t_val is not None:
                t_plot = t_val

        if not curves:
            continue

        x_min = min(c[1].min() for c in curves)
        x_max = max(c[1].max() for c in curves)
        x_ana = np.linspace(x_min, x_max, N_ANALYTICAL_POINTS)
        if t_plot is None:
            t_plot = 0.0
        u_ana = c_gaussian_ic(x_ana, t_plot)

        plt.figure(figsize=(9, 5))
        plt.plot(x_ana, u_ana, 'k-', lw=2.2, label=f"Analytical (t={t_plot:.4f})")

        # plot coarse to fine by nnx
        curves.sort(key=lambda c: c[3])
        for p, x, u, nnx in curves:
            label = f"{os.path.basename(p)} (nnx={nnx})"
            plt.plot(x, u, marker='o', ms=2.5, lw=1.1, label=label)

        plt.xlabel("x")
        plt.ylabel("u")
        plt.title(f"TADR vs Analytical, index={t_idx}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()

        out_png = os.path.join(OUT_DIR, f"TADR_vs_Analytical_t{t_idx}.png")
        plt.savefig(out_png, dpi=180)
        plt.close()
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
