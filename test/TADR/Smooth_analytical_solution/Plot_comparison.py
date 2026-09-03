import argparse
import os
import xml.etree.ElementTree as ET

import h5py
import numpy as np


def c_gaussian_ic(x, t, c0=1.0, vel=0.3, diff=0.007, center=0.5, sigma=0.08):
    x = np.asarray(x)
    var_t = sigma**2 + 2.0 * diff * t
    amp = c0 * np.sqrt((sigma**2) / var_t)
    dx = x - (center + vel * t)
    return amp * np.exp(-0.5 * (dx * dx) / var_t)


def get_time_from_h5(f, t_idx):
    mesh_key = f"Mesh_Spatial_Domain_{t_idx}"
    if mesh_key not in f:
        return None
    root = ET.fromstring(f[mesh_key][:])
    node = root.find("Time")
    if node is None:
        return None
    return float(node.attrib["Value"])


def load_line_from_h5(h5_path, t_idx):
    if not os.path.exists(h5_path):
        return None
    with h5py.File(h5_path, "r") as f:
        u_key = f"u_t{t_idx}"
        if u_key not in f:
            return None
        nodes = f["nodesSpatial_Domain0"][:]
        u = f[u_key][:]
        t_val = get_time_from_h5(f, t_idx)

    x = nodes[:, 0]
    order = np.argsort(x)
    x = x[order]
    u = u[order]
    return x, u, t_val


def parse_time_indices(time_indices_arg):
    if not time_indices_arg:
        return [10, 20, 50, 100, 150, 199]
    out = []
    for item in time_indices_arg:
        parts = [p.strip() for p in item.split(",") if p.strip()]
        out.extend(int(p) for p in parts)
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot analytical vs FCT vs low-order solution for a given refinement "
            "and selected time indices."
        )
    )
    parser.add_argument("--refinement", type=int, required=True, help="Refinement level N for tadr_level_N.h5")
    parser.add_argument(
        "--time-indices",
        nargs="+",
        default=None,
        help="Time indices (space/comma separated), e.g. --time-indices 10 50 100 or 10,50,100",
    )
    parser.add_argument("--fct-dir", default=".", help="Directory containing FCT run H5")
    parser.add_argument("--low-dir", default="Stab_2", help="Directory containing low-order run H5")
    parser.add_argument("--out-dir", default="plots_comparison", help="Output directory for PNG files")

    parser.add_argument("--C0", type=float, default=1.0)
    parser.add_argument("--v", type=float, default=0.3)
    parser.add_argument("--D", type=float, default=0.007)
    parser.add_argument("--x0", type=float, default=0.5)
    parser.add_argument("--sigma0", type=float, default=0.08)

    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "matplotlib is required. Install it and rerun Plot_comparison.py"
        ) from e

    time_indices = parse_time_indices(args.time_indices)
    os.makedirs(args.out_dir, exist_ok=True)

    fct_h5 = os.path.join(args.fct_dir, f"tadr_level_{args.refinement}.h5")
    low_h5 = os.path.join(args.low_dir, f"tadr_level_{args.refinement}.h5")

    if not os.path.exists(fct_h5):
        raise FileNotFoundError(f"FCT file not found: {fct_h5}")
    if not os.path.exists(low_h5):
        raise FileNotFoundError(f"Low-order file not found: {low_h5}")

    n_times = len(time_indices)
    ncols = 3 if n_times >= 3 else n_times
    nrows = int(np.ceil(n_times / float(ncols)))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)

    for idx, t_idx in enumerate(time_indices):
        ax = axes[idx // ncols][idx % ncols]

        fct = load_line_from_h5(fct_h5, t_idx)
        low = load_line_from_h5(low_h5, t_idx)

        if fct is None or low is None:
            ax.set_title(f"t_idx={t_idx} (missing in one file)")
            ax.axis("off")
            continue

        x_fct, u_fct, t_fct = fct
        x_low, u_low, t_low = low

        # Use FCT time metadata as reference; fallback to low-order if needed
        t_val = t_fct if t_fct is not None else (t_low if t_low is not None else 0.0)

        x_min = min(np.min(x_fct), np.min(x_low))
        x_max = max(np.max(x_fct), np.max(x_low))
        x_ana = np.linspace(x_min, x_max, 1600)
        u_ana = c_gaussian_ic(
            x_ana,
            t_val,
            c0=args.C0,
            vel=args.v,
            diff=args.D,
            center=args.x0,
            sigma=args.sigma0,
        )

        # Keep analytical style consistent with Plot_TADR_analytical.py
        ax.plot(x_ana, u_ana, "k-", lw=2.2, label=f"Analytical (t={t_val:.4f})", zorder=5)
        ax.plot(x_fct, u_fct, color="tab:blue", lw=1.3, marker="o", ms=2.0, label="FCT", zorder=2)
        ax.plot(x_low, u_low, color="tab:red", lw=1.3, marker="s", ms=1.9, label="Low-order", zorder=2)

        ax.set_title(f"ref={args.refinement}, t_idx={t_idx}, t={t_val:.6f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Turn off unused axes
    for j in range(n_times, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    out_png = os.path.join(args.out_dir, f"comparison_ref{args.refinement}.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
