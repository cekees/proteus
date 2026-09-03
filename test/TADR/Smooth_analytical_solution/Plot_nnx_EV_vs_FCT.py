import argparse
import glob
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


def load_line(h5_path, t_idx):
    if not os.path.exists(h5_path):
        return None
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
    nnx = int(np.unique(np.round(nodes[:, 0], 14)).size)
    return x, u, t_val, nnx


def find_available_refinements(fct_dir, ev_dir):
    fct = {
        int(os.path.basename(p).split("_")[-1].split(".")[0])
        for p in glob.glob(os.path.join(fct_dir, "tadr_level_*.h5"))
    }
    ev = {
        int(os.path.basename(p).split("_")[-1].split(".")[0])
        for p in glob.glob(os.path.join(ev_dir, "tadr_level_*.h5"))
    }
    return sorted(fct.intersection(ev))


def parse_refinements(values):
    if not values:
        return None
    out = []
    for item in values:
        out.extend(int(v.strip()) for v in item.split(",") if v.strip())
    return sorted(set(out))


def main():
    parser = argparse.ArgumentParser(
        description="Subplots at fixed time index: Entropy-viscosity vs FCT for multiple refinements"
    )
    parser.add_argument("--time-index", type=int, required=True, help="H5 solution index, i.e. u_t<index>")
    parser.add_argument("--fct-dir", default=".", help="Directory containing FCT runs")
    parser.add_argument("--ev-dir", default="Stab_2", help="Directory containing entropy-viscosity runs")
    parser.add_argument(
        "--refinements",
        nargs="+",
        default=None,
        help="Optional refinement levels; if omitted, auto-detect intersection",
    )
    parser.add_argument("--out-dir", default="plots_nnx_comparison", help="Output directory")
    parser.add_argument("--C0", type=float, default=1.0)
    parser.add_argument("--v", type=float, default=0.3)
    parser.add_argument("--D", type=float, default=0.007)
    parser.add_argument("--x0", type=float, default=0.5)
    parser.add_argument("--sigma0", type=float, default=0.08)

    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("matplotlib is required for plotting") from e

    requested = parse_refinements(args.refinements)
    available = find_available_refinements(args.fct_dir, args.ev_dir)

    if not available:
        raise FileNotFoundError(
            f"No common tadr_level_*.h5 found between {args.fct_dir} and {args.ev_dir}"
        )

    if requested is None:
        refs = available
    else:
        refs = [r for r in requested if r in available]
        if not refs:
            raise ValueError(
                f"Requested refinements {requested} not found in both dirs. Available intersection: {available}"
            )

    os.makedirs(args.out_dir, exist_ok=True)

    n = len(refs)
    ncols = min(3, n)
    nrows = int(np.ceil(n / float(ncols)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)

    valid_count = 0
    suptitle_time = None
    for i, ref in enumerate(refs):
        ax = axes[i // ncols][i % ncols]
        fct_h5 = os.path.join(args.fct_dir, f"tadr_level_{ref}.h5")
        ev_h5 = os.path.join(args.ev_dir, f"tadr_level_{ref}.h5")

        fct = load_line(fct_h5, args.time_index)
        ev = load_line(ev_h5, args.time_index)

        if fct is None or ev is None:
            ax.set_title(f"ref={ref}, t_idx={args.time_index} (missing)")
            ax.axis("off")
            continue

        x_fct, u_fct, t_fct, nnx_fct = fct
        x_ev, u_ev, t_ev, nnx_ev = ev

        t_val = t_fct if t_fct is not None else (t_ev if t_ev is not None else np.nan)
        nnx = nnx_fct if nnx_fct == nnx_ev else min(nnx_fct, nnx_ev)
        x_min = min(np.min(x_fct), np.min(x_ev))
        x_max = max(np.max(x_fct), np.max(x_ev))
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

        ax.plot(x_ana, u_ana, "k-", lw=2.2, label="Analytical")
        ax.plot(x_fct, u_fct, color="tab:blue", lw=1.4, marker="o", ms=2.0, label="FCT")
        ax.plot(x_ev, u_ev, color="tab:red", lw=1.4, marker="s", ms=2.0, label="Entropy_viscosity")

        ax.set_title(f"nnx={nnx}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        valid_count += 1
        if suptitle_time is None and np.isfinite(t_val):
            suptitle_time = t_val

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    time_label = f"{suptitle_time:.2f}" if suptitle_time is not None else "N/A"
    fig.suptitle(f"time = {time_label}", y=1.02)
    fig.tight_layout()

    out_png = os.path.join(args.out_dir, f"EV_vs_FCT_tidx{args.time_index}.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    if valid_count == 0:
        raise ValueError(
            f"No refinement had both files with u_t{args.time_index}."
        )

    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
