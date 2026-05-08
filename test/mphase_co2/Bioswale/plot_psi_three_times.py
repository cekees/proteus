from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


BASE = Path(__file__).resolve().parent
H5_PATH = BASE / "Raingarden.h5"
XMF_PATH = BASE / "Raingarden.xmf"
OUT_PATH = BASE / "pressure_head_three_times.png"
FIELD = "pressure_head"   # dataset prefix written by mphase_co2 (was "u" in salinity case)


def available_steps(h5_file) -> list[int]:
    prefix = f"{FIELD}_t"
    return sorted(
        int(k[len(prefix):]) for k in h5_file.keys() if k.startswith(prefix)
    )


def load_times(xmf_path: Path) -> dict[int, float]:
    if not xmf_path.exists():
        return {}
    root = ET.parse(xmf_path).getroot()
    out: dict[int, float] = {}
    for grid in root.findall('.//Grid[@GridType="Uniform"]'):
        time_el = grid.find("Time")
        if time_el is None:
            continue
        name = time_el.attrib.get("Name")
        if name is None:
            continue
        try:
            out[int(name)] = float(time_el.attrib["Value"])
        except (KeyError, ValueError):
            continue
    return out


def choose_steps(steps: list[int]) -> list[int]:
    last = steps[-1]
    if last >= 400:
        requested = [100, 400, last]
    else:
        requested = [max(1, last // 4), max(1, last // 2), last]

    chosen = []
    for step in requested:
        nearest = min(steps, key=lambda s: (abs(s - step), s))
        if nearest not in chosen:
            chosen.append(nearest)
    return chosen


def main() -> None:
    xmf_times = load_times(XMF_PATH)

    with h5py.File(H5_PATH, "r") as h5_file:
        steps_in_h5 = available_steps(h5_file)
        if not steps_in_h5:
            raise RuntimeError(f"no {FIELD}_t* datasets found in {H5_PATH}")
        steps = choose_steps(steps_in_h5)

        # Geometry extent from union of all chosen frames.
        all_nodes = [h5_file[f"nodesSpatial_Domain{step}"][:] for step in steps]
        stacked_nodes = np.vstack(all_nodes)
        x_limits = (
            float(stacked_nodes[:, 0].min()) - 0.05,
            float(stacked_nodes[:, 0].max()) + 0.05,
        )
        z_limits = (
            float(stacked_nodes[:, 1].min()) - 0.05,
            float(stacked_nodes[:, 1].max()) + 0.05,
        )

        # Auto color range across the chosen frames (ψ may be negative or positive).
        u_global_min = np.inf
        u_global_max = -np.inf
        for step in steps:
            u = h5_file[f"{FIELD}_t{step}"][:]
            u_global_min = min(u_global_min, float(u.min()))
            u_global_max = max(u_global_max, float(u.max()))
        if u_global_min < 0.0 < u_global_max:
            cmap = "RdBu_r"
            vabs = max(abs(u_global_min), abs(u_global_max))
            vmin, vmax = -vabs, vabs
        else:
            cmap = "turbo"
            vmin, vmax = u_global_min, u_global_max

        n = len(steps)
        x_span = x_limits[1] - x_limits[0]
        z_span = z_limits[1] - z_limits[0]
        panel_width = 4.0
        panel_height = panel_width * z_span / max(x_span, 1e-9)
        fig, axes = plt.subplots(
            nrows=1,
            ncols=n,
            figsize=(panel_width * n + 1.2, panel_height + 0.8),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        if n == 1:
            axes = [axes]

        mappable = None
        for ax, step in zip(axes, steps):
            nodes = h5_file[f"nodesSpatial_Domain{step}"][:]
            elements = h5_file[f"elementsSpatial_Domain{step}"][:]
            u = h5_file[f"{FIELD}_t{step}"][:]

            tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
            mappable = ax.tripcolor(
                tri,
                u,
                shading="gouraud",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            ax.set_xlim(*x_limits)
            ax.set_ylim(*z_limits)
            ax.set_aspect("equal")
            ax.set_xlabel("x [m]")
            t_str = f"t approx {xmf_times[step]:.4f} d, " if step in xmf_times else ""
            ax.set_title(f"time, {t_str}step {step}")

    axes[0].set_ylabel("z [m]")
    colorbar = fig.colorbar(mappable, ax=axes, fraction=0.04, pad=0.02)
    colorbar.set_label(r"pressure head $\psi$ [m]")
    fig.suptitle("Bioswale (mphase_co2) -- pressure head", fontsize=17)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")

    print(f"wrote {OUT_PATH}")
    print(f"steps {steps}")
    print(f"psi range over chosen frames: [{u_global_min:.4g}, {u_global_max:.4g}]")


if __name__ == "__main__":
    main()
