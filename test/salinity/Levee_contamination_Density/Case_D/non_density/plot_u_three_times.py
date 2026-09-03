from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


BASE = Path(__file__).resolve().parent
H5_PATH = BASE / "ls_CCS_so.h5"
XMF_PATH = BASE / "ls_CCS_so.xmf"
OUT_PATH = BASE / "u_three_times.png"

X_LIMITS = (0.0, 160.0)
Z_LIMITS = (0.0, 60.0)


def load_times(xmf_path: Path) -> np.ndarray:
    root = ET.parse(xmf_path).getroot()
    return np.array(
        [
            float(grid.find("Time").attrib["Value"])
            for grid in root.findall('.//Grid[@GridType="Uniform"]')
        ]
    )


def choose_steps(times: np.ndarray) -> list[int]:
    requested_steps = [100, 400, len(times) - 1]
    return [min(step, len(times) - 1) for step in requested_steps]


def main() -> None:
    times = load_times(XMF_PATH)
    steps = choose_steps(times)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(10, 13),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    with h5py.File(H5_PATH, "r") as h5_file:
        mappable = None
        for ax, step in zip(axes, steps):
            nodes = h5_file[f"nodesSpatial_Domain{step}"][:]
            elements = h5_file[f"elementsSpatial_Domain{step}"][:]
            u = h5_file[f"u_t{step}"][:]

            tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
            mappable = ax.tripcolor(
                tri,
                u,
                shading="gouraud",
                cmap="viridis", # "turbo",
                vmin=0.0,
                vmax=0.7,
                rasterized=True,
            )
            ax.set_xlim(*X_LIMITS)
            ax.set_ylim(*Z_LIMITS)
            ax.set_aspect("equal")
            ax.set_ylabel("z [m]")
            ax.set_title(
                f"step {step},  t ≈ {times[step]:.1f} d,   "
                f"u ∈ [{max(0.0, float(u.min())):.3f}, {float(u.max()):.3f}]"
            )

    axes[-1].set_xlabel("x [m]")
    colorbar = fig.colorbar(mappable, ax=axes, fraction=0.06, pad=0.04)
    colorbar.set_label("u (normalized concentration)")
    fig.suptitle(
        "Levee_contamination_Density / Case_D  —  u(x,z) at three times",
        fontsize=18,
    )
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")

    print(f"wrote {OUT_PATH}")
    print(f"steps {steps}")


if __name__ == "__main__":
    main()
