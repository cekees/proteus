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


def available_steps(h5_file) -> list[int]:
    return sorted(
        int(k[len("u_t"):]) for k in h5_file.keys() if k.startswith("u_t")
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
    requested = [100, 400, last]
    return [s for s in requested if s <= last]


def main() -> None:
    xmf_times = load_times(XMF_PATH)

    with h5py.File(H5_PATH, "r") as h5_file:
        steps_in_h5 = available_steps(h5_file)
        if not steps_in_h5:
            raise RuntimeError(f"no u_t* datasets found in {H5_PATH}")
        steps = choose_steps(steps_in_h5)

        n = len(steps)
        fig, axes = plt.subplots(
            nrows=n,
            ncols=1,
            figsize=(10, 4.3 * n + 0.5),
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
            u = h5_file[f"u_t{step}"][:]

            tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
            mappable = ax.tripcolor(
                tri,
                u,
                shading="gouraud",
                cmap="viridis",
                vmin=0.0,
                vmax=0.7,
                rasterized=True,
            )
            ax.set_xlim(*X_LIMITS)
            ax.set_ylim(*Z_LIMITS)
            ax.set_aspect("equal")
            ax.set_ylabel("z [m]")
            t_str = f"t ≈ {xmf_times[step]:.1f} d, " if step in xmf_times else ""
            ax.set_title(
                f"step {step}, {t_str}"
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
