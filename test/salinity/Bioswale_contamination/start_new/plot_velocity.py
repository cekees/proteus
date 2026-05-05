#!/usr/bin/env python3
"""
Plot Richards velocity snapshots for Stab_0 and Stab_2 at selected times.

By default, the script compares the nearest common snapshots to t=0.2 and t=0.3.
Set TARGET_TIMES = [] to use only the last common time instead.

Run from this directory with:
    python plot_velocity.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "matplotlib is required to run plot_velocity.py. "
        "Please activate an environment with matplotlib installed."
    ) from exc


TIME_RE = re.compile(r"_t([0-9.e+-]+)\.txt$")

HERE = Path(__file__).resolve().parent
STAB0_DIR = HERE / "Stab_0"
STAB2_DIR = HERE / "Stab_2"

TARGET_TIMES = [0.2, 0.3]
OUTPUT_FILE = HERE / "velocity_selected_times_compare_new.png"
MARKER_SIZE = 4.0
ARROW_WIDTH = 0.004
POINT_ALPHA = 0.18
QUIVER_SCALE = None
DPI = 200


def load_profile_index(base_dir: Path, stab_tag: str) -> dict[float, Path]:
    files = {}
    pattern = f"richards_q_velocity_profile_{stab_tag}_t*.txt"
    for path in base_dir.glob(pattern):
        match = TIME_RE.search(path.name)
        if match:
            files[float(match.group(1))] = path
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} were found in {base_dir}")
    return files


def select_last_common_time(common_times: list[float]) -> float:
    if not common_times:
        raise RuntimeError("No common snapshot times were found between Stab_0 and Stab_2.")
    return common_times[-1]


def select_nearest_common_time(common_times: list[float], target_t: float) -> float:
    if not common_times:
        raise RuntimeError("No common snapshot times were found between Stab_0 and Stab_2.")
    return min(common_times, key=lambda t: (abs(t - target_t), t))


def load_snapshot(path: Path) -> np.ndarray:
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 5:
        raise ValueError(f"Unexpected snapshot format in {path}")
    return data


def compute_shared_scale(snapshots: list[np.ndarray]) -> float:
    all_speeds = np.concatenate([
        np.linalg.norm(snapshot[:, 3:5], axis=1) for snapshot in snapshots
    ])
    q95 = np.percentile(all_speeds, 95.0)
    return max(q95 / 0.35, 1.0e-12)


def plot_panel(ax, data: np.ndarray, label: str, title_suffix: str,
               marker_size: float, point_alpha: float, quiver_scale: float,
               arrow_width: float, vmin: float, vmax: float, xlim, ylim):
    x = data[:, 0]
    y = data[:, 1]
    vx = data[:, 3]
    vy = data[:, 4]
    speed = np.sqrt(vx * vx + vy * vy)

    ax.scatter(x, y, s=marker_size, c="k", alpha=point_alpha, linewidths=0)
    q = ax.quiver(
        x,
        y,
        vx,
        vy,
        speed,
        cmap="turbo",
        clim=(vmin, vmax),
        angles="xy",
        scale_units="xy",
        scale=quiver_scale,
        width=arrow_width,
        pivot="mid",
    )
    ax.set_title(f"{label} | {title_suffix}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    return q


files0 = load_profile_index(STAB0_DIR, "stab0")
files2 = load_profile_index(STAB2_DIR, "stab2")
common_times = sorted(set(files0) & set(files2))

if TARGET_TIMES:
    selected = [(target_t, select_nearest_common_time(common_times, target_t))
                for target_t in TARGET_TIMES]
else:
    actual_t = select_last_common_time(common_times)
    selected = [(actual_t, actual_t)]

snapshots = []
coords_blocks = []
for _, actual_t in selected:
    snap0 = load_snapshot(files0[actual_t])
    snap2 = load_snapshot(files2[actual_t])
    snapshots.extend((snap0, snap2))
    coords_blocks.extend((snap0[:, :2], snap2[:, :2]))

coords = np.vstack(coords_blocks)
x_pad = 0.05 * max(np.ptp(coords[:, 0]), 1.0)
y_pad = 0.05 * max(np.ptp(coords[:, 1]), 1.0)
xlim = (coords[:, 0].min() - x_pad, coords[:, 0].max() + x_pad)
ylim = (coords[:, 1].min() - y_pad, coords[:, 1].max() + y_pad)

speeds = np.concatenate([
    np.linalg.norm(snapshot[:, 3:5], axis=1) for snapshot in snapshots
])
vmin = 0.0
vmax = speeds.max()
quiver_scale = QUIVER_SCALE if QUIVER_SCALE is not None else compute_shared_scale(snapshots)

fig, axes = plt.subplots(
    nrows=len(selected),
    ncols=2,
    figsize=(12, 5 * len(selected)),
    squeeze=False,
)

mappable = None
for row, (target_t, actual_t) in enumerate(selected):
    snap0 = load_snapshot(files0[actual_t])
    snap2 = load_snapshot(files2[actual_t])
    title_suffix = f"target t={target_t:.3f}, actual t={actual_t:.9e}"
    mappable = plot_panel(
        axes[row, 0],
        snap0,
        "Stab_0",
        title_suffix,
        MARKER_SIZE,
        POINT_ALPHA,
        quiver_scale,
        ARROW_WIDTH,
        vmin,
        vmax,
        xlim,
        ylim,
    )
    plot_panel(
        axes[row, 1],
        snap2,
        "Stab_2",
        title_suffix,
        MARKER_SIZE,
        POINT_ALPHA,
        quiver_scale,
        ARROW_WIDTH,
        vmin,
        vmax,
        xlim,
        ylim,
    )

if TARGET_TIMES:
    fig.suptitle("Richards velocity comparison near selected common times", fontsize=15)
else:
    fig.suptitle("Richards velocity comparison at last common time", fontsize=15)

fig.tight_layout(rect=(0.0, 0.0, 0.94, 0.97))
cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
cbar.set_label("|v| (speed)")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches="tight")

for target_t, actual_t in selected:
    print(f"target t={target_t:.6f} -> actual common t={actual_t:.9e}")
print(f"Saved plot to {OUTPUT_FILE}")
