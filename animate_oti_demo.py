#!/usr/bin/env python3
"""Visualize an OTI heat-equation run: temperature + parameter sensitivities.

Reads slice_snapshots.csv (top z-slice) produced by oti_heat_analysis and renders
a 2x2 panel of the OTI fields over time:

    T (solution)          dT/dalpha
    dT/damplitude         dT/dsigma

Outputs into <output_dir>/demo/:
    final_frame.png   the last time step, all four fields
    oti_fields.mp4    animation over all snapshot frames (GIF fallback if no ffmpeg)

Run the solver first with enough frames, e.g.:
    ./build/oti_heat_analysis_double demo_out --N 81 --snapshots 60
    python3 animate_oti_demo.py demo_out
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

# (column, title, colormap, signed?) -- signed fields use a 0-centered diverging map.
PANELS = [
    ("solution_oti", "T  (temperature)", "inferno", False),
    ("dalpha_oti", r"$\partial T/\partial\alpha$  (diffusivity)", "RdBu_r", True),
    ("damplitude_oti", r"$\partial T/\partial A$  (source amplitude)", "RdBu_r", True),
    ("dsigma_oti", r"$\partial T/\partial\sigma$  (source width)", "RdBu_r", True),
]


def load_frames(csv_path: Path):
    """Return (steps, times, x, y, {field: [grid per step]})."""
    rows = list(csv.DictReader(csv_path.open(newline="")))
    steps = sorted({int(r["step"]) for r in rows})
    nx = max(int(r["i"]) for r in rows) + 1
    ny = max(int(r["j"]) for r in rows) + 1

    by_step = {s: [] for s in steps}
    for r in rows:
        by_step[int(r["step"])].append(r)

    x = np.zeros((ny, nx))
    y = np.zeros((ny, nx))
    fields = {col: [] for col, *_ in PANELS}
    times = []
    for s in steps:
        srows = by_step[s]
        times.append(float(srows[0]["time"]))
        grids = {col: np.zeros((ny, nx)) for col, *_ in PANELS}
        for r in srows:
            i, j = int(r["i"]), int(r["j"])
            x[j, i] = float(r["x"])
            y[j, i] = float(r["y"])
            for col, *_ in PANELS:
                grids[col][j, i] = float(r[col])
        for col, *_ in PANELS:
            fields[col].append(grids[col])
    return steps, times, x, y, fields


def color_limits(field_grids, signed):
    """Global vmin/vmax across all frames so colors are stable in the animation."""
    stack = np.array(field_grids)
    if signed:
        m = float(np.abs(stack).max()) or 1.0
        return -m, m
    return float(stack.min()), float(stack.max() or 1.0)


def build_figure(x, y, fields, frame):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    ims = {}
    for ax, (col, title, cmap, signed) in zip(axes.flat, PANELS):
        vmin, vmax = color_limits(fields[col], signed)
        im = ax.pcolormesh(x, y, fields[col][frame], cmap=cmap,
                           vmin=vmin, vmax=vmax, shading="auto")
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ims[col] = im
    return fig, ims


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output_dir", nargs="?", default="oti_analysis_output")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--gif", action="store_true", help="also/instead write a GIF")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    csv_path = out_dir / "slice_snapshots.csv"
    if not csv_path.exists():
        raise SystemExit(f"missing {csv_path} -- run the solver WITHOUT --skip-fd")
    demo_dir = out_dir / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    steps, times, x, y, fields = load_frames(csv_path)
    print(f"loaded {len(steps)} frames, grid {x.shape[1]}x{x.shape[0]} (top z-slice)")

    # Static final-frame figure.
    fig, ims = build_figure(x, y, fields, len(steps) - 1)
    fig.suptitle(f"OTI heat solve, t = {times[-1]:.5f}  (final frame)", fontsize=14)
    final_png = demo_dir / "final_frame.png"
    fig.savefig(final_png, dpi=140)
    print("wrote", final_png)
    plt.close(fig)

    # Animation.
    fig, ims = build_figure(x, y, fields, 0)
    suptitle = fig.suptitle("", fontsize=14)

    def update(frame):
        for col, *_ in PANELS:
            ims[col].set_array(fields[col][frame].ravel())
        suptitle.set_text(f"OTI heat solve, t = {times[frame]:.5f}  (frame {frame + 1}/{len(steps)})")
        return list(ims.values()) + [suptitle]

    anim = FuncAnimation(fig, update, frames=len(steps), blit=False)

    if not args.gif:
        try:
            mp4 = demo_dir / "oti_fields.mp4"
            anim.save(mp4, writer=FFMpegWriter(fps=args.fps, bitrate=4000))
            print("wrote", mp4)
        except Exception as exc:  # noqa: BLE001
            print("ffmpeg failed, falling back to GIF:", exc)
            args.gif = True
    if args.gif:
        gif = demo_dir / "oti_fields.gif"
        anim.save(gif, writer=PillowWriter(fps=args.fps))
        print("wrote", gif)
    plt.close(fig)


if __name__ == "__main__":
    main()
