#!/usr/bin/env python3
"""Plot OTI heat-equation solution, derivative, and error snapshots."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


FIELDS = [
    ("solution_scalar", "Solution"),
    ("solution_abs_error", "|OTI real - scalar|"),
    ("dalpha_oti", "dT/dalpha OTI"),
    ("dalpha_fd", "dT/dalpha finite difference"),
    ("dalpha_abs_error", "|dT/dalpha error|"),
    ("damplitude_oti", "dT/damplitude OTI"),
    ("damplitude_fd", "dT/damplitude finite difference"),
    ("damplitude_abs_error", "|dT/damplitude error|"),
    ("dsigma_oti", "dT/dsigma OTI"),
    ("dsigma_fd", "dT/dsigma finite difference"),
    ("dsigma_abs_error", "|dT/dsigma error|"),
]


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def unique_sorted(rows: list[dict[str, str]], key: str) -> list[int]:
    return sorted({int(row[key]) for row in rows})


def plot_field(rows: list[dict[str, str]], step: int, field: str, title: str, out_dir: Path) -> None:
    step_rows = [row for row in rows if int(row["step"]) == step]
    if not step_rows:
        return

    nx = max(int(row["i"]) for row in step_rows) + 1
    ny = max(int(row["j"]) for row in step_rows) + 1
    x = np.zeros((ny, nx))
    y = np.zeros((ny, nx))
    values = np.zeros((ny, nx))

    for row in step_rows:
        i = int(row["i"])
        j = int(row["j"])
        x[j, i] = float(row["x"])
        y[j, i] = float(row["y"])
        values[j, i] = float(row[field])

    time = float(step_rows[0]["time"])
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    mesh = ax.contourf(x, y, values, levels=40, cmap="viridis")
    fig.colorbar(mesh, ax=ax)
    ax.set_title(f"{title}, t={time:.5f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    safe_field = field.replace("/", "_")
    fig.savefig(out_dir / f"{safe_field}_step_{step:05d}.png", dpi=180)
    plt.close(fig)


def write_summary(rows: list[dict[str, str]], out_dir: Path) -> None:
    steps = unique_sorted(rows, "step")
    error_fields = [
        "solution_abs_error",
        "dalpha_abs_error",
        "damplitude_abs_error",
        "dsigma_abs_error",
    ]
    with (out_dir / "error_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time", "field", "max_abs_error", "mean_abs_error", "rms_abs_error"])
        for step in steps:
            step_rows = [row for row in rows if int(row["step"]) == step]
            time = float(step_rows[0]["time"])
            for field in error_fields:
                values = np.array([float(row[field]) for row in step_rows])
                writer.writerow(
                    [
                        step,
                        time,
                        field,
                        float(np.max(values)),
                        float(np.mean(values)),
                        float(np.sqrt(np.mean(values * values))),
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", default="oti_analysis_output")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    csv_path = out_dir / "slice_snapshots.csv"
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    steps = unique_sorted(rows, "step")
    for step in steps:
        for field, title in FIELDS:
            plot_field(rows, step, field, title, plot_dir)

    write_summary(rows, out_dir)
    print(f"Wrote plots to {plot_dir}")
    print(f"Wrote summary to {out_dir / 'error_summary.csv'}")


if __name__ == "__main__":
    main()
