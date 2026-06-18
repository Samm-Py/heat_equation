#!/usr/bin/env python3
"""Plot OTI solve overhead (``oti_over_base``) vs problem size, one line per
optimization stage.

This complements ``plot_heat_optimization_benchmarks.py``: that script plots the
metrics *against the optimization stage* at fixed grid sizes, whereas here the
x-axis is problem size, so each cumulative stage gets its own overhead-vs-size
curve -- the per-stage version of the "Why the OTI overhead grows with problem
size" figure.

Run the optimization study over a grid sweep with every variant first:

  python3 benchmarks/run_heat_optimization_benchmarks.py --build --build-dir build-cuda \\
    --grid-sizes 21 31 41 51 61 81 \\
    --variants naive lookup unrolled aligned fused_aos fused_soa

then point this at the results directory:

  python3 benchmarks/plot_oti_overhead_by_stage.py benchmarks/results/optimization_study

It writes oti_overhead_by_stage.{png,pdf} into a figures/ subdirectory (override
with --output-dir), faceted by source-division (rows) x precision (columns).
"""

import argparse
import csv
import statistics
import os
from collections import defaultdict
from pathlib import Path


VARIANTS = ["naive", "lookup", "unrolled", "aligned", "fused_aos", "fused_soa"]
LABELS = {
    "naive": "no product table",
    "lookup": "runtime lookup",
    "unrolled": "compile-time fold",
    "aligned": "aligned AoS",
    "fused_aos": "fused AoS",
    "fused_soa": "fused SoA",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path,
                        help="optimization_study dir or its heat_optimization_results.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--x", choices=["work", "num_nodes"], default="work",
                        help="x-axis: total node-updates (work) or node count")
    return parser.parse_args()


def load(path):
    if path.is_dir():
        path = path / "heat_optimization_results.csv"
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            for key in ("run", "N", "num_nodes", "num_steps", "work", "checksum_match"):
                row[key] = int(row[key])
            row["oti_over_base"] = float(row["oti_over_base"])
            rows.append(row)
    if not rows:
        raise SystemExit(f"no rows in {path}")
    if any(not row["checksum_match"] for row in rows):
        raise SystemExit("results contain a checksum mismatch")
    return path, rows


def aggregate(rows, xkey):
    """Median oti_over_base per (scenario, precision, variant, grid), with its x."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source_division"], row["precision"], row["variant"],
                 row["N"])].append(row)
    points = []
    for (scenario, precision, variant, _grid), group in grouped.items():
        points.append({
            "scenario": scenario,
            "precision": precision,
            "variant": variant,
            "x": group[0][xkey],
            "oti_over_base": statistics.median(r["oti_over_base"] for r in group),
        })
    return points


def main():
    args = parse_args()
    input_path, rows = load(args.input.resolve())
    output_dir = (args.output_dir.resolve() if args.output_dir
                  else input_path.parent / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    points = aggregate(rows, args.x)
    scenarios = sorted({p["scenario"] for p in points})
    precisions = sorted({p["precision"] for p in points})

    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(scenarios), len(precisions),
        figsize=(6.5 * len(precisions), 4.8 * len(scenarios)),
        squeeze=False)

    xlabel = ("total node-updates (num_nodes x num_steps)"
              if args.x == "work" else "num_nodes")
    for r, scenario in enumerate(scenarios):
        for c, precision in enumerate(precisions):
            ax = axes[r][c]
            for variant in VARIANTS:
                series = sorted(
                    (p["x"], p["oti_over_base"]) for p in points
                    if p["scenario"] == scenario and p["precision"] == precision
                    and p["variant"] == variant)
                if not series:
                    continue
                xs = [s[0] for s in series]
                ys = [s[1] for s in series]
                ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.6,
                        label=LABELS[variant])
            ax.axhline(1.0, color="black", linewidth=0.8)  # OTI == base
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{precision}, source division {scenario}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("OTI / base wall-time ratio")
            ax.grid(True, which="both", linestyle=":", alpha=0.5)
            ax.legend(fontsize=8)

    fig.suptitle("OTI solve overhead vs problem size, by optimization stage")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        path = output_dir / f"oti_overhead_by_stage.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
