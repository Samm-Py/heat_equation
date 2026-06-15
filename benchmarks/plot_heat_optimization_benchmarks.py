#!/usr/bin/env python3
"""Aggregate and plot heat-application optimization benchmark results."""

import argparse
import csv
import os
import statistics
from collections import defaultdict
from pathlib import Path


VARIANTS = ["naive", "lookup", "unrolled", "aligned", "fused_aos", "fused_soa"]
LABELS = {
    "naive": "No product\ntable",
    "lookup": "Runtime\nlookup",
    "unrolled": "Compile-time\nfold",
    "aligned": "Aligned\nAoS",
    "fused_aos": "Fused\nAoS",
    "fused_soa": "Fused\nSoA",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load(path):
    if path.is_dir():
        path = path / "heat_optimization_results.csv"
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            for key in ("run", "N", "num_nodes", "num_steps", "work", "checksum_match"):
                row[key] = int(row[key])
            for key in (
                "base_seconds", "oti_seconds", "oti_over_base",
                "speedup_vs_naive", "incremental_speedup",
            ):
                row[key] = float(row[key])
            rows.append(row)
    if not rows or any(not row["checksum_match"] for row in rows):
        raise SystemExit("results are empty or contain a checksum mismatch")
    return path, rows


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source_division"], row["precision"], row["N"],
                 row["variant"])].append(row)

    summary = []
    for key, group in grouped.items():
        source_division, precision, grid_size, variant = key
        item = {
            "source_division": source_division,
            "precision": precision,
            "N": grid_size,
            "variant": variant,
            "runs": len(group),
        }
        for metric in (
            "base_seconds", "oti_seconds", "oti_over_base",
            "speedup_vs_naive", "incremental_speedup",
        ):
            values = [row[metric] for row in group]
            item[f"{metric}_median"] = statistics.median(values)
            item[f"{metric}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(item)
    summary.sort(key=lambda item: (
        item["source_division"], item["precision"], item["N"],
        VARIANTS.index(item["variant"])))
    return summary


def write_summary(summary, output_dir):
    path = output_dir / "heat_optimization_summary.csv"
    columns = list(summary[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {path}")


def select(summary, scenario, precision, grid_size):
    by_variant = {
        row["variant"]: row for row in summary
        if row["source_division"] == scenario
        and row["precision"] == precision
        and row["N"] == grid_size
    }
    return [by_variant[variant] for variant in VARIANTS if variant in by_variant]


def plot_metric(summary, output_dir, metric, ylabel, basename, log_scale):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = sorted({row["source_division"] for row in summary})
    precisions = sorted({row["precision"] for row in summary})
    grid_sizes = sorted({row["N"] for row in summary})
    fig, axes = plt.subplots(
        len(scenarios), len(precisions),
        figsize=(6.5 * len(precisions), 4.8 * len(scenarios)),
        squeeze=False)

    for row_index, scenario in enumerate(scenarios):
        for col_index, precision in enumerate(precisions):
            ax = axes[row_index][col_index]
            for grid_size in grid_sizes:
                points = select(summary, scenario, precision, grid_size)
                if not points:
                    continue
                xs = list(range(len(points)))
                ys = [point[f"{metric}_median"] for point in points]
                errors = [point[f"{metric}_stdev"] for point in points]
                ax.errorbar(xs, ys, yerr=errors, marker="o", capsize=3,
                            label=f"N={grid_size}")
                ax.set_xticks(xs, [LABELS[point["variant"]] for point in points])
            if metric != "oti_seconds":
                ax.axhline(1.0, color="black", linewidth=0.8)
            if log_scale:
                ax.set_yscale("log")
            ax.set_title(f"{precision}, source division {scenario}")
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)
            ax.legend()

    fig.suptitle("OTI heat-equation application optimization")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        path = output_dir / f"{basename}.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        print(f"wrote {path}")
    plt.close(fig)


def main():
    args = parse_args()
    input_path, rows = load(args.input.resolve())
    output_dir = (args.output_dir.resolve() if args.output_dir
                  else input_path.parent / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(rows)
    write_summary(summary, output_dir)
    plot_metric(
        summary, output_dir, "oti_seconds", "OTI solve wall time [s]",
        "heat_optimization_stages", True)
    plot_metric(
        summary, output_dir, "speedup_vs_naive", "speedup vs no-table application",
        "heat_optimization_speedups", False)
    plot_metric(
        summary, output_dir, "incremental_speedup", "incremental stage speedup",
        "heat_optimization_incremental", False)


if __name__ == "__main__":
    main()
