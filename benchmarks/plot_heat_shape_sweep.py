#!/usr/bin/env python3
"""Plot the OTI shape-sweep: solve wall time and speedup vs algebra size, one
line per optimization variant, faceted by precision x source-division.

Reads heat_shape_sweep_results.csv (from run_heat_shape_sweep.py)."""

import argparse
import csv
import os
import statistics
from collections import defaultdict
from pathlib import Path


VARIANTS = ["naive", "lookup", "unrolled", "aligned", "fused_aos", "fused_soa"]
LABELS = {
    "naive": "No product table",
    "lookup": "Runtime lookup",
    "unrolled": "Compile-time fold",
    "aligned": "Aligned AoS",
    "fused_aos": "Fused AoS",
    "fused_soa": "Fused SoA",
}
MARKERS = {"naive": "o", "lookup": "s", "unrolled": "^",
           "aligned": "D", "fused_aos": "v", "fused_soa": "P"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path,
                        help="heat_shape_sweep_results.csv or its directory")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--x", choices=["ncoeffs", "M", "nproducts"], default="ncoeffs")
    return parser.parse_args()


def load(path):
    if path.is_dir():
        path = path / "heat_shape_sweep_results.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for r in rows:
        for k in ("M", "N", "ncoeffs", "nproducts", "checksum_match"):
            r[k] = int(r[k])
        for k in ("oti_seconds", "speedup_vs_naive", "incremental_speedup"):
            r[k] = float(r[k])
    return rows, path


def medians(rows, metric, xkey):
    # (precision, division, variant) -> sorted [(x, median_metric), ...]
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["precision"], r["source_division"], r["variant"])
        acc[key][r[xkey]].append(r[metric])
    out = {}
    for key, byx in acc.items():
        out[key] = sorted((x, statistics.median(v)) for x, v in byx.items())
    return out


def main():
    args = parse_args()
    rows, path = load(args.input)
    output_dir = args.output_dir or path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    mism = [r for r in rows if not r["checksum_match"]]
    if mism:
        print(f"WARNING: {len(mism)} rows failed the checksum gate "
              "(variant result diverged from naive); plotting anyway")

    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precisions = sorted({r["precision"] for r in rows})
    divisions = sorted({r["source_division"] for r in rows})
    xlabel = {"ncoeffs": "coefficients  C(M+1, 1) = M+1",
              "M": "variables  M", "nproducts": "product terms"}[args.x]

    panels = [
        ("oti_seconds", "OTI solve wall time (s)", True,
         "heat_shape_sweep_walltime"),
        ("speedup_vs_naive", "speedup vs naive", False,
         "heat_shape_sweep_speedup"),
    ]

    for metric, ylabel, ylog, stem in panels:
        data = medians(rows, metric, args.x)
        nrow, ncol = len(precisions), len(divisions)
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.4 * nrow),
                                 squeeze=False, sharex=True)
        for i, precision in enumerate(precisions):
            for j, division in enumerate(divisions):
                ax = axes[i][j]
                for variant in VARIANTS:
                    series = data.get((precision, division, variant))
                    if not series:
                        continue
                    xs = [p[0] for p in series]
                    ys = [p[1] for p in series]
                    ax.plot(xs, ys, marker=MARKERS[variant], markersize=5,
                            linewidth=1.6, label=LABELS[variant])
                ax.set_xscale("log")
                if ylog:
                    ax.set_yscale("log")
                ax.grid(True, which="both", alpha=0.3)
                ax.set_title(f"{precision}, {division} source")
                if i == nrow - 1:
                    ax.set_xlabel(xlabel)
                if j == 0:
                    ax.set_ylabel(ylabel)
        axes[0][0].legend(fontsize=8, loc="best")
        fig.suptitle("Heat-solver OTI optimization sweep over algebra size <M, 1>")
        fig.tight_layout()
        for suffix in ("pdf", "png"):
            out = output_dir / f"{stem}.{suffix}"
            fig.savefig(out, dpi=220 if suffix == "png" else None)
            print(f"wrote {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
