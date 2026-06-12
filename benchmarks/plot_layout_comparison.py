#!/usr/bin/env python3
"""Overlay GPU OTI solve performance for the AoS and SoA storage layouts.

Reads the archived array-of-structs GPU sweep (benchmark_results_gpu_aos.csv,
a copy of the canonical results) and the coefficient-major sweep produced by
``SWEEP_DEVICES=gpu GPU_VARIANT=_soa run_benchmark.sh``
(benchmark_results_gpu_soa.csv), collapses reps to medians, and plots the OTI
solve wall time and the SoA/AoS ratio against work (node-updates), matching
the axes of wall_vs_complexity.png.

Output: results/gpu_layout_comparison.png
"""
import csv
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

FILES = {
    "aos": os.path.join(RESULTS, "benchmark_results_gpu_aos.csv"),
    "soa": os.path.join(RESULTS, "benchmark_results_gpu_soa.csv"),
}
STYLE = {
    ("aos", "double"): dict(color="#d62728", marker="o", ls="-", label="AoS double"),
    ("aos", "float"): dict(color="#d62728", marker="s", ls="--", label="AoS float"),
    ("soa", "double"): dict(color="#2ca02c", marker="o", ls="-", label="SoA double"),
    ("soa", "float"): dict(color="#2ca02c", marker="s", ls="--", label="SoA float"),
}


def medians(path):
    by_key = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["device"] != "gpu":
                continue
            by_key[(r["precision"], int(r["N"]))].append(
                (int(r["work"]), float(r["oti_solve"])))
    out = {}
    for (precision, n), rows in by_key.items():
        work = rows[0][0]
        out[(precision, n)] = (work, statistics.median(t for _, t in rows))
    return out


def series(med, precision):
    keys = sorted(n for p, n in med if p == precision)
    work = [med[(precision, n)][0] for n in keys]
    secs = [med[(precision, n)][1] for n in keys]
    return keys, work, secs


def main():
    med = {layout: medians(path) for layout, path in FILES.items()}

    fig, (ax_time, ax_ratio) = plt.subplots(1, 2, figsize=(12, 5))

    for layout in ("aos", "soa"):
        for precision in ("double", "float"):
            _, work, secs = series(med[layout], precision)
            ax_time.loglog(work, secs, **STYLE[(layout, precision)])
    ax_time.set_xlabel("work (node-updates = num_nodes * num_steps)")
    ax_time.set_ylabel("OTI solve wall time [s]")
    ax_time.set_title("GPU OTI solve: storage layout comparison")
    ax_time.grid(True, which="both", alpha=0.3)
    ax_time.legend()

    for precision, style_key in (("double", ("soa", "double")), ("float", ("soa", "float"))):
        ns_a, work_a, secs_a = series(med["aos"], precision)
        ns_s, _, secs_s = series(med["soa"], precision)
        common = [n for n in ns_a if n in ns_s]
        work = [work_a[ns_a.index(n)] for n in common]
        ratio = [secs_s[ns_s.index(n)] / secs_a[ns_a.index(n)] for n in common]
        ax_ratio.semilogx(work, ratio, **STYLE[style_key])
    ax_ratio.axhline(1.0, color="k", lw=0.8)
    ax_ratio.set_xlabel("work (node-updates = num_nodes * num_steps)")
    ax_ratio.set_ylabel("SoA / AoS OTI solve time")
    ax_ratio.set_title("SoA cost relative to AoS (1.0 = parity)")
    ax_ratio.grid(True, which="both", alpha=0.3)
    ax_ratio.legend()

    fig.tight_layout()
    out = os.path.join(RESULTS, "gpu_layout_comparison.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
