#!/usr/bin/env python3
"""Aggregate and plot OTI heat-solver benchmark results.

Reads benchmarks/results/benchmark_results.csv (one row per rep, produced by
run_benchmark.sh), collapses reps to the median per (device, precision, N), and
plots wall-clock time against computational complexity.

Computational complexity (x-axis) = node-updates = num_nodes * num_steps, i.e.
the total number of per-node solver updates performed in a run. Because the
physical total_time is fixed and dt is CFL-bound, this captures the real work
even though it grows ~N^5 with grid size.

Outputs:
    results/benchmark_median.csv          collapsed medians
    results/wall_vs_complexity.png        OTI solve wall time vs work, 4 configs
    results/wall_vs_complexity_base.png   base scalar solve vs work, 4 configs
    results/oti_overhead.png              OTI/base ratio vs work
"""
import csv
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CSV = os.path.join(RESULTS, "benchmark_results.csv")

CONFIGS = [
    ("cpu", "double"),
    ("cpu", "float"),
    ("gpu", "double"),
    ("gpu", "float"),
]
STYLE = {
    ("cpu", "double"): dict(color="#1f77b4", marker="o", ls="-", label="CPU (OpenMP) double"),
    ("cpu", "float"):  dict(color="#1f77b4", marker="s", ls="--", label="CPU (OpenMP) float"),
    ("gpu", "double"): dict(color="#d62728", marker="o", ls="-", label="GPU (CUDA) double"),
    ("gpu", "float"):  dict(color="#d62728", marker="s", ls="--", label="GPU (CUDA) float"),
}


def load():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            for k in ("N", "num_nodes", "num_steps", "work"):
                r[k] = int(r[k])
            for k in ("base_scalar_solve", "oti_solve", "oti_ratio"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def collapse(rows):
    """median over reps, keyed by (device, precision, N)."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["device"], r["precision"], r["N"])].append(r)
    out = []
    for (device, precision, N), rs in groups.items():
        rep = rs[0]
        out.append({
            "device": device, "precision": precision, "N": N,
            "num_nodes": rep["num_nodes"], "num_steps": rep["num_steps"],
            "work": rep["work"], "n_reps": len(rs),
            "base_scalar_solve": statistics.median(x["base_scalar_solve"] for x in rs),
            "oti_solve": statistics.median(x["oti_solve"] for x in rs),
            "oti_ratio": statistics.median(x["oti_ratio"] for x in rs),
        })
    out.sort(key=lambda d: (d["device"], d["precision"], d["N"]))
    return out


def write_median(med):
    path = os.path.join(RESULTS, "benchmark_median.csv")
    cols = ["device", "precision", "N", "num_nodes", "num_steps", "work",
            "n_reps", "base_scalar_solve", "oti_solve", "oti_ratio"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for d in med:
            w.writerow(d)
    print("wrote", path)


def series(med, device, precision, ykey):
    pts = [(d["work"], d[ykey]) for d in med
           if d["device"] == device and d["precision"] == precision]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def plot(med, ykey, title, fname, ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    for cfg in CONFIGS:
        xs, ys = series(med, cfg[0], cfg[1], ykey)
        if xs:
            ax.plot(xs, ys, **STYLE[cfg])
    ax.set_xscale("log")
    if ykey != "oti_ratio":
        ax.set_yscale("log")
    ax.set_xlabel("computational complexity  (node-updates = num_nodes x num_steps)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(RESULTS, fname)
    fig.savefig(path, dpi=130)
    print("wrote", path)


def plot_saturation(med, fname):
    """Per-node-update time for base vs OTI on the GPU. This is the decomposition
    behind the overhead curve: the base scalar solve is latency-bound at small
    sizes and its per-node-update time falls steeply as the GPU saturates, while
    the (4x heavier) OTI solve saturates earlier and is much flatter -- so the
    OTI/base ratio grows mostly because the *base* denominator shrinks, not
    because OTI gets more expensive."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    for precision, color in (("double", "#d62728"), ("float", "#ff7f0e")):
        for ykey, ls, tag in (("base_scalar_solve", "--", "base"),
                              ("oti_solve", "-", "OTI")):
            xs, ys = series(med, "gpu", precision, ykey)
            if not xs:
                continue
            ns = [y / w * 1e9 for w, y in zip(xs, ys)]
            ax.plot(xs, ns, color=color, ls=ls, marker="o", markersize=4,
                    label=f"{tag} {precision}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("computational complexity  (node-updates = num_nodes x num_steps)")
    ax.set_ylabel("wall time per node-update (ns)  -- lower = better utilized")
    ax.set_title("GPU per-node-update time: base saturates late, OTI saturates early")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(RESULTS, fname)
    fig.savefig(path, dpi=130)
    print("wrote", path)


def main():
    rows = load()
    med = collapse(rows)
    write_median(med)
    plot(med, "oti_solve", "OTI heat solve: wall time vs computational complexity",
         "wall_vs_complexity.png", "OTI solve wall time (s)")
    plot(med, "base_scalar_solve", "Base scalar solve: wall time vs computational complexity",
         "wall_vs_complexity_base.png", "base scalar solve wall time (s)")
    plot(med, "oti_ratio", "OTI overhead (otinum<3,1>) vs computational complexity",
         "oti_overhead.png", "OTI / base wall-time ratio")
    plot_saturation(med, "oti_overhead_saturation.png")


if __name__ == "__main__":
    main()
