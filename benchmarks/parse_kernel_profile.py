#!/usr/bin/env python3
"""Parse space-time-stack reports into a per-kernel OTI-vs-scalar table.

Reads benchmarks/results/prof_<device>_<precision>.txt (produced by
profile_kernels.sh) and extracts, from the TOP-DOWN TIME TREE, the time each
named parallel_for spent inside the base_scalar_solve vs oti_solve regions.

Outputs:
    results/kernel_profile.csv   tidy: device,precision,kernel,base_s,oti_s,oti_over_base,calls
and prints a readable table.
"""
import csv
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

CONFIGS = [("cpu", "double"), ("cpu", "float"), ("gpu", "double"), ("gpu", "float")]
KERNELS = ["ComputeStiffnessForce", "ComputeSource", "UpdateTemperature"]

# A tree line: "... <time> sec <pct>% ... <calls> <name> [for|region]"
TIME_RE = re.compile(r"([0-9.eE+-]+)\s+sec\b")
TAIL_RE = re.compile(r"([0-9]+)\s+(\w+)\s+\[(for|region)\]\s*$")


def parse_report(path):
    """Return {(region, kernel): (time_s, calls)} from the top-down tree."""
    out = {}
    if not os.path.exists(path):
        return out
    in_topdown = False
    region = None
    with open(path) as f:
        for line in f:
            if "TOP-DOWN TIME TREE" in line:
                in_topdown = True
                continue
            if "BOTTOM-UP TIME TREE" in line:
                break
            if not in_topdown:
                continue
            tm = TIME_RE.search(line)
            tl = TAIL_RE.search(line)
            if not tm or not tl:
                continue
            time_s = float(tm.group(1))
            calls, name, kind = float(tl.group(1)), tl.group(2), tl.group(3)
            if kind == "region":
                region = name           # base_scalar_solve / oti_solve
            elif kind == "for" and region is not None:
                out[(region, name)] = (time_s, int(calls))
    return out


def main():
    rows = []
    for device, precision in CONFIGS:
        data = parse_report(os.path.join(RESULTS, f"prof_{device}_{precision}.txt"))
        for kernel in KERNELS:
            base = data.get(("base_scalar_solve", kernel))
            oti = data.get(("oti_solve", kernel))
            if not base or not oti:
                continue
            base_s, calls = base
            oti_s, _ = oti
            rows.append({
                "device": device, "precision": precision, "kernel": kernel,
                "base_s": base_s, "oti_s": oti_s,
                "oti_over_base": oti_s / base_s if base_s else float("nan"),
                "calls": calls,
            })

    if not rows:
        sys.exit("no reports parsed -- run profile_kernels.sh first")

    out_csv = os.path.join(RESULTS, "kernel_profile.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["device", "precision", "kernel",
                                          "base_s", "oti_s", "oti_over_base", "calls"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", out_csv, "\n")

    hdr = f"{'device':4} {'prec':6} {'kernel':22} {'base(s)':>9} {'oti(s)':>9} {'OTI/base':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['device']:4} {r['precision']:6} {r['kernel']:22} "
              f"{r['base_s']:9.3f} {r['oti_s']:9.3f} {r['oti_over_base']:8.2f}x")


if __name__ == "__main__":
    main()
