#!/usr/bin/env python3
"""Sweep the OTI algebra size (number of variables M, order N=1) over the heat
solver and record the optimization variant ladder at each shape.

The analysis seeds exactly three physical parameters into variables 0, 1, 2, so
for M >= 3 the three sensitivities are unchanged while the algebra -- and the
solve cost -- grows with M. Each shape <M, 1> is a real, end-to-end heat solve,
directly comparable to the single-shape optimization study.

Not every variant compiles at every shape: the unrolled/aligned/fused variants
fold the product table inline, which the compiler cannot do for the largest
shapes, so those rows are simply absent (naive and lookup, the runtime paths,
build everywhere). Builds are therefore attempted, not required.

Output: heat_shape_sweep_results.csv (one row per shape/variant/precision/
division/run) plus metadata.json.
"""

import argparse
import csv
import datetime as dt
import json
import math
import platform
import subprocess
from pathlib import Path


VARIANTS = ["naive", "lookup", "unrolled", "aligned", "fused_aos", "fused_soa"]


def parse_args():
    here = Path(__file__).resolve().parent
    root = here.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=root / "build-cuda")
    parser.add_argument("--output", type=Path, default=here / "results" / "shape_sweep")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--nvars", type=int, nargs="+",
                        default=[3, 5, 10, 20, 50, 100],
                        help="OTI variable counts M to sweep (each >= 3)")
    parser.add_argument("--order", type=int, default=1, help="OTI order N (fixed)")
    parser.add_argument("--grid", type=int, default=41, help="PDE spatial grid size")
    parser.add_argument("--total-time", type=float, default=0.01)
    parser.add_argument("--precisions", nargs="+", choices=["float", "double"],
                        default=["float", "double"])
    parser.add_argument("--source-divisions", nargs="+", choices=["hoisted", "per-node"],
                        default=["hoisted", "per-node"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--allow-non-cuda", action="store_true")
    return parser.parse_args()


def read_key_value(path):
    with path.open(newline="") as handle:
        return {row["key"]: row["value"] for row in csv.DictReader(handle)}


def read_timings(path):
    with path.open(newline="") as handle:
        return {row["phase"]: float(row["seconds"]) for row in csv.DictReader(handle)}


def read_checksum(path):
    with path.open(newline="") as handle:
        return [float(row["sum"]) for row in csv.DictReader(handle)]


def git_value(cwd, *args):
    result = subprocess.run(["git", *args], cwd=cwd, text=True,
                            capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def command_output(command):
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def checksums_match(reference, candidate, precision):
    rtol = 5.0e-5 if precision == "float" else 1.0e-10
    atol = 1.0e-8 if precision == "float" else 1.0e-12
    return len(reference) == len(candidate) and all(
        abs(a - b) <= atol + rtol * abs(a) for a, b in zip(reference, candidate))


def target_name(variant, precision, nvars):
    return f"oti_heat_bench_{variant}_{precision}_n{nvars}"


def try_build(build_dir, target):
    """Build one target. Returns True on success; non-fatal on failure (the
    variant simply will not be available at that shape)."""
    result = subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", target, "--parallel", "1"],
        text=True, capture_output=True, check=False)
    return result.returncode == 0 and (build_dir / target).is_file()


def main():
    args = parse_args()
    if args.runs < 1 or any(m < 3 for m in args.nvars):
        raise SystemExit("--runs must be positive and every --nvars must be >= 3")
    if "naive" not in args.variants:
        raise SystemExit("--variants must include naive for the speedup/checksum baseline")

    args.build_dir = args.build_dir.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    N = args.order

    # Determine which (variant, precision, M) binaries are available, building
    # them first if asked. The unrolled/aligned/fused variants may not compile
    # at the largest M; that is expected and recorded as an absent target.
    available = {}
    build_status = {}
    for precision in args.precisions:
        for nvars in args.nvars:
            for variant in args.variants:
                target = target_name(variant, precision, nvars)
                ok = try_build(args.build_dir, target) if args.build \
                    else (args.build_dir / target).is_file()
                available[(variant, precision, nvars)] = ok
                build_status[target] = "ok" if ok else "unavailable"
                if not ok and variant in ("naive", "lookup"):
                    print(f"WARNING: runtime-path target did not build: {target}")

    rows = []
    for source_division in args.source_divisions:
        for precision in args.precisions:
            for nvars in args.nvars:
                shape_variants = [v for v in args.variants
                                  if available[(v, precision, nvars)]]
                if "naive" not in shape_variants:
                    print(f"SKIP M={nvars} {precision}: no naive baseline binary")
                    continue
                ncoeffs = math.comb(nvars + N, N)
                nproducts = math.comb(2 * nvars + N, N)

                for run_index in range(1, args.runs + 1):
                    ordered = list(shape_variants)
                    if run_index % 2 == 0:
                        ordered.reverse()

                    group = []
                    for variant in ordered:
                        binary = args.build_dir / target_name(variant, precision, nvars)
                        run_dir = (args.output / "runs" / source_division / precision /
                                   f"M{nvars}" / f"run_{run_index:02d}" / variant)
                        run_dir.mkdir(parents=True, exist_ok=True)
                        command = [
                            str(binary), str(run_dir),
                            "--N", str(args.grid),
                            "--total-time", str(args.total_time),
                            "--snapshots", "1",
                            "--skip-fd",
                            "--source-division", source_division,
                        ]
                        result = subprocess.run(command, text=True,
                                                capture_output=True, check=False)
                        (run_dir / "run.log").write_text(result.stdout + result.stderr)
                        if result.returncode != 0:
                            raise SystemExit(
                                f"{binary.name} failed; see {run_dir / 'run.log'}")

                        config = read_key_value(run_dir / "run_config.csv")
                        timings = read_timings(run_dir / "timing_summary.csv")
                        backend = config["execution_backend"]
                        if backend != "Cuda" and not args.allow_non_cuda:
                            raise SystemExit(
                                f"expected CUDA backend, got {backend}; "
                                "use --allow-non-cuda to override")

                        nodes = int(config["num_nodes"])
                        steps = int(config["num_steps"])
                        group.append({
                            "run": run_index,
                            "variant": variant,
                            "precision": precision,
                            "source_division": source_division,
                            "backend": backend,
                            "M": nvars,
                            "N": N,
                            "ncoeffs": ncoeffs,
                            "nproducts": nproducts,
                            "grid": args.grid,
                            "num_nodes": nodes,
                            "num_steps": steps,
                            "work": nodes * steps,
                            "base_seconds": timings["base_scalar_solve"],
                            "oti_seconds": timings["oti_solve"],
                            "oti_over_base": timings["oti_solve"] /
                                             timings["base_scalar_solve"],
                            "checksum": read_checksum(run_dir / "solution_checksum.csv"),
                        })
                        print(f"{source_division:8s} {precision:6s} M={nvars:<4d} "
                              f"run={run_index:02d} {variant:10s} "
                              f"{group[-1]['oti_seconds']:.6f} s")

                    reference = next(r for r in group if r["variant"] == "naive")
                    reference_checksum = reference["checksum"]
                    by_variant = {r["variant"]: r for r in group}
                    for row in group:
                        idx = VARIANTS.index(row["variant"])
                        prev = None
                        for j in range(idx - 1, -1, -1):
                            if VARIANTS[j] in by_variant:
                                prev = by_variant[VARIANTS[j]]
                                break
                        row["speedup_vs_naive"] = reference["oti_seconds"] / row["oti_seconds"]
                        row["incremental_speedup"] = (
                            prev["oti_seconds"] / row["oti_seconds"] if prev else 1.0)
                        row["checksum_match"] = int(checksums_match(
                            reference_checksum, row["checksum"], precision))
                        del row["checksum"]
                        rows.append(row)

    columns = [
        "run", "variant", "precision", "source_division", "backend",
        "M", "N", "ncoeffs", "nproducts", "grid", "num_nodes", "num_steps", "work",
        "base_seconds", "oti_seconds", "oti_over_base",
        "speedup_vs_naive", "incremental_speedup", "checksum_match",
    ]
    results_path = args.output / "heat_shape_sweep_results.csv"
    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    heat_repo = Path(__file__).resolve().parents[1]
    library_repo = heat_repo.parent
    metadata = {
        "collected_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "build_dir": str(args.build_dir),
        "runs": args.runs,
        "nvars": args.nvars,
        "order": N,
        "grid": args.grid,
        "total_time": args.total_time,
        "precisions": args.precisions,
        "source_divisions": args.source_divisions,
        "variants": args.variants,
        "build_status": build_status,
        "heat_git_revision": git_value(heat_repo, "rev-parse", "HEAD"),
        "heat_git_status": git_value(heat_repo, "status", "--short"),
        "library_git_revision": git_value(library_repo, "rev-parse", "HEAD"),
        "library_git_status": git_value(library_repo, "status", "--short"),
        "nvidia_smi": command_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader"]),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
