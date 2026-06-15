#!/usr/bin/env python3
"""Collect repeated end-to-end heat-solver optimization measurements."""

import argparse
import csv
import datetime as dt
import json
import platform
import subprocess
from pathlib import Path


VARIANTS = ["naive", "lookup", "unrolled", "aligned", "fused_aos", "fused_soa"]


def parse_args():
    here = Path(__file__).resolve().parent
    root = here.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=root / "build-cuda")
    parser.add_argument("--output", type=Path, default=here / "results" / "optimization_study")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[41, 61])
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
    result = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def command_output(command):
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def checksums_match(reference, candidate, precision):
    rtol = 5.0e-5 if precision == "float" else 1.0e-10
    atol = 1.0e-8 if precision == "float" else 1.0e-12
    return len(reference) == len(candidate) and all(
        abs(a - b) <= atol + rtol * abs(a) for a, b in zip(reference, candidate))


def build_targets(build_dir, variants, precisions):
    for precision in precisions:
        for variant in variants:
            target = f"oti_heat_bench_{variant}_{precision}"
            subprocess.run(
                ["cmake", "--build", str(build_dir), "--target", target, "--parallel", "1"],
                check=True)


def main():
    args = parse_args()
    if args.runs < 1 or any(size < 2 for size in args.grid_sizes):
        raise SystemExit("--runs must be positive and grid sizes must be at least 2")
    if "naive" not in args.variants:
        raise SystemExit("--variants must include naive for checksum and speedup baselines")

    args.build_dir = args.build_dir.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.build:
        build_targets(args.build_dir, args.variants, args.precisions)

    rows = []
    for source_division in args.source_divisions:
        for precision in args.precisions:
            for grid_size in args.grid_sizes:
                for run_index in range(1, args.runs + 1):
                    variants = list(args.variants)
                    if run_index % 2 == 0:
                        variants.reverse()

                    group = []
                    for variant in variants:
                        binary = args.build_dir / f"oti_heat_bench_{variant}_{precision}"
                        if not binary.is_file():
                            raise SystemExit(f"missing benchmark binary: {binary}")

                        run_dir = (args.output / "runs" / source_division / precision /
                                   f"N{grid_size}" / f"run_{run_index:02d}" / variant)
                        run_dir.mkdir(parents=True, exist_ok=True)
                        command = [
                            str(binary), str(run_dir),
                            "--N", str(grid_size),
                            "--total-time", str(args.total_time),
                            "--snapshots", "1",
                            "--skip-fd",
                            "--source-division", source_division,
                        ]
                        result = subprocess.run(command, text=True, capture_output=True, check=False)
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
                        row = {
                            "run": run_index,
                            "variant": variant,
                            "precision": precision,
                            "source_division": source_division,
                            "backend": backend,
                            "N": grid_size,
                            "num_nodes": nodes,
                            "num_steps": steps,
                            "work": nodes * steps,
                            "base_seconds": timings["base_scalar_solve"],
                            "oti_seconds": timings["oti_solve"],
                            "oti_over_base": timings["oti_solve"] /
                                             timings["base_scalar_solve"],
                            "checksum": read_checksum(run_dir / "solution_checksum.csv"),
                        }
                        group.append(row)
                        print(
                            f"{source_division:8s} {precision:6s} N={grid_size:<3d} "
                            f"run={run_index:02d} {variant:10s} "
                            f"{row['oti_seconds']:.6f} s")

                    reference = next(row for row in group if row["variant"] == "naive")
                    reference_checksum = reference["checksum"]
                    by_variant = {row["variant"]: row for row in group}
                    for row in group:
                        previous_index = VARIANTS.index(row["variant"]) - 1
                        previous = (by_variant.get(VARIANTS[previous_index])
                                    if previous_index >= 0 else None)
                        row["speedup_vs_naive"] = (
                            reference["oti_seconds"] / row["oti_seconds"])
                        row["incremental_speedup"] = (
                            previous["oti_seconds"] / row["oti_seconds"]
                            if previous is not None else 1.0)
                        row["checksum_match"] = int(checksums_match(
                            reference_checksum, row["checksum"], precision))
                        del row["checksum"]
                        rows.append(row)

    columns = [
        "run", "variant", "precision", "source_division", "backend", "N",
        "num_nodes", "num_steps", "work", "base_seconds", "oti_seconds",
        "oti_over_base", "speedup_vs_naive", "incremental_speedup",
        "checksum_match",
    ]
    results_path = args.output / "heat_optimization_results.csv"
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
        "grid_sizes": args.grid_sizes,
        "total_time": args.total_time,
        "precisions": args.precisions,
        "source_divisions": args.source_divisions,
        "variants": args.variants,
        "heat_git_revision": git_value(heat_repo, "rev-parse", "HEAD"),
        "heat_git_status": git_value(heat_repo, "status", "--short"),
        "library_git_revision": git_value(library_repo, "rev-parse", "HEAD"),
        "library_git_status": git_value(library_repo, "status", "--short"),
        "nvidia_smi": command_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
