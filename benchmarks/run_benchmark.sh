#!/usr/bin/env bash
#
# Sweep the OTI heat-equation solver over grid sizes for four configurations:
#   CPU (Kokkos OpenMP)  x  {double, float}
#   GPU (Kokkos CUDA)    x  {double, float}
#
# Each run uses --skip-fd so only the base scalar solve and the OTI solve are
# timed. The physical total_time is held fixed, so dt is CFL-bound and the step
# count grows ~N^2; we record num_nodes and num_steps and let the analysis plot
# wall time against the actual work performed (node-updates = num_nodes*num_steps).
#
# Output: benchmarks/results/benchmark_results.csv (tidy, one row per rep).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
RESULTS_DIR="$HERE/results"
RUN_DIR="${RUN_DIR:-/tmp/oti_bench_runs}"
CSV="${CSV_OUT:-$RESULTS_DIR/benchmark_results.csv}"

CPU_BIN_PREFIX="$ROOT/build/oti_heat_analysis"
GPU_BIN_PREFIX="$ROOT/build-cuda/oti_heat_analysis"

# SWEEP_DEVICES selects which halves to run ("cpu gpu" by default).
# GPU_VARIANT selects an alternative GPU OTI binary suffix, e.g. "_soa" for
# the coefficient-major storage variants; the CPU binaries have no variants.
SWEEP_DEVICES="${SWEEP_DEVICES:-cpu gpu}"
GPU_VARIANT="${GPU_VARIANT:-}"

# CPU thread pinning: use the 8 physical cores, avoid SMT contention noise.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"

# Grid sweeps. CPU is capped lower because fixed-total-time work grows ~N^5.
CPU_NS=(21 31 41 51 61 71 81 91 101)
GPU_NS=(21 31 41 51 61 71 81 101 121 151)

# Repetitions: more for small/noisy grids, fewer for the long runs.
reps_for_N() {
    local n="$1"
    if   [ "$n" -le 41 ]; then echo 3
    elif [ "$n" -le 71 ]; then echo 2
    else echo 1
    fi
}

mkdir -p "$RESULTS_DIR" "$RUN_DIR"
echo "device,precision,N,num_nodes,num_steps,work,rep,base_scalar_solve,oti_solve,oti_ratio" > "$CSV"

get_cfg() { # file key  -> value
    grep -E "^$2," "$1" | head -1 | cut -d, -f2
}
get_timing() { # file phase -> seconds
    grep -E "^$2," "$1" | head -1 | cut -d, -f2
}

run_one() { # device precision binary N rep
    local device="$1" precision="$2" binary="$3" n="$4" rep="$5"
    local out="$RUN_DIR/${device}_${precision}_N${n}_r${rep}"
    rm -rf "$out"
    "$binary" "$out" --N "$n" --skip-fd > "$out.log" 2>&1 || {
        echo "  !! FAILED ${device} ${precision} N=${n} rep=${rep} (see $out.log)" >&2
        return 1
    }
    local cfg="$out/run_config.csv" tim="$out/timing_summary.csv"
    local nodes steps base oti work ratio
    nodes="$(get_cfg "$cfg" num_nodes)"
    steps="$(get_cfg "$cfg" num_steps)"
    base="$(get_timing "$tim" base_scalar_solve)"
    oti="$(get_timing "$tim" oti_solve)"
    work=$(( nodes * steps ))
    ratio="$(awk -v a="$oti" -v b="$base" 'BEGIN{ printf (b>0)? "%.6f" : "nan", a/b }')"
    echo "${device},${precision},${n},${nodes},${steps},${work},${rep},${base},${oti},${ratio}" >> "$CSV"
    printf "  %-4s %-6s N=%-4s rep=%s  nodes=%-8s steps=%-5s  base=%8.4fs  oti=%8.4fs  (%sx)\n" \
        "$device" "$precision" "$n" "$rep" "$nodes" "$steps" "$base" "$oti" "$ratio"
}

sweep() { # device precision binary  N...
    local device="$1" precision="$2" binary="$3"; shift 3
    echo "== sweeping ${device} ${precision} =="
    for n in "$@"; do
        local r reps; reps="$(reps_for_N "$n")"
        for r in $(seq 1 "$reps"); do
            run_one "$device" "$precision" "$binary" "$n" "$r" || true
        done
    done
}

case " $SWEEP_DEVICES " in *" cpu "*)
    sweep cpu double "${CPU_BIN_PREFIX}_double" "${CPU_NS[@]}"
    sweep cpu float  "${CPU_BIN_PREFIX}_float"  "${CPU_NS[@]}"
;; esac
case " $SWEEP_DEVICES " in *" gpu "*)
    sweep gpu double "${GPU_BIN_PREFIX}${GPU_VARIANT}_double" "${GPU_NS[@]}"
    sweep gpu float  "${GPU_BIN_PREFIX}${GPU_VARIANT}_float"  "${GPU_NS[@]}"
;; esac

echo
echo "Done. Wrote $CSV"
