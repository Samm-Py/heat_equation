#!/usr/bin/env bash
#
# Per-kernel profiling of the OTI heat solver using the Kokkos Tools
# space-time-stack connector. Unlike the wall-clock benchmark, this attributes
# time to each named parallel_for (ComputeStiffnessForce / ComputeSource /
# UpdateTemperature), separated into the base_scalar_solve and oti_solve
# profiling regions, so the OTI overhead can be read per kernel.
#
# The connector fences around each kernel, so GPU (CUDA) times are true device
# kernel times, not async launch overhead (at a small, known perturbation).
#
# Requires the connector built from kokkos/kokkos-tools, e.g.:
#   make -C <kokkos-tools>/profiling/space-time-stack
# Point KP_TOOL at the resulting kp_space_time_stack.so (or set it in the env).
#
# Output: benchmarks/results/prof_<device>_<precision>.txt (raw reports)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
RESULTS_DIR="$HERE/results"
RUN_DIR="${RUN_DIR:-/tmp/oti_prof_runs}"

N="${1:-61}"
KP_TOOL="${KP_TOOL:-/root/Research/kokkos-tools/profiling/space-time-stack/kp_space_time_stack.so}"

if [ ! -f "$KP_TOOL" ]; then
    echo "Kokkos Tools connector not found at: $KP_TOOL" >&2
    echo "Build it: make -C <kokkos-tools>/profiling/space-time-stack" >&2
    exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"

mkdir -p "$RESULTS_DIR" "$RUN_DIR"

profile_one() { # device precision binary
    local device="$1" precision="$2" binary="$3"
    local report="$RESULTS_DIR/prof_${device}_${precision}.txt"
    echo "== profiling ${device} ${precision} (N=${N}) =="
    KOKKOS_TOOLS_LIBS="$KP_TOOL" "$binary" "$RUN_DIR/${device}_${precision}" \
        --N "$N" --skip-fd > "$report" 2>&1
    # Echo just the top-down region/kernel tree to the console.
    sed -n '/TOP-DOWN TIME TREE/,/BOTTOM-UP/p' "$report" | grep -E "\[region\]|\[for\]" || true
}

profile_one cpu double "$ROOT/build/oti_heat_analysis_double"
profile_one cpu float  "$ROOT/build/oti_heat_analysis_float"
profile_one gpu double "$ROOT/build-cuda/oti_heat_analysis_double"
profile_one gpu float  "$ROOT/build-cuda/oti_heat_analysis_float"

echo
echo "Raw reports in $RESULTS_DIR/prof_*.txt"
echo "Parse with: python3 $HERE/parse_kernel_profile.py $N"
