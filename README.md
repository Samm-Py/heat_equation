# Heat Equation

This code solves the heat equation on the unit cube. The heat source is a
gaussian that describes a circle. The code only depends on Kokkos. 

This program was mostly written by gemini.

## OTI Parameter Analysis

This clone includes an additional `oti_heat_analysis` executable that runs the
Kokkos heat solver with OTI-valued model parameters:

- thermal diffusivity `alpha`
- source `amplitude`
- source width `sigma`

The executable compares the OTI sensitivities against central finite
differences of the same discrete solver and writes slice data at several time
steps.

The analysis currently uses `oti::otinum<3,1>`: three active parameters and
first-order terms only. The reported OTI timings therefore measure first-order
sensitivity propagation, not higher-order OTI algebra.

Build against a Kokkos installation and the neighboring `cpp_oti_lib` headers:

```sh
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=g++-10 \
  -DCMAKE_PREFIX_PATH=/root/Research/kokkos-install

cmake --build build --parallel 2
```

Run the analysis:

```sh
./build/oti_heat_analysis oti_analysis_output
```

Higher-fidelity stress runs can increase the uniform mesh size and final time:

```sh
./build-cuda/oti_heat_analysis oti_analysis_output_cuda_N41 --N 41 --total-time 0.05
./build-cuda/oti_heat_analysis oti_analysis_output_cuda_N61 --N 61 --total-time 0.05
```

For performance-only stress tests, skip the six finite-difference validation
solves and compare only the base scalar solve against the OTI solve:

```sh
./build-cuda/oti_heat_analysis oti_analysis_output_cuda_N61_perf --N 61 --total-time 0.05 --skip-fd
```

The default `oti_heat_analysis` target uses double coefficients. The CMake build
also creates explicit precision targets for GPU comparisons:

```sh
./build-cuda/oti_heat_analysis_double oti_analysis_output_cuda_double_N61_perf --N 61 --total-time 0.05 --skip-fd
./build-cuda/oti_heat_analysis_float oti_analysis_output_cuda_float_N61_perf --N 61 --total-time 0.05 --skip-fd
```

Each run writes `coefficient_precision` to `run_config.csv` and records the
base scalar and OTI wall times in `timing_summary.csv`. It also writes the final
OTI coefficient sums to `solution_checksum.csv`.

## Library Optimization Study

The heat application is also the end-to-end benchmark for the OTI library
optimization sequence. CMake provides six cumulative variants in both float
and double precision:

1. `naive`: no product table, natural alignment, operator chains
2. `lookup`: runtime product lookup tables
3. `unrolled`: compile-time-unrolled tables
4. `aligned`: conditionally aligned AoS coefficients
5. `fused_aos`: fused timestep operations
6. `fused_soa`: coefficient-major storage

Collect repeated CUDA runs and save their native outputs:

```sh
python3 benchmarks/run_heat_optimization_benchmarks.py \
  --build \
  --build-dir build-cuda \
  --runs 5 \
  --grid-sizes 41 61 \
  --output ../benchmark_results/heat_optimization_gpu
```

The collector runs both the normal hoisted source denominator and a
mathematically equivalent `per-node` mode that exercises OTI division inside
the timed source kernel. It verifies final solution checksums against the naive
variant and writes `heat_optimization_results.csv`.

Generate the application-stage, cumulative-speedup, and incremental-speedup
figures:

```sh
python3 benchmarks/plot_heat_optimization_benchmarks.py \
  ../benchmark_results/heat_optimization_gpu
```

Generate plots:

```sh
python3 plot_oti_analysis.py oti_analysis_output
```

Outputs:

- `oti_analysis_output/slice_snapshots.csv`
- `oti_analysis_output/timing_summary.csv`
- `oti_analysis_output/plots/*.png`
