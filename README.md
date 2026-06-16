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

Build against a Kokkos installation and the neighboring `cpp_oti_lib` headers
(the headers are expected at `../include`; override with
`-DCPP_OTI_LIB_INCLUDE_DIR=/path/to/cpp_oti_lib/include`). For a CPU
(Serial/OpenMP) Kokkos:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/kokkos-install
cmake --build build --parallel
```

### Building for a GPU

The optimization study and the algebra-size sweep run on CUDA, which needs a
CUDA-enabled Kokkos built for your GPU's compute architecture. See the
"Kokkos GPU" tutorial in the neighboring `cpp_oti_lib` for building one. Two
things commonly bite here:

- CUDA Kokkos requires C++20, so NVCC's host compiler must be **g++ 11 or
  newer** (g++ 10 emits `-std=c++2a`, which NVCC rejects).
- Configure with the Kokkos install's `nvcc_wrapper` as the C++ compiler and
  point its host compiler at g++ 11 with `NVCC_WRAPPER_DEFAULT_COMPILER`.

```sh
export NVCC_WRAPPER_DEFAULT_COMPILER=g++-11
cmake -S . -B build-cuda \
  -DCMAKE_CXX_COMPILER=/path/to/kokkos-cuda-install/bin/nvcc_wrapper \
  -DCMAKE_PREFIX_PATH=/path/to/kokkos-cuda-install
cmake --build build-cuda --target oti_heat_analysis --parallel
```

Confirm the backend really is CUDA before trusting any timing -- every run
writes `execution_backend` to `run_config.csv`:

```sh
./build-cuda/oti_heat_analysis /tmp/check --N 21 --total-time 0.005 --skip-fd
grep execution_backend /tmp/check/run_config.csv   # -> execution_backend,Cuda
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

## Algebra Size Sweep

The study above runs at one fixed shape, `otinum<3,1>`. The sweep answers a
different question: **how does each library optimization's benefit scale with
the size of the OTI algebra?** It runs the same heat sensitivity solve at a
range of algebra sizes and times the end-to-end GPU solve at each.

What it computes, and why the comparison is fair:

- The solver always seeds the same three physical parameters (into variables
  0, 1, 2), but the OTI variable count `M` is swept -- 3, 5, 10, 20, 50, 100 --
  through the `OTI_HEAT_NVARS` build setting (order is fixed at `N = 1`). Any
  variable beyond the three seeded ones stays zero, so the three sensitivities
  are **bit-identical at every `M`** while the algebra `<M,1>` -- and therefore
  the solve cost -- grows. The arithmetic is dense, so those extra coefficients
  cost real work: the sweep measures the price of a larger algebra in the real,
  memory-bound application, not in a synthetic kernel.
- At each size the solver is built as the six cumulative optimization variants
  (`naive` ... `fused_soa`, listed above) and the OTI solve wall time is
  recorded. The metric is end-to-end seconds and speedup vs. `naive`; the
  `M = 3` point is exactly the single-shape study's measurement.
- Each variant's final solution is checksum-validated against `naive` at the
  same size, so a regression surfaces as a checksum mismatch, not a silent
  wrong number.

Run it (it builds the targets it needs first; variants that do not compile at a
given size -- the inlined-table variants on the largest shapes -- are skipped,
not required):

```sh
python3 benchmarks/run_heat_shape_sweep.py \
  --build \
  --build-dir build-cuda \
  --runs 5 \
  --output benchmarks/results/shape_sweep
```

Useful flags: `--nvars 3 5 10 20 50 100` (the sizes), `--precisions double float`,
`--source-divisions hoisted per-node`, `--runs N`. This is a long run -- it
compiles up to six variants in two precisions at each size -- so start with
`--nvars 3 10` and `--runs 1` to smoke-test your setup.

It writes `heat_shape_sweep_results.csv` (one row per
size/variant/precision/division/run) and `metadata.json` (GPU, driver,
compiler, and git revisions, for reproducibility). Key columns:

- `M`, `ncoeffs`, `nproducts` -- the algebra size (the plot's x-axis).
- `variant`, `precision`, `source_division` -- the configuration.
- `oti_seconds` -- the measured end-to-end OTI solve time.
- `speedup_vs_naive` -- relative to `naive` at the same size.
- `checksum_match` -- 1 if the result matched `naive` (it must be 1).

Plot solve time and speedup-vs-naive against algebra size, one line per variant,
faceted by precision and source division:

```sh
python3 benchmarks/plot_heat_shape_sweep.py benchmarks/results/shape_sweep
```

Generate plots:

```sh
python3 plot_oti_analysis.py oti_analysis_output
```

Outputs:

- `oti_analysis_output/slice_snapshots.csv`
- `oti_analysis_output/timing_summary.csv`
- `oti_analysis_output/plots/*.png`
