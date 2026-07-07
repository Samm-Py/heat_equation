# Heat Equation

This code solves the heat equation on the unit cube. The heat source is a
gaussian that describes a circle. The code only depends on Kokkos. 

This program was mostly written by gemini.

## Prerequisites

The base heat solver depends only on Kokkos. The OTI sensitivity analysis and
the optimization/sweep benchmarks additionally need:

- **`cpp_oti_lib`** -- the header-only OTI library this project is built to
  exercise. Clone it and place it beside this repository (so its headers resolve
  at `../include`), or pass `-DCPP_OTI_LIB_INCLUDE_DIR=/path/to/cpp_oti_lib/include`.
- **Kokkos** -- a Serial/OpenMP build for CPU runs, or a **CUDA-enabled build
  compiled for your GPU's compute architecture** for the GPU studies. The
  "Kokkos GPU" tutorial in `cpp_oti_lib` walks through building one (CUDA
  toolkit, host compiler, and arch flag).
- **CMake 3.16+** and a C++17 compiler -- g++ 11 or newer if you build for CUDA,
  since Kokkos's CUDA backend needs C++20 on the NVCC host pass.
- **Python 3** with `matplotlib`, for the benchmark runners and plotters.

## Running the studies

This repository is both a standalone heat solver and the **end-to-end benchmark
for the `cpp_oti_lib` optimization sequence**. The per-optimization *isolation*
benchmarks live in `cpp_oti_lib` (its "GPU Optimization Benchmark Workflow"
docs); this solve is where those optimizations are *stacked* in a real
application (the "Heat Equation: Optimizations Stacked End-to-End" page there).

There are three studies, each detailed in its own section below. After a CUDA
Kokkos build, the full workflow is:

```sh
# 0. Build (CUDA Kokkos -- see "Building for a GPU" for the toolchain notes)
export NVCC_WRAPPER_DEFAULT_COMPILER=g++-11
cmake -S . -B build-cuda \
  -DCMAKE_CXX_COMPILER=/path/to/kokkos-cuda-install/bin/nvcc_wrapper \
  -DCMAKE_PREFIX_PATH=/path/to/kokkos-cuda-install
cmake --build build-cuda --parallel

# 1. OTI sensitivity analysis: OTI derivatives vs central finite differences
./build-cuda/oti_heat_analysis oti_analysis_output_cuda_N41 --N 41 --total-time 0.05
python3 plot_oti_analysis.py oti_analysis_output_cuda_N41

# 2. Optimization study: the six cumulative library optimizations, end-to-end
python3 benchmarks/run_heat_optimization_benchmarks.py --build \
  --build-dir build-cuda --runs 5 --grid-sizes 41 \
  --output ../benchmark_results/heat_optimization_gpu
python3 benchmarks/plot_heat_optimization_benchmarks.py \
  ../benchmark_results/heat_optimization_gpu

# 3. Algebra-size sweep: how each optimization's benefit scales with algebra size
python3 benchmarks/run_heat_shape_sweep.py --build \
  --build-dir build-cuda --runs 5 --output benchmarks/results/shape_sweep
python3 benchmarks/plot_heat_shape_sweep.py benchmarks/results/shape_sweep

# 4. Scaling sweep: OTI/base overhead vs problem size (and why it grows)
benchmarks/run_benchmark.sh
python3 benchmarks/plot_benchmark.py     # -> oti_overhead.png, oti_overhead_saturation.png

# 5. UQ of the max temperature: one otinum<3,2> solve vs a Monte Carlo reference
./build-cuda/uq_max_temperature --N 41 --mc-samples 40000 --output uq_output
python3 uq_moment_analysis.py uq_output   # -> uq_output/uq_results.txt + figure

# 6. Adaptive surrogate reuse: validity gate over a drifting diffusivity
./build-cuda/uq_adaptive_reuse --output reuse_output
python3 plot_adaptive_reuse.py reuse_output figures/adaptive_reuse.png
```

A CPU (Serial/OpenMP) Kokkos build works for studies 0 and 1; the optimization
study and the sweep are written for CUDA. Each runner takes `--runs N` and
pools the runs, and writes a `metadata.json` recording the GPU, driver, and both
git revisions for reproducibility.

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

### Testing a specific optimization

`--variants` runs any subset of the six stages instead of the whole chain (and
`--precisions` / `--source-divisions` narrow further). Each variant is its own
CMake target `oti_heat_bench_<variant>_<precision>`, runnable directly with the
analysis CLI:

```sh
# just the aligned stage, float, end-to-end
python3 benchmarks/run_heat_optimization_benchmarks.py --build \
  --build-dir build-cuda --runs 5 --grid-sizes 41 \
  --variants aligned --precisions float \
  --output ../benchmark_results/heat_aligned_only
```

Note the stages are **cumulative**: `aligned` = product tables + unrolling +
alignment, not alignment alone. To attribute one optimization's contribution,
run adjacent stages and read the ratio -- e.g. `--variants unrolled aligned`
isolates alignment's marginal effect, which is exactly the `incremental_speedup`
column. For a single optimization flipped in true isolation (everything else
fixed), use the `bench_*` suite in `cpp_oti_lib` instead: this study is the
*stacked, end-to-end* view, those are the *isolated per-kernel* view.

## Scaling Sweep and OTI Overhead

This sweep answers a different question: **how does the OTI solve's overhead
scale with problem size?** It runs the base scalar solve and the
`otinum<3,1>` OTI solve over a range of grid sizes for all four
device x precision configurations, using `--skip-fd` so only the two solves are
timed. The x-axis is total node-updates (`num_nodes * num_steps`); the physical
`total_time` is fixed and `dt` is CFL-bound, so larger grids do more work.

```sh
benchmarks/run_benchmark.sh                 # writes results/benchmark_results.csv
python3 benchmarks/plot_benchmark.py        # medians + the figures below
```

`plot_benchmark.py` writes `results/benchmark_median.csv` and three figures:
`wall_vs_complexity.png` (OTI solve time), `oti_overhead.png` (the OTI/base
ratio), and `oti_overhead_saturation.png` (per-node-update time for base vs
OTI). The overhead ratio *rises* with problem size and plateaus (~3.2x double,
~2.8x float on a GTX 1650). The saturation plot shows why: it is the base scalar
solve being under-utilized at small sizes -- its per-node-update time falls
steeply as the GPU saturates while the 4x-heavier OTI solve saturates earlier
and is much flatter, so the ratio grows because the *base* denominator shrinks,
not because OTI gets more expensive. It plateaus once both are saturated.

To confirm the plateau is device memory bandwidth (not host-device copies), the
per-kernel profile shows no `cudaMemcpy` in the timed loop and concentrates the
overhead in the memory-bound stencil gather (`ComputeStiffnessForce`):

```sh
benchmarks/profile_kernels.sh 61
python3 benchmarks/parse_kernel_profile.py 61   # writes results/kernel_profile.csv
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

## UQ and Certified Surrogate Reuse

Two examples reuse the sensitivity jet from a single solve as a Taylor
surrogate; both are documented as worked pages in the `cpp_oti_lib`
"Numerical Examples" docs section.

**`uq_max_temperature`** propagates input uncertainty through the solve
analytically: the three physical parameters are given independent normal
distributions (5% CoV by default; `--cov`), one `otinum<3,2>` solve yields a
second-order expansion of the peak temperature, and
`uq_moment_analysis.py` integrates it with Gauss--Hermite quadrature --
mean, standard deviation, skewness, kurtosis, and a KL divergence against the
brute-force reference. The Monte Carlo reference (`--mc-samples`, genuine
re-solves) exists only to validate the method; skewness is the discriminating
moment (a first-order expansion reports exactly zero).

**`uq_adaptive_reuse`** is the digital-twin pattern: a drifting diffusivity is
served by anchor jets, and `oti::validity::is_trusted` decides per query
whether the linear surrogate is still certified within `--tau` (default 0.02)
or a fresh anchor solve is needed. A dense truth sweep audits the gate (max
reuse error vs budget, false-positive count); it is not part of the method
cost. Defaults: 200 queries across alpha in [0.7, 1.6] -> 4 solves,
0/197 false positives. `plot_adaptive_reuse.py <output> <figure.png>` renders
the sweep and the error-vs-budget panel.

Both run on any Kokkos backend; CUDA is not required.

**`uq_gp_bank`** exports the *jet bank* behind the GP digital-twin study (the
"Digital Twin II" page of the `cpp_oti_lib` docs, built on
[JetGP](https://github.com/Samm-Py/jetgp)): a nested Halton anchor set with
one full `otinum<3,2>` jet per point (value, gradient, and complete Hessian as
true derivatives), a uniform Monte Carlo truth set over the (alpha, A, sigma)
box, and a closed drifting-parameter query path (traversed 1.5x, so late
queries revisit early territory) with both truth values and jets. Everything a
derivative-enhanced GP needs to be trained and audited, precomputed so the GP
experiments reproduce offline without a PDE solver in the loop:

```sh
./build/uq_gp_bank --output gp_bank   # --N --anchors --mc --queries
```
