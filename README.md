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
base scalar and OTI wall times in `timing_summary.csv`.

Generate plots:

```sh
python3 plot_oti_analysis.py oti_analysis_output
```

Outputs:

- `oti_analysis_output/slice_snapshots.csv`
- `oti_analysis_output/timing_summary.csv`
- `oti_analysis_output/plots/*.png`
