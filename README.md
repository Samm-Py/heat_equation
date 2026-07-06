# Heat Equation

This code solves the heat equation on the unit cube. The heat source is a
gaussian that describes a circle. The code only depends on Kokkos. 

This program was mostly written by gemini.

## Building

```sh
cmake -S . -B build -DKokkos_ROOT=/path/to/kokkos
cmake --build build
./build/heat_solver
```

## Optional: forward-mode sensitivities (OTI automatic differentiation)

`heat_oti.cpp` is the same solver instantiated on an *order-truncated imaginary*
(OTI) number instead of `double`, using the header-only
[cpp_oti_lib](https://github.com/Samm-Py/cpp_oti_lib) (vendored as a submodule
under `external/`). An `otinum<M, N>` is a truncated multivariate Taylor
polynomial, so each temperature value carries its exact derivatives with respect
to the seeded parameters — the thermal diffusivity, the source amplitude and the
source width — from a **single** solve, with no finite differencing and no change
to the kernels (`heat_solver.hpp` is templated on the scalar type).

```sh
git submodule update --init --recursive
cmake -S . -B build -DKokkos_ROOT=/path/to/kokkos -DHEAT_ENABLE_AD=ON
cmake --build build
./build/heat_solver_oti      # writes output_*.vtk plus dTdalpha/dTdamplitude/dTdsigma_*.vtk
```

The OTI algebra shape is configurable. Setting the truncation order to zero makes
`otinum<M, 0>` store only the real coefficient, so it reproduces the plain
`double` solve bit-for-bit — `double` is just the zeroth-order case of the OTI
algebra:

```sh
cmake -S . -B build -DHEAT_ENABLE_AD=ON -DHEAT_OTI_ORDER=0   # double-equivalent
cmake -S . -B build -DHEAT_ENABLE_AD=ON -DHEAT_OTI_ORDER=1   # first-order sensitivities (default)
```
