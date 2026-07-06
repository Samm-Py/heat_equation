cmake_minimum_required(VERSION 3.16)
project(HeatEquation3D CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Kokkos REQUIRED)

add_executable(heat_solver main.cpp)
target_link_libraries(heat_solver Kokkos::kokkos)

# Enable optimization
target_compile_options(heat_solver PRIVATE -O3)

# ---------------------------------------------------------------------------
# Optional: hypercomplex (OTI) heat solver with forward-mode sensitivities.
#
# Builds heat_solver_oti from heat_oti.cpp, which reuses the same templated
# heat_solver.hpp but instantiates it on an OTI (order-truncated imaginary)
# number so each temperature carries its exact derivatives w.r.t. the source
# parameters. Header-only; provided as a git submodule under external/.
#
#   cmake -S . -B build -DHEAT_ENABLE_AD=ON
#
# The OTI algebra shape is configurable; OTI_ORDER=0 reproduces the plain
# double solve bit-for-bit:
#   -DHEAT_OTI_NVARS=3   number of parameters carried (>= 3)
#   -DHEAT_OTI_ORDER=1   Taylor order (0 = double-equivalent, 1 = sensitivities)
# ---------------------------------------------------------------------------
option(HEAT_ENABLE_AD "Build the OTI automatic-differentiation heat solver" OFF)
set(HEAT_OTI_NVARS 3 CACHE STRING "OTI number of variables for heat_solver_oti")
set(HEAT_OTI_ORDER 1 CACHE STRING "OTI truncation order for heat_solver_oti (0 = double)")

if(HEAT_ENABLE_AD)
    set(CPP_OTI_LIB_INCLUDE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/external/cpp_oti_lib/include"
        CACHE PATH "Path to cpp_oti_lib include directory")
    if(NOT EXISTS "${CPP_OTI_LIB_INCLUDE_DIR}/otinum/otinum.hpp")
        message(FATAL_ERROR
            "cpp_oti_lib headers not found at ${CPP_OTI_LIB_INCLUDE_DIR}.\n"
            "Fetch the submodule first: git submodule update --init --recursive")
    endif()

    add_executable(heat_solver_oti heat_oti.cpp)
    target_include_directories(heat_solver_oti PRIVATE ${CPP_OTI_LIB_INCLUDE_DIR})
    target_link_libraries(heat_solver_oti Kokkos::kokkos)
    target_compile_definitions(heat_solver_oti PRIVATE
        OTI_ENABLE_KOKKOS
        OTI_NVARS=${HEAT_OTI_NVARS}
        OTI_ORDER=${HEAT_OTI_ORDER})
    # Clang caps fold-expression expansion at 256 arguments; the otinum product
    # folds can exceed that for larger shapes.
    target_compile_options(heat_solver_oti PRIVATE -O3
        $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fbracket-depth=65536>)
endif()
