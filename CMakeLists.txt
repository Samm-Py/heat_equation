cmake_minimum_required(VERSION 3.16)
project(HeatEquation3D CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Kokkos REQUIRED)

set(CPP_OTI_LIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include" CACHE PATH
    "Path to cpp_oti_lib include directory")

# Clang caps fold-expression expansion at 256 arguments by default; the otinum
# product folds exceed that for larger shapes (the microbenchmark instantiates
# <4,4> with 495 product terms). Matches the flag on cpp_oti_lib's own CMake
# interface target, which this project bypasses by using a raw include path.
set(OTI_CLANG_FOLD_FLAGS $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fbracket-depth=65536>)

add_executable(heat_solver main.cpp)
target_link_libraries(heat_solver Kokkos::kokkos)

# Enable optimization
target_compile_options(heat_solver PRIVATE -O3)

function(add_oti_heat_analysis_target target coeff_type)
    add_executable(${target} oti_heat_analysis.cpp)
    target_include_directories(${target} PRIVATE ${CPP_OTI_LIB_INCLUDE_DIR})
    target_compile_definitions(${target} PRIVATE
        OTI_ENABLE_KOKKOS
        OTI_HEAT_COEFF_TYPE=${coeff_type})
    target_link_libraries(${target} Kokkos::kokkos)
    target_compile_options(${target} PRIVATE -O3 ${OTI_CLANG_FOLD_FLAGS})
endfunction()

add_oti_heat_analysis_target(oti_heat_analysis double)
add_oti_heat_analysis_target(oti_heat_analysis_double double)
add_oti_heat_analysis_target(oti_heat_analysis_float float)

# Coefficient-major (SoA) OTI storage variants for layout comparison on GPU
# backends; the default targets and the base scalar solve are unaffected.
add_oti_heat_analysis_target(oti_heat_analysis_soa_double double)
target_compile_definitions(oti_heat_analysis_soa_double PRIVATE OTI_HEAT_SOA)
add_oti_heat_analysis_target(oti_heat_analysis_soa_float float)
target_compile_definitions(oti_heat_analysis_soa_float PRIVATE OTI_HEAT_SOA)

# The cumulative-stage compile definitions for one variant. Shared by the
# single-shape (<3,1>) targets and the per-M shape-sweep targets.
function(oti_heat_apply_variant_defs target variant)
    if(variant STREQUAL "naive")
        target_compile_definitions(${target} PRIVATE
            OTI_BENCHMARK_ARITHMETIC_PATH=0
            OTI_BENCHMARK_NATURAL_ALIGNMENT
            OTI_HEAT_OPERATOR_CHAINS)
    elseif(variant STREQUAL "lookup")
        target_compile_definitions(${target} PRIVATE
            OTI_BENCHMARK_ARITHMETIC_PATH=1
            OTI_BENCHMARK_NATURAL_ALIGNMENT
            OTI_HEAT_OPERATOR_CHAINS)
    elseif(variant STREQUAL "unrolled")
        target_compile_definitions(${target} PRIVATE
            OTI_BENCHMARK_ARITHMETIC_PATH=2
            OTI_BENCHMARK_NATURAL_ALIGNMENT
            OTI_HEAT_OPERATOR_CHAINS)
    elseif(variant STREQUAL "aligned")
        target_compile_definitions(${target} PRIVATE
            OTI_BENCHMARK_ARITHMETIC_PATH=2
            OTI_HEAT_OPERATOR_CHAINS)
    elseif(variant STREQUAL "fused_aos")
        target_compile_definitions(${target} PRIVATE OTI_BENCHMARK_ARITHMETIC_PATH=2)
    elseif(variant STREQUAL "fused_soa")
        target_compile_definitions(${target} PRIVATE
            OTI_BENCHMARK_ARITHMETIC_PATH=2
            OTI_HEAT_SOA)
    else()
        message(FATAL_ERROR "Unknown OTI heat benchmark variant: ${variant}")
    endif()
endfunction()

function(add_oti_heat_benchmark_variant variant coeff_type)
    set(target "oti_heat_bench_${variant}_${coeff_type}")
    add_executable(${target} EXCLUDE_FROM_ALL oti_heat_analysis.cpp)
    target_include_directories(${target} PRIVATE ${CPP_OTI_LIB_INCLUDE_DIR})
    target_compile_definitions(${target} PRIVATE
        OTI_ENABLE_KOKKOS
        OTI_HEAT_COEFF_TYPE=${coeff_type}
        OTI_HEAT_VARIANT_NAME="${variant}")
    target_link_libraries(${target} Kokkos::kokkos)
    target_compile_options(${target} PRIVATE -O3 ${OTI_CLANG_FOLD_FLAGS})
    oti_heat_apply_variant_defs(${target} ${variant})
endfunction()

foreach(variant naive lookup unrolled aligned fused_aos fused_soa)
    add_oti_heat_benchmark_variant(${variant} double)
    add_oti_heat_benchmark_variant(${variant} float)
endforeach()

# Shape-sweep targets: the same six variants at a chosen OTI variable count
# (algebra <nvars, 1>). The analysis seeds three parameters; extra variables
# stay zero, so the three sensitivities are unchanged while the solve cost grows
# with nvars. Named oti_heat_bench_<variant>_<coeff>_n<nvars>.
function(add_oti_heat_sweep_variant variant coeff_type nvars)
    set(target "oti_heat_bench_${variant}_${coeff_type}_n${nvars}")
    add_executable(${target} EXCLUDE_FROM_ALL oti_heat_analysis.cpp)
    target_include_directories(${target} PRIVATE ${CPP_OTI_LIB_INCLUDE_DIR})
    target_compile_definitions(${target} PRIVATE
        OTI_ENABLE_KOKKOS
        OTI_HEAT_COEFF_TYPE=${coeff_type}
        OTI_HEAT_NVARS=${nvars}
        OTI_HEAT_VARIANT_NAME="${variant}")
    target_link_libraries(${target} Kokkos::kokkos)
    target_compile_options(${target} PRIVATE -O3 ${OTI_CLANG_FOLD_FLAGS})
    oti_heat_apply_variant_defs(${target} ${variant})
endfunction()

# M >= 3 (three parameters are always seeded). N = 1.
foreach(nvars 3 5 10 20 50 100)
    foreach(variant naive lookup unrolled aligned fused_aos fused_soa)
        add_oti_heat_sweep_variant(${variant} double ${nvars})
        add_oti_heat_sweep_variant(${variant} float ${nvars})
    endforeach()
endforeach()

# Device-side otinum arithmetic throughput microbenchmark. Runs on whatever
# Kokkos backend the build targets; point CPP_OTI_LIB_INCLUDE_DIR at different
# header trees to compare an otinum baseline against a candidate.
add_executable(oti_kernel_microbench benchmarks/oti_kernel_microbench.cpp)
target_include_directories(oti_kernel_microbench PRIVATE ${CPP_OTI_LIB_INCLUDE_DIR})
target_compile_definitions(oti_kernel_microbench PRIVATE OTI_ENABLE_KOKKOS)
target_link_libraries(oti_kernel_microbench Kokkos::kokkos)
target_compile_options(oti_kernel_microbench PRIVATE -O3 ${OTI_CLANG_FOLD_FLAGS})

# Streaming alignment/padding microbenchmark: natural vs the library's
# conditional alignment vs always-pad-to-16, in useful GB/s.
add_executable(oti_alignment_bench benchmarks/bench_alignment_padding.cpp)
target_include_directories(oti_alignment_bench PRIVATE ${CPP_OTI_LIB_INCLUDE_DIR})
target_compile_definitions(oti_alignment_bench PRIVATE OTI_ENABLE_KOKKOS)
target_link_libraries(oti_alignment_bench Kokkos::kokkos)
target_compile_options(oti_alignment_bench PRIVATE -O3 ${OTI_CLANG_FOLD_FLAGS})
