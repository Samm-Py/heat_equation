cmake_minimum_required(VERSION 3.16)
project(HeatEquation3D CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Kokkos REQUIRED)

set(CPP_OTI_LIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include" CACHE PATH
    "Path to cpp_oti_lib include directory")

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
    target_compile_options(${target} PRIVATE -O3)
endfunction()

add_oti_heat_analysis_target(oti_heat_analysis double)
add_oti_heat_analysis_target(oti_heat_analysis_double double)
add_oti_heat_analysis_target(oti_heat_analysis_float float)
