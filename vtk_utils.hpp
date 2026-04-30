#ifndef VTK_UTILS_HPP
#define VTK_UTILS_HPP

#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>

/**
 * @brief Exports temperature data to a Legacy VTK file (STRUCTURED_POINTS).
 */
inline void export_vtk(const std::string& filename, 
                       int Nx, int Ny, int Nz, 
                       double dx, double dy, double dz,
                       Kokkos::View<double*>::host_mirror_type u_host) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }

    ofs << "# vtk DataFile Version 3.0" << std::endl;
    ofs << "Heat Equation 3D Temperature" << std::endl;
    ofs << "ASCII" << std::endl;
    ofs << "DATASET STRUCTURED_POINTS" << std::endl;
    ofs << "DIMENSIONS " << Nx << " " << Ny << " " << Nz << std::endl;
    ofs << "ORIGIN 0 0 0" << std::endl;
    ofs << "SPACING " << dx << " " << dy << " " << dz << std::endl;
    ofs << "POINT_DATA " << Nx * Ny * Nz << std::endl;
    ofs << "SCALARS Temperature double 1" << std::endl;
    ofs << "LOOKUP_TABLE default" << std::endl;

    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int idx = i + j * Nx + k * Nx * Ny;
                ofs << std::scientific << std::setprecision(6) << u_host(idx) << "\n";
            }
        }
    }

    ofs.close();
}

#endif // VTK_UTILS_HPP
