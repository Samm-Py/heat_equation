#include <Kokkos_Core.hpp>
#include "heat_solver.hpp"
#include "vtk_utils.hpp"
#include <iostream>
#include <vector>
#include <string>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // Simulation Parameters
        const int N = 31; // Number of nodes per dimension
        const double L = 1.0; // Cube side length
        const double alpha = 1.0; // Thermal diffusivity
        const double total_time = 0.5;
        const double amplitude = 100.0;
        const double sigma = 0.05;
        const int vtk_interval = 100;

        Mesh mesh(N, N, N, L, L, L);
        
        // CFL stability condition for Forward Euler: dt <= dx^2 / (6 * alpha)
        double dx = mesh.dx;
        double dt = 0.8 * (dx * dx) / (6.0 * alpha);
        int num_steps = static_cast<int>(total_time / dt);
        
        std::cout << "Grid: " << N << "x" << N << "x" << N << std::endl;
        std::cout << "dx: " << dx << ", dt: " << dt << std::endl;
        std::cout << "Total steps: " << num_steps << std::endl;

        // Kokkos Views
        Kokkos::View<double*> u("temperature", mesh.num_nodes);
        Kokkos::View<double*> u_new("temperature_next", mesh.num_nodes);
        Kokkos::View<double*> M_lumped("mass_lumped", mesh.num_nodes);
        Kokkos::View<double*> f("source_vector", mesh.num_nodes);
        Kokkos::View<double*> Ku("stiffness_force", mesh.num_nodes);
        
        // Precompute local stiffness matrix on host then copy to device
        double h_K_local[8][8];
        compute_local_stiffness(mesh.dx, mesh.dy, mesh.dz, h_K_local);
        Kokkos::View<double[8][8], Kokkos::HostSpace> host_K("host_K");
        for(int i=0; i<8; ++i) for(int j=0; j<8; ++j) host_K(i,j) = h_K_local[i][j];
        auto device_K = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), host_K);

        // Initial conditions: u = 0
        Kokkos::deep_copy(u, 0.0);
        
        // Precompute lumped mass
        compute_lumped_mass(mesh, M_lumped);

        auto u_host = Kokkos::create_mirror_view(u);

        // Time-stepping loop (Forward Euler)
        for (int step = 0; step <= num_steps; ++step) {
            double t = step * dt;

            // Source position: moving in a circle in the XY plane at z = 1.0
            double xc = 0.5 + 0.3 * std::cos(2.0 * M_PI * t/total_time);
            double yc = 0.5 + 0.3 * std::sin(2.0 * M_PI * t/total_time);
            double zc = 1.0;

            // Compute f^n
            Kokkos::parallel_for("ComputeSource", mesh.num_nodes, KOKKOS_LAMBDA(int n_idx) {
                double x, y, z;
                mesh.get_node_coords(n_idx, x, y, z);
                double r2 = (x-xc)*(x-xc) + (y-yc)*(y-yc) + (z-zc)*(z-zc);
                double val = amplitude * std::exp(-r2 / (2.0 * sigma * sigma));
                
                // Approximate integral(q * phi_i) as q_i * M_i
                // This is consistent with lumped mass approximation
                f(n_idx) = val * M_lumped(n_idx);
            });

            // Compute K * u^n
            compute_stiffness_force(mesh, device_K, u, Ku);

            // Update: u^{n+1} = u^n + dt * M_L^{-1} * (f^n - alpha * Ku^n)
            Kokkos::parallel_for("UpdateTemperature", mesh.num_nodes, KOKKOS_LAMBDA(int n_idx) {
                u_new(n_idx) = u(n_idx) + dt * (1.0 / M_lumped(n_idx)) * (f(n_idx) - alpha * Ku(n_idx));
            });

            // Advance time
            Kokkos::deep_copy(u, u_new);

            // VTK Output
            if (step % vtk_interval == 0) {
                std::cout << "Step " << step << " / " << num_steps << " (t = " << t << ")" << std::endl;
                Kokkos::deep_copy(u_host, u);
                std::string filename = "output_" + std::to_string(step) + ".vtk";
                export_vtk(filename, N, N, N, mesh.dx, mesh.dy, mesh.dz, u_host);
            }
        }
    }
    Kokkos::finalize();
    return 0;
}
