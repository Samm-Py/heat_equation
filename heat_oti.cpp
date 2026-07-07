// Hypercomplex (OTI) heat solver: forward-mode sensitivity analysis.
//
// This is the original main.cpp solver with one change: the scalar type is an
// order-truncated imaginary (OTI) number instead of `double`. An OTI number
// oti::otinum<M, N> is a truncated multivariate Taylor polynomial, so every
// temperature value simultaneously carries its exact derivatives with respect
// to the seeded parameters -- no finite differences, no extra solves, and no
// kernel rewrite (the solver in heat_solver.hpp is templated on the scalar).
//
// The three seeded parameters are the thermal diffusivity (alpha), the source
// amplitude, and the source width (sigma). After one solve, u carries T along
// with dT/d(alpha), dT/d(amplitude) and dT/d(sigma) at every node.
//
// Recovering the plain double solve:
//   otinum<M, 0> stores only the real coefficient, so building with
//   -DOTI_ORDER=0 reproduces the original double result bit-for-bit (all
//   sensitivities are simply absent). OTI_ORDER=1 turns them on. This is the
//   sense in which double is just the zeroth-order case of the OTI algebra.

#include <Kokkos_Core.hpp>
#include "heat_solver.hpp"
#include "vtk_utils.hpp"
#include "otinum/otinum.hpp"

#include <iostream>
#include <string>

// Compile-time OTI shape, injected by CMake (HEAT_OTI_NVARS / HEAT_OTI_ORDER).
//   OTI_NVARS : number of parameters carried (must be >= the seeded count, 3).
//   OTI_ORDER : Taylor truncation order. 0 -> plain double solve; 1 -> first
//               derivatives (sensitivities); higher -> higher-order terms.
#if !defined(OTI_NVARS) || !defined(OTI_ORDER)
#error "OTI_NVARS and OTI_ORDER must be defined; configure with -DHEAT_OTI_NVARS / -DHEAT_OTI_ORDER"
#endif

using Scalar = oti::otinum<OTI_NVARS, OTI_ORDER>;

namespace {

// Parameter -> OTI variable index.
enum Param { P_ALPHA = 0, P_AMPLITUDE = 1, P_SIGMA = 2 };

// First-order sensitivity d(x)/d(parameter p) carried by an OTI value. Returns
// 0 when the algebra has no first-order slot (OTI_ORDER == 0).
double sensitivity(Scalar const& x, int p) {
    typename Scalar::alpha_type a{};
    a[p] = 1;
    return x.partial(a);
}

// Copy the real part of a Scalar host field into a plain double host buffer so
// the untouched double-only export_vtk can write it.
template <class HostView>
void extract_real(HostView const& src, Kokkos::View<double*, Kokkos::HostSpace> dst) {
    for (int i = 0; i < static_cast<int>(src.extent(0)); ++i) {
        dst(i) = src(i).real();
    }
}

} // namespace

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // Simulation Parameters
        const int N = 31; // Number of nodes per dimension
        const double L = 1.0; // Cube side length
        const double total_time = 0.5;
        const int vtk_interval = 100;

        // Physical parameters, seeded as the OTI variables. At OTI_ORDER == 0
        // variable() simply returns the value (no derivative slot).
        const Scalar alpha     = Scalar::variable(P_ALPHA, 1.0);   // Thermal diffusivity
        const Scalar amplitude = Scalar::variable(P_AMPLITUDE, 100.0);
        const Scalar sigma     = Scalar::variable(P_SIGMA, 0.05);

        Mesh mesh(N, N, N, L, L, L);

        // CFL stability condition for Forward Euler: dt <= dx^2 / (6 * alpha).
        double dx = mesh.dx;
        double dt = 0.8 * (dx * dx) / (6.0 * alpha.real());
        int num_steps = static_cast<int>(total_time / dt);

        std::cout << "Grid: " << N << "x" << N << "x" << N << std::endl;
        std::cout << "dx: " << dx << ", dt: " << dt << std::endl;
        std::cout << "Total steps: " << num_steps << std::endl;
        std::cout << "OTI shape: otinum<" << OTI_NVARS << ", " << OTI_ORDER << ">"
                  << (OTI_ORDER == 0 ? "  (double-equivalent)" : "  (sensitivities on)")
                  << std::endl;

        // Kokkos Views. The temperature u, source f and stiffness force Ku carry
        // the parameter sensitivities, so they use the OTI scalar. The lumped
        // mass depends only on the geometry, so it stays double (as upstream).
        Kokkos::View<Scalar*> u("temperature", mesh.num_nodes);
        Kokkos::View<Scalar*> u_new("temperature_next", mesh.num_nodes);
        Kokkos::View<double*> M_lumped("mass_lumped", mesh.num_nodes);
        Kokkos::View<Scalar*> f("source_vector", mesh.num_nodes);
        Kokkos::View<Scalar*> Ku("stiffness_force", mesh.num_nodes);

        // Precompute local stiffness matrix on host then copy to device. The
        // stiffness matrix depends only on the geometry, so it stays double.
        double h_K_local[8][8];
        compute_local_stiffness(mesh.dx, mesh.dy, mesh.dz, h_K_local);
        Kokkos::View<double[8][8], Kokkos::HostSpace> host_K("host_K");
        for(int i=0; i<8; ++i) for(int j=0; j<8; ++j) host_K(i,j) = h_K_local[i][j];
        auto device_K = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), host_K);

        // Initial conditions: u = 0
        Kokkos::deep_copy(u, Scalar(0.0));

        // Precompute lumped mass
        compute_lumped_mass(mesh, M_lumped);

        auto u_host = Kokkos::create_mirror_view(u);
        // Plain-double buffer used for all VTK export (real part / sensitivities).
        Kokkos::View<double*, Kokkos::HostSpace> out_host("out_host", mesh.num_nodes);

        // Time-stepping loop (Forward Euler)
        for (int step = 0; step <= num_steps; ++step) {
            double t = step * dt;

            // VTK Output: Output current state u^n at time t
            if (step % vtk_interval == 0) {
                std::cout << "Step " << step << " / " << num_steps << " (t = " << t << ")" << std::endl;
                Kokkos::deep_copy(u_host, u);

                extract_real(u_host, out_host);
                export_vtk("output_" + std::to_string(step) + ".vtk",
                           N, N, N, mesh.dx, mesh.dy, mesh.dz, out_host);

                // Also export the sensitivity fields recovered from the OTI
                // coefficients. Skipped at order 0, where they are identically 0.
                if constexpr (OTI_ORDER > 0) {
                    const char* names[3] = {"dTdalpha", "dTdamplitude", "dTdsigma"};
                    for (int p = 0; p < 3; ++p) {
                        for (int i = 0; i < mesh.num_nodes; ++i) {
                            out_host(i) = sensitivity(u_host(i), p);
                        }
                        export_vtk(std::string(names[p]) + "_" + std::to_string(step) + ".vtk",
                                   N, N, N, mesh.dx, mesh.dy, mesh.dz, out_host);
                    }
                }
            }

            if (step == num_steps) break;

            // Source position: moving in a circle in the XY plane at z = 1.0
            double xc = 0.5 + 0.3 * std::cos(2.0 * M_PI * t/total_time);
            double yc = 0.5 + 0.3 * std::sin(2.0 * M_PI * t/total_time);
            double zc = 1.0;

            // Compute f^n. Position and radius are geometry (double); amplitude
            // and sigma are OTI parameters, so f carries their sensitivities.
            Kokkos::parallel_for("ComputeSource", mesh.num_nodes, KOKKOS_LAMBDA(int n_idx) {
                double x, y, z;
                mesh.get_node_coords(n_idx, x, y, z);
                double r2 = (x-xc)*(x-xc) + (y-yc)*(y-yc) + (z-zc)*(z-zc);
                Scalar val = amplitude * oti::exp(Scalar(-r2) / (Scalar(2.0) * sigma * sigma));

                // Approximate integral(q * phi_i) as q_i * M_i
                // This is consistent with lumped mass approximation
                f(n_idx) = val * M_lumped(n_idx);
            });

            // Compute K * u^n
            compute_stiffness_force(mesh, device_K, u, Ku);

            // Update: u^{n+1} = u^n + dt * M_L^{-1} * (f^n - alpha * Ku^n)
            Kokkos::parallel_for("UpdateTemperature", mesh.num_nodes, KOKKOS_LAMBDA(int n_idx) {
                u_new(n_idx) = u(n_idx) + dt * (Scalar(1.0) / M_lumped(n_idx)) * (f(n_idx) - alpha * Ku(n_idx));
            });

            // Advance time
            Kokkos::deep_copy(u, u_new);
        }

        // Report the peak temperature and its exact sensitivities to the three
        // parameters -- all recovered from the single OTI solve.
        Kokkos::deep_copy(u_host, u);
        int peak = 0;
        for (int i = 1; i < mesh.num_nodes; ++i) {
            if (u_host(i).real() > u_host(peak).real()) peak = i;
        }
        std::cout << "\nPeak temperature: " << u_host(peak).real()
                  << " at node " << peak << std::endl;
        if constexpr (OTI_ORDER > 0) {
            std::cout << "  d(T_peak)/d(alpha)     = " << sensitivity(u_host(peak), P_ALPHA) << std::endl;
            std::cout << "  d(T_peak)/d(amplitude) = " << sensitivity(u_host(peak), P_AMPLITUDE) << std::endl;
            std::cout << "  d(T_peak)/d(sigma)     = " << sensitivity(u_host(peak), P_SIGMA) << std::endl;
        }
    }
    Kokkos::finalize();
    return 0;
}
