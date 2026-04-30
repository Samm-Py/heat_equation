#ifndef HEAT_SOLVER_HPP
#define HEAT_SOLVER_HPP

#include <Kokkos_Core.hpp>
#include <cmath>

struct Mesh {
    int Nx, Ny, Nz;
    double dx, dy, dz;
    int num_nodes;
    int num_elements;

    Mesh(int nx, int ny, int nz, double lx, double ly, double lz)
        : Nx(nx), Ny(ny), Nz(nz), 
          dx(lx/(nx-1)), dy(ly/(ny-1)), dz(lz/(nz-1)) {
        num_nodes = Nx * Ny * Nz;
        num_elements = (Nx-1) * (Ny-1) * (Nz-1);
    }

    KOKKOS_INLINE_FUNCTION
    void get_node_coords(int idx, double& x, double& y, double& z) const {
        int i = idx % Nx;
        int j = (idx / Nx) % Ny;
        int k = idx / (Nx * Ny);
        x = i * dx;
        y = j * dy;
        z = k * dz;
    }

    KOKKOS_INLINE_FUNCTION
    void get_element_nodes(int e_idx, int nodes[8]) const {
        int ex = e_idx % (Nx - 1);
        int ey = (e_idx / (Nx - 1)) % (Ny - 1);
        int ez = e_idx / ((Nx - 1) * (Ny - 1));

        int n0 = ex + ey * Nx + ez * Nx * Ny;
        nodes[0] = n0;
        nodes[1] = n0 + 1;
        nodes[2] = n0 + 1 + Nx;
        nodes[3] = n0 + Nx;
        nodes[4] = n0 + Nx * Ny;
        nodes[5] = n0 + 1 + Nx * Ny;
        nodes[6] = n0 + 1 + Nx + Nx * Ny;
        nodes[7] = n0 + Nx + Nx * Ny;
    }
};

// Precompute local stiffness matrix for a unit cube element
// K_e = integral( grad(phi_i) . grad(phi_j) dV )
// For a uniform cube of size dx, dy, dz, the integration involves scaling.
inline void compute_local_stiffness(double dx, double dy, double dz, double K_local[8][8]) {
    // 1D shape function derivatives integrated: 
    // int(phi'_i * phi'_j) dx
    double k1d[2][2] = {{1.0, -1.0}, {-1.0, 1.0}};
    // int(phi_i * phi_j) dx
    double m1d[2][2] = {{1.0/3.0, 1.0/6.0}, {1.0/6.0, 1.0/3.0}};

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int i_x = i & 1, i_y = (i >> 1) & 1, i_z = (i >> 2) & 1;
            int j_x = j & 1, j_y = (j >> 1) & 1, j_z = (j >> 2) & 1;

            // K_ij = int(dphi_i/dx * dphi_j/dx + dphi_i/dy * dphi_j/dy + dphi_i/dz * dphi_j/dz) dV
            // Using tensor product property of shape functions: phi(x,y,z) = N(x)N(y)N(z)
            double term_x = (k1d[i_x][j_x] / dx) * (m1d[i_y][j_y] * dy) * (m1d[i_z][j_z] * dz);
            double term_y = (m1d[i_x][j_x] * dx) * (k1d[i_y][j_y] / dy) * (m1d[i_z][j_z] * dz);
            double term_z = (m1d[i_x][j_x] * dx) * (m1d[i_y][j_y] * dy) * (k1d[i_z][j_z] / dz);

            K_local[i][j] = term_x + term_y + term_z;
        }
    }
}

// Compute lumped mass matrix: diagonal elements are sum of rows of consistent mass matrix
inline void compute_lumped_mass(const Mesh& mesh, Kokkos::View<double*> M_lumped) {
    double vol_elem = mesh.dx * mesh.dy * mesh.dz;
    double node_share = vol_elem / 8.0;

    Kokkos::parallel_for("ComputeLumpedMass", mesh.num_elements, KOKKOS_LAMBDA(int e_idx) {
        int nodes[8];
        mesh.get_element_nodes(e_idx, nodes);
        for (int i = 0; i < 8; ++i) {
            Kokkos::atomic_add(&M_lumped(nodes[i]), node_share);
        }
    });
}

// Matrix-free internal force computation: R = K * u
inline void compute_stiffness_force(const Mesh& mesh, 
                                   Kokkos::View<const double[8][8]> K_local,
                                   Kokkos::View<const double*> u,
                                   Kokkos::View<double*> R) {
    Kokkos::deep_copy(R, 0.0);
    Kokkos::parallel_for("ComputeStiffnessForce", mesh.num_elements, KOKKOS_LAMBDA(int e_idx) {
        int nodes[8];
        mesh.get_element_nodes(e_idx, nodes);
        
        double u_elem[8];
        for (int i = 0; i < 8; ++i) u_elem[i] = u(nodes[i]);

        for (int i = 0; i < 8; ++i) {
            double val = 0.0;
            for (int j = 0; j < 8; ++j) {
                val += K_local(i, j) * u_elem[j];
            }
            Kokkos::atomic_add(&R(nodes[i]), val);
        }
    });
}

#endif // HEAT_SOLVER_HPP
