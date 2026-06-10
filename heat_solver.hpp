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

    template <class Scalar>
    KOKKOS_INLINE_FUNCTION
    void get_node_coords(int idx, Scalar& x, Scalar& y, Scalar& z) const {
        int i = idx % Nx;
        int j = (idx / Nx) % Ny;
        int k = idx / (Nx * Ny);
        x = static_cast<Scalar>(i) * static_cast<Scalar>(dx);
        y = static_cast<Scalar>(j) * static_cast<Scalar>(dy);
        z = static_cast<Scalar>(k) * static_cast<Scalar>(dz);
    }

    KOKKOS_INLINE_FUNCTION
    void get_element_nodes(int e_idx, int nodes[8]) const {
        int ex = e_idx % (Nx - 1);
        int ey = (e_idx / (Nx - 1)) % (Ny - 1);
        int ez = e_idx / ((Nx - 1) * (Ny - 1));

        int n0 = ex + ey * Nx + ez * Nx * Ny;
        nodes[0] = n0;                      // (0,0,0)
        nodes[1] = n0 + 1;                  // (1,0,0)
        nodes[2] = n0 + Nx;                 // (0,1,0)
        nodes[3] = n0 + 1 + Nx;             // (1,1,0)
        nodes[4] = n0 + Nx * Ny;            // (0,0,1)
        nodes[5] = n0 + 1 + Nx * Ny;        // (1,0,1)
        nodes[6] = n0 + Nx + Nx * Ny;       // (0,1,1)
        nodes[7] = n0 + 1 + Nx + Nx * Ny;   // (1,1,1)
    }
};

// Precompute local stiffness matrix for a unit cube element
// K_e = integral( grad(phi_i) . grad(phi_j) dV )
// For a uniform cube of size dx, dy, dz, the integration involves scaling.
template <class Scalar>
inline void compute_local_stiffness(Scalar dx, Scalar dy, Scalar dz, Scalar K_local[8][8]) {
    // 1D shape function derivatives integrated: 
    // int(phi'_i * phi'_j) dx
    Scalar k1d[2][2] = {{Scalar(1), Scalar(-1)}, {Scalar(-1), Scalar(1)}};
    // int(phi_i * phi_j) dx
    Scalar m1d[2][2] = {
        {Scalar(1) / Scalar(3), Scalar(1) / Scalar(6)},
        {Scalar(1) / Scalar(6), Scalar(1) / Scalar(3)},
    };

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int i_x = i & 1, i_y = (i >> 1) & 1, i_z = (i >> 2) & 1;
            int j_x = j & 1, j_y = (j >> 1) & 1, j_z = (j >> 2) & 1;

            // K_ij = int(dphi_i/dx * dphi_j/dx + dphi_i/dy * dphi_j/dy + dphi_i/dz * dphi_j/dz) dV
            // Using tensor product property of shape functions: phi(x,y,z) = N(x)N(y)N(z)
            Scalar term_x = (k1d[i_x][j_x] / dx) * (m1d[i_y][j_y] * dy) * (m1d[i_z][j_z] * dz);
            Scalar term_y = (m1d[i_x][j_x] * dx) * (k1d[i_y][j_y] / dy) * (m1d[i_z][j_z] * dz);
            Scalar term_z = (m1d[i_x][j_x] * dx) * (m1d[i_y][j_y] * dy) * (k1d[i_z][j_z] / dz);

            K_local[i][j] = term_x + term_y + term_z;
        }
    }
}

// Compute lumped mass matrix: diagonal elements are sum of rows of consistent mass matrix
template <class MView>
inline void compute_lumped_mass(const Mesh& mesh, MView M_lumped) {
    using Scalar = typename MView::non_const_value_type;
    Scalar vol_elem = static_cast<Scalar>(mesh.dx) * static_cast<Scalar>(mesh.dy) *
                      static_cast<Scalar>(mesh.dz);
    Scalar node_share = vol_elem / Scalar(8);

    Kokkos::parallel_for("ComputeLumpedMass", mesh.num_elements, KOKKOS_LAMBDA(int e_idx) {
        int nodes[8];
        mesh.get_element_nodes(e_idx, nodes);
        for (int i = 0; i < 8; ++i) {
            Kokkos::atomic_add(&M_lumped(nodes[i]), node_share);
        }
    });
}

// Matrix-free internal force computation: R = K * u.
//
// This is written as a node gather instead of an element scatter. The scatter
// form needs atomic adds into R, which is especially expensive for composite
// scalar types such as oti::otinum on CUDA.
template <class KView, class UView, class RView>
inline void compute_stiffness_force(const Mesh& mesh,
                                   KView K_local,
                                   UView u,
                                   RView R) {
    using Scalar = typename RView::non_const_value_type;
    Kokkos::parallel_for("ComputeStiffnessForce", mesh.num_nodes, KOKKOS_LAMBDA(int n_idx) {
        int ix = n_idx % mesh.Nx;
        int iy = (n_idx / mesh.Nx) % mesh.Ny;
        int iz = n_idx / (mesh.Nx * mesh.Ny);

        Scalar sum = 0.0;
        for (int oz = 0; oz <= 1; ++oz) {
            int ez = iz - oz;
            if (ez < 0 || ez >= mesh.Nz - 1) {
                continue;
            }
            for (int oy = 0; oy <= 1; ++oy) {
                int ey = iy - oy;
                if (ey < 0 || ey >= mesh.Ny - 1) {
                    continue;
                }
                for (int ox = 0; ox <= 1; ++ox) {
                    int ex = ix - ox;
                    if (ex < 0 || ex >= mesh.Nx - 1) {
                        continue;
                    }

                    int row = ox + 2 * oy + 4 * oz;
                    int elem_base = ex + ey * mesh.Nx + ez * mesh.Nx * mesh.Ny;
                    int nodes[8] = {
                        elem_base,
                        elem_base + 1,
                        elem_base + mesh.Nx,
                        elem_base + 1 + mesh.Nx,
                        elem_base + mesh.Nx * mesh.Ny,
                        elem_base + 1 + mesh.Nx * mesh.Ny,
                        elem_base + mesh.Nx + mesh.Nx * mesh.Ny,
                        elem_base + 1 + mesh.Nx + mesh.Nx * mesh.Ny,
                    };

                    for (int col = 0; col < 8; ++col) {
                        sum += K_local(row, col) * u(nodes[col]);
                    }
                }
            }
        }
        R(n_idx) = sum;
    });
}

#endif // HEAT_SOLVER_HPP
