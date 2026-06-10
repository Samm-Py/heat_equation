// Device-side microbenchmark for otinum arithmetic throughput.
//
// Runs a Kokkos parallel_for in which each thread performs many otinum
// operations (multiply, divide, and exp+sin+log) in registers, then times the
// fenced kernel and reports effective nanoseconds per operation. This isolates
// the cost of the otinum kernels themselves, on whatever Kokkos backend the
// build targets (CUDA, OpenMP, ...), away from the memory-bound behavior of the
// full heat solver.
//
// It was written to validate the compile-time-unrolled product convolutions in
// cpp_oti_lib (operator*, inv/division, trunc_mul, and the Taylor composition
// behind the elementary functions). To compare a baseline against a candidate,
// build twice with CPP_OTI_LIB_INCLUDE_DIR pointing at the two header trees and
// diff the reported ns/op (the printed checksums must match -- the optimization
// is numerically identical). The win grows with the coefficient count C(M+N,N);
// first-order shapes (<3,1>) are essentially unchanged.
//
// Note: ns/op is amortized over all threads, so absolute values reflect the
// backend's parallelism -- only ratios between builds are meaningful.

#include <Kokkos_Core.hpp>

#include "otinum/otinum.hpp"

#include <chrono>
#include <cstdio>

namespace {

enum Op { MUL, DIV, FUNC };

template <class T, Op op>
void bench(char const* tag, int n_elem, int reps)
{
    Kokkos::View<double*> out("out", n_elem);
    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();
    Kokkos::parallel_for("oti_microbench", n_elem, KOKKOS_LAMBDA(int i) {
        T a, b;
        for (int k = 0; k < T::ncoeffs; ++k) {
            a[k] = static_cast<double>(0.30 + 0.05 * ((i + k) % 5));
            b[k] = static_cast<double>(0.20 + 0.04 * ((i + 2 * k) % 6));
        }
        a[0] = 1.5 + 1e-6 * i;   // nonzero real parts for div / log
        b[0] = 2.0 + 1e-6 * i;
        T acc{};
        for (int r = 0; r < reps; ++r) {
            a[0] += 1e-9;        // perturb to defeat hoisting / CSE
            if constexpr (op == MUL) {
                acc += a * b;
            } else if constexpr (op == DIV) {
                acc += a / b;
            } else {
                acc += oti::exp(a) + oti::sin(a) + oti::log(b);
            }
        }
        double s = 0;
        for (int k = 0; k < T::ncoeffs; ++k) {
            s += acc[k];
        }
        out(i) = s;
    });
    Kokkos::fence();
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    auto host = Kokkos::create_mirror_view(out);
    Kokkos::deep_copy(host, out);
    double total = 0;
    for (int i = 0; i < n_elem; ++i) {
        total += host(i);
    }
    std::printf("%-12s %12.2f ns/op  (checksum %.4e)\n",
                tag, secs * 1e9 / (double(n_elem) * reps), total);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
        int const ne = 16384;
        std::printf("backend: %s\n", Kokkos::DefaultExecutionSpace::name());

        std::printf("== MUL ==\n");
        bench<oti::otinum<3, 1, double>, MUL>("<3,1> mul", ne, 6104);
        bench<oti::otinum<3, 3, double>, MUL>("<3,3> mul", ne, 610);
        bench<oti::otinum<4, 4, double>, MUL>("<4,4> mul", ne, 15);
        std::printf("== DIV ==\n");
        bench<oti::otinum<3, 1, double>, DIV>("<3,1> div", ne, 6104);
        bench<oti::otinum<3, 3, double>, DIV>("<3,3> div", ne, 610);
        bench<oti::otinum<4, 4, double>, DIV>("<4,4> div", ne, 15);
        std::printf("== FUNC (exp+sin+log) ==\n");
        bench<oti::otinum<3, 1, double>, FUNC>("<3,1> func", ne, 6104);
        bench<oti::otinum<3, 3, double>, FUNC>("<3,3> func", ne, 610);
        bench<oti::otinum<4, 4, double>, FUNC>("<4,4> func", ne, 15);
    }
    Kokkos::finalize();
    return 0;
}
