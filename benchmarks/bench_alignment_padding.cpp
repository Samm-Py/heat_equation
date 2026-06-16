// Targeted alignment/padding microbenchmark for arrays of OTI coefficient
// blocks. It answers one question: when a jet's byte size is not a multiple of
// 16, is it worth PADDING it up to 16 (and aligning to 16) to get wide vector
// loads, or does the padding cost more bandwidth than the wider load saves?
//
// Three storage policies over the same NC-coefficient block and the same
// streaming kernel (y = a*x + y, one read of x, one read+write of y per
// element, arrays far larger than cache, so it is memory bound):
//
//   natural  - alignof(Coeff); no promotion, no padding.
//   aligned  - the current otinum rule: promote to 16/8 only when it divides
//              the byte size, so sizeof never grows (the library default).
//   padded   - alignas(16): forces 16-byte alignment, which rounds sizeof up
//              to a multiple of 16, i.e. adds padding bytes.
//
// The reported rate is USEFUL bandwidth = useful coefficient bytes / time, so
// padding shows up as a lower number (it moves dead bytes). aligned vs natural
// isolates load width at equal size; padded vs aligned is the padding trade.
// All three compute identical results, so the checksums must match.
//
// Requires OTI_ENABLE_KOKKOS; runs on whatever backend the build targets.

#include <Kokkos_Core.hpp>

#include "otinum/otinum.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

enum Policy { NATURAL, ALIGNED, PADDED };

char const* policy_name(Policy p)
{
    return p == NATURAL ? "natural" : p == ALIGNED ? "aligned" : "padded";
}

template <int NC, class Coeff, Policy P>
constexpr std::size_t jet_alignment()
{
    if (P == NATURAL) {
        return alignof(Coeff);
    } else if (P == ALIGNED) {
        return oti::detail::otinum_alignment<Coeff, NC>();  // the library's rule
    } else {
        return 16;  // force 16-byte alignment -> sizeof padded up to a multiple
    }
}

template <int NC, class Coeff, Policy P>
struct alignas(jet_alignment<NC, Coeff, P>()) jet {
    Coeff c[NC];
};

struct result {
    double useful_gbps;
    std::size_t bytes_per_jet;
    double checksum;
};

template <int NC, class Coeff, Policy P>
result bench(std::size_t n, int reps)
{
    using J = jet<NC, Coeff, P>;
    Kokkos::View<J*> x("x", n);
    Kokkos::View<J*> y("y", n);
    Kokkos::parallel_for("init", n, KOKKOS_LAMBDA(std::size_t i) {
        for (int k = 0; k < NC; ++k) {
            x(i).c[k] = static_cast<Coeff>(0.25 + 0.01 * ((i + k) % 13));
            y(i).c[k] = static_cast<Coeff>(0.50 + 0.01 * ((i + 2 * k) % 11));
        }
    });
    Kokkos::fence();

    Coeff const a = static_cast<Coeff>(1.5);
    auto pass = [&](char const* label) {
        Kokkos::parallel_for(label, n, KOKKOS_LAMBDA(std::size_t i) {
            J xi = x(i);
            J yi = y(i);
            for (int k = 0; k < NC; ++k) {
                yi.c[k] = a * xi.c[k] + yi.c[k];
            }
            y(i) = yi;
        });
    };

    pass("warmup");
    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        pass("timed");
    }
    Kokkos::fence();
    double const secs = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();

    double checksum = 0;
    Kokkos::parallel_reduce("checksum", n, KOKKOS_LAMBDA(std::size_t i, double& acc) {
        for (int k = 0; k < NC; ++k) {
            acc += static_cast<double>(y(i).c[k]);
        }
    }, checksum);

    // Useful bytes only (the NC coefficients), so padding lowers the rate.
    double const useful = 3.0 * double(NC) * double(sizeof(Coeff)) * double(n) * reps;
    return {useful / secs * 1e-9, sizeof(J), checksum};
}

template <int NC, class Coeff>
void compare(char const* tag, int reps)
{
    // Same element count for all three; size to ~96 MiB of useful data so every
    // pass misses cache. The padded array is physically larger (that is the
    // point) but holds the same n elements.
    std::size_t const n = (96u << 20) / (NC * sizeof(Coeff));

    result const nat = bench<NC, Coeff, NATURAL>(n, reps);
    result const ali = bench<NC, Coeff, ALIGNED>(n, reps);
    result const pad = bench<NC, Coeff, PADDED>(n, reps);

    std::size_t const align = oti::detail::otinum_alignment<Coeff, NC>();
    bool const match = nat.checksum == ali.checksum && ali.checksum == pad.checksum;
    std::printf("%-13s  size %4zu B  align %2zu   natural %6.1f  aligned %6.1f  "
                "padded %6.1f GB/s   pad->%zuB   pad/aligned %4.2fx   %s\n",
                tag, ali.bytes_per_jet, align,
                nat.useful_gbps, ali.useful_gbps, pad.useful_gbps,
                pad.bytes_per_jet, pad.useful_gbps / ali.useful_gbps,
                match ? "ok" : "MISMATCH");
    std::fflush(stdout);
    if (!match) {
        std::exit(1);
    }
}

} // namespace

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
        int const reps = (argc > 1) ? std::atoi(argv[1]) : 30;
        std::printf("backend: %s   (~96 MiB useful per array, %d timed passes)\n",
                    Kokkos::DefaultExecutionSpace::name(), reps);
        std::printf("pad/aligned > 1 means padding wins. align 16 rows are "
                    "controls (padding is a no-op).\n");

        // Sweep byte size finely for each precision. The near-miss shapes the
        // rule leaves at 8-byte alignment (double: odd coeff count; float: count
        // == 2 mod 4) are where padding to 16 could pay off; a few 16-aligned
        // controls bracket them.
        std::printf("\n== double (8 B coeff): odd count -> align 8 (near-miss) ==\n");
        compare<3, double>("d nc=3", reps);     // 24 B
        compare<5, double>("d nc=5", reps);     // 40
        compare<7, double>("d nc=7", reps);     // 56
        compare<9, double>("d nc=9", reps);     // 72
        compare<11, double>("d nc=11", reps);   // 88
        compare<13, double>("d nc=13", reps);   // 104
        compare<15, double>("d nc=15", reps);   // 120
        compare<17, double>("d nc=17", reps);   // 136
        compare<19, double>("d nc=19", reps);   // 152
        compare<21, double>("d nc=21", reps);   // 168
        compare<25, double>("d nc=25", reps);   // 200
        compare<31, double>("d nc=31", reps);   // 248
        compare<41, double>("d nc=41", reps);   // 328
        compare<51, double>("d nc=51", reps);   // 408
        compare<10, double>("d nc=10 (ctl)", reps);  // 80 B, align 16
        compare<20, double>("d nc=20 (ctl)", reps);  // 160 B, align 16

        std::printf("\n== float (4 B coeff): count == 2 mod 4 -> align 8 (near-miss) ==\n");
        compare<6, float>("f nc=6", reps);      // 24 B
        compare<10, float>("f nc=10", reps);    // 40
        compare<14, float>("f nc=14", reps);    // 56
        compare<18, float>("f nc=18", reps);    // 72
        compare<22, float>("f nc=22", reps);    // 88
        compare<26, float>("f nc=26", reps);    // 104
        compare<30, float>("f nc=30", reps);    // 120
        compare<34, float>("f nc=34", reps);    // 136
        compare<42, float>("f nc=42", reps);    // 168
        compare<50, float>("f nc=50", reps);    // 200
        compare<62, float>("f nc=62", reps);    // 248
        compare<82, float>("f nc=82", reps);    // 328
        compare<102, float>("f nc=102", reps);  // 408
        compare<20, float>("f nc=20 (ctl)", reps);   // 80 B, align 16
        compare<40, float>("f nc=40 (ctl)", reps);   // 160 B, align 16
    }
    Kokkos::finalize();
    return 0;
}
