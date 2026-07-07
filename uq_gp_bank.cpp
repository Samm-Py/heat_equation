// Export a "jet bank" for the GP digital-twin study: everything a
// derivative-enhanced Gaussian process (JetGP) needs to be trained and audited
// over the full 3-D parameter box (alpha, A, sigma), precomputed so the GP
// experiments are reproducible offline with no PDE solver in the loop.
//
// Three data sets, one shared sensor QoI (final-time temperature at the peak
// node of the nominal solve, same convention as uq_adaptive_reuse):
//
//   bank_anchors.csv      Halton space-filling points in the box (nested: any
//                         prefix of the sequence is itself space-filling, so a
//                         convergence study can grow the training set by
//                         taking the first k rows). One otinum<3,2> solve per
//                         point exports the complete jet AS DERIVATIVES
//                         (partial() applies the factorial): value, gradient,
//                         and full Hessian including mixed terms -- 10 numbers.
//   bank_mc_truth.csv     Uniform Monte Carlo points in the box with the true
//                         (plain double) QoI: the validation set for measuring
//                         surrogate error.
//   bank_drift_truth.csv  An operational drift path -- a closed curve through
//                         the box traversed 1.5 times, so the second half
//                         REVISITS earlier territory -- with the true QoI at
//                         each of the 200 queries: the twin-loop scenario.
//
// dt is sized for explicit stability at the largest alpha in the box and held
// fixed, so every solve shares one discretization and the QoI is a smooth
// function of the parameters only.

#include <Kokkos_Core.hpp>

#include "heat_solver.hpp"
#include "otinum/otinum.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using Coeff = double;
using OTI = oti::otinum<3, 2, Coeff>;   // (alpha, A, sigma) through order 2

struct Config {
    int N = 21;
    double total_time = 0.05;
    double alpha0 = 1.0, amplitude0 = 100.0, sigma0 = 0.05;
    // the parameter box (alpha range matches uq_adaptive_reuse; A, sigma +/-30%)
    double alpha_min = 0.7,   alpha_max = 1.6;
    double amp_min   = 70.0,  amp_max   = 130.0;
    double sigma_min = 0.035, sigma_max = 0.065;
    int n_anchors = 64;
    int n_mc = 400;
    int n_queries = 200;
    unsigned mc_seed = 20260706;
    std::string output_dir = "gp_bank";
};

template <class Scalar> struct scalar_value { using type = Scalar; };
template <int M, int N, class S> struct scalar_value<oti::otinum<M, N, S>> { using type = S; };
template <class Scalar> using scalar_value_t = typename scalar_value<Scalar>::type;

template <class Scalar, std::enable_if_t<std::is_arithmetic_v<Scalar>, int> = 0>
KOKKOS_INLINE_FUNCTION Scalar scalar_exp(Scalar v) { return Kokkos::exp(v); }
template <int M, int N, class S>
KOKKOS_INLINE_FUNCTION oti::otinum<M, N, S> scalar_exp(oti::otinum<M, N, S> const& v) { return oti::exp(v); }

int compute_num_steps(double total_time, double dt) { return static_cast<int>(std::ceil((total_time / dt) - 1.0e-12)); }

template <class Scalar>
std::vector<Scalar> solve_final_field(Config const& c, Scalar alpha, Scalar amplitude, Scalar sigma, double dt_fixed)
{
    using Real = scalar_value_t<Scalar>;
    constexpr Real pi = static_cast<Real>(3.14159265358979323846);
    Mesh mesh(c.N, c.N, c.N, 1.0, 1.0, 1.0);
    int num_steps = compute_num_steps(c.total_time, dt_fixed);
    Kokkos::View<Scalar*> u("u", mesh.num_nodes), u_new("u_new", mesh.num_nodes);
    Kokkos::View<Real*> M_lumped("M", mesh.num_nodes);
    Kokkos::View<Scalar*> f("f", mesh.num_nodes), Ku("Ku", mesh.num_nodes);
    Real h_K_local[8][8];
    compute_local_stiffness(static_cast<Real>(mesh.dx), static_cast<Real>(mesh.dy), static_cast<Real>(mesh.dz), h_K_local);
    Kokkos::View<Real[8][8], Kokkos::HostSpace> host_K("host_K");
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) host_K(i, j) = h_K_local[i][j];
    auto device_K = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), host_K);
    Kokkos::deep_copy(u, Scalar(0.0));
    compute_lumped_mass(mesh, M_lumped);
    Scalar inv_two_sigma2 = Scalar(1) / (Scalar(2) * sigma * sigma);
    Scalar neg_alpha = -alpha;
    Real dt = static_cast<Real>(dt_fixed), total_time = static_cast<Real>(c.total_time);
    for (int step = 0; step < num_steps; ++step) {
        Real t = static_cast<Real>(step) * dt;
        Real xc = Real(0.5) + Real(0.3) * Kokkos::cos(Real(2) * pi * t / total_time);
        Real yc = Real(0.5) + Real(0.3) * Kokkos::sin(Real(2) * pi * t / total_time);
        Real zc = Real(1);
        Kokkos::parallel_for("Source", mesh.num_nodes, KOKKOS_LAMBDA(int n) {
            Real x, y, z; mesh.get_node_coords(n, x, y, z);
            Real r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc) + (z - zc) * (z - zc);
            f(n) = amplitude * scalar_exp(Scalar(-r2) * inv_two_sigma2) * M_lumped(n);
        });
        compute_stiffness_force(mesh, device_K, u, Ku);
        Kokkos::parallel_for("Update", mesh.num_nodes, KOKKOS_LAMBDA(int n) {
            Scalar acc = f(n); oti::fma_into(acc, neg_alpha, Ku(n));
            Real scale = dt * (Real(1) / M_lumped(n));
            u_new(n) = oti::scale_add(u(n), scale, acc);
        });
        std::swap(u, u_new);
    }
    auto u_host = Kokkos::create_mirror_view(u);
    Kokkos::deep_copy(u_host, u);
    std::vector<Scalar> fld(static_cast<std::size_t>(mesh.num_nodes));
    for (int i = 0; i < mesh.num_nodes; ++i) fld[static_cast<std::size_t>(i)] = u_host(i);
    return fld;
}

// Halton sequence in base b (van der Corput radical inverse), 1-indexed so the
// first point is not the box corner.
double halton(int i, int b)
{
    double f = 1.0, r = 0.0;
    for (int n = i; n > 0; n /= b) {
        f /= b;
        r += f * (n % b);
    }
    return r;
}

struct Point3 { double a, A, s; };

Point3 box_point(Config const& c, double u0, double u1, double u2)
{
    return {c.alpha_min + u0 * (c.alpha_max - c.alpha_min),
            c.amp_min   + u1 * (c.amp_max   - c.amp_min),
            c.sigma_min + u2 * (c.sigma_max - c.sigma_min)};
}

// The operational drift path: a closed curve traversed 1.5 times over the
// queries, so queries in the second traversal REVISIT parameter territory from
// the first -- the regime where a surrogate with memory needs no new solves.
Point3 drift_point(Config const& c, double t)   // t in [0, 1.5]
{
    constexpr double two_pi = 2.0 * 3.14159265358979323846;
    Point3 p;
    p.a = 1.15  + 0.35  * std::cos(two_pi * t);
    p.A = 100.0 + 25.0  * std::sin(two_pi * t);
    p.s = 0.05  + 0.012 * std::sin(2.0 * two_pi * t + 0.7);
    return p;
}

void parse_args(int argc, char* argv[], Config& c)
{
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i]; auto next = [&]() { return argv[++i]; };
        if (a == "--output") c.output_dir = next();
        else if (a == "--N") c.N = std::atoi(next());
        else if (a == "--anchors") c.n_anchors = std::atoi(next());
        else if (a == "--mc") c.n_mc = std::atoi(next());
        else if (a == "--queries") c.n_queries = std::atoi(next());
        else { std::cerr << "unknown arg " << a << "\n"; std::exit(1); }
    }
}

} // namespace

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {
        Config cfg; parse_args(argc, argv, cfg);
        std::filesystem::create_directories(cfg.output_dir);
        Mesh mesh(cfg.N, cfg.N, cfg.N, 1.0, 1.0, 1.0);
        double const dt = 0.8 * (mesh.dx * mesh.dx) / (6.0 * cfg.alpha_max);

        // fixed sensor = peak node of the nominal field (uq_adaptive_reuse convention)
        std::vector<double> f0 = solve_final_field<double>(cfg, cfg.alpha0, cfg.amplitude0, cfg.sigma0, dt);
        std::size_t sensor = 0; double pk = -1e300;
        for (std::size_t n = 0; n < f0.size(); ++n) if (f0[n] > pk) { pk = f0[n]; sensor = n; }
        std::cout << std::setprecision(8) << "sensor node " << sensor << " (nominal QoI=" << pk << ")\n";

        // ---- 1. anchors: Halton points, one otinum<3,2> jet each -----------
        // partial() applies the factorial, so these columns are DERIVATIVES.
        std::ofstream an(cfg.output_dir + "/bank_anchors.csv");
        an << std::setprecision(17)
           << "alpha,amplitude,sigma,q,dq_da,dq_dA,dq_ds,"
              "d2q_da2,d2q_dadA,d2q_dads,d2q_dA2,d2q_dAds,d2q_ds2\n";
        for (int i = 0; i < cfg.n_anchors; ++i) {
            Point3 p = box_point(cfg, halton(i + 1, 2), halton(i + 1, 3), halton(i + 1, 5));
            OTI jet = solve_final_field<OTI>(cfg, OTI::variable(0, p.a), OTI::variable(1, p.A),
                                             OTI::variable(2, p.s), dt)[sensor];
            an << p.a << ',' << p.A << ',' << p.s << ',' << jet.real() << ','
               << jet.partial(oti::sparse({{0, 1}})) << ','
               << jet.partial(oti::sparse({{1, 1}})) << ','
               << jet.partial(oti::sparse({{2, 1}})) << ','
               << jet.partial(oti::sparse({{0, 2}})) << ','
               << jet.partial(oti::sparse({{0, 1}, {1, 1}})) << ','
               << jet.partial(oti::sparse({{0, 1}, {2, 1}})) << ','
               << jet.partial(oti::sparse({{1, 2}})) << ','
               << jet.partial(oti::sparse({{1, 1}, {2, 1}})) << ','
               << jet.partial(oti::sparse({{2, 2}})) << '\n';
            if ((i + 1) % 16 == 0) std::cout << "  anchors: " << (i + 1) << "/" << cfg.n_anchors << "\n";
        }

        // ---- 2. Monte Carlo truth set: uniform box points, double solves ---
        std::mt19937 rng(cfg.mc_seed);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::ofstream mc(cfg.output_dir + "/bank_mc_truth.csv");
        mc << std::setprecision(17) << "alpha,amplitude,sigma,q_true\n";
        for (int i = 0; i < cfg.n_mc; ++i) {
            Point3 p = box_point(cfg, uni(rng), uni(rng), uni(rng));
            double q = solve_final_field<double>(cfg, p.a, p.A, p.s, dt)[sensor];
            mc << p.a << ',' << p.A << ',' << p.s << ',' << q << '\n';
            if ((i + 1) % 100 == 0) std::cout << "  mc truth: " << (i + 1) << "/" << cfg.n_mc << "\n";
        }

        // ---- 3. drift-path truth + jets: the twin-loop query stream ---------
        // The truth column audits every prediction; the jet columns exist so an
        // adaptive twin can "solve" at any query it chooses to anchor (the jet
        // is only consulted for queries the gate promotes to anchors).
        std::ofstream dr(cfg.output_dir + "/bank_drift_truth.csv");
        dr << std::setprecision(17)
           << "t,alpha,amplitude,sigma,q_true,q,dq_da,dq_dA,dq_ds,"
              "d2q_da2,d2q_dadA,d2q_dads,d2q_dA2,d2q_dAds,d2q_ds2\n";
        for (int i = 0; i < cfg.n_queries; ++i) {
            double t = 1.5 * i / (cfg.n_queries - 1);
            Point3 p = drift_point(cfg, t);
            double q = solve_final_field<double>(cfg, p.a, p.A, p.s, dt)[sensor];
            OTI jet = solve_final_field<OTI>(cfg, OTI::variable(0, p.a), OTI::variable(1, p.A),
                                             OTI::variable(2, p.s), dt)[sensor];
            dr << t << ',' << p.a << ',' << p.A << ',' << p.s << ',' << q << ','
               << jet.real() << ','
               << jet.partial(oti::sparse({{0, 1}})) << ','
               << jet.partial(oti::sparse({{1, 1}})) << ','
               << jet.partial(oti::sparse({{2, 1}})) << ','
               << jet.partial(oti::sparse({{0, 2}})) << ','
               << jet.partial(oti::sparse({{0, 1}, {1, 1}})) << ','
               << jet.partial(oti::sparse({{0, 1}, {2, 1}})) << ','
               << jet.partial(oti::sparse({{1, 2}})) << ','
               << jet.partial(oti::sparse({{1, 1}, {2, 1}})) << ','
               << jet.partial(oti::sparse({{2, 2}})) << '\n';
            if ((i + 1) % 50 == 0) std::cout << "  drift: " << (i + 1) << "/" << cfg.n_queries << "\n";
        }

        std::ofstream co(cfg.output_dir + "/bank_config.csv");
        co << std::setprecision(17) << "key,value\n"
           << "N," << cfg.N << "\ntotal_time," << cfg.total_time << "\ndt," << dt
           << "\nsensor_node," << sensor << "\nnominal_qoi," << pk
           << "\nalpha_min," << cfg.alpha_min << "\nalpha_max," << cfg.alpha_max
           << "\namp_min," << cfg.amp_min << "\namp_max," << cfg.amp_max
           << "\nsigma_min," << cfg.sigma_min << "\nsigma_max," << cfg.sigma_max
           << "\nn_anchors," << cfg.n_anchors << "\nn_mc," << cfg.n_mc
           << "\nn_queries," << cfg.n_queries << "\nmc_seed," << cfg.mc_seed << "\n";

        std::cout << "Wrote " << cfg.output_dir << "/bank_{anchors,mc_truth,drift_truth,config}.csv\n";
    }
    Kokkos::finalize();
    return 0;
}
