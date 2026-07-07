// On-the-fly error analysis doing something useful: ADAPTIVE SURROGATE REUSE.
//
// A parameter (diffusivity alpha) drifts across a range; at each step we need a
// QoI (temperature at a fixed sensor). Instead of re-solving the PDE for every
// query, we hold an "anchor" OTI jet and, on the fly, ask the validity gate
// whether the new alpha is within the jet's trusted region:
//   * is_trusted(jet, h, tau)  -> reuse: evaluate the QoI from the surrogate (no solve)
//   * not trusted              -> re-solve at the new alpha (a fresh anchor), continue
// So a whole sweep of N queries is covered by K << N PDE solves, and every reused
// answer is certified within tau. This is the amortized win (one solve, many
// certified evaluations) with the gate making the reuse safe.
//
// A dense true sweep (N scalar solves) is computed ONLY to validate the gate
// (max error vs budget, false-positive rate); it is not part of the method cost.
//
// dt is held fixed (sized for stability at alpha_max) so the surrogate is a clean
// function of alpha and truth uses the identical discretization.
//
// Outputs <out>/reuse_sweep.csv (per query) and <out>/reuse_summary.csv.

#include <Kokkos_Core.hpp>

#include "heat_solver.hpp"
#include "otinum/otinum.hpp"
#include "otinum/validity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using Coeff = double;
using OTI = oti::otinum<3, 2, Coeff>;  // linear model (m=1) + order-2 error band
namespace val = oti::validity;

struct Config {
    int N = 21;
    double total_time = 0.05;
    double alpha0 = 1.0, amplitude0 = 100.0, sigma0 = 0.05;
    double alpha_min = 0.7, alpha_max = 1.6;
    int n_queries = 200;
    double tau = 0.02;
    std::string output_dir = "reuse_output";
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

void parse_args(int argc, char* argv[], Config& c)
{
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i]; auto next = [&]() { return argv[++i]; };
        if (a == "--output") c.output_dir = next();
        else if (a == "--N") c.N = std::atoi(next());
        else if (a == "--alpha-min") c.alpha_min = std::atof(next());
        else if (a == "--alpha-max") c.alpha_max = std::atof(next());
        else if (a == "--queries") c.n_queries = std::atoi(next());
        else if (a == "--tau") c.tau = std::atof(next());
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
        // dt sized for explicit-FE stability at the LARGEST alpha, then held fixed.
        double const dt = 0.8 * (mesh.dx * mesh.dx) / (6.0 * cfg.alpha_max);

        // fixed sensor = peak node of the nominal-alpha field
        std::vector<double> f0 = solve_final_field<double>(cfg, cfg.alpha0, cfg.amplitude0, cfg.sigma0, dt);
        std::size_t sensor = 0; double pk = -1e300;
        for (std::size_t n = 0; n < f0.size(); ++n) if (f0[n] > pk) { pk = f0[n]; sensor = n; }
        std::cout << std::setprecision(8) << "sensor node " << sensor << " (nominal QoI=" << pk << ")\n";

        std::vector<double> aq(cfg.n_queries);
        for (int i = 0; i < cfg.n_queries; ++i)
            aq[i] = cfg.alpha_min + (cfg.alpha_max - cfg.alpha_min) * i / (cfg.n_queries - 1);

        // ground-truth sweep (validation only, not method cost)
        std::vector<double> truth(cfg.n_queries);
        for (int i = 0; i < cfg.n_queries; ++i)
            truth[i] = solve_final_field<double>(cfg, aq[i], cfg.amplitude0, cfg.sigma0, dt)[sensor];

        // Adaptive reuse: an ATLAS of anchor jets + the on-the-fly validity
        // gate. Every anchor is kept and each query is served by the nearest
        // one; a certified twin must never re-solve territory it has already
        // certified, so a new solve happens only when NO stored anchor's
        // trusted region covers the query. (On this monotone sweep the nearest
        // stored anchor is always the newest, so the atlas behaves identically
        // to holding a single anchor -- but on a query stream that revisits
        // old territory the atlas keeps reusing where a single-anchor twin
        // would pointlessly re-solve. Jets are 10 doubles each; keeping them
        // all is free.)
        std::ofstream sw(cfg.output_dir + "/reuse_sweep.csv");
        sw << std::setprecision(10) << "alpha,truth,pred,abs_err,budget,reused,is_anchor,anchor_alpha\n";
        int solves = 0, reused = 0, false_pos = 0;
        double max_reuse_err = 0.0;
        std::vector<double> atlas_alpha;
        std::vector<OTI> atlas_jet;
        auto add_anchor = [&](double a) {
            atlas_jet.push_back(solve_final_field<OTI>(cfg, OTI::variable(0, a),
                                                       OTI::variable(1, cfg.amplitude0),
                                                       OTI::variable(2, cfg.sigma0), dt)[sensor]);
            atlas_alpha.push_back(a);
            ++solves;
            return atlas_jet.size() - 1;
        };
        auto nearest_anchor = [&](double a) {
            std::size_t best = 0;
            for (std::size_t k = 1; k < atlas_alpha.size(); ++k)
                if (std::abs(a - atlas_alpha[k]) < std::abs(a - atlas_alpha[best])) best = k;
            return best;
        };
        add_anchor(aq[0]);
        for (int i = 0; i < cfg.n_queries; ++i) {
            std::size_t k = nearest_anchor(aq[i]);
            oti::detail::array<double, 3> h{aq[i] - atlas_alpha[k], 0.0, 0.0};
            bool is_anchor = false;
            if (!val::is_trusted(atlas_jet[k], h, cfg.tau, 0.0, 1)) {
                k = add_anchor(aq[i]);
                h = {0.0, 0.0, 0.0};
                is_anchor = true;
            }
            double pred = val::evaluate(atlas_jet[k], h, 1);
            double budget = cfg.tau * std::abs(atlas_jet[k][0]);
            double err = std::abs(pred - truth[i]);
            if (!is_anchor) { ++reused; max_reuse_err = std::max(max_reuse_err, err); if (err > budget) ++false_pos; }
            sw << aq[i] << ',' << truth[i] << ',' << pred << ',' << err << ',' << budget << ','
               << (is_anchor ? 0 : 1) << ',' << (is_anchor ? 1 : 0) << ',' << atlas_alpha[k] << '\n';
        }

        std::cout << "queries=" << cfg.n_queries << "  PDE solves (anchors)=" << solves
                  << "  reused=" << reused << "  speedup=" << double(cfg.n_queries) / solves << "x\n";
        std::cout << "max reuse error=" << max_reuse_err << "  false positives (err>budget)=" << false_pos
                  << "/" << reused << "\n";

        std::ofstream su(cfg.output_dir + "/reuse_summary.csv");
        su << std::setprecision(10) << "key,value\n";
        su << "tau," << cfg.tau << "\nqueries," << cfg.n_queries << "\nsolves," << solves
           << "\nreused," << reused << "\nfalse_positives," << false_pos
           << "\nmax_reuse_err," << max_reuse_err << "\nalpha_min," << cfg.alpha_min
           << "\nalpha_max," << cfg.alpha_max << "\nsensor_node," << sensor << "\n";
        std::cout << "Wrote " << cfg.output_dir << "/reuse_sweep.csv\n";
    }
    Kokkos::finalize();
    return 0;
}
