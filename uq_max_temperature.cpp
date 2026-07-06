// UQ example: distribution of the maximum temperature under uncertain
// parameters (alpha, amplitude, sigma), comparing
//   (1) a Taylor-surrogate (TSE) moment propagation built from ONE OTI solve, and
//   (2) a Monte Carlo reference of many true PDE re-solves.
//
// This program is the data generator. It writes:
//   <out>/qoi_jet.csv      the truncated-Taylor jet of the QoI (max temperature)
//                          at the nominal point: one row per multi-index alpha,
//                          columns a0,a1,a2,coeff where coeff = c[alpha] is the
//                          NORMALIZED Taylor coefficient (1/alpha!) d^alpha Q.
//   <out>/mc_samples.csv   one true-resolve max temperature per Monte Carlo draw.
//   <out>/uq_config.csv    run metadata (nominal values, CoV, peak node, etc.).
//
// The Python post-process (uq_moment_analysis.py) forms the order-1 and order-2
// surrogates from the same jet (order-1 = the |alpha|<=1 subset) and propagates
// the Gaussian input moments through them exactly via Gauss-Hermite quadrature.
//
// QoI: the maximum nodal temperature of the final-time field. The OTI jet is
// taken at the node where the nominal solution is maximal (exact to first order
// by the envelope theorem); the Monte Carlo reference takes the true moving
// maximum, so any residual from the shifting argmax shows up honestly as part
// of the measured TSE error.
//
// dt is held fixed at its nominal value for every solve (the solver treats dt
// as a constant, undifferentiated), isolating the parametric Taylor error from
// time-discretization effects.

#include <Kokkos_Core.hpp>

#include "heat_solver.hpp"
#include "otinum/otinum.hpp"

#include <algorithm>
#include <array>
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

#ifndef OTI_UQ_ORDER
#define OTI_UQ_ORDER 2
#endif

using Coeff = double;
using OTI = oti::otinum<3, OTI_UQ_ORDER, Coeff>;

struct UqConfig {
    int N = 21;
    double total_time = 0.05;
    double alpha0 = 1.0;
    double amplitude0 = 100.0;
    double sigma0 = 0.05;
    double cov = 0.05;          // coefficient of variation (std / mean)
    long mc_samples = 5000;
    unsigned long seed = 20260629UL;
    std::string output_dir = "uq_output";
};

template <class Scalar>
struct scalar_value { using type = Scalar; };
template <int M, int N, class S>
struct scalar_value<oti::otinum<M, N, S>> { using type = S; };
template <class Scalar>
using scalar_value_t = typename scalar_value<Scalar>::type;

template <class Scalar, std::enable_if_t<std::is_arithmetic_v<Scalar>, int> = 0>
KOKKOS_INLINE_FUNCTION Scalar scalar_exp(Scalar v) { return Kokkos::exp(v); }
template <int M, int N, class S>
KOKKOS_INLINE_FUNCTION oti::otinum<M, N, S> scalar_exp(oti::otinum<M, N, S> const& v)
{
    return oti::exp(v);
}

template <class Scalar>
double real_value(Scalar const& v)
{
    if constexpr (std::is_arithmetic_v<Scalar>) {
        return static_cast<double>(v);
    } else {
        return v.real();
    }
}

int compute_num_steps(double total_time, double fixed_dt)
{
    return static_cast<int>(std::ceil((total_time / fixed_dt) - 1.0e-12));
}

// Time-steps the explicit FE heat solve and returns the final-time field.
// Mirrors the default (fused, hoisted-source) path of oti_heat_analysis.cpp.
template <class Scalar>
std::vector<Scalar> solve_final_field(UqConfig const& config,
                                      Scalar alpha,
                                      Scalar amplitude,
                                      Scalar sigma,
                                      double fixed_dt)
{
    using Real = scalar_value_t<Scalar>;
    constexpr Real pi = static_cast<Real>(3.14159265358979323846);

    Mesh mesh(config.N, config.N, config.N, 1.0, 1.0, 1.0);
    int num_steps = compute_num_steps(config.total_time, fixed_dt);

    Kokkos::View<Scalar*> u("temperature", mesh.num_nodes);
    Kokkos::View<Scalar*> u_new("temperature_next", mesh.num_nodes);
    Kokkos::View<Real*> M_lumped("mass_lumped", mesh.num_nodes);
    Kokkos::View<Scalar*> f("source_vector", mesh.num_nodes);
    Kokkos::View<Scalar*> Ku("stiffness_force", mesh.num_nodes);

    Real h_K_local[8][8];
    compute_local_stiffness(static_cast<Real>(mesh.dx), static_cast<Real>(mesh.dy),
                            static_cast<Real>(mesh.dz), h_K_local);
    Kokkos::View<Real[8][8], Kokkos::HostSpace> host_K("host_K");
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) host_K(i, j) = h_K_local[i][j];
    auto device_K = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), host_K);

    Kokkos::deep_copy(u, Scalar(0.0));
    compute_lumped_mass(mesh, M_lumped);

    Scalar inv_two_sigma2 = Scalar(1) / (Scalar(2) * sigma * sigma);
    Scalar neg_alpha = -alpha;
    Real dt = static_cast<Real>(fixed_dt);
    Real total_time = static_cast<Real>(config.total_time);

    for (int step = 0; step < num_steps; ++step) {
        Real t = static_cast<Real>(step) * dt;
        Real xc = Real(0.5) + Real(0.3) * Kokkos::cos(Real(2) * pi * t / total_time);
        Real yc = Real(0.5) + Real(0.3) * Kokkos::sin(Real(2) * pi * t / total_time);
        Real zc = Real(1);

        Kokkos::parallel_for("ComputeSource", mesh.num_nodes, KOKKOS_LAMBDA(int n_idx) {
            Real x, y, z;
            mesh.get_node_coords(n_idx, x, y, z);
            Real r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc) + (z - zc) * (z - zc);
            Scalar val = amplitude * scalar_exp(Scalar(-r2) * inv_two_sigma2);
            f(n_idx) = val * M_lumped(n_idx);
        });

        compute_stiffness_force(mesh, device_K, u, Ku);

        Kokkos::parallel_for("UpdateTemperature", mesh.num_nodes, KOKKOS_LAMBDA(int n_idx) {
            Scalar acc = f(n_idx);
            oti::fma_into(acc, neg_alpha, Ku(n_idx));
            Real scale = dt * (Real(1) / M_lumped(n_idx));
            u_new(n_idx) = oti::scale_add(u(n_idx), scale, acc);
        });

        std::swap(u, u_new);
    }

    auto u_host = Kokkos::create_mirror_view(u);
    Kokkos::deep_copy(u_host, u);
    std::vector<Scalar> field(static_cast<std::size_t>(mesh.num_nodes));
    for (int i = 0; i < mesh.num_nodes; ++i)
        field[static_cast<std::size_t>(i)] = u_host(i);
    return field;
}

void parse_args(int argc, char* argv[], UqConfig& c)
{
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](char const* opt) -> char const* {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + opt);
            return argv[++i];
        };
        if (a == "--output") c.output_dir = next("--output");
        else if (a == "--N") c.N = std::atoi(next("--N"));
        else if (a == "--total-time") c.total_time = std::atof(next("--total-time"));
        else if (a == "--alpha") c.alpha0 = std::atof(next("--alpha"));
        else if (a == "--amplitude") c.amplitude0 = std::atof(next("--amplitude"));
        else if (a == "--sigma") c.sigma0 = std::atof(next("--sigma"));
        else if (a == "--cov") c.cov = std::atof(next("--cov"));
        else if (a == "--mc-samples") c.mc_samples = std::atol(next("--mc-samples"));
        else if (a == "--seed") c.seed = std::strtoul(next("--seed"), nullptr, 10);
        else throw std::runtime_error("unknown option: " + a);
    }
}

} // namespace

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {
        UqConfig config;
        try {
            parse_args(argc, argv, config);
        } catch (std::exception const& ex) {
            std::cerr << "Error: " << ex.what() << "\n";
            Kokkos::finalize();
            return 1;
        }

        std::filesystem::create_directories(config.output_dir);

        Mesh mesh(config.N, config.N, config.N, 1.0, 1.0, 1.0);
        double fixed_dt = 0.8 * (mesh.dx * mesh.dx) / (6.0 * config.alpha0);
        int num_steps = compute_num_steps(config.total_time, fixed_dt);

        std::cout << "UQ max-temperature analysis (OTI order " << OTI_UQ_ORDER << ")\n"
                  << "  N=" << config.N << ", nodes=" << mesh.num_nodes
                  << ", dt=" << fixed_dt << ", steps=" << num_steps
                  << ", backend=" << Kokkos::DefaultExecutionSpace::name() << "\n"
                  << "  nominal: alpha=" << config.alpha0
                  << ", amplitude=" << config.amplitude0
                  << ", sigma=" << config.sigma0
                  << ", CoV=" << config.cov << "\n"
                  << "  MC samples=" << config.mc_samples << ", seed=" << config.seed << "\n";

        // ---- One OTI solve at the nominal point -> QoI jet ------------------
        OTI alpha = OTI::variable(0, config.alpha0);
        OTI amplitude = OTI::variable(1, config.amplitude0);
        OTI sigma = OTI::variable(2, config.sigma0);
        std::vector<OTI> oti_field = solve_final_field<OTI>(config, alpha, amplitude, sigma, fixed_dt);

        std::size_t peak = 0;
        double peak_val = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < oti_field.size(); ++i) {
            double v = oti_field[i].real();
            if (v > peak_val) { peak_val = v; peak = i; }
        }
        OTI const& qoi = oti_field[peak];
        double px, py, pz;
        mesh.get_node_coords(static_cast<int>(peak), px, py, pz);
        std::cout << "  nominal peak temperature = " << std::setprecision(10) << peak_val
                  << " at node " << peak << " (x=" << px << ", y=" << py << ", z=" << pz << ")\n";

        // Dump every Taylor coefficient up to the compiled order.
        {
            std::ofstream out(config.output_dir + "/qoi_jet.csv");
            out << std::setprecision(17);
            out << "a0,a1,a2,coeff\n";
            for (int a0 = 0; a0 <= OTI_UQ_ORDER; ++a0)
                for (int a1 = 0; a1 + a0 <= OTI_UQ_ORDER; ++a1)
                    for (int a2 = 0; a2 + a1 + a0 <= OTI_UQ_ORDER; ++a2) {
                        OTI::alpha_type alpha_idx{a0, a1, a2};
                        out << a0 << ',' << a1 << ',' << a2 << ','
                            << qoi.coeff(alpha_idx) << '\n';
                    }
        }

        // ---- Monte Carlo reference: true re-solves --------------------------
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(config.mc_samples));
        std::mt19937_64 rng(config.seed);
        std::normal_distribution<double> n_alpha(config.alpha0, config.cov * config.alpha0);
        std::normal_distribution<double> n_amp(config.amplitude0, config.cov * config.amplitude0);
        std::normal_distribution<double> n_sigma(config.sigma0, config.cov * config.sigma0);

        for (long s = 0; s < config.mc_samples; ++s) {
            double a = n_alpha(rng);
            double am = n_amp(rng);
            double sg = n_sigma(rng);
            // Parameters are physically positive; a 5% CoV makes draws <= 0
            // astronomically unlikely, but guard anyway so a pathological draw
            // cannot blow up the solve.
            if (a <= 0.0 || am <= 0.0 || sg <= 0.0) { --s; continue; }
            std::vector<double> field = solve_final_field<double>(config, a, am, sg, fixed_dt);
            double m = *std::max_element(field.begin(), field.end());
            samples.push_back(m);
            if ((s + 1) % 500 == 0)
                std::cout << "  MC " << (s + 1) << "/" << config.mc_samples << "\r" << std::flush;
        }
        std::cout << "  MC " << config.mc_samples << "/" << config.mc_samples << " done\n";

        {
            std::ofstream out(config.output_dir + "/mc_samples.csv");
            out << std::setprecision(17) << "tmax\n";
            for (double v : samples) out << v << '\n';
        }
        {
            std::ofstream out(config.output_dir + "/uq_config.csv");
            out << std::setprecision(17) << "key,value\n";
            out << "oti_order," << OTI_UQ_ORDER << '\n';
            out << "N," << config.N << '\n';
            out << "num_nodes," << mesh.num_nodes << '\n';
            out << "dt," << fixed_dt << '\n';
            out << "num_steps," << num_steps << '\n';
            out << "total_time," << config.total_time << '\n';
            out << "alpha0," << config.alpha0 << '\n';
            out << "amplitude0," << config.amplitude0 << '\n';
            out << "sigma0," << config.sigma0 << '\n';
            out << "cov," << config.cov << '\n';
            out << "mc_samples," << config.mc_samples << '\n';
            out << "seed," << config.seed << '\n';
            out << "nominal_peak_value," << peak_val << '\n';
            out << "peak_node," << peak << '\n';
            out << "peak_x," << px << '\n';
            out << "peak_y," << py << '\n';
            out << "peak_z," << pz << '\n';
        }

        std::cout << "Wrote " << config.output_dir << "/{qoi_jet.csv,mc_samples.csv,uq_config.csv}\n";
    }
    Kokkos::finalize();
    return 0;
}
