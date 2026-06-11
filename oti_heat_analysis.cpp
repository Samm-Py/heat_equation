#include <Kokkos_Core.hpp>

#include "heat_solver.hpp"
#include "otinum/otinum.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

#ifndef OTI_HEAT_COEFF_TYPE
#define OTI_HEAT_COEFF_TYPE double
#endif

using Coeff = OTI_HEAT_COEFF_TYPE;
using OTI = oti::otinum<3, 1, Coeff>;

struct AnalysisConfig {
    int N = 21;
    double L = 1.0;
    double total_time = 0.05;
    double alpha0 = 1.0;
    double amplitude0 = 100.0;
    double sigma0 = 0.05;
    double h_alpha = 1.0e-4;
    double h_amplitude = 1.0e-3;
    double h_sigma = 1.0e-5;
    int num_snapshots = 5;
    bool run_finite_differences = true;
    std::string output_dir = "oti_analysis_output";
};

struct ParameterSweep {
    std::string name;
    double h;
    OTI::alpha_type derivative_alpha;
};

struct PhaseTiming {
    std::string name;
    double seconds = 0.0;
};

class ScopedTimer {
public:
    explicit ScopedTimer(double& seconds_out)
        : seconds_out_(seconds_out)
    {
        Kokkos::fence();
        start_ = clock_type::now();
    }

    ~ScopedTimer()
    {
        Kokkos::fence();
        seconds_out_ = std::chrono::duration<double>(clock_type::now() - start_).count();
    }

private:
    using clock_type = std::chrono::steady_clock;

    double& seconds_out_;
    clock_type::time_point start_;
};

template <class Scalar>
struct RunResult {
    Mesh mesh;
    double dt = 0.0;
    int num_steps = 0;
    std::vector<int> snapshot_steps;
    std::vector<double> snapshot_times;
    std::vector<std::vector<Scalar>> snapshots;

    RunResult(Mesh mesh_in) : mesh(mesh_in) {}
};

template <class Scalar>
struct scalar_value {
    using type = Scalar;
};

template <int M, int N, class Scalar>
struct scalar_value<oti::otinum<M, N, Scalar>> {
    using type = Scalar;
};

template <class Scalar>
using scalar_value_t = typename scalar_value<Scalar>::type;

std::string precision_name()
{
    if constexpr (std::is_same_v<Coeff, float>) {
        return "float";
    } else if constexpr (std::is_same_v<Coeff, double>) {
        return "double";
    } else {
        return "custom";
    }
}

template <class Scalar, std::enable_if_t<std::is_arithmetic_v<Scalar>, int> = 0>
KOKKOS_INLINE_FUNCTION
Scalar scalar_exp(Scalar value)
{
    return Kokkos::exp(value);
}

template <int M, int N, class Scalar>
KOKKOS_INLINE_FUNCTION oti::otinum<M, N, Scalar>
scalar_exp(oti::otinum<M, N, Scalar> const& value)
{
    return oti::exp(value);
}

template <class Scalar>
double real_value(Scalar const& value)
{
    if constexpr (std::is_arithmetic_v<Scalar>) {
        return static_cast<double>(value);
    } else {
        return value.real();
    }
}

std::vector<int> make_snapshot_steps(int num_steps, int num_snapshots)
{
    std::vector<int> steps;
    if (num_snapshots <= 1) {
        steps = {num_steps};
    } else {
        // num_snapshots frames evenly spaced over [0, num_steps] inclusive.
        for (int s = 0; s < num_snapshots; ++s) {
            long long step = static_cast<long long>(s) * num_steps / (num_snapshots - 1);
            steps.push_back(static_cast<int>(step));
        }
    }
    std::sort(steps.begin(), steps.end());
    steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
    return steps;
}

int parse_int_arg(char const* value, std::string const& option)
{
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::runtime_error("Invalid integer for " + option + ": " + value);
    }
    return static_cast<int>(parsed);
}

double parse_double_arg(char const* value, std::string const& option)
{
    char* end = nullptr;
    double parsed = std::strtod(value, &end);
    if (end == value || *end != '\0') {
        throw std::runtime_error("Invalid floating-point value for " + option + ": " + value);
    }
    return parsed;
}

char const* require_value(int& i, int argc, char* argv[], std::string const& option)
{
    if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + option);
    }
    ++i;
    return argv[i];
}

void print_usage(char const* program)
{
    std::cout
        << "Usage: " << program << " [output_dir] [options]\n"
        << "\n"
        << "Options:\n"
        << "  --output DIR          Output directory\n"
        << "  --N VALUE             Uniform grid nodes per direction, default 21\n"
        << "  --total-time VALUE    Final simulation time, default 0.05\n"
        << "  --alpha VALUE         Base thermal diffusivity, default 1.0\n"
        << "  --amplitude VALUE     Base source amplitude, default 100.0\n"
        << "  --sigma VALUE         Base source width, default 0.05\n"
        << "  --h-alpha VALUE       Finite-difference alpha step, default 1e-4\n"
        << "  --h-amplitude VALUE   Finite-difference amplitude step, default 1e-3\n"
        << "  --h-sigma VALUE       Finite-difference sigma step, default 1e-5\n"
        << "  --snapshots VALUE     Number of evenly spaced time snapshots, default 5\n"
        << "  --skip-fd             Skip finite-difference validation solves and slice CSV\n"
        << "  --help                Show this help\n";
}

bool parse_command_line(int argc, char* argv[], AnalysisConfig& config)
{
    bool output_dir_set = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        } else if (arg == "--output") {
            config.output_dir = require_value(i, argc, argv, arg);
            output_dir_set = true;
        } else if (arg == "--N") {
            config.N = parse_int_arg(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--total-time") {
            config.total_time = parse_double_arg(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--alpha") {
            config.alpha0 = parse_double_arg(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--amplitude") {
            config.amplitude0 = parse_double_arg(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--sigma") {
            config.sigma0 = parse_double_arg(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--h-alpha") {
            config.h_alpha = parse_double_arg(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--h-amplitude") {
            config.h_amplitude = parse_double_arg(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--h-sigma") {
            config.h_sigma = parse_double_arg(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--snapshots") {
            config.num_snapshots = parse_int_arg(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--skip-fd") {
            config.run_finite_differences = false;
        } else if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error("Unknown option: " + arg);
        } else if (!output_dir_set) {
            config.output_dir = arg;
            output_dir_set = true;
        } else {
            throw std::runtime_error("Unexpected positional argument: " + arg);
        }
    }

    if (config.N < 2) {
        throw std::runtime_error("--N must be at least 2");
    }
    if (config.L <= 0.0 || config.total_time <= 0.0 || config.alpha0 <= 0.0 ||
        config.amplitude0 <= 0.0 || config.sigma0 <= 0.0) {
        throw std::runtime_error("L, total time, alpha, amplitude, and sigma must be positive");
    }
    if (config.h_alpha <= 0.0 || config.h_amplitude <= 0.0 || config.h_sigma <= 0.0) {
        throw std::runtime_error("Finite-difference step sizes must be positive");
    }
    if (config.num_snapshots < 1) {
        throw std::runtime_error("--snapshots must be at least 1");
    }
    return true;
}

int compute_num_steps(double total_time, double fixed_dt)
{
    return static_cast<int>(std::ceil((total_time / fixed_dt) - 1.0e-12));
}

template <class Scalar>
RunResult<Scalar> run_heat_solver(AnalysisConfig const& config,
                                  Scalar alpha,
                                  Scalar amplitude,
                                  Scalar sigma,
                                  double fixed_dt)
{
    using Real = scalar_value_t<Scalar>;
    constexpr Real pi = static_cast<Real>(3.14159265358979323846);

    Mesh mesh(config.N, config.N, config.N, config.L, config.L, config.L);
    RunResult<Scalar> result(mesh);
    result.dt = fixed_dt;
    result.num_steps = compute_num_steps(config.total_time, fixed_dt);
    result.snapshot_steps = make_snapshot_steps(result.num_steps, config.num_snapshots);

    Kokkos::View<Scalar*> u("temperature", mesh.num_nodes);
    Kokkos::View<Scalar*> u_new("temperature_next", mesh.num_nodes);
    Kokkos::View<Real*> M_lumped("mass_lumped", mesh.num_nodes);
    Kokkos::View<Scalar*> f("source_vector", mesh.num_nodes);
    Kokkos::View<Scalar*> Ku("stiffness_force", mesh.num_nodes);

    Real h_K_local[8][8];
    compute_local_stiffness(
        static_cast<Real>(mesh.dx), static_cast<Real>(mesh.dy), static_cast<Real>(mesh.dz), h_K_local);
    Kokkos::View<Real[8][8], Kokkos::HostSpace> host_K("host_K");
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            host_K(i, j) = h_K_local[i][j];
        }
    }
    auto device_K = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), host_K);

    Kokkos::deep_copy(u, Scalar(0.0));
    compute_lumped_mass(mesh, M_lumped);

    Scalar inv_two_sigma2 = Scalar(1) / (Scalar(2) * sigma * sigma);
    Scalar neg_alpha = -alpha;
    Real dt = static_cast<Real>(fixed_dt);
    Real total_time = static_cast<Real>(config.total_time);

    auto u_host = Kokkos::create_mirror_view(u);
    std::size_t next_snapshot = 0;

    for (int step = 0; step <= result.num_steps; ++step) {
        Real t = static_cast<Real>(step) * dt;

        if (next_snapshot < result.snapshot_steps.size() &&
            step == result.snapshot_steps[next_snapshot]) {
            Kokkos::deep_copy(u_host, u);
            std::vector<Scalar> snapshot(static_cast<std::size_t>(mesh.num_nodes));
            for (int idx = 0; idx < mesh.num_nodes; ++idx) {
                snapshot[static_cast<std::size_t>(idx)] = u_host(idx);
            }
            result.snapshot_times.push_back(t);
            result.snapshots.push_back(std::move(snapshot));
            ++next_snapshot;
        }

        if (step == result.num_steps) {
            break;
        }

        Real xc = Real(0.5) + Real(0.3) * Kokkos::cos(Real(2) * pi * t / total_time);
        Real yc = Real(0.5) + Real(0.3) * Kokkos::sin(Real(2) * pi * t / total_time);
        Real zc = Real(1);

        Kokkos::parallel_for("ComputeSource", mesh.num_nodes, KOKKOS_LAMBDA(int n_idx) {
            Real x, y, z;
            mesh.get_node_coords(n_idx, x, y, z);
            Real r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc) + (z - zc) * (z - zc);
            Scalar exponent = Scalar(-r2) * inv_two_sigma2;
            Scalar val = amplitude * scalar_exp(exponent);
            f(n_idx) = val * M_lumped(n_idx);
        });

        compute_stiffness_force(mesh, device_K, u, Ku);

        Kokkos::parallel_for("UpdateTemperature", mesh.num_nodes, KOKKOS_LAMBDA(int n_idx) {
            // Fused form of  u + dt*(1/M) * (f - alpha*Ku).  fma_into avoids
            // the alpha*Ku temporary, and folding dt into a plain Real scale
            // skips the jet-lift of dt (a full jet x jet product per node).
            Scalar acc = f(n_idx);
            oti::fma_into(acc, neg_alpha, Ku(n_idx));
            Real scale = dt * (Real(1) / M_lumped(n_idx));
            u_new(n_idx) = oti::scale_add(u(n_idx), scale, acc);
        });

        std::swap(u, u_new);
    }

    return result;
}

RunResult<Coeff> run_scalar_variant(AnalysisConfig const& config,
                                    Coeff alpha,
                                    Coeff amplitude,
                                    Coeff sigma,
                                    double fixed_dt)
{
    return run_heat_solver<Coeff>(config, alpha, amplitude, sigma, fixed_dt);
}

void write_slice_csv(AnalysisConfig const& config,
                     RunResult<Coeff> const& base,
                     RunResult<OTI> const& oti_result,
                     std::array<RunResult<Coeff>, 3> const& fd_plus,
                     std::array<RunResult<Coeff>, 3> const& fd_minus,
                     std::array<ParameterSweep, 3> const& params)
{
    std::filesystem::create_directories(config.output_dir);
    std::ofstream out(config.output_dir + "/slice_snapshots.csv");
    out << "step,time,i,j,k,x,y,z,"
           "solution_scalar,solution_oti,solution_abs_error,"
           "dalpha_oti,dalpha_fd,dalpha_abs_error,"
           "damplitude_oti,damplitude_fd,damplitude_abs_error,"
           "dsigma_oti,dsigma_fd,dsigma_abs_error\n";

    int k = base.mesh.Nz - 1;
    for (std::size_t s = 0; s < base.snapshots.size(); ++s) {
        int step = base.snapshot_steps[s];
        double time = base.snapshot_times[s];
        for (int j = 0; j < base.mesh.Ny; ++j) {
            for (int i = 0; i < base.mesh.Nx; ++i) {
                int idx = i + j * base.mesh.Nx + k * base.mesh.Nx * base.mesh.Ny;
                std::size_t uidx = static_cast<std::size_t>(idx);

                double solution_scalar = real_value(base.snapshots[s][uidx]);
                OTI const& solution_oti_value = oti_result.snapshots[s][uidx];
                double solution_oti = solution_oti_value.real();

                std::array<double, 3> oti_deriv{};
                std::array<double, 3> fd_deriv{};
                std::array<double, 3> deriv_error{};

                for (std::size_t p = 0; p < params.size(); ++p) {
                    oti_deriv[p] = solution_oti_value.partial(params[p].derivative_alpha);
                    fd_deriv[p] = (real_value(fd_plus[p].snapshots[s][uidx]) -
                                   real_value(fd_minus[p].snapshots[s][uidx])) /
                                  (2.0 * params[p].h);
                    deriv_error[p] = std::abs(oti_deriv[p] - fd_deriv[p]);
                }

                out << step << ','
                    << time << ','
                    << i << ','
                    << j << ','
                    << k << ','
                    << static_cast<double>(i) * base.mesh.dx << ','
                    << static_cast<double>(j) * base.mesh.dy << ','
                    << static_cast<double>(k) * base.mesh.dz << ','
                    << solution_scalar << ','
                    << solution_oti << ','
                    << std::abs(solution_oti - solution_scalar) << ','
                    << oti_deriv[0] << ','
                    << fd_deriv[0] << ','
                    << deriv_error[0] << ','
                    << oti_deriv[1] << ','
                    << fd_deriv[1] << ','
                    << deriv_error[1] << ','
                    << oti_deriv[2] << ','
                    << fd_deriv[2] << ','
                    << deriv_error[2] << '\n';
            }
        }
    }
}

void write_metrics_csv(AnalysisConfig const& config, Mesh const& mesh, double fixed_dt, int num_steps)
{
    std::filesystem::create_directories(config.output_dir);
    std::ofstream out(config.output_dir + "/run_config.csv");
    out << "key,value\n";
    out << "N," << config.N << '\n';
    out << "L," << config.L << '\n';
    out << "num_nodes," << mesh.num_nodes << '\n';
    out << "num_elements," << mesh.num_elements << '\n';
    out << "dt," << fixed_dt << '\n';
    out << "num_steps," << num_steps << '\n';
    out << "total_time," << config.total_time << '\n';
    out << "alpha0," << config.alpha0 << '\n';
    out << "amplitude0," << config.amplitude0 << '\n';
    out << "sigma0," << config.sigma0 << '\n';
    out << "h_alpha," << config.h_alpha << '\n';
    out << "h_amplitude," << config.h_amplitude << '\n';
    out << "h_sigma," << config.h_sigma << '\n';
    out << "run_finite_differences," << (config.run_finite_differences ? 1 : 0) << '\n';
    out << "coefficient_precision," << precision_name() << '\n';
}

void write_timing_csv(AnalysisConfig const& config, std::vector<PhaseTiming> const& timings)
{
    std::filesystem::create_directories(config.output_dir);
    std::ofstream out(config.output_dir + "/timing_summary.csv");
    out << "phase,seconds\n";
    out << std::setprecision(17);
    for (PhaseTiming const& timing : timings) {
        out << timing.name << ',' << timing.seconds << '\n';
    }
}

} // namespace

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {
        AnalysisConfig config;
        try {
            if (!parse_command_line(argc, argv, config)) {
                Kokkos::finalize();
                return 0;
            }
        } catch (std::exception const& ex) {
            std::cerr << "Error: " << ex.what() << "\n\n";
            print_usage(argv[0]);
            Kokkos::finalize();
            return 1;
        }

        Mesh mesh(config.N, config.N, config.N, config.L, config.L, config.L);
        double fixed_dt = 0.8 * (mesh.dx * mesh.dx) / (6.0 * config.alpha0);
        int num_steps = compute_num_steps(config.total_time, fixed_dt);

        std::cout << "Configuration: N=" << config.N
                  << ", nodes=" << mesh.num_nodes
                  << ", elements=" << mesh.num_elements
                  << ", coefficient_precision=" << precision_name()
                  << ", dt=" << fixed_dt
                  << ", steps=" << num_steps
                  << ", total_time=" << config.total_time
                  << ", finite_differences=" << (config.run_finite_differences ? "on" : "off")
                  << '\n';

        std::array<ParameterSweep, 3> params = {{
            {"alpha", config.h_alpha, {1, 0, 0}},
            {"amplitude", config.h_amplitude, {0, 1, 0}},
            {"sigma", config.h_sigma, {0, 0, 1}},
        }};

        std::vector<PhaseTiming> timings;

        std::cout << "Running base " << precision_name() << " solve\n";
        RunResult<Coeff> base(Mesh(config.N, config.N, config.N, config.L, config.L, config.L));
        timings.push_back({"base_scalar_solve", 0.0});
        {
            Kokkos::Profiling::pushRegion("base_scalar_solve");
            {
                ScopedTimer timer(timings.back().seconds);
                base = run_scalar_variant(
                    config,
                    static_cast<Coeff>(config.alpha0),
                    static_cast<Coeff>(config.amplitude0),
                    static_cast<Coeff>(config.sigma0),
                    fixed_dt);
            }
            Kokkos::Profiling::popRegion();
        }
        std::cout << "Base " << precision_name() << " solve wall time: "
                  << timings.back().seconds << " s\n";

        std::cout << "Running OTI " << precision_name() << " solve\n";
        OTI alpha = OTI::variable(0, static_cast<Coeff>(config.alpha0));
        OTI amplitude = OTI::variable(1, static_cast<Coeff>(config.amplitude0));
        OTI sigma = OTI::variable(2, static_cast<Coeff>(config.sigma0));
        RunResult<OTI> oti_result(Mesh(config.N, config.N, config.N, config.L, config.L, config.L));
        timings.push_back({"oti_solve", 0.0});
        {
            Kokkos::Profiling::pushRegion("oti_solve");
            {
                ScopedTimer timer(timings.back().seconds);
                oti_result = run_heat_solver<OTI>(config, alpha, amplitude, sigma, fixed_dt);
            }
            Kokkos::Profiling::popRegion();
        }
        std::cout << "OTI " << precision_name() << " solve wall time: "
                  << timings.back().seconds << " s\n";
        std::cout << "OTI/base wall-time ratio: "
                  << timings[1].seconds / timings[0].seconds << "x\n";

        if (config.run_finite_differences) {
            std::optional<std::array<RunResult<Coeff>, 3>> fd_plus;
            std::optional<std::array<RunResult<Coeff>, 3>> fd_minus;

            timings.push_back({"finite_difference_solves", 0.0});
            {
                ScopedTimer timer(timings.back().seconds);
                fd_plus = std::array<RunResult<Coeff>, 3> {
                    run_scalar_variant(config,
                                       static_cast<Coeff>(config.alpha0 + config.h_alpha),
                                       static_cast<Coeff>(config.amplitude0),
                                       static_cast<Coeff>(config.sigma0),
                                       fixed_dt),
                    run_scalar_variant(config,
                                       static_cast<Coeff>(config.alpha0),
                                       static_cast<Coeff>(config.amplitude0 + config.h_amplitude),
                                       static_cast<Coeff>(config.sigma0),
                                       fixed_dt),
                    run_scalar_variant(config,
                                       static_cast<Coeff>(config.alpha0),
                                       static_cast<Coeff>(config.amplitude0),
                                       static_cast<Coeff>(config.sigma0 + config.h_sigma),
                                       fixed_dt),
                };
                fd_minus = std::array<RunResult<Coeff>, 3> {
                    run_scalar_variant(config,
                                       static_cast<Coeff>(config.alpha0 - config.h_alpha),
                                       static_cast<Coeff>(config.amplitude0),
                                       static_cast<Coeff>(config.sigma0),
                                       fixed_dt),
                    run_scalar_variant(config,
                                       static_cast<Coeff>(config.alpha0),
                                       static_cast<Coeff>(config.amplitude0 - config.h_amplitude),
                                       static_cast<Coeff>(config.sigma0),
                                       fixed_dt),
                    run_scalar_variant(config,
                                       static_cast<Coeff>(config.alpha0),
                                       static_cast<Coeff>(config.amplitude0),
                                       static_cast<Coeff>(config.sigma0 - config.h_sigma),
                                       fixed_dt),
                };
            }
            std::cout << "Finite-difference solve wall time: " << timings.back().seconds << " s\n";
            write_slice_csv(config, base, oti_result, *fd_plus, *fd_minus, params);
        }

        write_metrics_csv(config, mesh, fixed_dt, num_steps);
        write_timing_csv(config, timings);

        if (config.run_finite_differences) {
            std::cout << "Wrote " << config.output_dir << "/slice_snapshots.csv\n";
        }
        std::cout << "Wrote " << config.output_dir << "/timing_summary.csv\n";
    }
    Kokkos::finalize();
    return 0;
}
