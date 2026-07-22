#include "y26_k1x_full_executor.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace {

struct Seed {
    int operation;
    int native_h_640;
    const char* shape_class;
};

constexpr std::array<Seed, 8> kSeeds {{
    {5, 160, "dense_1x1_k32"},
    {12, 160, "dense_1x1_k48"},
    {124, 80, "dense_1x1_k96"},
    {133, 80, "dense_1x1_n80_k64"},
    {144, 80, "dense_1x1_n80_k80"},
    {161, 40, "dense_3x3s1_k1152"},
    {201, 20, "dense_3x3s1_k2304"},
    {206, 20, "dense_1x1_n80_k256"},
}};

int integer_argument(const char* text, const char* name) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 1000000) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return static_cast<int>(value);
}

std::set<int> selected_operations() {
    std::set<int> selected;
    const char* text = std::getenv("Y26_STAGE60_SHAPE_OPERATIONS");
    if (text == nullptr || text[0] == '\0') return selected;
    std::string values(text);
    std::size_t begin = 0;
    while (begin < values.size()) {
        const std::size_t comma = values.find(',', begin);
        const std::size_t end = comma == std::string::npos ? values.size() : comma;
        const std::string value = values.substr(begin, end - begin);
        selected.insert(integer_argument(value.c_str(), "shape operation"));
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return selected;
}

bool m4_tail_only() {
    const char* value = std::getenv("Y26_STAGE60_M4_TAIL");
    return value != nullptr && std::string_view(value) == "1";
}

const char* route_name() {
    if (m4_tail_only()) return "m4tail";
    const char* dense_m8 = std::getenv("Y26_STAGE54_DENSE_M8");
    return dense_m8 != nullptr && std::string_view(dense_m8) == "1"
        ? "m8n16"
        : "m12n16";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            std::cerr << "usage: stage60_shape_matrix PACKAGE MANIFEST WARMUP RUNS\n";
            return 2;
        }
        const std::filesystem::path package = argv[1];
        const std::string manifest = argv[2];
        const int warmup = integer_argument(argv[3], "warmup");
        const int runs = integer_argument(argv[4], "runs");
        if (runs == 0) throw std::runtime_error("runs must be positive");

        y26::stage52::FullExecutor executor;
        y26::stage52::RunConfig config;
        config.workers = 4;
        config.worker_cpu_begin = 0;
        config.controller_cpu = 4;
        config.scheduler = y26::stage52::SchedulerMode::safe;
        config.wake_policy = y26::stage52::WakePolicy::frame_gated_spin;
        config.compute = y26::stage52::ComputeMode::optimized;
        config.allow_stage60_static_profiles = true;
        if (executor.prepare(package, manifest, config) != 0) {
            throw std::runtime_error("prepare failed: " + executor.last_error());
        }
        const int resolution = executor.input_width();
        if (resolution <= 0 || resolution != executor.input_height()) {
            throw std::runtime_error("Stage60 package input must be square");
        }
        const std::set<int> operations = selected_operations();

        std::cout << std::setprecision(12)
                  << "resolution\troute\tshape_class\toperation_index\toperation_name\tkind\t"
                     "output_h\toutput_w\tm\tn\tk\tkernel\tstride\tinput_c\toutput_c\t"
                     "working_set_bytes\tpacked_weight_bytes\tmacs\tgmac_per_s\tmean_us\t"
                     "median_us\tp95_us\tp99_us\tmax_us\toutput_hash\tcorrectness_status\n";
        for (const Seed& seed : kSeeds) {
            if (!operations.empty() && !operations.contains(seed.operation)) continue;
            const int side = seed.native_h_640 * resolution / 640;
            const int output_h = m4_tail_only() ? 1 : side;
            const int output_w = m4_tail_only() ? 4 : side;
            y26::stage52::DiagnosticConvShapeResult value;
            if (executor.diagnostic_benchmark_conv_shape(
                    seed.operation, output_h, output_w, 0, warmup, runs, &value) != 0) {
                throw std::runtime_error("shape benchmark failed: " + executor.last_error());
            }
            const std::uint64_t macs = static_cast<std::uint64_t>(value.m) * value.n * value.k;
            const double gmac_per_s = value.mean_us > 0.0
                ? static_cast<double>(macs) / value.mean_us / 1000.0
                : 0.0;
            std::cout << resolution << '\t' << route_name() << '\t' << seed.shape_class << '\t'
                      << value.operation_index << '\t' << value.operation_name << '\t'
                      << value.operation_kind << '\t' << value.output_h << '\t' << value.output_w
                      << '\t' << value.m << '\t' << value.n << '\t' << value.k << '\t'
                      << value.kernel_h << 'x' << value.kernel_w << '\t' << value.stride_h << 'x'
                      << value.stride_w << '\t' << value.input_c << '\t' << value.output_c << '\t'
                      << value.working_set_bytes << '\t' << value.packed_weight_bytes << '\t'
                      << macs << '\t' << gmac_per_s << '\t' << value.mean_us << '\t'
                      << value.median_us << '\t' << value.p95_us << '\t' << value.p99_us << '\t'
                      << value.maximum_us << "\t0x" << std::hex << value.output_hash << std::dec
                      << '\t' << (value.deterministic ? "exact-component-contract" : "failed")
                      << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
