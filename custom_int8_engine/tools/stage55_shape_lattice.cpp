#include "y26_k1x_full_executor.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Seed {
    int operation;
    int native_h;
    const char* shape_class;
};

constexpr std::array<int, 6> kResolutions {384, 416, 448, 480, 512, 640};

// These are graph-backed package operations. The diagnostic API only shrinks
// their spatial dimensions or output-channel prefix, preserving packed weights,
// quantization assets, layout, and the selected board kernel.
constexpr std::array<Seed, 20> kSeeds {{
    {5, 160, "dense_1x1_k32"},
    {12, 160, "dense_1x1_k48"},
    {108, 80, "dense_1x1_k256"},
    {124, 80, "dense_1x1_k96"},
    {133, 80, "dense_1x1_n80_k64"},
    {144, 80, "dense_1x1_n80_k80"},
    {167, 40, "dense_1x1_n80_k128"},
    {90, 40, "dense_1x1_k384"},
    {206, 20, "dense_1x1_n80_k256"},
    {9, 160, "dense_3x3s1_k72"},
    {113, 80, "dense_3x3s1_k144"},
    {127, 80, "dense_3x3s1_k576"},
    {161, 40, "dense_3x3s1_k1152"},
    {201, 20, "dense_3x3s1_k2304"},
    {3, 160, "dense_3x3s2_k144"},
    {14, 80, "dense_3x3s2_k576"},
    {128, 80, "depthwise_c64"},
    {162, 40, "depthwise_c128"},
    {173, 40, "depthwise_c80"},
    {202, 20, "depthwise_c256"},
}};

int integer_argument(const char* text, const char* name) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 1000000) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return static_cast<int>(value);
}

void print_result(int resolution, const char* shape_class,
                  const y26::stage52::DiagnosticConvShapeResult& value) {
    std::cout << resolution << '\t' << shape_class << '\t' << value.operation_index << '\t'
              << value.operation_name << '\t' << value.operation_kind << '\t'
              << value.output_h << '\t' << value.output_w << '\t' << value.m << '\t'
              << value.n << '\t' << value.k << '\t' << value.kernel_h << 'x'
              << value.kernel_w << '\t' << value.stride_h << 'x' << value.stride_w << '\t'
              << value.input_c << '\t' << value.output_c << '\t'
              << value.working_set_bytes << '\t' << value.packed_weight_bytes << '\t'
              << value.mean_us << '\t' << value.median_us << '\t' << value.p95_us << '\t'
              << value.maximum_us << "\t0x" << std::hex << value.output_hash << std::dec
              << '\t' << (value.deterministic ? "exact-component-contract" : "failed") << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            std::cerr << "usage: stage55_shape_lattice PACKAGE MANIFEST WARMUP RUNS\n";
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
        config.compute = y26::stage52::ComputeMode::optimized;
        if (executor.prepare(package, manifest, config) != 0) {
            throw std::runtime_error("prepare failed: " + executor.last_error());
        }

        std::cout << "resolution\tshape_class\toperation_index\toperation_name\tkind\t"
                     "output_h\toutput_w\tm\tn\tk\tkernel\tstride\tinput_c\toutput_c\t"
                     "working_set_bytes\tpacked_weight_bytes\tmean_us\tmedian_us\tp95_us\t"
                     "max_us\toutput_hash\tcorrectness_status\n";
        for (const Seed& seed : kSeeds) {
            for (int resolution : kResolutions) {
                const int side = seed.native_h * resolution / 640;
                y26::stage52::DiagnosticConvShapeResult result;
                if (executor.diagnostic_benchmark_conv_shape(
                        seed.operation, side, side, 0, warmup, runs, &result) != 0) {
                    throw std::runtime_error("shape benchmark failed: " + executor.last_error());
                }
                print_result(resolution, seed.shape_class, result);
            }
        }

        constexpr std::array<int, 12> kTailChannels {
            4, 8, 16, 24, 32, 48, 64, 80, 96, 128, 192, 256,
        };
        for (int channels : kTailChannels) {
            y26::stage52::DiagnosticConvShapeResult result;
            if (executor.diagnostic_benchmark_conv_shape(
                    63, 20, 20, channels, warmup, runs, &result) != 0) {
                throw std::runtime_error("channel-tail benchmark failed: " + executor.last_error());
            }
            print_result(640, "dense_1x1_n_tail_k384", result);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
