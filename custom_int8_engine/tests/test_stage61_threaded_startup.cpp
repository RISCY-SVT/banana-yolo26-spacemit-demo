#include "y26_k1x_threaded_conv.h"

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

int parse_iterations(int argc, char** argv) {
    if (argc == 1) return 100;
    if (argc != 3 || std::string_view(argv[1]) != "--iterations") return -1;
    int iterations = 0;
    const std::string_view text(argv[2]);
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), iterations);
    if (error != std::errc() || end != text.data() + text.size() || iterations < 1 ||
        iterations > 100000) {
        return -1;
    }
    return iterations;
}

}  // namespace

int main(int argc, char** argv) {
    const int iterations = parse_iterations(argc, argv);
    if (iterations < 1) {
        std::cerr << "usage: " << argv[0] << " [--iterations 1..100000]\n";
        return 2;
    }

    std::array<std::int8_t, 8 * 8> weights {};
    std::array<std::int32_t, 8> bias {};
    std::array<float, 8> weight_scales {};
    weight_scales.fill(1.0F);

    Y26Stage7ConvNodeConfig config {};
    config.node_name = "stage61/startup-readiness";
    config.params = Y26Conv2DParams {8, 8, 8, 8, 1, 1, 0, 0};
    config.kernel_h = 1;
    config.kernel_w = 1;
    config.input_scale = 1.0F;
    config.output_scale = 1.0F;
    config.weight_scales = weight_scales.data();
    config.weight_scale_count = weight_scales.size();
    config.weights_ohwi_s8 = weights.data();
    config.weight_count = weights.size();
    config.bias_i32 = bias.data();
    config.bias_count = bias.size();

    std::uint64_t workspaces = 0;
    std::uint64_t readiness_transitions = 0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int workers = 1; workers <= 4; ++workers) {
            Y26ThreadedConvWorkspace* workspace =
                y26_threaded_conv_create_spatial_rows(&config, workers);
            if (workspace == nullptr) {
                std::cerr << "workspace creation failed iteration=" << iteration
                          << " workers=" << workers << '\n';
                return 1;
            }
            Y26ThreadedConvPlan plan {};
            if (y26_threaded_conv_get_plan(workspace, &plan) != Y26_CONV_STATUS_SUCCESS ||
                plan.thread_count != workers) {
                y26_threaded_conv_destroy(workspace);
                std::cerr << "workspace plan mismatch iteration=" << iteration
                          << " workers=" << workers << '\n';
                return 1;
            }
            y26_threaded_conv_destroy(workspace);
            ++workspaces;
            readiness_transitions += static_cast<std::uint64_t>(workers);
        }
    }

    std::cout << "stage61_threaded_startup iterations=" << iterations
              << " workspaces=" << workspaces
              << " worker_readiness_transitions=" << readiness_transitions
              << " failures=0\n";
    return 0;
}
