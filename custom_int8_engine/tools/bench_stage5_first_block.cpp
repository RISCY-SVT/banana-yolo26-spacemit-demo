#include "stage5_block0_fixture.h"
#include "y26_k1x_block_runner.h"
#include "y26_k1x_vmadot.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct BenchResult {
    double mean_us;
    std::int64_t checksum;
    int status;
};

template <typename Fn>
BenchResult measure(int iterations, Fn&& fn) {
    int status = 0;
    std::int64_t checksum = 0;
    const auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto result = fn();
        status = result.first;
        checksum += result.second;
    }
    const auto end = Clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return {static_cast<double>(ns) / 1000.0 / static_cast<double>(iterations), checksum, status};
}

std::int64_t checksum_i32(const std::vector<std::int32_t>& values) {
    return std::accumulate(values.begin(), values.end(), std::int64_t{0});
}

Y26Stage5Block0Config full_block0_config() {
    const auto& fixture = y26_stage5_block0_fixture::kSyntheticSeededFixture;
    Y26Conv2DParams params = fixture.params;
    params.input_h = 640;
    params.input_w = 640;
    return Y26Stage5Block0Config{
        "block0_conv_only",
        fixture.node_name,
        params,
        fixture.kernel_h,
        fixture.kernel_w,
        fixture.activation_zero_point_u8,
        fixture.input_storage_zero_point_s8,
        fixture.weights_ohwi_s8,
        fixture.weight_count,
        fixture.bias_i32,
        fixture.bias_count,
    };
}

std::vector<std::int8_t> make_input(const Y26Conv2DParams& params) {
    std::vector<std::int8_t> input(static_cast<std::size_t>(params.input_h * params.input_w * params.input_c), 0);
    for (std::size_t i = 0; i < input.size(); ++i) {
        const int q = static_cast<int>((i * 29 + 17) & 255);
        input[i] = static_cast<std::int8_t>(q - 128);
    }
    return input;
}

int align_up(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

void store_a_value(std::vector<std::int8_t>& workspace, int m, int flat_k, std::int8_t value) {
    const int k_tile = flat_k / 8;
    const int k_lane = flat_k % 8;
    workspace[static_cast<std::size_t>(k_tile * 32 + m * 8 + k_lane)] = value;
}

std::int64_t pack_a_probe_checksum(const std::vector<std::int8_t>& input,
                                   const Y26Stage5Block0Config& cfg,
                                   std::vector<std::int8_t>& workspace) {
    const int output_h = y26_stage5_block0_output_h(&cfg);
    const int output_w = y26_stage5_block0_output_w(&cfg);
    const int output_m = output_h * output_w;
    const int kernel_k = cfg.kernel_h * cfg.kernel_w * cfg.conv0_params.input_c;
    const int k_padded = align_up(kernel_k, 8);
    std::int64_t checksum = 0;
    for (int m0 = 0; m0 < output_m; m0 += 4) {
        std::fill(workspace.begin(), workspace.begin() + 4 * k_padded, std::int8_t{0});
        for (int m = 0; m < 4; ++m) {
            const int flat_m = m0 + m;
            if (flat_m >= output_m) {
                continue;
            }
            const int oh = flat_m / output_w;
            const int ow = flat_m - oh * output_w;
            int flat_k = 0;
            for (int kh = 0; kh < cfg.kernel_h; ++kh) {
                const int ih = oh * cfg.conv0_params.stride_h + kh - cfg.conv0_params.pad_h;
                const bool valid_h = ih >= 0 && ih < cfg.conv0_params.input_h;
                for (int kw = 0; kw < cfg.kernel_w; ++kw) {
                    const int iw = ow * cfg.conv0_params.stride_w + kw - cfg.conv0_params.pad_w;
                    const bool inside = valid_h && iw >= 0 && iw < cfg.conv0_params.input_w;
                    const std::int8_t* src =
                        inside ? input.data() + (ih * cfg.conv0_params.input_w + iw) * cfg.conv0_params.input_c
                               : nullptr;
                    for (int ic = 0; ic < cfg.conv0_params.input_c; ++ic, ++flat_k) {
                        const std::int8_t value =
                            inside ? src[ic] : static_cast<std::int8_t>(cfg.input_storage_zero_point_s8);
                        store_a_value(workspace, m, flat_k, value);
                        checksum += value;
                    }
                }
            }
        }
    }
    return checksum;
}

}  // namespace

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::max(1, std::atoi(argv[1])) : 3;
    Y26Stage5Block0Config cfg = full_block0_config();
    const int output_h = y26_stage5_block0_output_h(&cfg);
    const int output_w = y26_stage5_block0_output_w(&cfg);
    const std::size_t output_count = y26_stage5_block0_output_count(&cfg);
    std::vector<std::int8_t> input = make_input(cfg.conv0_params);
    std::vector<std::int32_t> output(output_count, 0);
    std::vector<std::int8_t> pack_probe_workspace(
        y26_conv_mmt4d_a_workspace_bytes(&cfg.conv0_params, cfg.kernel_h, cfg.kernel_w), 0);

    const auto prepack_once = measure(iterations, [&]() {
        Y26Stage5Block0Workspace tmp {};
        const int status = y26_stage5_block0_prepare(&cfg, &tmp);
        const std::int64_t checksum =
            status == Y26_CONV_STATUS_SUCCESS
                ? static_cast<std::int64_t>(tmp.prepacked_bytes + tmp.workspace_bytes + tmp.raw_i32_bytes)
                : -1;
        y26_stage5_block0_release(&tmp);
        return std::make_pair(status, checksum);
    });

    Y26Stage5Block0Workspace ws {};
    const int prepare_status = y26_stage5_block0_prepare(&cfg, &ws);
    if (prepare_status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage5 block prepare failed status=" << prepare_status << "\n";
        return 1;
    }

    const auto pack_a_probe = measure(iterations, [&]() {
        return std::make_pair(0, pack_a_probe_checksum(input, cfg, pack_probe_workspace));
    });

    const auto scalar = measure(iterations, [&]() {
        const int status = y26_stage5_block0_run_scalar(&cfg, &ws, input.data(), output.data());
        return std::make_pair(status, checksum_i32(output));
    });

    BenchResult ime {0.0, 0, Y26_CONV_STATUS_NOT_BUILT_WITH_IME};
    if (y26_vmadot_4x4x8_ime_available_buildtime()) {
        ime = measure(iterations, [&]() {
            const int status = y26_stage5_block0_run_ime_cluster0_hotpath(&cfg, &ws, input.data(), output.data());
            return std::make_pair(status, checksum_i32(output));
        });
    }

    const double residual_compute_us =
        ime.status == Y26_CONV_STATUS_SUCCESS ? std::max(0.0, ime.mean_us - pack_a_probe.mean_us) : 0.0;

    std::cout << "STAGE5_BLOCK_BENCH_BEGIN\n";
    std::cout << "case=block0_conv_only node=/model.0/conv/Conv shape=" << cfg.conv0_params.input_h << "x"
              << cfg.conv0_params.input_w << "x" << cfg.conv0_params.input_c << "->" << output_h << "x"
              << output_w << "x" << cfg.conv0_params.output_c << " iterations=" << iterations << "\n";
    std::cout << "scalar_total_us=" << scalar.mean_us << " status=" << scalar.status
              << " checksum=" << scalar.checksum << "\n";
    std::cout << "ime_prepack_one_time_us=" << prepack_once.mean_us << " status=" << prepack_once.status
              << " checksum=" << prepack_once.checksum << "\n";
    std::cout << "ime_packA_probe_us=" << pack_a_probe.mean_us << " status=" << pack_a_probe.status
              << " checksum=" << pack_a_probe.checksum << "\n";
    std::cout << "ime_total_packing_included_us=" << ime.mean_us << " status=" << ime.status
              << " checksum=" << ime.checksum << "\n";
    std::cout << "ime_compute_plus_correction_residual_us=" << residual_compute_us << "\n";
    std::cout << "stage4_nearest_conv3x3_reference_us=37097.9 caveat=not_same_shape\n";
    std::cout << "prepacked_bytes=" << ws.prepacked_bytes << " workspace_bytes=" << ws.workspace_bytes
              << " raw_i32_bytes=" << ws.raw_i32_bytes << "\n";
    std::cout << "microbench_scope=selected_block_only_no_yolo26_fps\n";
    std::cout << "STAGE5_BLOCK_BENCH_END\n";

    y26_stage5_block0_release(&ws);
    return (scalar.status == Y26_CONV_STATUS_SUCCESS &&
            (!y26_vmadot_4x4x8_ime_available_buildtime() || ime.status == Y26_CONV_STATUS_SUCCESS))
               ? 0
               : 1;
}
