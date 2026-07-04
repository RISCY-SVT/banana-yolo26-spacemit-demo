#include "stage6_multiblock_fixture.h"
#include "y26_k1x_block_runner.h"
#include "y26_k1x_multiblock_runner.h"
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
    double conv0_us;
    double activation_us;
    double conv1_us;
    std::int64_t checksum;
    int status;
};

int output_h(const Y26Conv2DParams& params, int kernel_h) {
    return kernel_h == 1 ? y26_conv1x1_output_h(&params) : y26_conv3x3_output_h(&params);
}

int output_w(const Y26Conv2DParams& params, int kernel_w) {
    return kernel_w == 1 ? y26_conv1x1_output_w(&params) : y26_conv3x3_output_w(&params);
}

int align_up(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

std::int64_t checksum_i32(const std::vector<std::int32_t>& values) {
    return std::accumulate(values.begin(), values.end(), std::int64_t{0});
}

std::int64_t checksum_i8(const std::vector<std::int8_t>& values) {
    std::int64_t checksum = 0;
    for (std::int8_t value : values) {
        checksum += static_cast<int>(value);
    }
    return checksum;
}

std::vector<std::int8_t> make_input(const Y26Conv2DParams& params, int seed) {
    std::vector<std::int8_t> input(static_cast<std::size_t>(params.input_h * params.input_w * params.input_c), 0);
    for (std::size_t i = 0; i < input.size(); ++i) {
        const int q = static_cast<int>((i * 37 + seed * 19) & 255);
        input[i] = static_cast<std::int8_t>(q - 128);
    }
    return input;
}

Y26Stage6ConvNodeConfig conv0_config_from_fixture(const y26_stage6_multiblock_fixture::MultiblockFixture& fixture,
                                                  const Y26Conv2DParams& params) {
    return Y26Stage6ConvNodeConfig{
        fixture.conv0_node_name,
        params,
        fixture.conv0_kernel_h,
        fixture.conv0_kernel_w,
        fixture.conv0_activation_zero_point_u8,
        fixture.conv0_input_storage_zero_point_s8,
        fixture.images_scale,
        fixture.conv0_output_scale,
        fixture.conv0_output_zero_point_u8,
        fixture.conv0_weight_scales,
        fixture.conv0_weight_scale_count,
        fixture.conv0_weights_ohwi_s8,
        fixture.conv0_weight_count,
        fixture.conv0_bias_i32,
        fixture.conv0_bias_count,
    };
}

Y26Stage6ConvNodeConfig conv1_config_from_fixture(const y26_stage6_multiblock_fixture::MultiblockFixture& fixture,
                                                  const Y26Conv2DParams& params) {
    return Y26Stage6ConvNodeConfig{
        fixture.conv1_node_name,
        params,
        fixture.conv1_kernel_h,
        fixture.conv1_kernel_w,
        fixture.act0_output_zero_point_u8,
        fixture.conv1_input_storage_zero_point_s8,
        fixture.act0_output_scale,
        fixture.conv1_output_scale,
        fixture.conv1_output_zero_point_u8,
        fixture.conv1_weight_scales,
        fixture.conv1_weight_scale_count,
        fixture.conv1_weights_ohwi_s8,
        fixture.conv1_weight_count,
        fixture.conv1_bias_i32,
        fixture.conv1_bias_count,
    };
}

Y26Stage6MultiblockConfig full_shape_config(const y26_stage6_multiblock_fixture::MultiblockFixture& fixture) {
    const Y26Conv2DParams conv0 {640, 640, 3, 16, 2, 2, 1, 1};
    const Y26Conv2DParams conv1 {320, 320, 16, 32, 2, 2, 1, 1};
    return Y26Stage6MultiblockConfig{
        fixture.subset_id,
        conv0_config_from_fixture(fixture, conv0),
        conv1_config_from_fixture(fixture, conv1),
        fixture.act0_output_scale,
        fixture.act0_output_zero_point_u8,
    };
}

Y26Stage5Block0Config stage5_full_shape_config(const y26_stage6_multiblock_fixture::MultiblockFixture& fixture) {
    return Y26Stage5Block0Config{
        "block0_conv_only",
        fixture.conv0_node_name,
        Y26Conv2DParams{640, 640, 3, 16, 2, 2, 1, 1},
        fixture.conv0_kernel_h,
        fixture.conv0_kernel_w,
        fixture.conv0_activation_zero_point_u8,
        fixture.conv0_input_storage_zero_point_s8,
        fixture.conv0_weights_ohwi_s8,
        fixture.conv0_weight_count,
        fixture.conv0_bias_i32,
        fixture.conv0_bias_count,
    };
}

template <typename Fn>
BenchResult measure(int iterations, Fn&& fn) {
    int status = 0;
    std::int64_t checksum = 0;
    double conv0_us = 0.0;
    double activation_us = 0.0;
    double conv1_us = 0.0;
    const auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto result = fn();
        status = result.first;
        checksum += result.second;
        conv0_us += result.third.conv0_us;
        activation_us += result.third.activation_us;
        conv1_us += result.third.conv1_us;
    }
    const auto end = Clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return {static_cast<double>(ns) / 1000.0 / static_cast<double>(iterations),
            conv0_us / static_cast<double>(iterations),
            activation_us / static_cast<double>(iterations),
            conv1_us / static_cast<double>(iterations),
            checksum,
            status};
}

struct RunResult {
    int first;
    std::int64_t second;
    Y26Stage6TimingUs third;
};

void store_a_value(std::vector<std::int8_t>& workspace, int m, int flat_k, std::int8_t value) {
    const int k_tile = flat_k / 8;
    const int k_lane = flat_k % 8;
    workspace[static_cast<std::size_t>(k_tile * 32 + m * 8 + k_lane)] = value;
}

std::int64_t pack_a_probe_checksum(const std::vector<std::int8_t>& input,
                                   const Y26Conv2DParams& params,
                                   int kernel_h,
                                   int kernel_w,
                                   std::int8_t storage_zero_point,
                                   std::vector<std::int8_t>& workspace) {
    const int oh_count = output_h(params, kernel_h);
    const int ow_count = output_w(params, kernel_w);
    const int output_m = oh_count * ow_count;
    const int kernel_k = kernel_h * kernel_w * params.input_c;
    const int k_padded = align_up(kernel_k, 8);
    std::int64_t checksum = 0;
    for (int m0 = 0; m0 < output_m; m0 += 4) {
        std::fill(workspace.begin(), workspace.begin() + 4 * k_padded, std::int8_t{0});
        for (int m = 0; m < 4; ++m) {
            const int flat_m = m0 + m;
            if (flat_m >= output_m) {
                continue;
            }
            const int oh = flat_m / ow_count;
            const int ow = flat_m - oh * ow_count;
            int flat_k = 0;
            for (int kh = 0; kh < kernel_h; ++kh) {
                const int ih = oh * params.stride_h + kh - params.pad_h;
                const bool valid_h = ih >= 0 && ih < params.input_h;
                for (int kw = 0; kw < kernel_w; ++kw) {
                    const int iw = ow * params.stride_w + kw - params.pad_w;
                    const bool inside = valid_h && iw >= 0 && iw < params.input_w;
                    const std::int8_t* src =
                        inside ? input.data() + (ih * params.input_w + iw) * params.input_c : nullptr;
                    for (int ic = 0; ic < params.input_c; ++ic, ++flat_k) {
                        const std::int8_t value = inside ? src[ic] : storage_zero_point;
                        store_a_value(workspace, m, flat_k, value);
                        checksum += value;
                    }
                }
            }
        }
    }
    return checksum;
}

double measure_pack_a_probe(int iterations,
                            const std::vector<std::int8_t>& input,
                            const Y26Conv2DParams& params,
                            int kernel_h,
                            int kernel_w,
                            int storage_zero_point,
                            std::int64_t& checksum_out) {
    std::vector<std::int8_t> workspace(
        y26_conv_mmt4d_a_workspace_bytes(&params, kernel_h, kernel_w), std::int8_t{0});
    checksum_out = 0;
    const auto begin = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        checksum_out += pack_a_probe_checksum(
            input, params, kernel_h, kernel_w, static_cast<std::int8_t>(storage_zero_point), workspace);
    }
    const auto end = Clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    return static_cast<double>(ns) / 1000.0 / static_cast<double>(iterations);
}

}  // namespace

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::max(1, std::atoi(argv[1])) : 1;
    const auto& fixture = y26_stage6_multiblock_fixture::kSyntheticSeededFixture;
    Y26Stage6MultiblockConfig cfg = full_shape_config(fixture);
    Y26Stage6MultiblockWorkspace ws {};
    const int prepare_status = y26_stage6_multiblock_prepare(&cfg, &ws);
    if (prepare_status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "prepare_status=" << prepare_status << "\n";
        return 1;
    }
    std::vector<std::int8_t> input = make_input(cfg.conv0.params, 31);
    std::vector<std::int32_t> output(y26_stage6_multiblock_conv1_output_count(&cfg), 0);
    std::vector<std::int32_t> stage5_output;
    Y26Stage5Block0Workspace stage5_ws {};
    Y26Stage5Block0Config stage5_cfg = stage5_full_shape_config(fixture);
    const int stage5_prepare_status = y26_stage5_block0_prepare(&stage5_cfg, &stage5_ws);
    if (stage5_prepare_status == Y26_CONV_STATUS_SUCCESS) {
        stage5_output.assign(y26_stage5_block0_output_count(&stage5_cfg), 0);
    }

    const auto scalar = measure(iterations, [&]() {
        Y26Stage6TimingUs timing {};
        const int status = y26_stage6_multiblock_run_scalar(&cfg, &ws, input.data(), output.data(), &timing);
        return RunResult{status, checksum_i32(output), timing};
    });

    BenchResult ime {0.0, 0.0, 0.0, 0.0, 0, Y26_CONV_STATUS_NOT_BUILT_WITH_IME};
    BenchResult stage5_replay {
        0.0, 0.0, 0.0, 0.0, 0, stage5_prepare_status == Y26_CONV_STATUS_SUCCESS
                                           ? Y26_CONV_STATUS_NOT_BUILT_WITH_IME
                                           : stage5_prepare_status};
    if (y26_vmadot_4x4x8_ime_available_buildtime()) {
        (void)y26_k1x_ime_probe_once();
        ime = measure(iterations, [&]() {
            Y26Stage6TimingUs timing {};
            const int status =
                y26_stage6_multiblock_run_ime_cluster0_hotpath(&cfg, &ws, input.data(), output.data(), &timing);
            return RunResult{status, checksum_i32(output), timing};
        });
        if (stage5_prepare_status == Y26_CONV_STATUS_SUCCESS) {
            const auto begin = Clock::now();
            int status = 0;
            std::int64_t checksum = 0;
            for (int i = 0; i < iterations; ++i) {
                status =
                    y26_stage5_block0_run_ime_cluster0_hotpath(&stage5_cfg, &stage5_ws, input.data(), stage5_output.data());
                checksum += checksum_i32(stage5_output);
            }
            const auto end = Clock::now();
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
            stage5_replay = {static_cast<double>(ns) / 1000.0 / static_cast<double>(iterations),
                             0.0,
                             0.0,
                             0.0,
                             checksum,
                             status};
        }
    }

    std::int64_t conv0_pack_checksum = 0;
    const double conv0_pack_a_us = measure_pack_a_probe(iterations,
                                                        input,
                                                        cfg.conv0.params,
                                                        cfg.conv0.kernel_h,
                                                        cfg.conv0.kernel_w,
                                                        cfg.conv0.input_storage_zero_point_s8,
                                                        conv0_pack_checksum);
    std::vector<std::int8_t> conv1_input_copy(ws.conv1_input_s8, ws.conv1_input_s8 + ws.conv1_input_count);
    std::int64_t conv1_pack_checksum = 0;
    const double conv1_pack_a_us = measure_pack_a_probe(iterations,
                                                        conv1_input_copy,
                                                        cfg.conv1.params,
                                                        cfg.conv1.kernel_h,
                                                        cfg.conv1.kernel_w,
                                                        cfg.conv1.input_storage_zero_point_s8,
                                                        conv1_pack_checksum);

    std::cout << "STAGE6_MULTIBLOCK_BENCH_BEGIN\n";
    std::cout << "note=selected-subset microbenchmark only, not YOLO26 inference FPS\n";
    std::cout << "subset=" << fixture.subset_id << " iterations=" << iterations << "\n";
    std::cout << "shape=640x640x3->320x320x16_silu_requant->160x160x32\n";
    std::cout << "scalar_total_us=" << scalar.mean_us << " status=" << scalar.status << " checksum="
              << scalar.checksum << " conv0_us=" << scalar.conv0_us << " activation_us=" << scalar.activation_us
              << " conv1_us=" << scalar.conv1_us << "\n";
    std::cout << "ime_total_us=" << ime.mean_us << " status=" << ime.status << " checksum=" << ime.checksum
              << " conv0_us=" << ime.conv0_us << " activation_us=" << ime.activation_us
              << " conv1_us=" << ime.conv1_us << "\n";
    std::cout << "stage5_block0_replay_ime_us=" << stage5_replay.mean_us << " status=" << stage5_replay.status
              << " checksum=" << stage5_replay.checksum << "\n";
    std::cout << "conv0_packA_probe_us=" << conv0_pack_a_us << " checksum=" << conv0_pack_checksum << "\n";
    std::cout << "conv1_packA_probe_us=" << conv1_pack_a_us << " checksum=" << conv1_pack_checksum << "\n";
    std::cout << "prepacked_bytes=" << ws.prepacked_bytes << " workspace_bytes=" << ws.workspace_bytes
              << " conv1_input_checksum=" << checksum_i8(conv1_input_copy) << "\n";
    std::cout << "STAGE6_MULTIBLOCK_BENCH_END\n";

    y26_stage5_block0_release(&stage5_ws);
    y26_stage6_multiblock_release(&ws);
    return (scalar.status == Y26_CONV_STATUS_SUCCESS &&
            (!y26_vmadot_4x4x8_ime_available_buildtime() || ime.status == Y26_CONV_STATUS_SUCCESS))
               ? 0
               : 1;
}
