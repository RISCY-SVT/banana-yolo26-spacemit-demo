#include "stage6_multiblock_fixture.h"
#include "stage7_backbone_subset_fixture.h"
#include "y26_k1x_backbone_subset_runner.h"
#include "y26_k1x_multiblock_runner.h"
#include "y26_k1x_vmadot.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct BenchResult {
    double mean_us;
    Y26Stage7TimingUs timing;
    std::int64_t checksum;
    int status;
};

struct Stage6BenchResult {
    double mean_us;
    Y26Stage6TimingUs timing;
    std::int64_t checksum;
    int status;
};

std::int64_t checksum_i32(const std::vector<std::int32_t>& values) {
    return std::accumulate(values.begin(), values.end(), std::int64_t{0});
}

std::vector<std::int8_t> make_input(const Y26Conv2DParams& params, int seed) {
    std::vector<std::int8_t> input(static_cast<std::size_t>(params.input_h * params.input_w * params.input_c), 0);
    for (std::size_t i = 0; i < input.size(); ++i) {
        const int q = static_cast<int>((i * 37 + seed * 19) & 255);
        input[i] = static_cast<std::int8_t>(q - 128);
    }
    return input;
}

Y26Stage7ConvNodeConfig stage7_conv0_config(
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture,
    const Y26Conv2DParams& params) {
    return Y26Stage7ConvNodeConfig{fixture.conv0_node_name,
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
                                   fixture.conv0_bias_count};
}

Y26Stage7ConvNodeConfig stage7_conv1_config(
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture,
    const Y26Conv2DParams& params) {
    return Y26Stage7ConvNodeConfig{fixture.conv1_node_name,
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
                                   fixture.conv1_bias_count};
}

Y26Stage7ConvNodeConfig stage7_conv2_config(
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture,
    const Y26Conv2DParams& params) {
    return Y26Stage7ConvNodeConfig{fixture.conv2_node_name,
                                   params,
                                   fixture.conv2_kernel_h,
                                   fixture.conv2_kernel_w,
                                   fixture.act1_output_zero_point_u8,
                                   fixture.conv2_input_storage_zero_point_s8,
                                   fixture.act1_output_scale,
                                   fixture.conv2_output_scale,
                                   fixture.conv2_output_zero_point_u8,
                                   fixture.conv2_weight_scales,
                                   fixture.conv2_weight_scale_count,
                                   fixture.conv2_weights_ohwi_s8,
                                   fixture.conv2_weight_count,
                                   fixture.conv2_bias_i32,
                                   fixture.conv2_bias_count};
}

Y26Stage7BackboneSubsetConfig stage7_full_shape_config(
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture) {
    return Y26Stage7BackboneSubsetConfig{
        fixture.subset_id,
        stage7_conv0_config(fixture, Y26Conv2DParams{640, 640, 3, 16, 2, 2, 1, 1}),
        stage7_conv1_config(fixture, Y26Conv2DParams{320, 320, 16, 32, 2, 2, 1, 1}),
        stage7_conv2_config(fixture, Y26Conv2DParams{160, 160, 32, 32, 1, 1, 0, 0}),
        fixture.act0_output_scale,
        fixture.act0_output_zero_point_u8,
        fixture.act1_output_scale,
        fixture.act1_output_zero_point_u8,
    };
}

Y26Stage6ConvNodeConfig stage6_conv0_config(const y26_stage6_multiblock_fixture::MultiblockFixture& fixture,
                                            const Y26Conv2DParams& params) {
    return Y26Stage6ConvNodeConfig{fixture.conv0_node_name,
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
                                   fixture.conv0_bias_count};
}

Y26Stage6ConvNodeConfig stage6_conv1_config(const y26_stage6_multiblock_fixture::MultiblockFixture& fixture,
                                            const Y26Conv2DParams& params) {
    return Y26Stage6ConvNodeConfig{fixture.conv1_node_name,
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
                                   fixture.conv1_bias_count};
}

Y26Stage6MultiblockConfig stage6_full_shape_config(
    const y26_stage6_multiblock_fixture::MultiblockFixture& fixture) {
    return Y26Stage6MultiblockConfig{fixture.subset_id,
                                     stage6_conv0_config(fixture, Y26Conv2DParams{640, 640, 3, 16, 2, 2, 1, 1}),
                                     stage6_conv1_config(fixture, Y26Conv2DParams{320, 320, 16, 32, 2, 2, 1, 1}),
                                     fixture.act0_output_scale,
                                     fixture.act0_output_zero_point_u8};
}

template <typename Fn>
BenchResult measure_stage7(int iterations, Fn&& fn) {
    int status = 0;
    std::int64_t checksum = 0;
    Y26Stage7TimingUs timing_sum {};
    const auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        Y26Stage7TimingUs timing {};
        status = fn(timing);
        checksum += fn.checksum();
        timing_sum.conv0_us += timing.conv0_us;
        timing_sum.act0_requant_us += timing.act0_requant_us;
        timing_sum.conv1_us += timing.conv1_us;
        timing_sum.act1_requant_us += timing.act1_requant_us;
        timing_sum.conv2_us += timing.conv2_us;
        timing_sum.total_us += timing.total_us;
    }
    const auto end = Clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    const double denom = static_cast<double>(iterations);
    timing_sum.conv0_us /= denom;
    timing_sum.act0_requant_us /= denom;
    timing_sum.conv1_us /= denom;
    timing_sum.act1_requant_us /= denom;
    timing_sum.conv2_us /= denom;
    timing_sum.total_us /= denom;
    return {static_cast<double>(ns) / 1000.0 / denom, timing_sum, checksum, status};
}

template <typename Fn>
Stage6BenchResult measure_stage6(int iterations, Fn&& fn) {
    int status = 0;
    std::int64_t checksum = 0;
    Y26Stage6TimingUs timing_sum {};
    const auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        Y26Stage6TimingUs timing {};
        status = fn(timing);
        checksum += fn.checksum();
        timing_sum.conv0_us += timing.conv0_us;
        timing_sum.activation_us += timing.activation_us;
        timing_sum.conv1_us += timing.conv1_us;
        timing_sum.total_us += timing.total_us;
    }
    const auto end = Clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    const double denom = static_cast<double>(iterations);
    timing_sum.conv0_us /= denom;
    timing_sum.activation_us /= denom;
    timing_sum.conv1_us /= denom;
    timing_sum.total_us /= denom;
    return {static_cast<double>(ns) / 1000.0 / denom, timing_sum, checksum, status};
}

struct Stage7ScalarCall {
    const Y26Stage7BackboneSubsetConfig* cfg;
    Y26Stage7BackboneSubsetWorkspace* ws;
    const std::vector<std::int8_t>* input;
    std::vector<std::int32_t>* output;
    int operator()(Y26Stage7TimingUs& timing) const {
        return y26_stage7_backbone_subset_run_scalar(cfg, ws, input->data(), output->data(), &timing);
    }
    std::int64_t checksum() const { return checksum_i32(*output); }
};

struct Stage7ImeCall {
    const Y26Stage7BackboneSubsetConfig* cfg;
    Y26Stage7BackboneSubsetWorkspace* ws;
    const std::vector<std::int8_t>* input;
    std::vector<std::int32_t>* output;
    int operator()(Y26Stage7TimingUs& timing) const {
        return y26_stage7_backbone_subset_run_ime_cluster0_hotpath(cfg, ws, input->data(), output->data(), &timing);
    }
    std::int64_t checksum() const { return checksum_i32(*output); }
};

struct Stage6ImeCall {
    const Y26Stage6MultiblockConfig* cfg;
    Y26Stage6MultiblockWorkspace* ws;
    const std::vector<std::int8_t>* input;
    std::vector<std::int32_t>* output;
    int operator()(Y26Stage6TimingUs& timing) const {
        return y26_stage6_multiblock_run_ime_cluster0_hotpath(cfg, ws, input->data(), output->data(), &timing);
    }
    std::int64_t checksum() const { return checksum_i32(*output); }
};

}  // namespace

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::max(1, std::atoi(argv[1])) : 1;
    const auto& fixture = y26_stage7_backbone_subset_fixture::kSyntheticSeededFixture;
    Y26Stage7BackboneSubsetConfig cfg = stage7_full_shape_config(fixture);
    Y26Stage7BackboneSubsetWorkspace ws {};
    const int prepare_status = y26_stage7_backbone_subset_prepare(&cfg, &ws);
    if (prepare_status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage7_prepare_status=" << prepare_status << "\n";
        return 1;
    }

    std::vector<std::int8_t> input = make_input(cfg.conv0.params, 31);
    std::vector<std::int32_t> output(y26_stage7_backbone_subset_conv2_output_count(&cfg), 0);

    const auto scalar = measure_stage7(iterations, Stage7ScalarCall{&cfg, &ws, &input, &output});

    BenchResult ime {0.0, {}, 0, Y26_CONV_STATUS_NOT_BUILT_WITH_IME};
    if (y26_vmadot_4x4x8_ime_available_buildtime()) {
        (void)y26_k1x_ime_probe_once();
        ime = measure_stage7(iterations, Stage7ImeCall{&cfg, &ws, &input, &output});
    }

    const auto& stage6_fixture = y26_stage6_multiblock_fixture::kSyntheticSeededFixture;
    Y26Stage6MultiblockConfig stage6_cfg = stage6_full_shape_config(stage6_fixture);
    Y26Stage6MultiblockWorkspace stage6_ws {};
    Stage6BenchResult stage6_replay {0.0, {}, 0, Y26_CONV_STATUS_NOT_BUILT_WITH_IME};
    if (y26_stage6_multiblock_prepare(&stage6_cfg, &stage6_ws) == Y26_CONV_STATUS_SUCCESS &&
        y26_vmadot_4x4x8_ime_available_buildtime()) {
        std::vector<std::int32_t> stage6_output(y26_stage6_multiblock_conv1_output_count(&stage6_cfg), 0);
        stage6_replay = measure_stage6(iterations, Stage6ImeCall{&stage6_cfg, &stage6_ws, &input, &stage6_output});
    }

    std::cout << "STAGE7_BACKBONE_SUBSET_BENCH_BEGIN\n";
    std::cout << "note=selected-subset microbenchmark only, not YOLO26 inference FPS\n";
    std::cout << "subset=" << fixture.subset_id << " iterations=" << iterations << "\n";
    std::cout << "shape=640x640x3->320x320x16_silu->160x160x32_silu->160x160x32\n";
    std::cout << "scalar_total_us=" << scalar.mean_us << " status=" << scalar.status
              << " checksum=" << scalar.checksum << " conv0_us=" << scalar.timing.conv0_us
              << " act0_requant_us=" << scalar.timing.act0_requant_us
              << " conv1_us=" << scalar.timing.conv1_us
              << " act1_requant_us=" << scalar.timing.act1_requant_us
              << " conv2_us=" << scalar.timing.conv2_us << "\n";
    std::cout << "ime_total_us=" << ime.mean_us << " status=" << ime.status
              << " checksum=" << ime.checksum << " conv0_us=" << ime.timing.conv0_us
              << " act0_requant_us=" << ime.timing.act0_requant_us
              << " conv1_us=" << ime.timing.conv1_us
              << " act1_requant_us=" << ime.timing.act1_requant_us
              << " conv2_us=" << ime.timing.conv2_us << "\n";
    std::cout << "stage6_replay_ime_total_us=" << stage6_replay.mean_us
              << " status=" << stage6_replay.status << " checksum=" << stage6_replay.checksum
              << " conv0_us=" << stage6_replay.timing.conv0_us
              << " activation_us=" << stage6_replay.timing.activation_us
              << " conv1_us=" << stage6_replay.timing.conv1_us << "\n";
    std::cout << "prepacked_bytes=" << ws.prepacked_bytes << " workspace_bytes=" << ws.workspace_bytes << "\n";
    if (scalar.status == Y26_CONV_STATUS_SUCCESS && ime.status == Y26_CONV_STATUS_SUCCESS && ime.mean_us > 0.0) {
        std::cout << "speedup_ime_vs_scalar=" << (scalar.mean_us / ime.mean_us) << "\n";
        const double activation_sum = ime.timing.act0_requant_us + ime.timing.act1_requant_us;
        std::cout << "activation_requant_total_us=" << activation_sum
                  << " activation_requant_pct_of_ime_total=" << (100.0 * activation_sum / ime.mean_us) << "\n";
    }
    std::cout << "STAGE7_BACKBONE_SUBSET_BENCH_END\n";

    y26_stage6_multiblock_release(&stage6_ws);
    y26_stage7_backbone_subset_release(&ws);
    return scalar.status == Y26_CONV_STATUS_SUCCESS ? 0 : 1;
}
