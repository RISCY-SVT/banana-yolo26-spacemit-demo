#include "stage10_backbone_expansion_fixture.h"
#include "y26_k1x_backbone_stage10_runner.h"
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
    Y26Stage10TimingUs timing;
    std::int64_t checksum;
    int status;
    std::size_t mismatches;
};

struct Stage9BenchResult {
    double mean_us;
    Y26Stage7TimingUs timing;
    std::int64_t checksum;
    int status;
    std::size_t mismatches;
};

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000.0;
}

std::int64_t checksum_i32(const std::vector<std::int32_t>& values) {
    return std::accumulate(values.begin(), values.end(), std::int64_t{0});
}

std::size_t mismatches_i32(const std::vector<std::int32_t>& actual, const std::vector<std::int32_t>& expected) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < actual.size() && i < expected.size(); ++i) {
        mismatches += actual[i] != expected[i] ? 1U : 0U;
    }
    return mismatches + (actual.size() > expected.size() ? actual.size() - expected.size()
                                                         : expected.size() - actual.size());
}

std::vector<std::int8_t> make_input(const Y26Conv2DParams& params, int seed) {
    std::vector<std::int8_t> input(static_cast<std::size_t>(params.input_h * params.input_w * params.input_c), 0);
    for (std::size_t i = 0; i < input.size(); ++i) {
        const int q = static_cast<int>((i * 37 + seed * 19) & 255);
        input[i] = static_cast<std::int8_t>(q - 128);
    }
    return input;
}

Y26Stage7ConvNodeConfig conv0_config(
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

Y26Stage7ConvNodeConfig conv1_config(
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

Y26Stage7ConvNodeConfig conv2_config(
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

Y26Stage7BackboneSubsetConfig full_stage9_config(
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture,
    int activation_mode) {
    return Y26Stage7BackboneSubsetConfig{
        fixture.subset_id,
        conv0_config(fixture, Y26Conv2DParams{640, 640, 3, 16, 2, 2, 1, 1}),
        conv1_config(fixture, Y26Conv2DParams{320, 320, 16, 32, 2, 2, 1, 1}),
        conv2_config(fixture, Y26Conv2DParams{160, 160, 32, 32, 1, 1, 0, 0}),
        fixture.act0_output_scale,
        fixture.act0_output_zero_point_u8,
        fixture.act1_output_scale,
        fixture.act1_output_zero_point_u8,
        activation_mode,
    };
}

Y26Stage7ConvNodeConfig full_branch0_config(
    const y26_stage10_backbone_expansion_fixture::BackboneExpansionFixture& fixture) {
    return Y26Stage7ConvNodeConfig{fixture.branch0_node_name,
                                   Y26Conv2DParams{160, 160, 16, 8, 1, 1, 1, 1},
                                   fixture.branch0_kernel_h,
                                   fixture.branch0_kernel_w,
                                   fixture.branch0_activation_zero_point_u8,
                                   fixture.branch0_input_storage_zero_point_s8,
                                   fixture.split_output1_scale,
                                   fixture.branch0_output_scale,
                                   fixture.branch0_output_zero_point_u8,
                                   fixture.branch0_weight_scales,
                                   fixture.branch0_weight_scale_count,
                                   fixture.branch0_weights_ohwi_s8,
                                   fixture.branch0_weight_count,
                                   fixture.branch0_bias_i32,
                                   fixture.branch0_bias_count};
}

Y26Stage10BackboneExpansionConfig full_stage10_config(
    const y26_stage10_backbone_expansion_fixture::BackboneExpansionFixture& fixture,
    int activation_mode) {
    return Y26Stage10BackboneExpansionConfig{fixture.subset_id,
                                             full_stage9_config(*fixture.stage9_fixture, activation_mode),
                                             full_branch0_config(fixture),
                                             fixture.split_output1_scale,
                                             fixture.branch0_activation_zero_point_u8,
                                             1,
                                             16,
                                             16,
                                             activation_mode};
}

Stage9BenchResult run_stage9(const Y26Stage7BackboneSubsetConfig& cfg,
                             const std::vector<std::int8_t>& input,
                             const std::vector<std::int32_t>& expected,
                             int iterations,
                             bool use_ime) {
    Y26Stage7BackboneSubsetWorkspace ws {};
    int status = y26_stage7_backbone_subset_prepare(&cfg, &ws);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return {0.0, {}, 0, status, expected.size()};
    }
    std::vector<std::int32_t> output(y26_stage7_backbone_subset_conv2_output_count(&cfg), 0);
    Y26Stage7TimingUs timing_sum {};
    std::int64_t checksum = 0;
    const auto begin = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        Y26Stage7TimingUs timing {};
        status = use_ime ? y26_stage7_backbone_subset_run_ime_cluster0_hotpath(&cfg, &ws, input.data(), output.data(), &timing)
                         : y26_stage7_backbone_subset_run_scalar(&cfg, &ws, input.data(), output.data(), &timing);
        checksum += checksum_i32(output);
        timing_sum.conv0_us += timing.conv0_us;
        timing_sum.act0_requant_us += timing.act0_requant_us;
        timing_sum.conv1_us += timing.conv1_us;
        timing_sum.act1_requant_us += timing.act1_requant_us;
        timing_sum.conv2_us += timing.conv2_us;
        timing_sum.total_us += timing.total_us;
        if (status != Y26_CONV_STATUS_SUCCESS) {
            break;
        }
    }
    const auto end = Clock::now();
    const double denom = static_cast<double>(std::max(1, iterations));
    timing_sum.conv0_us /= denom;
    timing_sum.act0_requant_us /= denom;
    timing_sum.conv1_us /= denom;
    timing_sum.act1_requant_us /= denom;
    timing_sum.conv2_us /= denom;
    timing_sum.total_us /= denom;
    const std::size_t mismatches = expected.empty() ? 0 : mismatches_i32(output, expected);
    y26_stage7_backbone_subset_release(&ws);
    return {elapsed_us(begin, end) / denom, timing_sum, checksum, status, mismatches};
}

BenchResult run_stage10(const Y26Stage10BackboneExpansionConfig& cfg,
                        const std::vector<std::int8_t>& input,
                        const std::vector<std::int32_t>& expected,
                        int iterations,
                        bool use_ime) {
    Y26Stage10BackboneExpansionWorkspace ws {};
    int status = y26_stage10_backbone_expansion_prepare(&cfg, &ws);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return {0.0, {}, 0, status, expected.size()};
    }
    std::vector<std::int32_t> output(y26_stage10_backbone_expansion_output_count(&cfg), 0);
    Y26Stage10TimingUs timing_sum {};
    std::int64_t checksum = 0;
    const auto begin = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        Y26Stage10TimingUs timing {};
        status = use_ime ? y26_stage10_backbone_expansion_run_ime_cluster0_hotpath(&cfg, &ws, input.data(), output.data(), &timing)
                         : y26_stage10_backbone_expansion_run_scalar(&cfg, &ws, input.data(), output.data(), &timing);
        checksum += checksum_i32(output);
        timing_sum.conv0_ime_us += timing.conv0_ime_us;
        timing_sum.act0_requant_lut_us += timing.act0_requant_lut_us;
        timing_sum.conv1_ime_us += timing.conv1_ime_us;
        timing_sum.act1_requant_lut_us += timing.act1_requant_lut_us;
        timing_sum.conv2_ime_us += timing.conv2_ime_us;
        timing_sum.act2_requant_lut_us += timing.act2_requant_lut_us;
        timing_sum.split_us += timing.split_us;
        timing_sum.branch_conv_us += timing.branch_conv_us;
        timing_sum.branch_correction_us += timing.branch_correction_us;
        timing_sum.pack_layout_us += timing.pack_layout_us;
        timing_sum.total_us += timing.total_us;
        if (status != Y26_CONV_STATUS_SUCCESS) {
            break;
        }
    }
    const auto end = Clock::now();
    const double denom = static_cast<double>(std::max(1, iterations));
    timing_sum.conv0_ime_us /= denom;
    timing_sum.act0_requant_lut_us /= denom;
    timing_sum.conv1_ime_us /= denom;
    timing_sum.act1_requant_lut_us /= denom;
    timing_sum.conv2_ime_us /= denom;
    timing_sum.act2_requant_lut_us /= denom;
    timing_sum.split_us /= denom;
    timing_sum.branch_conv_us /= denom;
    timing_sum.branch_correction_us /= denom;
    timing_sum.pack_layout_us /= denom;
    timing_sum.total_us /= denom;
    const std::size_t mismatches = expected.empty() ? 0 : mismatches_i32(output, expected);
    y26_stage10_backbone_expansion_release(&ws);
    return {elapsed_us(begin, end) / denom, timing_sum, checksum, status, mismatches};
}

void print_stage9(const char* label, const Stage9BenchResult& result) {
    const double activation = result.timing.act0_requant_us + result.timing.act1_requant_us;
    std::cout << label << " total_us=" << result.mean_us << " status=" << result.status
              << " mismatches=" << result.mismatches << " checksum=" << result.checksum
              << " conv0_us=" << result.timing.conv0_us << " act0_requant_us=" << result.timing.act0_requant_us
              << " conv1_us=" << result.timing.conv1_us << " act1_requant_us=" << result.timing.act1_requant_us
              << " conv2_us=" << result.timing.conv2_us << " activation_total_us=" << activation
              << " activation_share_pct=" << (result.mean_us > 0.0 ? 100.0 * activation / result.mean_us : 0.0)
              << "\n";
}

void print_stage10(const char* label, const BenchResult& result) {
    const double activation =
        result.timing.act0_requant_lut_us + result.timing.act1_requant_lut_us + result.timing.act2_requant_lut_us;
    const double conv = result.timing.conv0_ime_us + result.timing.conv1_ime_us + result.timing.conv2_ime_us +
                        result.timing.branch_conv_us;
    std::cout << label << " total_us=" << result.mean_us << " status=" << result.status
              << " mismatches=" << result.mismatches << " checksum=" << result.checksum
              << " conv0_ime_us=" << result.timing.conv0_ime_us
              << " act0_requant_lut_us=" << result.timing.act0_requant_lut_us
              << " conv1_ime_us=" << result.timing.conv1_ime_us
              << " act1_requant_lut_us=" << result.timing.act1_requant_lut_us
              << " conv2_ime_us=" << result.timing.conv2_ime_us
              << " act2_requant_lut_us=" << result.timing.act2_requant_lut_us
              << " split_us=" << result.timing.split_us << " branch_conv_us=" << result.timing.branch_conv_us
              << " branch_correction_us=" << result.timing.branch_correction_us
              << " pack_layout_us=" << result.timing.pack_layout_us << " activation_total_us=" << activation
              << " activation_share_pct=" << (result.mean_us > 0.0 ? 100.0 * activation / result.mean_us : 0.0)
              << " conv_share_pct=" << (result.mean_us > 0.0 ? 100.0 * conv / result.mean_us : 0.0)
              << " pack_layout_share_pct="
              << (result.mean_us > 0.0 ? 100.0 * result.timing.pack_layout_us / result.mean_us : 0.0)
              << " split_branch_share_pct="
              << (result.mean_us > 0.0 ? 100.0 * (result.timing.split_us + result.timing.branch_conv_us) / result.mean_us : 0.0)
              << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::max(1, std::atoi(argv[1])) : 1;
    const auto& fixture = y26_stage10_backbone_expansion_fixture::kSyntheticSeededFixture;
    const auto& stage9_fixture = *fixture.stage9_fixture;
    const std::vector<std::int8_t> input = make_input(Y26Conv2DParams{640, 640, 3, 16, 2, 2, 1, 1}, 0);

    Y26Stage7BackboneSubsetConfig stage9_scalar_cfg =
        full_stage9_config(stage9_fixture, Y26_ACTIVATION_MODE_INT8_LUT);
    Stage9BenchResult stage9_scalar = run_stage9(stage9_scalar_cfg, input, {}, 1, false);
    std::vector<std::int32_t> stage9_expected(y26_stage7_backbone_subset_conv2_output_count(&stage9_scalar_cfg), 0);
    {
        Y26Stage7BackboneSubsetWorkspace ws {};
        (void)y26_stage7_backbone_subset_prepare(&stage9_scalar_cfg, &ws);
        Y26Stage7TimingUs timing {};
        (void)y26_stage7_backbone_subset_run_scalar(&stage9_scalar_cfg, &ws, input.data(), stage9_expected.data(), &timing);
        y26_stage7_backbone_subset_release(&ws);
    }

    Y26Stage10BackboneExpansionConfig stage10_scalar_cfg =
        full_stage10_config(fixture, Y26_ACTIVATION_MODE_INT8_LUT);
    BenchResult stage10_scalar = run_stage10(stage10_scalar_cfg, input, {}, 1, false);
    std::vector<std::int32_t> stage10_expected(y26_stage10_backbone_expansion_output_count(&stage10_scalar_cfg), 0);
    {
        Y26Stage10BackboneExpansionWorkspace ws {};
        (void)y26_stage10_backbone_expansion_prepare(&stage10_scalar_cfg, &ws);
        Y26Stage10TimingUs timing {};
        (void)y26_stage10_backbone_expansion_run_scalar(&stage10_scalar_cfg, &ws, input.data(), stage10_expected.data(), &timing);
        y26_stage10_backbone_expansion_release(&ws);
    }

    std::cout << "subset=candidate_E_branch1_stage9_split_model2_m0_cv1_conv iterations=" << iterations << "\n";
    print_stage9("stage9_scalar_reference", stage9_scalar);
    print_stage10("stage10_scalar_reference", stage10_scalar);

    if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
        std::cout << "ime_not_built\n";
        return 0;
    }
    (void)y26_k1x_ime_probe_once();
    Stage9BenchResult stage9_a0 =
        run_stage9(full_stage9_config(stage9_fixture, Y26_ACTIVATION_MODE_INT8_LUT), input, stage9_expected, iterations, true);
    Stage9BenchResult stage9_a2 =
        run_stage9(full_stage9_config(stage9_fixture, Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT),
                   input,
                   stage9_expected,
                   iterations,
                   true);
    BenchResult stage10_a2 =
        run_stage10(full_stage10_config(fixture, Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT),
                    input,
                    stage10_expected,
                    iterations,
                    true);
    print_stage9("stage9_A0_int8_lut", stage9_a0);
    print_stage9("stage9_A2_rvv_f32_lut", stage9_a2);
    print_stage10("stage10_A2_rvv_f32_lut", stage10_a2);
    return stage9_a0.status == Y26_CONV_STATUS_SUCCESS && stage9_a2.status == Y26_CONV_STATUS_SUCCESS &&
                   stage10_a2.status == Y26_CONV_STATUS_SUCCESS && stage9_a0.mismatches == 0 &&
                   stage9_a2.mismatches == 0 && stage10_a2.mismatches == 0
               ? 0
               : 1;
}
