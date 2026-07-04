#include "stage11_branch_block_fixture.h"
#include "y26_k1x_branch_block_runner.h"

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
    Y26Stage11TimingUs timing;
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

Y26Stage7ConvNodeConfig full_branch1_config(
    const y26_stage11_branch_block_fixture::BranchBlockFixture& fixture) {
    return Y26Stage7ConvNodeConfig{fixture.branch1_node_name,
                                   Y26Conv2DParams{160, 160, 8, 16, 1, 1, 1, 1},
                                   fixture.branch1_kernel_h,
                                   fixture.branch1_kernel_w,
                                   fixture.branch1_activation_zero_point_u8,
                                   fixture.branch1_input_storage_zero_point_s8,
                                   fixture.branch0_act_output_scale,
                                   fixture.branch1_output_scale,
                                   fixture.branch1_output_zero_point_u8,
                                   fixture.branch1_weight_scales,
                                   fixture.branch1_weight_scale_count,
                                   fixture.branch1_weights_ohwi_s8,
                                   fixture.branch1_weight_count,
                                   fixture.branch1_bias_i32,
                                   fixture.branch1_bias_count};
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

Y26Stage11BranchBlockConfig full_stage11_config(
    const y26_stage11_branch_block_fixture::BranchBlockFixture& fixture,
    int activation_mode) {
    return Y26Stage11BranchBlockConfig{fixture.subset_id,
                                       full_stage10_config(*fixture.stage10_fixture, activation_mode),
                                       full_branch1_config(fixture),
                                       fixture.branch0_act_output_scale,
                                       fixture.branch1_activation_zero_point_u8,
                                       activation_mode};
}

BenchResult run_stage11(const Y26Stage11BranchBlockConfig& cfg,
                        const std::vector<std::int8_t>& input,
                        const std::vector<std::int32_t>& expected,
                        int iterations,
                        bool use_ime) {
    Y26Stage11BranchBlockWorkspace ws {};
    int status = y26_stage11_branch_block_prepare(&cfg, &ws);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return {0.0, {}, 0, status, expected.size()};
    }
    std::vector<std::int32_t> output(y26_stage11_branch_block_output_count(&cfg), 0);
    Y26Stage11TimingUs timing_sum {};
    std::int64_t checksum = 0;
    const auto begin = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        Y26Stage11TimingUs timing {};
        status = use_ime ? y26_stage11_branch_block_run_ime_cluster0_hotpath(&cfg, &ws, input.data(), output.data(), &timing)
                         : y26_stage11_branch_block_run_scalar(&cfg, &ws, input.data(), output.data(), &timing);
        checksum += checksum_i32(output);
        timing_sum.conv0_ime_us += timing.conv0_ime_us;
        timing_sum.act0_requant_lut_us += timing.act0_requant_lut_us;
        timing_sum.conv1_ime_us += timing.conv1_ime_us;
        timing_sum.act1_requant_lut_us += timing.act1_requant_lut_us;
        timing_sum.conv2_ime_us += timing.conv2_ime_us;
        timing_sum.act2_requant_lut_us += timing.act2_requant_lut_us;
        timing_sum.split_us += timing.split_us;
        timing_sum.branch_cv1_conv_us += timing.branch_cv1_conv_us;
        timing_sum.branch_cv1_activation_us += timing.branch_cv1_activation_us;
        timing_sum.branch_cv2_conv_us += timing.branch_cv2_conv_us;
        timing_sum.branch_cv2_correction_us += timing.branch_cv2_correction_us;
        timing_sum.layout_or_pack_us += timing.layout_or_pack_us;
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
    timing_sum.branch_cv1_conv_us /= denom;
    timing_sum.branch_cv1_activation_us /= denom;
    timing_sum.branch_cv2_conv_us /= denom;
    timing_sum.branch_cv2_correction_us /= denom;
    timing_sum.layout_or_pack_us /= denom;
    timing_sum.total_us /= denom;
    const std::size_t mismatches = expected.empty() ? 0 : mismatches_i32(output, expected);
    y26_stage11_branch_block_release(&ws);
    return {elapsed_us(begin, end) / denom, timing_sum, checksum, status, mismatches};
}

void print_stage11(const char* label, const BenchResult& result) {
    const double activation = result.timing.act0_requant_lut_us + result.timing.act1_requant_lut_us +
                              result.timing.act2_requant_lut_us + result.timing.branch_cv1_activation_us;
    const double conv = result.timing.conv0_ime_us + result.timing.conv1_ime_us + result.timing.conv2_ime_us +
                        result.timing.branch_cv1_conv_us + result.timing.branch_cv2_conv_us;
    std::cout << label << " total_us=" << result.mean_us << " status=" << result.status
              << " mismatches=" << result.mismatches << " checksum=" << result.checksum
              << " conv0_ime_us=" << result.timing.conv0_ime_us
              << " act0_requant_lut_us=" << result.timing.act0_requant_lut_us
              << " conv1_ime_us=" << result.timing.conv1_ime_us
              << " act1_requant_lut_us=" << result.timing.act1_requant_lut_us
              << " conv2_ime_us=" << result.timing.conv2_ime_us
              << " act2_requant_lut_us=" << result.timing.act2_requant_lut_us
              << " split_us=" << result.timing.split_us
              << " branch_cv1_conv_us=" << result.timing.branch_cv1_conv_us
              << " branch_cv1_activation_us=" << result.timing.branch_cv1_activation_us
              << " branch_cv2_conv_us=" << result.timing.branch_cv2_conv_us
              << " branch_cv2_correction_us=" << result.timing.branch_cv2_correction_us
              << " residual_add_us=" << result.timing.residual_add_us
              << " concat_copy_us=" << result.timing.concat_copy_us
              << " layout_or_pack_us=" << result.timing.layout_or_pack_us
              << " activation_total_us=" << activation
              << " activation_share_pct=" << (result.mean_us > 0.0 ? 100.0 * activation / result.mean_us : 0.0)
              << " conv_share_pct=" << (result.mean_us > 0.0 ? 100.0 * conv / result.mean_us : 0.0)
              << " pack_layout_share_pct="
              << (result.mean_us > 0.0 ? 100.0 * result.timing.layout_or_pack_us / result.mean_us : 0.0)
              << " split_copy_share_pct="
              << (result.mean_us > 0.0 ? 100.0 * result.timing.split_us / result.mean_us : 0.0)
              << " add_share_pct=0 concat_share_pct=0"
              << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::max(1, std::atoi(argv[1])) : 1;
    const auto& fixture = y26_stage11_branch_block_fixture::kSyntheticSeededFixture;
    const std::vector<std::int8_t> input = make_input(Y26Conv2DParams{640, 640, 3, 16, 2, 2, 1, 1}, 0);

    Y26Stage11BranchBlockConfig scalar_cfg = full_stage11_config(fixture, Y26_ACTIVATION_MODE_INT8_LUT);
    BenchResult scalar = run_stage11(scalar_cfg, input, {}, 1, false);
    std::vector<std::int32_t> expected(y26_stage11_branch_block_output_count(&scalar_cfg), 0);
    {
        Y26Stage11BranchBlockWorkspace ws {};
        (void)y26_stage11_branch_block_prepare(&scalar_cfg, &ws);
        Y26Stage11TimingUs timing {};
        (void)y26_stage11_branch_block_run_scalar(&scalar_cfg, &ws, input.data(), expected.data(), &timing);
        y26_stage11_branch_block_release(&ws);
    }

    Y26Stage11BranchBlockConfig a2_cfg =
        full_stage11_config(fixture, Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT);
    BenchResult scalar_a2 = run_stage11(a2_cfg, input, expected, iterations, false);
    BenchResult ime_a2 = run_stage11(a2_cfg, input, expected, iterations, true);

    std::cout << "subset=candidate_F_model2_m0_cv1_act_cv2_conv iterations=" << iterations << "\n";
    print_stage11("stage11_scalar_reference", scalar);
    print_stage11("stage11_scalar_A2_rvv_f32_lut", scalar_a2);
    print_stage11("stage11_IME_A2_rvv_f32_lut", ime_a2);
    return (scalar.status == Y26_CONV_STATUS_SUCCESS && scalar_a2.status == Y26_CONV_STATUS_SUCCESS &&
            ime_a2.status == Y26_CONV_STATUS_SUCCESS && scalar_a2.mismatches == 0 && ime_a2.mismatches == 0)
               ? 0
               : 1;
}
