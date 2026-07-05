#include "stage12_c2f_block_fixture.h"
#include "y26_k1x_c2f_block_runner.h"

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
    Y26Stage12TimingUs timing;
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

Y26Stage7ConvNodeConfig full_model2_cv2_config(
    const y26_stage12_c2f_block_fixture::C2fBlockFixture& fixture) {
    return Y26Stage7ConvNodeConfig{fixture.model2_cv2_node_name,
                                   Y26Conv2DParams{160, 160, 48, 64, 1, 1, 0, 0},
                                   fixture.model2_cv2_kernel_h,
                                   fixture.model2_cv2_kernel_w,
                                   fixture.concat_output_zero_point_u8,
                                   fixture.concat_input_storage_zero_point_s8,
                                   fixture.concat_output_scale,
                                   fixture.model2_cv2_output_scale,
                                   fixture.model2_cv2_output_zero_point_u8,
                                   fixture.model2_cv2_weight_scales,
                                   fixture.model2_cv2_weight_scale_count,
                                   fixture.model2_cv2_weights_ohwi_s8,
                                   fixture.model2_cv2_weight_count,
                                   fixture.model2_cv2_bias_i32,
                                   fixture.model2_cv2_bias_count};
}

Y26Stage12C2fBlockConfig full_stage12_config(
    const y26_stage12_c2f_block_fixture::C2fBlockFixture& fixture,
    int activation_mode,
    int merge_mode) {
    return Y26Stage12C2fBlockConfig{fixture.subset_id,
                                    full_stage11_config(*fixture.stage11_fixture, activation_mode),
                                    full_model2_cv2_config(fixture),
                                    fixture.stage11_fixture->stage10_fixture->split_output1_scale,
                                    fixture.stage11_fixture->stage10_fixture->branch0_activation_zero_point_u8,
                                    fixture.concat_output_scale,
                                    fixture.concat_output_zero_point_u8,
                                    activation_mode,
                                    merge_mode};
}

void accumulate_timing(Y26Stage12TimingUs& dst, const Y26Stage12TimingUs& src) {
    dst.conv_us += src.conv_us;
    dst.activation_requant_us += src.activation_requant_us;
    dst.split_us += src.split_us;
    dst.add_us += src.add_us;
    dst.concat_us += src.concat_us;
    dst.post_concat_qdq_us += src.post_concat_qdq_us;
    dst.pack_layout_us += src.pack_layout_us;
    dst.correction_us += src.correction_us;
    dst.model2_cv2_conv_us += src.model2_cv2_conv_us;
    dst.total_us += src.total_us;
    dst.split_copy_us += src.split_copy_us;
    dst.add_compute_us += src.add_compute_us;
    dst.concat_materialize_us += src.concat_materialize_us;
    dst.pack_for_model2_cv2_us += src.pack_for_model2_cv2_us;
    dst.layout_copy_us += src.layout_copy_us;
    dst.other_us += src.other_us;
    dst.merge_total_us += src.merge_total_us;
}

void divide_timing(Y26Stage12TimingUs& timing, double denom) {
    timing.conv_us /= denom;
    timing.activation_requant_us /= denom;
    timing.split_us /= denom;
    timing.add_us /= denom;
    timing.concat_us /= denom;
    timing.post_concat_qdq_us /= denom;
    timing.pack_layout_us /= denom;
    timing.correction_us /= denom;
    timing.model2_cv2_conv_us /= denom;
    timing.total_us /= denom;
    timing.split_copy_us /= denom;
    timing.add_compute_us /= denom;
    timing.concat_materialize_us /= denom;
    timing.pack_for_model2_cv2_us /= denom;
    timing.layout_copy_us /= denom;
    timing.other_us /= denom;
    timing.merge_total_us /= denom;
    timing.activation_share_pct = timing.total_us > 0.0 ? 100.0 * timing.activation_requant_us / timing.total_us : 0.0;
    timing.conv_share_pct = timing.total_us > 0.0 ? 100.0 * timing.conv_us / timing.total_us : 0.0;
    const double add_concat = timing.add_compute_us + timing.concat_materialize_us + timing.post_concat_qdq_us;
    timing.add_concat_share_pct = timing.total_us > 0.0 ? 100.0 * add_concat / timing.total_us : 0.0;
    timing.pack_layout_share_pct = timing.total_us > 0.0 ? 100.0 * timing.pack_layout_us / timing.total_us : 0.0;
    timing.merge_share_pct = timing.total_us > 0.0 ? 100.0 * timing.merge_total_us / timing.total_us : 0.0;
}

BenchResult run_stage12(const Y26Stage12C2fBlockConfig& cfg,
                        const std::vector<std::int8_t>& input,
                        const std::vector<std::int32_t>& expected,
                        int iterations,
                        bool use_ime) {
    Y26Stage12C2fBlockWorkspace ws {};
    int status = y26_stage12_c2f_block_prepare(&cfg, &ws);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return {0.0, {}, 0, status, expected.size()};
    }
    std::vector<std::int32_t> output(y26_stage12_c2f_block_output_count(&cfg), 0);
    Y26Stage12TimingUs timing_sum {};
    std::int64_t checksum = 0;
    const auto begin = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        Y26Stage12TimingUs timing {};
        status = use_ime ? y26_stage12_c2f_block_run_ime_cluster0_hotpath(&cfg, &ws, input.data(), output.data(), &timing)
                         : y26_stage12_c2f_block_run_scalar(&cfg, &ws, input.data(), output.data(), &timing);
        checksum += checksum_i32(output);
        accumulate_timing(timing_sum, timing);
        if (status != Y26_CONV_STATUS_SUCCESS) {
            break;
        }
    }
    const auto end = Clock::now();
    const double denom = static_cast<double>(std::max(1, iterations));
    divide_timing(timing_sum, denom);
    const std::size_t mismatches = expected.empty() ? 0 : mismatches_i32(output, expected);
    y26_stage12_c2f_block_release(&ws);
    return {elapsed_us(begin, end) / denom, timing_sum, checksum, status, mismatches};
}

void print_stage13(const char* label, const BenchResult& result) {
    const char* correctness = result.status == Y26_CONV_STATUS_SUCCESS && result.mismatches == 0 ? "pass" : "fail";
    std::cout << "candidate=" << label
              << " correctness_status=" << correctness
              << " total_us=" << result.mean_us
              << " status=" << result.status
              << " mismatches=" << result.mismatches
              << " checksum=" << result.checksum
              << " conv_us=" << result.timing.conv_us
              << " activation_requant_us=" << result.timing.activation_requant_us
              << " split_copy_us=" << result.timing.split_copy_us
              << " add_compute_us=" << result.timing.add_compute_us
              << " concat_materialize_us=" << result.timing.concat_materialize_us
              << " post_concat_qdq_us=" << result.timing.post_concat_qdq_us
              << " pack_for_model2_cv2_us=" << result.timing.pack_for_model2_cv2_us
              << " layout_copy_us=" << result.timing.layout_copy_us
              << " correction_us=" << result.timing.correction_us
              << " model2_cv2_conv_us=" << result.timing.model2_cv2_conv_us
              << " merge_total_us=" << result.timing.merge_total_us
              << " merge_share_pct=" << result.timing.merge_share_pct
              << " pack_layout_us=" << result.timing.pack_layout_us
              << " pack_layout_share_pct=" << result.timing.pack_layout_share_pct
              << " activation_share_pct=" << result.timing.activation_share_pct
              << " conv_share_pct=" << result.timing.conv_share_pct
              << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::max(1, std::atoi(argv[1])) : 1;
    const auto& fixture = y26_stage12_c2f_block_fixture::kSyntheticSeededFixture;
    const std::vector<std::int8_t> input = make_input(Y26Conv2DParams{640, 640, 3, 16, 2, 2, 1, 1}, 0);

    Y26Stage12C2fBlockConfig scalar_cfg = full_stage12_config(
        fixture, Y26_ACTIVATION_MODE_INT8_LUT, Y26_STAGE12_MERGE_MODE_A0_MATERIALIZED_FLOAT);
    std::vector<std::int32_t> expected(y26_stage12_c2f_block_output_count(&scalar_cfg), 0);
    {
        Y26Stage12C2fBlockWorkspace ws {};
        (void)y26_stage12_c2f_block_prepare(&scalar_cfg, &ws);
        Y26Stage12TimingUs timing {};
        (void)y26_stage12_c2f_block_run_scalar(&scalar_cfg, &ws, input.data(), expected.data(), &timing);
        y26_stage12_c2f_block_release(&ws);
    }

    Y26Stage12C2fBlockConfig a0_cfg = full_stage12_config(
        fixture, Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT, Y26_STAGE12_MERGE_MODE_A0_MATERIALIZED_FLOAT);
    Y26Stage12C2fBlockConfig a1_cfg = full_stage12_config(
        fixture, Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT, Y26_STAGE13_MERGE_MODE_A1_FUSED_ADD_CONCAT);
    Y26Stage12C2fBlockConfig a2_cfg = full_stage12_config(
        fixture, Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT, Y26_STAGE13_MERGE_MODE_A2_FUSED_QDQ_NHWC);

    BenchResult scalar_reference = run_stage12(scalar_cfg, input, expected, 1, false);
    BenchResult a0_ime = run_stage12(a0_cfg, input, expected, iterations, true);
    BenchResult a1_ime = run_stage12(a1_cfg, input, expected, iterations, true);
    BenchResult a2_ime = run_stage12(a2_cfg, input, expected, iterations, true);

    std::cout << "subset=candidate_G_model2_c2f_add_concat_cv2_conv iterations=" << iterations << "\n";
    std::cout << "timing_bucket_contract=non_overlapping_stage13 split_copy/add_compute/concat_materialize/"
                 "post_concat_qdq are merge-local; pack_layout excludes split_copy\n";
    print_stage13("scalar_reference_A0_materialized_float_merge", scalar_reference);
    print_stage13("A0_materialized_float_merge", a0_ime);
    print_stage13("A1_fused_add_concat", a1_ime);
    print_stage13("A2_fused_qdq_nhwc", a2_ime);
    return (scalar_reference.status == Y26_CONV_STATUS_SUCCESS && a0_ime.status == Y26_CONV_STATUS_SUCCESS &&
            a1_ime.status == Y26_CONV_STATUS_SUCCESS && a2_ime.status == Y26_CONV_STATUS_SUCCESS &&
            a0_ime.mismatches == 0 && a1_ime.mismatches == 0 && a2_ime.mismatches == 0)
               ? 0
               : 1;
}
