#include "stage11_branch_block_fixture.h"
#include "y26_k1x_branch_block_runner.h"
#include "y26_k1x_vmadot.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

Y26Stage7ConvNodeConfig conv0_config_from_fixture(
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture) {
    return Y26Stage7ConvNodeConfig{fixture.conv0_node_name,
                                   fixture.conv0_params,
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

Y26Stage7ConvNodeConfig conv1_config_from_fixture(
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture) {
    return Y26Stage7ConvNodeConfig{fixture.conv1_node_name,
                                   fixture.conv1_params,
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

Y26Stage7ConvNodeConfig conv2_config_from_fixture(
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture) {
    return Y26Stage7ConvNodeConfig{fixture.conv2_node_name,
                                   fixture.conv2_params,
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

Y26Stage7BackboneSubsetConfig stage9_config_from_fixture(
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture,
    int activation_mode) {
    return Y26Stage7BackboneSubsetConfig{fixture.subset_id,
                                         conv0_config_from_fixture(fixture),
                                         conv1_config_from_fixture(fixture),
                                         conv2_config_from_fixture(fixture),
                                         fixture.act0_output_scale,
                                         fixture.act0_output_zero_point_u8,
                                         fixture.act1_output_scale,
                                         fixture.act1_output_zero_point_u8,
                                         activation_mode};
}

Y26Stage7ConvNodeConfig branch0_config_from_fixture(
    const y26_stage10_backbone_expansion_fixture::BackboneExpansionFixture& fixture) {
    return Y26Stage7ConvNodeConfig{fixture.branch0_node_name,
                                   fixture.branch0_params,
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

Y26Stage10BackboneExpansionConfig stage10_config_from_fixture(
    const y26_stage10_backbone_expansion_fixture::BackboneExpansionFixture& fixture,
    int activation_mode) {
    return Y26Stage10BackboneExpansionConfig{fixture.subset_id,
                                             stage9_config_from_fixture(*fixture.stage9_fixture, activation_mode),
                                             branch0_config_from_fixture(fixture),
                                             fixture.split_output1_scale,
                                             fixture.branch0_activation_zero_point_u8,
                                             1,
                                             16,
                                             16,
                                             activation_mode};
}

Y26Stage7ConvNodeConfig branch1_config_from_fixture(
    const y26_stage11_branch_block_fixture::BranchBlockFixture& fixture) {
    return Y26Stage7ConvNodeConfig{fixture.branch1_node_name,
                                   fixture.branch1_params,
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

Y26Stage11BranchBlockConfig config_from_fixture(
    const y26_stage11_branch_block_fixture::BranchBlockFixture& fixture,
    int activation_mode) {
    return Y26Stage11BranchBlockConfig{fixture.subset_id,
                                       stage10_config_from_fixture(*fixture.stage10_fixture, activation_mode),
                                       branch1_config_from_fixture(fixture),
                                       fixture.branch0_act_output_scale,
                                       fixture.branch1_activation_zero_point_u8,
                                       activation_mode};
}

std::size_t mismatches_i32(const std::int32_t* actual, const std::int32_t* expected, std::size_t count) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < count; ++i) {
        mismatches += actual[i] != expected[i] ? 1U : 0U;
    }
    return mismatches;
}

std::size_t mismatches_i8(const std::int8_t* actual, const std::int8_t* expected, std::size_t count) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < count; ++i) {
        mismatches += actual[i] != expected[i] ? 1U : 0U;
    }
    return mismatches;
}

int verify_scalar_mode(const y26_stage11_branch_block_fixture::BranchBlockFixture& fixture,
                       int activation_mode,
                       const char* mode_name) {
    Y26Stage11BranchBlockConfig cfg = config_from_fixture(fixture, activation_mode);
    Y26Stage11BranchBlockWorkspace ws {};
    const int prepare_status = y26_stage11_branch_block_prepare(&cfg, &ws);
    if (prepare_status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage11 prepare failed label=" << fixture.label << " mode=" << mode_name
                  << " status=" << prepare_status << "\n";
        return 1;
    }
    std::vector<std::int32_t> output(y26_stage11_branch_block_output_count(&cfg), 0);
    Y26Stage11TimingUs timing {};
    const int status = y26_stage11_branch_block_run_scalar(
        &cfg, &ws, fixture.stage10_fixture->stage9_fixture->input_nhwc_s8, output.data(), &timing);
    const std::size_t act_mismatches =
        mismatches_i8(y26_stage11_branch_block_branch0_activation_s8(&ws),
                      fixture.expected_branch0_act_s8_nhwc,
                      fixture.expected_branch0_act_count);
    const std::size_t output_mismatches =
        mismatches_i32(output.data(), fixture.expected_branch1_i32_nhwc, fixture.expected_branch1_count);
    std::cout << "stage11_scalar label=" << fixture.label << " mode=" << mode_name << " status=" << status
              << " branch0_act_mismatches=" << act_mismatches
              << " branch1_mismatches=" << output_mismatches << " total_us=" << timing.total_us
              << " branch_cv2_conv_us=" << timing.branch_cv2_conv_us << "\n";
    y26_stage11_branch_block_release(&ws);
    return status == Y26_CONV_STATUS_SUCCESS && act_mismatches == 0 && output_mismatches == 0 ? 0 : 1;
}

int verify_ime_a2(const y26_stage11_branch_block_fixture::BranchBlockFixture& fixture) {
    if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
        std::cout << "stage11_ime skipped_not_built label=" << fixture.label << "\n";
        return 0;
    }
    (void)y26_k1x_ime_probe_once();
    Y26Stage11BranchBlockConfig cfg =
        config_from_fixture(fixture, Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT);
    Y26Stage11BranchBlockWorkspace ws {};
    const int prepare_status = y26_stage11_branch_block_prepare(&cfg, &ws);
    if (prepare_status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage11 ime prepare failed label=" << fixture.label << " status=" << prepare_status << "\n";
        return 1;
    }
    std::vector<std::int32_t> output(y26_stage11_branch_block_output_count(&cfg), 0);
    Y26Stage11TimingUs timing {};
    const int status = y26_stage11_branch_block_run_ime_cluster0_hotpath(
        &cfg, &ws, fixture.stage10_fixture->stage9_fixture->input_nhwc_s8, output.data(), &timing);
    const std::size_t act_mismatches =
        mismatches_i8(y26_stage11_branch_block_branch0_activation_s8(&ws),
                      fixture.expected_branch0_act_s8_nhwc,
                      fixture.expected_branch0_act_count);
    const std::size_t output_mismatches =
        mismatches_i32(output.data(), fixture.expected_branch1_i32_nhwc, fixture.expected_branch1_count);
    std::cout << "stage11_ime label=" << fixture.label << " mode=A2_rvv_f32_lut status=" << status
              << " branch0_act_mismatches=" << act_mismatches
              << " branch1_mismatches=" << output_mismatches << " total_us=" << timing.total_us
              << " branch_cv2_conv_us=" << timing.branch_cv2_conv_us << "\n";
    y26_stage11_branch_block_release(&ws);
    return status == Y26_CONV_STATUS_SUCCESS && act_mismatches == 0 && output_mismatches == 0 ? 0 : 1;
}

}  // namespace

int main() {
    int failures = 0;
    for (const auto* fixture : y26_stage11_branch_block_fixture::kFixtures) {
        failures += verify_scalar_mode(*fixture, Y26_ACTIVATION_MODE_INT8_LUT, "A0_int8_lut");
#if defined(__riscv_vector)
        failures += verify_scalar_mode(*fixture, Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT, "A2_rvv_f32_lut");
#endif
        failures += verify_ime_a2(*fixture);
    }
    return failures == 0 ? 0 : 1;
}
