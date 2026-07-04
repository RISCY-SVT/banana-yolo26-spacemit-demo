#include "stage7_backbone_subset_fixture.h"
#include "y26_k1x_backbone_subset_runner.h"
#include "y26_k1x_vmadot.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

Y26Stage7ConvNodeConfig conv0_config(const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture) {
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

Y26Stage7ConvNodeConfig conv1_config(const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture) {
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

Y26Stage7ConvNodeConfig conv2_config(const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture) {
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

Y26Stage7BackboneSubsetConfig config_for_mode(const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture,
                                              int mode) {
    return Y26Stage7BackboneSubsetConfig{fixture.subset_id,
                                         conv0_config(fixture),
                                         conv1_config(fixture),
                                         conv2_config(fixture),
                                         fixture.act0_output_scale,
                                         fixture.act0_output_zero_point_u8,
                                         fixture.act1_output_scale,
                                         fixture.act1_output_zero_point_u8,
                                         mode};
}

std::size_t mismatches_i8(const std::int8_t* actual, const std::int8_t* expected, std::size_t count) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (actual[i] != expected[i]) {
            ++mismatches;
        }
    }
    return mismatches;
}

std::size_t mismatches_i32(const std::int32_t* actual, const std::int32_t* expected, std::size_t count) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (actual[i] != expected[i]) {
            ++mismatches;
        }
    }
    return mismatches;
}

int verify_mode(const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture,
                int mode,
                const char* label) {
    Y26Stage7BackboneSubsetConfig cfg = config_for_mode(fixture, mode);
    Y26Stage7BackboneSubsetWorkspace ws {};
    const int prepare_status = y26_stage7_backbone_subset_prepare(&cfg, &ws);
    if (prepare_status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << label << " prepare failed status=" << prepare_status << "\n";
        return 1;
    }
    std::vector<std::int32_t> output(fixture.expected_conv2_count, 0);
    const int status = y26_stage7_backbone_subset_run_scalar(&cfg, &ws, fixture.input_nhwc_s8, output.data(), nullptr);
    const std::size_t act0_mismatches =
        mismatches_i8(y26_stage7_backbone_subset_conv1_input_s8(&ws),
                      fixture.expected_act0_s8_nhwc,
                      fixture.expected_act0_count);
    const std::size_t act1_mismatches =
        mismatches_i8(y26_stage7_backbone_subset_conv2_input_s8(&ws),
                      fixture.expected_act1_s8_nhwc,
                      fixture.expected_act1_count);
    const std::size_t conv2_mismatches =
        mismatches_i32(output.data(), fixture.expected_conv2_i32_nhwc, fixture.expected_conv2_count);
    std::cout << "stage9_activation_mode label=" << label << " mode=" << mode << " status=" << status
              << " act0_mismatches=" << act0_mismatches << " act1_mismatches=" << act1_mismatches
              << " conv2_mismatches=" << conv2_mismatches << "\n";
    const bool ok = status == Y26_CONV_STATUS_SUCCESS && act0_mismatches == 0 && act1_mismatches == 0 &&
                    conv2_mismatches == 0;
    if (ok && y26_vmadot_4x4x8_ime_available_buildtime()) {
        (void)y26_k1x_ime_probe_once();
        std::fill(output.begin(), output.end(), 0);
        const int ime_status =
            y26_stage7_backbone_subset_run_ime_cluster0_hotpath(&cfg, &ws, fixture.input_nhwc_s8, output.data(), nullptr);
        const std::size_t ime_act0_mismatches =
            mismatches_i8(y26_stage7_backbone_subset_conv1_input_s8(&ws),
                          fixture.expected_act0_s8_nhwc,
                          fixture.expected_act0_count);
        const std::size_t ime_act1_mismatches =
            mismatches_i8(y26_stage7_backbone_subset_conv2_input_s8(&ws),
                          fixture.expected_act1_s8_nhwc,
                          fixture.expected_act1_count);
        const std::size_t ime_conv2_mismatches =
            mismatches_i32(output.data(), fixture.expected_conv2_i32_nhwc, fixture.expected_conv2_count);
        std::cout << "stage9_activation_mode_ime label=" << label << " mode=" << mode
                  << " status=" << ime_status << " act0_mismatches=" << ime_act0_mismatches
                  << " act1_mismatches=" << ime_act1_mismatches
                  << " conv2_mismatches=" << ime_conv2_mismatches << "\n";
        y26_stage7_backbone_subset_release(&ws);
        return ime_status == Y26_CONV_STATUS_SUCCESS && ime_act0_mismatches == 0 && ime_act1_mismatches == 0 &&
                       ime_conv2_mismatches == 0
                   ? 0
                   : 1;
    }
    y26_stage7_backbone_subset_release(&ws);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    const auto& fixture = y26_stage7_backbone_subset_fixture::kSyntheticSeededFixture;
    int failures = 0;
    failures += verify_mode(fixture, Y26_ACTIVATION_MODE_STAGE9_SCALAR_UNROLLED_LUT, "A1_scalar_unrolled_lut");
    failures += verify_mode(fixture, Y26_ACTIVATION_MODE_STAGE9_FIXED_REQUANT_LUT, "A3_fixed_requant_lut");
#if defined(__riscv_vector)
    failures += verify_mode(fixture, Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT, "A2_rvv_f32_lut");
#endif
    failures += verify_mode(fixture, Y26_ACTIVATION_MODE_STAGE9_FUSED_CURRENT_LAYOUT, "A4_fused_current_layout");
    return failures == 0 ? 0 : 1;
}
