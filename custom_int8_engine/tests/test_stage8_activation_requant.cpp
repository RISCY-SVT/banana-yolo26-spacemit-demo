#include "stage7_backbone_subset_fixture.h"
#include "y26_k1x_activation.h"
#include "y26_k1x_backbone_subset_runner.h"
#include "y26_k1x_conv_kernels.h"
#include "y26_k1x_vmadot.h"

#include <cmath>
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

Y26Stage7BackboneSubsetConfig config_from_fixture(
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

std::int8_t reference_lut_value(float conv_scale, int conv_zp, float act_scale, int act_zp, int q) {
    const float x = static_cast<float>(q - conv_zp) * conv_scale;
    const float y = x / (1.0f + std::exp(-x));
    const std::uint8_t qy = y26_quantize_u8_nearest_even_f32(y, act_scale, act_zp);
    return static_cast<std::int8_t>(static_cast<int>(qy) - 128);
}

int verify_lut_boundary(const char* label, float conv_scale, int conv_zp, float act_scale, int act_zp) {
    std::int8_t lut[256] {};
    const int status = y26_build_silu_u8_to_s8_lut(conv_scale, conv_zp, act_scale, act_zp, lut);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << label << " lut build failed status=" << status << "\n";
        return 1;
    }
    int mismatches = 0;
    for (int q = 0; q < 256; ++q) {
        const std::int8_t expected = reference_lut_value(conv_scale, conv_zp, act_scale, act_zp, q);
        if (lut[q] != expected) {
            ++mismatches;
        }
    }
    if (mismatches != 0) {
        std::cerr << label << " lut mismatches=" << mismatches << "\n";
        return 1;
    }
    return 0;
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

int verify_runner_lut_mode(const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture) {
    Y26Stage7BackboneSubsetConfig cfg = config_from_fixture(fixture, Y26_ACTIVATION_MODE_INT8_LUT);
    Y26Stage7BackboneSubsetWorkspace ws {};
    const int prepare_status = y26_stage7_backbone_subset_prepare(&cfg, &ws);
    if (prepare_status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "prepare failed label=" << fixture.label << " status=" << prepare_status << "\n";
        return 1;
    }
    std::vector<std::int32_t> output(fixture.expected_conv2_count, 0);
    const int status =
        y26_stage7_backbone_subset_run_scalar(&cfg, &ws, fixture.input_nhwc_s8, output.data(), nullptr);
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
    if (status != Y26_CONV_STATUS_SUCCESS || act0_mismatches != 0 || act1_mismatches != 0 ||
        conv2_mismatches != 0) {
        std::cerr << "lut runner failed label=" << fixture.label << " status=" << status
                  << " act0_mismatches=" << act0_mismatches << " act1_mismatches=" << act1_mismatches
                  << " conv2_mismatches=" << conv2_mismatches << "\n";
        y26_stage7_backbone_subset_release(&ws);
        return 1;
    }

    if (y26_vmadot_4x4x8_ime_available_buildtime()) {
        (void)y26_k1x_ime_probe_once();
        std::fill(output.begin(), output.end(), 0);
        const int ime_status = y26_stage7_backbone_subset_run_ime_cluster0_hotpath(
            &cfg, &ws, fixture.input_nhwc_s8, output.data(), nullptr);
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
        if (ime_status != Y26_CONV_STATUS_SUCCESS || ime_act0_mismatches != 0 || ime_act1_mismatches != 0 ||
            ime_conv2_mismatches != 0) {
            std::cerr << "lut ime runner failed label=" << fixture.label << " status=" << ime_status
                      << " act0_mismatches=" << ime_act0_mismatches
                      << " act1_mismatches=" << ime_act1_mismatches
                      << " conv2_mismatches=" << ime_conv2_mismatches << "\n";
            y26_stage7_backbone_subset_release(&ws);
            return 1;
        }
    }

    y26_stage7_backbone_subset_release(&ws);
    return 0;
}

int verify_activation_api_matches_scalar(const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture) {
    const Y26ActivationRequantParams act0_params{fixture.expected_conv0_count,
                                                 fixture.conv0_params.output_c,
                                                 fixture.images_scale,
                                                 fixture.conv0_weight_scales,
                                                 fixture.conv0_output_scale,
                                                 fixture.conv0_output_zero_point_u8,
                                                 fixture.act0_output_scale,
                                                 fixture.act0_output_zero_point_u8};
    std::int8_t lut[256] {};
    (void)y26_build_silu_u8_to_s8_lut(fixture.conv0_output_scale,
                                      fixture.conv0_output_zero_point_u8,
                                      fixture.act0_output_scale,
                                      fixture.act0_output_zero_point_u8,
                                      lut);
    std::vector<std::int8_t> scalar(fixture.expected_act0_count, 0);
    std::vector<std::int8_t> lut_output(fixture.expected_act0_count, 0);
    const int scalar_status = y26_activation_requant_silu_scalar_float(
        &act0_params, fixture.expected_conv0_i32_nhwc, scalar.data());
    const int lut_status =
        y26_activation_requant_silu_int8_lut(&act0_params, fixture.expected_conv0_i32_nhwc, lut, lut_output.data());
    const std::size_t mismatches = mismatches_i8(scalar.data(), lut_output.data(), scalar.size());
    if (scalar_status != Y26_CONV_STATUS_SUCCESS || lut_status != Y26_CONV_STATUS_SUCCESS || mismatches != 0) {
        std::cerr << "activation api mismatch label=" << fixture.label << " scalar_status=" << scalar_status
                  << " lut_status=" << lut_status << " mismatches=" << mismatches << "\n";
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    int failures = 0;
    const auto& fixture = y26_stage7_backbone_subset_fixture::kSyntheticSeededFixture;
    failures += verify_lut_boundary("act0",
                                    fixture.conv0_output_scale,
                                    fixture.conv0_output_zero_point_u8,
                                    fixture.act0_output_scale,
                                    fixture.act0_output_zero_point_u8);
    failures += verify_lut_boundary("act1",
                                    fixture.conv1_output_scale,
                                    fixture.conv1_output_zero_point_u8,
                                    fixture.act1_output_scale,
                                    fixture.act1_output_zero_point_u8);
    for (const auto* item : y26_stage7_backbone_subset_fixture::kFixtures) {
        failures += verify_activation_api_matches_scalar(*item);
        failures += verify_runner_lut_mode(*item);
    }
    return failures == 0 ? 0 : 1;
}
