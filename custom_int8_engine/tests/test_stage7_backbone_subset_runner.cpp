#include "stage7_backbone_subset_fixture.h"
#include "y26_k1x_backbone_subset_runner.h"
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

Y26Stage7BackboneSubsetConfig config_from_fixture(
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture) {
    return Y26Stage7BackboneSubsetConfig{fixture.subset_id,
                                         conv0_config_from_fixture(fixture),
                                         conv1_config_from_fixture(fixture),
                                         conv2_config_from_fixture(fixture),
                                         fixture.act0_output_scale,
                                         fixture.act0_output_zero_point_u8,
                                         fixture.act1_output_scale,
                                         fixture.act1_output_zero_point_u8};
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

std::size_t mismatches_i8(const std::int8_t* actual, const std::int8_t* expected, std::size_t count) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (actual[i] != expected[i]) {
            ++mismatches;
        }
    }
    return mismatches;
}

int verify_fixture(const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture) {
    Y26Stage7BackboneSubsetConfig cfg = config_from_fixture(fixture);
    Y26Stage7BackboneSubsetWorkspace ws {};
    const int prepare_status = y26_stage7_backbone_subset_prepare(&cfg, &ws);
    if (prepare_status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "prepare failed label=" << fixture.label << " status=" << prepare_status << "\n";
        return 1;
    }

    const std::size_t conv2_output_count = y26_stage7_backbone_subset_conv2_output_count(&cfg);
    if (fixture.expected_conv2_count != conv2_output_count) {
        std::cerr << "unexpected conv2 count label=" << fixture.label << " got=" << conv2_output_count
                  << " expected=" << fixture.expected_conv2_count << "\n";
        y26_stage7_backbone_subset_release(&ws);
        return 1;
    }

    std::vector<std::int32_t> output(conv2_output_count, 0);
    Y26Stage7TimingUs scalar_timing {};
    const int scalar_status = y26_stage7_backbone_subset_run_scalar(
        &cfg, &ws, fixture.input_nhwc_s8, output.data(), &scalar_timing);
    const std::size_t scalar_conv0_mismatches =
        mismatches_i32(y26_stage7_backbone_subset_conv0_i32(&ws),
                       fixture.expected_conv0_i32_nhwc,
                       fixture.expected_conv0_count);
    const std::size_t scalar_act0_mismatches =
        mismatches_i8(y26_stage7_backbone_subset_conv1_input_s8(&ws),
                      fixture.expected_act0_s8_nhwc,
                      fixture.expected_act0_count);
    const std::size_t scalar_conv1_mismatches =
        mismatches_i32(y26_stage7_backbone_subset_conv1_i32(&ws),
                       fixture.expected_conv1_i32_nhwc,
                       fixture.expected_conv1_count);
    const std::size_t scalar_act1_mismatches =
        mismatches_i8(y26_stage7_backbone_subset_conv2_input_s8(&ws),
                      fixture.expected_act1_s8_nhwc,
                      fixture.expected_act1_count);
    const std::size_t scalar_conv2_mismatches =
        mismatches_i32(output.data(), fixture.expected_conv2_i32_nhwc, fixture.expected_conv2_count);

    int ime_status = Y26_CONV_STATUS_NOT_BUILT_WITH_IME;
    std::size_t ime_conv0_mismatches = 0;
    std::size_t ime_act0_mismatches = 0;
    std::size_t ime_conv1_mismatches = 0;
    std::size_t ime_act1_mismatches = 0;
    std::size_t ime_conv2_mismatches = 0;
    Y26Stage7TimingUs ime_timing {};
    if (y26_vmadot_4x4x8_ime_available_buildtime()) {
        (void)y26_k1x_ime_probe_once();
        std::fill(output.begin(), output.end(), 0);
        ime_status = y26_stage7_backbone_subset_run_ime_cluster0_hotpath(
            &cfg, &ws, fixture.input_nhwc_s8, output.data(), &ime_timing);
        ime_conv0_mismatches = mismatches_i32(y26_stage7_backbone_subset_conv0_i32(&ws),
                                              fixture.expected_conv0_i32_nhwc,
                                              fixture.expected_conv0_count);
        ime_act0_mismatches = mismatches_i8(y26_stage7_backbone_subset_conv1_input_s8(&ws),
                                            fixture.expected_act0_s8_nhwc,
                                            fixture.expected_act0_count);
        ime_conv1_mismatches = mismatches_i32(y26_stage7_backbone_subset_conv1_i32(&ws),
                                              fixture.expected_conv1_i32_nhwc,
                                              fixture.expected_conv1_count);
        ime_act1_mismatches = mismatches_i8(y26_stage7_backbone_subset_conv2_input_s8(&ws),
                                            fixture.expected_act1_s8_nhwc,
                                            fixture.expected_act1_count);
        ime_conv2_mismatches = mismatches_i32(output.data(), fixture.expected_conv2_i32_nhwc, fixture.expected_conv2_count);
    }

    std::cout << "stage7_backbone_fixture label=" << fixture.label << " scalar_status=" << scalar_status
              << " scalar_conv0_mismatches=" << scalar_conv0_mismatches
              << " scalar_act0_mismatches=" << scalar_act0_mismatches
              << " scalar_conv1_mismatches=" << scalar_conv1_mismatches
              << " scalar_act1_mismatches=" << scalar_act1_mismatches
              << " scalar_conv2_mismatches=" << scalar_conv2_mismatches
              << " ime_status=" << ime_status << " ime_conv0_mismatches=" << ime_conv0_mismatches
              << " ime_act0_mismatches=" << ime_act0_mismatches
              << " ime_conv1_mismatches=" << ime_conv1_mismatches
              << " ime_act1_mismatches=" << ime_act1_mismatches
              << " ime_conv2_mismatches=" << ime_conv2_mismatches
              << " scalar_total_us=" << scalar_timing.total_us << " ime_total_us=" << ime_timing.total_us << "\n";

    const bool scalar_ok = scalar_status == Y26_CONV_STATUS_SUCCESS && scalar_conv0_mismatches == 0 &&
                           scalar_act0_mismatches == 0 && scalar_conv1_mismatches == 0 &&
                           scalar_act1_mismatches == 0 && scalar_conv2_mismatches == 0;
    const bool ime_ok = !y26_vmadot_4x4x8_ime_available_buildtime() ||
                        (ime_status == Y26_CONV_STATUS_SUCCESS && ime_conv0_mismatches == 0 &&
                         ime_act0_mismatches == 0 && ime_conv1_mismatches == 0 &&
                         ime_act1_mismatches == 0 && ime_conv2_mismatches == 0);
    y26_stage7_backbone_subset_release(&ws);
    return scalar_ok && ime_ok ? 0 : 1;
}

}  // namespace

int main() {
    int failures = 0;
    for (const auto* fixture : y26_stage7_backbone_subset_fixture::kFixtures) {
        failures += verify_fixture(*fixture);
    }
    return failures == 0 ? 0 : 1;
}
