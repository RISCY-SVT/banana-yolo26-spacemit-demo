#include "stage6_multiblock_fixture.h"
#include "y26_k1x_multiblock_runner.h"
#include "y26_k1x_vmadot.h"

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

Y26Stage6ConvNodeConfig conv0_config_from_fixture(const y26_stage6_multiblock_fixture::MultiblockFixture& fixture) {
    return Y26Stage6ConvNodeConfig{
        fixture.conv0_node_name,
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
        fixture.conv0_bias_count,
    };
}

Y26Stage6ConvNodeConfig conv1_config_from_fixture(const y26_stage6_multiblock_fixture::MultiblockFixture& fixture) {
    return Y26Stage6ConvNodeConfig{
        fixture.conv1_node_name,
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
        fixture.conv1_bias_count,
    };
}

Y26Stage6MultiblockConfig config_from_fixture(const y26_stage6_multiblock_fixture::MultiblockFixture& fixture) {
    return Y26Stage6MultiblockConfig{
        fixture.subset_id,
        conv0_config_from_fixture(fixture),
        conv1_config_from_fixture(fixture),
        fixture.act0_output_scale,
        fixture.act0_output_zero_point_u8,
    };
}

template <typename T>
int compare_array(const char* label, const char* name, const T* got, const T* expected, std::size_t count) {
    int mismatches = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (got[i] != expected[i]) {
            ++mismatches;
            if (mismatches <= 8) {
                std::cerr << label << " " << name << " mismatch index=" << i
                          << " got=" << static_cast<long long>(got[i])
                          << " expected=" << static_cast<long long>(expected[i]) << "\n";
            }
        }
    }
    return mismatches;
}

int verify_fixture(const y26_stage6_multiblock_fixture::MultiblockFixture& fixture) {
    Y26Stage6MultiblockConfig cfg = config_from_fixture(fixture);
    Y26Stage6MultiblockWorkspace ws {};
    int failures = 0;
    const int prepare_status = y26_stage6_multiblock_prepare(&cfg, &ws);
    if (prepare_status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << fixture.label << " prepare_status=" << prepare_status << "\n";
        return 1;
    }
    const std::size_t conv1_output_count = y26_stage6_multiblock_conv1_output_count(&cfg);
    if (conv1_output_count != fixture.expected_conv1_count || ws.conv0_output_count != fixture.expected_conv0_count ||
        ws.conv1_input_count != fixture.expected_act0_count || ws.prepacked_bytes == 0 || ws.workspace_bytes == 0) {
        std::cerr << fixture.label << " metadata mismatch conv1_output_count=" << conv1_output_count
                  << " expected=" << fixture.expected_conv1_count << "\n";
        failures += 1;
    }

    std::vector<std::int32_t> scalar_output(conv1_output_count, 0);
    Y26Stage6TimingUs scalar_timing {};
    const int scalar_status = y26_stage6_multiblock_run_scalar(
        &cfg, &ws, fixture.input_nhwc_s8, scalar_output.data(), &scalar_timing);
    const int scalar_conv0_mismatches =
        scalar_status == Y26_CONV_STATUS_SUCCESS
            ? compare_array(fixture.label,
                            "scalar_conv0",
                            y26_stage6_multiblock_conv0_i32(&ws),
                            fixture.expected_conv0_i32_nhwc,
                            fixture.expected_conv0_count)
            : 1;
    const int scalar_act_mismatches =
        scalar_status == Y26_CONV_STATUS_SUCCESS
            ? compare_array(fixture.label,
                            "scalar_act0",
                            y26_stage6_multiblock_conv1_input_s8(&ws),
                            fixture.expected_act0_s8_nhwc,
                            fixture.expected_act0_count)
            : 1;
    const int scalar_conv1_mismatches =
        scalar_status == Y26_CONV_STATUS_SUCCESS
            ? compare_array(fixture.label,
                            "scalar_conv1",
                            scalar_output.data(),
                            fixture.expected_conv1_i32_nhwc,
                            fixture.expected_conv1_count)
            : 1;
    failures += (scalar_status == Y26_CONV_STATUS_SUCCESS && scalar_conv0_mismatches == 0 &&
                 scalar_act_mismatches == 0 && scalar_conv1_mismatches == 0)
                    ? 0
                    : 1;

    int ime_status = Y26_CONV_STATUS_NOT_BUILT_WITH_IME;
    int ime_conv0_mismatches = 0;
    int ime_act_mismatches = 0;
    int ime_conv1_mismatches = 0;
    Y26Stage6TimingUs ime_timing {};
    if (y26_vmadot_4x4x8_ime_available_buildtime()) {
        std::vector<std::int32_t> ime_output(conv1_output_count, 0);
        ime_status = y26_stage6_multiblock_run_ime_cluster0_hotpath(
            &cfg, &ws, fixture.input_nhwc_s8, ime_output.data(), &ime_timing);
        ime_conv0_mismatches =
            ime_status == Y26_CONV_STATUS_SUCCESS
                ? compare_array(fixture.label,
                                "ime_conv0",
                                y26_stage6_multiblock_conv0_i32(&ws),
                                fixture.expected_conv0_i32_nhwc,
                                fixture.expected_conv0_count)
                : 1;
        ime_act_mismatches =
            ime_status == Y26_CONV_STATUS_SUCCESS
                ? compare_array(fixture.label,
                                "ime_act0",
                                y26_stage6_multiblock_conv1_input_s8(&ws),
                                fixture.expected_act0_s8_nhwc,
                                fixture.expected_act0_count)
                : 1;
        ime_conv1_mismatches =
            ime_status == Y26_CONV_STATUS_SUCCESS
                ? compare_array(fixture.label,
                                "ime_conv1",
                                ime_output.data(),
                                fixture.expected_conv1_i32_nhwc,
                                fixture.expected_conv1_count)
                : 1;
        failures += (ime_status == Y26_CONV_STATUS_SUCCESS && ime_conv0_mismatches == 0 &&
                     ime_act_mismatches == 0 && ime_conv1_mismatches == 0)
                        ? 0
                        : 1;
    }

    std::cout << "stage6_multiblock_fixture label=" << fixture.label << " scalar_status=" << scalar_status
              << " scalar_conv0_mismatches=" << scalar_conv0_mismatches
              << " scalar_act_mismatches=" << scalar_act_mismatches
              << " scalar_conv1_mismatches=" << scalar_conv1_mismatches << " ime_status=" << ime_status
              << " ime_conv0_mismatches=" << ime_conv0_mismatches << " ime_act_mismatches=" << ime_act_mismatches
              << " ime_conv1_mismatches=" << ime_conv1_mismatches << " scalar_total_us=" << scalar_timing.total_us
              << " ime_total_us=" << ime_timing.total_us << " prepacked_bytes=" << ws.prepacked_bytes
              << " workspace_bytes=" << ws.workspace_bytes << "\n";
    y26_stage6_multiblock_release(&ws);
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main() {
    int failures = 0;
    for (const auto* fixture : y26_stage6_multiblock_fixture::kFixtures) {
        failures += verify_fixture(*fixture);
    }
    return failures == 0 ? 0 : 1;
}

