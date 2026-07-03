#include "stage5_block0_fixture.h"
#include "y26_k1x_block_runner.h"
#include "y26_k1x_vmadot.h"

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

Y26Stage5Block0Config config_from_fixture(const y26_stage5_block0_fixture::Block0Fixture& fixture) {
    return Y26Stage5Block0Config{
        "block0_conv_only",
        fixture.node_name,
        fixture.params,
        fixture.kernel_h,
        fixture.kernel_w,
        fixture.activation_zero_point_u8,
        fixture.input_storage_zero_point_s8,
        fixture.weights_ohwi_s8,
        fixture.weight_count,
        fixture.bias_i32,
        fixture.bias_count,
    };
}

int compare_expected(const y26_stage5_block0_fixture::Block0Fixture& fixture,
                     const std::vector<std::int32_t>& output,
                     const char* path_name) {
    int mismatches = 0;
    for (std::size_t i = 0; i < fixture.expected_count; ++i) {
        if (output[i] != fixture.expected_i32_nhwc[i]) {
            ++mismatches;
            if (mismatches <= 8) {
                std::cerr << fixture.label << " " << path_name << " mismatch index=" << i << " got=" << output[i]
                          << " expected=" << fixture.expected_i32_nhwc[i] << "\n";
            }
        }
    }
    return mismatches;
}

int verify_fixture(const y26_stage5_block0_fixture::Block0Fixture& fixture) {
    Y26Stage5Block0Config cfg = config_from_fixture(fixture);
    Y26Stage5Block0Workspace ws {};
    int failures = 0;
    const int prepare_status = y26_stage5_block0_prepare(&cfg, &ws);
    if (prepare_status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << fixture.label << " prepare_status=" << prepare_status << "\n";
        return 1;
    }
    const std::size_t output_count = y26_stage5_block0_output_count(&cfg);
    if (output_count != fixture.expected_count || ws.prepacked_bytes == 0 || ws.workspace_bytes == 0 ||
        ws.raw_i32_count != output_count) {
        std::cerr << fixture.label << " metadata mismatch output_count=" << output_count
                  << " expected_count=" << fixture.expected_count << "\n";
        failures += 1;
    }

    std::vector<std::int32_t> scalar_output(output_count, 0);
    const int scalar_status =
        y26_stage5_block0_run_scalar(&cfg, &ws, fixture.input_nhwc_s8, scalar_output.data());
    const int scalar_mismatches =
        scalar_status == Y26_CONV_STATUS_SUCCESS ? compare_expected(fixture, scalar_output, "scalar") : 1;
    failures += (scalar_status == Y26_CONV_STATUS_SUCCESS && scalar_mismatches == 0) ? 0 : 1;

    int ime_status = Y26_CONV_STATUS_NOT_BUILT_WITH_IME;
    int ime_mismatches = 0;
    if (y26_vmadot_4x4x8_ime_available_buildtime()) {
        std::vector<std::int32_t> ime_output(output_count, 0);
        ime_status = y26_stage5_block0_run_ime_cluster0_hotpath(&cfg, &ws, fixture.input_nhwc_s8, ime_output.data());
        ime_mismatches =
            ime_status == Y26_CONV_STATUS_SUCCESS ? compare_expected(fixture, ime_output, "ime") : 1;
        failures += (ime_status == Y26_CONV_STATUS_SUCCESS && ime_mismatches == 0) ? 0 : 1;
    }

    std::cout << "stage5_block0_fixture label=" << fixture.label << " scalar_status=" << scalar_status
              << " scalar_mismatches=" << scalar_mismatches << " ime_status=" << ime_status
              << " ime_mismatches=" << ime_mismatches << " prepacked_bytes=" << ws.prepacked_bytes
              << " workspace_bytes=" << ws.workspace_bytes << " raw_i32_bytes=" << ws.raw_i32_bytes << "\n";
    y26_stage5_block0_release(&ws);
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main() {
    int failures = 0;
    for (const auto* fixture : y26_stage5_block0_fixture::kFixtures) {
        failures += verify_fixture(*fixture);
    }
    return failures == 0 ? 0 : 1;
}
