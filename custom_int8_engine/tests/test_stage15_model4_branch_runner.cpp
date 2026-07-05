#define Y26_STAGE14_NO_TEST_MAIN 1
#include "test_stage14_next_c2f_runner.cpp"

#include "stage15_model4_branch_fixture.h"
#include "y26_k1x_model4_branch_runner.h"
#include "y26_k1x_vmadot.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

Y26Stage7ConvNodeConfig stage15_branch0_config_from_fixture(
    const y26_stage15_model4_branch_fixture::Model4BranchFixture& fixture) {
    return Y26Stage7ConvNodeConfig{fixture.branch0_node_name,
                                   fixture.branch0_params,
                                   fixture.branch0_kernel_h,
                                   fixture.branch0_kernel_w,
                                   fixture.branch0_activation_zero_point_u8,
                                   fixture.branch0_input_storage_zero_point_s8,
                                   fixture.split1_output_scale,
                                   fixture.branch0_output_scale,
                                   fixture.branch0_output_zero_point_u8,
                                   fixture.branch0_weight_scales,
                                   fixture.branch0_weight_scale_count,
                                   fixture.branch0_weights_ohwi_s8,
                                   fixture.branch0_weight_count,
                                   fixture.branch0_bias_i32,
                                   fixture.branch0_bias_count};
}

Y26Stage15Model4BranchConfig stage15_config_from_fixture(
    const y26_stage15_model4_branch_fixture::Model4BranchFixture& fixture,
    int activation_mode) {
    return Y26Stage15Model4BranchConfig{fixture.subset_id,
                                        config_from_fixture(*fixture.stage14_fixture, activation_mode),
                                        stage15_branch0_config_from_fixture(fixture),
                                        fixture.split1_output_scale,
                                        fixture.split1_output_zero_point_u8,
                                        fixture.branch0_act_output_scale,
                                        fixture.branch0_act_output_zero_point_u8,
                                        activation_mode};
}

std::size_t mismatches_i32_stage15(const std::int32_t* actual, const std::int32_t* expected, std::size_t count) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < count; ++i) {
        mismatches += actual[i] != expected[i] ? 1U : 0U;
    }
    return mismatches;
}

std::size_t mismatches_i8_stage15(const std::int8_t* actual, const std::int8_t* expected, std::size_t count) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < count; ++i) {
        mismatches += actual[i] != expected[i] ? 1U : 0U;
    }
    return mismatches;
}

[[maybe_unused]] int verify_stage15_mode(const y26_stage15_model4_branch_fixture::Model4BranchFixture& fixture,
                                         int activation_mode,
                                         const char* label,
                                         bool use_ime) {
    Y26Stage15Model4BranchConfig cfg = stage15_config_from_fixture(fixture, activation_mode);
    Y26Stage15Model4BranchWorkspace ws {};
    const int prepare_status = y26_stage15_model4_branch_prepare(&cfg, &ws);
    if (prepare_status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage15 prepare failed fixture=" << fixture.label << " mode=" << label
                  << " status=" << prepare_status << "\n";
        return 1;
    }

    std::vector<std::int32_t> output(y26_stage15_model4_branch_output_count(&cfg), 0);
    Y26Stage15TimingUs timing {};
    const std::int8_t* input = fixture.stage14_fixture->stage12_fixture->stage11_fixture->stage10_fixture
                                   ->stage9_fixture->input_nhwc_s8;
    const int status =
        use_ime ? y26_stage15_model4_branch_run_ime_cluster0_hotpath(&cfg, &ws, input, output.data(), &timing)
                : y26_stage15_model4_branch_run_scalar(&cfg, &ws, input, output.data(), &timing);
    const std::size_t split1_mismatches =
        mismatches_i8_stage15(y26_stage15_model4_branch_split1_input_s8(&ws),
                              fixture.expected_split1_input_s8_nhwc,
                              fixture.expected_split1_input_count);
    const std::size_t branch0_mismatches =
        mismatches_i32_stage15(output.data(), fixture.expected_branch0_i32_nhwc, fixture.expected_branch0_count);
    const std::size_t branch0_act_mismatches =
        mismatches_i8_stage15(y26_stage15_model4_branch_branch0_act_s8(&ws),
                              fixture.expected_branch0_act_s8_nhwc,
                              fixture.expected_branch0_act_count);
    std::cout << "stage15_model4_branch fixture=" << fixture.label << " mode=" << label << " status=" << status
              << " split1_mismatches=" << split1_mismatches
              << " branch0_mismatches=" << branch0_mismatches
              << " branch0_act_mismatches=" << branch0_act_mismatches
              << " total_us=" << timing.total_us
              << " conv_us=" << timing.conv_us
              << " activation_requant_us=" << timing.activation_requant_us
              << " split_us=" << timing.split_us
              << " merge_us=" << timing.merge_us
              << "\n";
    y26_stage15_model4_branch_release(&ws);
    return status == Y26_CONV_STATUS_SUCCESS && split1_mismatches == 0 && branch0_mismatches == 0 &&
                   branch0_act_mismatches == 0
               ? 0
               : 1;
}

}  // namespace

#if !defined(Y26_STAGE15_NO_TEST_MAIN)
int main() {
    int failures = 0;
    for (const auto* fixture : y26_stage15_model4_branch_fixture::kFixtures) {
        failures += verify_stage15_mode(*fixture, Y26_ACTIVATION_MODE_INT8_LUT, "scalar_int8_lut", false);
#if defined(__riscv_vector)
        failures += verify_stage15_mode(*fixture, Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT, "rvv_f32_lut", false);
#endif
        if (y26_vmadot_4x4x8_ime_available_buildtime()) {
            (void)y26_k1x_ime_probe_once();
            failures += verify_stage15_mode(*fixture, Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT, "ime_rvv_f32_lut", true);
        }
    }
    return failures == 0 ? 0 : 1;
}
#endif
