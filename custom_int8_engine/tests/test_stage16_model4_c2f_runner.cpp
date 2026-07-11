#if defined(Y26_STAGE16_NO_TEST_MAIN)

#define Y26_STAGE15_NO_TEST_MAIN 1
#include "test_stage15_model4_branch_runner.cpp"

#include "stage16_model4_c2f_fixture.h"
#include "y26_k1x_model4_c2f_runner.h"

namespace {

Y26Stage7ConvNodeConfig branch1_config_from_fixture(
    const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture) {
    return Y26Stage7ConvNodeConfig{"/model.4/m.0/cv2/conv/Conv",
                                   fixture.branch1_params,
                                   fixture.branch1_kernel_h,
                                   fixture.branch1_kernel_w,
                                   fixture.branch1_activation_zero_point_u8,
                                   fixture.branch1_input_storage_zero_point_s8,
                                   fixture.branch1_input_scale,
                                   fixture.branch1_output_scale,
                                   fixture.branch1_output_zero_point_u8,
                                   fixture.branch1_weight_scales,
                                   fixture.branch1_weight_scale_count,
                                   fixture.branch1_weights_ohwi_s8,
                                   fixture.branch1_weight_count,
                                   fixture.branch1_bias_i32,
                                   fixture.branch1_bias_count};
}

Y26Stage7ConvNodeConfig model4_cv2_config_from_fixture(
    const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture) {
    return Y26Stage7ConvNodeConfig{"/model.4/cv2/conv/Conv",
                                   fixture.model4_cv2_params,
                                   fixture.model4_cv2_kernel_h,
                                   fixture.model4_cv2_kernel_w,
                                   fixture.model4_cv2_activation_zero_point_u8,
                                   fixture.model4_cv2_input_storage_zero_point_s8,
                                   fixture.concat_output_scale,
                                   fixture.model4_cv2_output_scale,
                                   fixture.model4_cv2_output_zero_point_u8,
                                   fixture.model4_cv2_weight_scales,
                                   fixture.model4_cv2_weight_scale_count,
                                   fixture.model4_cv2_weights_ohwi_s8,
                                   fixture.model4_cv2_weight_count,
                                   fixture.model4_cv2_bias_i32,
                                   fixture.model4_cv2_bias_count};
}

Y26Stage16Model4C2fConfig stage16_config_from_fixture(
    const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture,
    int activation_mode,
    int merge_mode = Y26_STAGE16_MERGE_MODE_A2_FUSED_QDQ_NHWC) {
    return Y26Stage16Model4C2fConfig{fixture.subset_id,
                                     stage15_config_from_fixture(*fixture.stage15_fixture, activation_mode),
                                     branch1_config_from_fixture(fixture),
                                     model4_cv2_config_from_fixture(fixture),
                                     fixture.concat_output_scale,
                                     fixture.concat_output_zero_point_u8,
                                     activation_mode,
                                     merge_mode};
}

std::size_t mismatches_i32_stage16(const std::int32_t* actual,
                                   const std::int32_t* expected,
                                   std::size_t count) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < count; ++i) {
        mismatches += actual[i] != expected[i] ? 1U : 0U;
    }
    return mismatches;
}

std::size_t mismatches_i8_stage16(const std::int8_t* actual,
                                  const std::int8_t* expected,
                                  std::size_t count) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < count; ++i) {
        mismatches += actual[i] != expected[i] ? 1U : 0U;
    }
    return mismatches;
}

}  // namespace

#else

#include "y26_k1x_model4_fixture_config.h"
#include "y26_k1x_vmadot.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

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

int verify_mode(int fixture_id, int activation_mode, const char* label, bool use_ime) {
    Y26Stage16Model4C2fConfig cfg {};
    Y26Model4FixtureView fixture {};
    const int fixture_status = y26_model4_fixture_make(fixture_id,
                                                        activation_mode,
                                                        Y26_STAGE16_MERGE_MODE_A2_FUSED_QDQ_NHWC,
                                                        &cfg,
                                                        &fixture);
    if (fixture_status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage16 fixture factory failed id=" << fixture_id << "\n";
        return 1;
    }

    Y26Stage16Model4C2fWorkspace ws {};
    const int prepare_status = y26_stage16_model4_c2f_prepare(&cfg, &ws);
    if (prepare_status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage16 prepare failed fixture=" << fixture.label << " mode=" << label
                  << " status=" << prepare_status << "\n";
        return 1;
    }

    std::vector<std::int32_t> output(y26_stage16_model4_c2f_output_count(&cfg), 0);
    Y26Stage16TimingUs timing {};
    const int status = use_ime
                           ? y26_stage16_model4_c2f_run_ime_cluster0_hotpath(
                                 &cfg, &ws, fixture.input_nhwc_s8, output.data(), &timing)
                           : y26_stage16_model4_c2f_run_scalar(
                                 &cfg, &ws, fixture.input_nhwc_s8, output.data(), &timing);
    const std::size_t branch1_mismatches =
        mismatches_i32(y26_stage16_model4_c2f_branch1_i32(&ws),
                       fixture.expected_branch1_i32_nhwc,
                       fixture.expected_branch1_count);
    const std::size_t concat_mismatches =
        mismatches_i8(y26_stage16_model4_c2f_concat_s8(&ws),
                      fixture.expected_concat_s8_nhwc,
                      fixture.expected_concat_count);
    const std::size_t model4_cv2_mismatches =
        mismatches_i32(output.data(), fixture.expected_model4_cv2_i32_nhwc, fixture.expected_model4_cv2_count);
    std::cout << "stage16_model4_c2f fixture=" << fixture.label << " mode=" << label << " status=" << status
              << " branch1_mismatches=" << branch1_mismatches
              << " concat_mismatches=" << concat_mismatches
              << " model4_cv2_mismatches=" << model4_cv2_mismatches
              << " total_us=" << timing.total_us
              << " conv_us=" << timing.conv_us
              << " activation_requant_us=" << timing.activation_requant_us
              << " merge_us=" << timing.merge_us
              << " post_qdq_us=" << timing.post_qdq_us << "\n";
    y26_stage16_model4_c2f_release(&ws);
    return status == Y26_CONV_STATUS_SUCCESS && branch1_mismatches == 0 && concat_mismatches == 0 &&
                   model4_cv2_mismatches == 0
               ? 0
               : 1;
}

}  // namespace

int main() {
    int failures = 0;
    for (int fixture_id = 0; fixture_id < y26_model4_fixture_count(); ++fixture_id) {
        failures += verify_mode(fixture_id, Y26_ACTIVATION_MODE_INT8_LUT, "scalar_int8_lut", false);
#if defined(__riscv_vector)
        failures += verify_mode(fixture_id, Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT, "rvv_f32_lut", false);
#endif
        if (y26_vmadot_4x4x8_ime_available_buildtime()) {
            (void)y26_k1x_ime_probe_once();
            failures += verify_mode(fixture_id,
                                    Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT,
                                    "ime_rvv_f32_lut",
                                    true);
        }
    }
    return failures == 0 ? 0 : 1;
}

#endif
