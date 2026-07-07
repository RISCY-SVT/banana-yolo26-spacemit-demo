#define Y26_STAGE16_NO_TEST_MAIN 1
#include "test_stage16_model4_c2f_runner.cpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

std::size_t mismatches_u8(const std::vector<std::uint8_t>& actual, const std::vector<std::uint8_t>& expected) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        mismatches += actual[i] != expected[i] ? 1U : 0U;
    }
    return mismatches;
}

int verify_stage26_branch1_add_lut_candidate(
    const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture) {
    Y26Stage16Model4C2fConfig baseline_cfg =
        stage16_config_from_fixture(fixture,
                                    Y26_ACTIVATION_MODE_INT8_LUT,
                                    Y26_STAGE16_MERGE_MODE_STAGE24_B3_SPLIT1_LUT);
    Y26Stage16Model4C2fConfig candidate_cfg =
        stage16_config_from_fixture(fixture,
                                    Y26_ACTIVATION_MODE_INT8_LUT,
                                    Y26_STAGE16_MERGE_MODE_STAGE26_BRANCH1_ADD_LUT);

    Y26Stage16Model4C2fWorkspace baseline_ws {};
    Y26Stage16Model4C2fWorkspace candidate_ws {};
    int status = y26_stage16_model4_c2f_prepare(&baseline_cfg, &baseline_ws);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage26 baseline prepare failed status=" << status << "\n";
        return 1;
    }
    status = y26_stage16_model4_c2f_prepare(&candidate_cfg, &candidate_ws);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage26 candidate prepare failed status=" << status << "\n";
        y26_stage16_model4_c2f_release(&baseline_ws);
        return 1;
    }

    const std::int8_t* upstream_input = fixture.stage15_fixture->stage14_fixture->stage12_fixture->stage11_fixture
                                            ->stage10_fixture->stage9_fixture->input_nhwc_s8;
    std::vector<std::int32_t> baseline_i32(y26_stage16_model4_c2f_output_count(&baseline_cfg), 0);
    std::vector<std::int32_t> candidate_i32(y26_stage16_model4_c2f_output_count(&candidate_cfg), 0);
    Y26Stage16TimingUs baseline_timing {};
    Y26Stage16TimingUs candidate_timing {};

    status = y26_stage16_model4_c2f_run_scalar(
        &baseline_cfg, &baseline_ws, upstream_input, baseline_i32.data(), &baseline_timing);
    if (status == Y26_CONV_STATUS_SUCCESS) {
        status = y26_stage16_model4_c2f_run_scalar(
            &candidate_cfg, &candidate_ws, upstream_input, candidate_i32.data(), &candidate_timing);
    }

    std::vector<std::uint8_t> baseline_q(baseline_i32.size(), 0);
    std::vector<std::uint8_t> candidate_q(candidate_i32.size(), 0);
    Y26ConvOutputQuantizeParams output_q_params{baseline_q.size(),
                                                baseline_cfg.model4_cv2.params.output_c,
                                                baseline_cfg.model4_cv2.input_scale,
                                                baseline_cfg.model4_cv2.weight_scales,
                                                baseline_cfg.model4_cv2.output_scale,
                                                baseline_cfg.model4_cv2.output_zero_point_u8};
    if (status == Y26_CONV_STATUS_SUCCESS) {
        status = y26_conv_output_quantize_i32_to_u8_scalar_unrolled(
            &output_q_params, baseline_i32.data(), baseline_q.data());
    }
    if (status == Y26_CONV_STATUS_SUCCESS) {
        status = y26_conv_output_quantize_i32_to_u8_scalar_unrolled(
            &output_q_params, candidate_i32.data(), candidate_q.data());
    }
    const std::size_t output_mismatches =
        status == Y26_CONV_STATUS_SUCCESS ? mismatches_u8(candidate_q, baseline_q) : 1U;
    const std::size_t concat_mismatches =
        mismatches_i8_stage16(y26_stage16_model4_c2f_concat_s8(&candidate_ws),
                              y26_stage16_model4_c2f_concat_s8(&baseline_ws),
                              fixture.expected_concat_count);

    std::cout << "stage26_activation_repair fixture=" << fixture.label
              << " status=" << status
              << " concat_mismatches=" << concat_mismatches
              << " output_mismatches=" << output_mismatches
              << " baseline_branch1_activation_us=" << baseline_timing.branch1_activation_us
              << " candidate_branch1_activation_us=" << candidate_timing.branch1_activation_us
              << " baseline_merge_us=" << baseline_timing.merge_us
              << " candidate_merge_us=" << candidate_timing.merge_us
              << "\n";

    y26_stage16_model4_c2f_release(&candidate_ws);
    y26_stage16_model4_c2f_release(&baseline_ws);
    return status == Y26_CONV_STATUS_SUCCESS && concat_mismatches == 0 && output_mismatches == 0 ? 0 : 1;
}

}  // namespace

int main() {
    return verify_stage26_branch1_add_lut_candidate(y26_stage16_model4_c2f_fixture::kSyntheticSeededFixture);
}
