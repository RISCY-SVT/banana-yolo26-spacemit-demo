#define Y26_STAGE16_NO_TEST_MAIN 1
#include "test_stage16_model4_c2f_runner.cpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

long long checksum_i32(const std::vector<std::int32_t>& values) {
    long long sum = 0;
    for (std::int32_t value : values) {
        sum += value;
    }
    return sum;
}

long long checksum_i8(const std::int8_t* values, std::size_t count) {
    long long sum = 0;
    for (std::size_t i = 0; i < count; ++i) {
        sum += values[i];
    }
    return sum;
}

int verify_c2_mode(const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture) {
    Y26Stage16Model4C2fConfig reference_cfg =
        stage16_config_from_fixture(fixture, Y26_ACTIVATION_MODE_INT8_LUT, Y26_STAGE16_MERGE_MODE_A2_FUSED_QDQ_NHWC);
    Y26Stage16Model4C2fConfig c2_cfg = stage16_config_from_fixture(
        fixture, Y26_ACTIVATION_MODE_INT8_LUT, Y26_STAGE16_MERGE_MODE_C2_SPLIT0_CONCAT_LUT);
    Y26Stage16Model4C2fWorkspace reference_ws {};
    Y26Stage16Model4C2fWorkspace c2_ws {};
    int failures = 0;
    int status = y26_stage16_model4_c2f_prepare(&reference_cfg, &reference_ws);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage21 reference prepare failed status=" << status << "\n";
        return 1;
    }
    status = y26_stage16_model4_c2f_prepare(&c2_cfg, &c2_ws);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage21 c2 prepare failed status=" << status << "\n";
        y26_stage16_model4_c2f_release(&reference_ws);
        return 1;
    }

    const std::int8_t* input = fixture.stage15_fixture->stage14_fixture->stage12_fixture->stage11_fixture
                                   ->stage10_fixture->stage9_fixture->input_nhwc_s8;
    std::vector<std::int32_t> reference_output(y26_stage16_model4_c2f_output_count(&reference_cfg), 0);
    std::vector<std::int32_t> c2_output(y26_stage16_model4_c2f_output_count(&c2_cfg), 0);
    Y26Stage16TimingUs reference_timing {};
    Y26Stage16TimingUs c2_timing {};
    status = y26_stage16_model4_c2f_run_scalar(
        &reference_cfg, &reference_ws, input, reference_output.data(), &reference_timing);
    failures += status == Y26_CONV_STATUS_SUCCESS ? 0 : 1;
    status = y26_stage16_model4_c2f_run_scalar(&c2_cfg, &c2_ws, input, c2_output.data(), &c2_timing);
    failures += status == Y26_CONV_STATUS_SUCCESS ? 0 : 1;

    const std::size_t concat_mismatches = mismatches_i8_stage16(y26_stage16_model4_c2f_concat_s8(&c2_ws),
                                                               y26_stage16_model4_c2f_concat_s8(&reference_ws),
                                                               fixture.expected_concat_count);
    const std::size_t concat_oracle_mismatches =
        mismatches_i8_stage16(y26_stage16_model4_c2f_concat_s8(&c2_ws),
                              fixture.expected_concat_s8_nhwc,
                              fixture.expected_concat_count);
    const std::size_t output_mismatches =
        mismatches_i32_stage16(c2_output.data(), reference_output.data(), c2_output.size());
    const std::size_t output_oracle_mismatches =
        mismatches_i32_stage16(c2_output.data(), fixture.expected_model4_cv2_i32_nhwc, fixture.expected_model4_cv2_count);
    std::cout << "stage21_c2f_merge_repair fixture=" << fixture.label
              << " concat_mismatches=" << concat_mismatches
              << " concat_oracle_mismatches=" << concat_oracle_mismatches
              << " model4_cv2_mismatches=" << output_mismatches
              << " model4_cv2_oracle_mismatches=" << output_oracle_mismatches
              << " reference_checksum=" << checksum_i32(reference_output)
              << " c2_checksum=" << checksum_i32(c2_output)
              << " concat_checksum=" << checksum_i8(y26_stage16_model4_c2f_concat_s8(&c2_ws), fixture.expected_concat_count)
              << " c2_activation_requant_us=" << c2_timing.activation_requant_us
              << " c2_merge_us=" << c2_timing.merge_us
              << "\n";

    failures += concat_mismatches == 0 && concat_oracle_mismatches == 0 && output_mismatches == 0 &&
                        output_oracle_mismatches == 0
                    ? 0
                    : 1;
    y26_stage16_model4_c2f_release(&c2_ws);
    y26_stage16_model4_c2f_release(&reference_ws);
    return failures;
}

}  // namespace

int main() {
    int failures = 0;
    for (const auto* fixture : y26_stage16_model4_c2f_fixture::kFixtures) {
        failures += verify_c2_mode(*fixture);
    }
    return failures == 0 ? 0 : 1;
}
