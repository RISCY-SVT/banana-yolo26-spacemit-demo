#define Y26_STAGE16_NO_TEST_MAIN 1
#include "test_stage16_model4_c2f_runner.cpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

unsigned long long checksum_u8(const std::vector<std::uint8_t>& values) {
    unsigned long long sum = 0;
    for (std::uint8_t value : values) {
        sum += value;
    }
    return sum;
}

std::size_t mismatches_u8(const std::vector<std::uint8_t>& actual, const std::vector<std::uint8_t>& expected) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        mismatches += actual[i] != expected[i] ? 1U : 0U;
    }
    return mismatches;
}

int verify_cut_api_from_compact_fixture(const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture) {
    Y26Stage16Model4C2fConfig cfg =
        stage16_config_from_fixture(fixture,
                                    Y26_ACTIVATION_MODE_INT8_LUT,
                                    Y26_STAGE16_MERGE_MODE_C2_SPLIT0_CONCAT_LUT);

    Y26Stage16Model4C2fWorkspace reference_ws {};
    Y26Stage16Model4C2fWorkspace cut_ws {};
    int status = y26_stage16_model4_c2f_prepare(&cfg, &reference_ws);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage23 reference prepare failed status=" << status << "\n";
        return 1;
    }
    status = y26_stage16_model4_c2f_prepare_cut(&cfg, &cut_ws);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage23 cut prepare failed status=" << status << "\n";
        y26_stage16_model4_c2f_release(&reference_ws);
        return 1;
    }

    const std::int8_t* upstream_input = fixture.stage15_fixture->stage14_fixture->stage12_fixture->stage11_fixture
                                            ->stage10_fixture->stage9_fixture->input_nhwc_s8;
    std::vector<std::int32_t> reference_i32(y26_stage16_model4_c2f_output_count(&cfg), 0);
    Y26Stage16TimingUs reference_timing {};
    status = y26_stage16_model4_c2f_run_scalar(&cfg, &reference_ws, upstream_input, reference_i32.data(), &reference_timing);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage23 reference run failed status=" << status << "\n";
        y26_stage16_model4_c2f_release(&cut_ws);
        y26_stage16_model4_c2f_release(&reference_ws);
        return 1;
    }

    std::vector<std::uint8_t> cut_input(y26_stage16_model4_c2f_cut_input_count(&cfg), 0);
    Y26ConvOutputQuantizeParams model4_cv1_q_params{cut_input.size(),
                                                    cfg.stage15.stage14.model4_cv1.params.output_c,
                                                    cfg.stage15.stage14.model4_cv1.input_scale,
                                                    cfg.stage15.stage14.model4_cv1.weight_scales,
                                                    cfg.stage15.stage14.model4_cv1.output_scale,
                                                    cfg.stage15.stage14.model4_cv1.output_zero_point_u8};
    status = y26_conv_output_quantize_i32_to_u8_scalar_unrolled(
        &model4_cv1_q_params,
        fixture.stage15_fixture->stage14_fixture->expected_model4_cv1_i32_nhwc,
        cut_input.data());
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage23 cut input quantize failed status=" << status << "\n";
        y26_stage16_model4_c2f_release(&cut_ws);
        y26_stage16_model4_c2f_release(&reference_ws);
        return 1;
    }

    std::vector<std::uint8_t> reference_q(reference_i32.size(), 0);
    std::vector<std::uint8_t> cut_q(reference_i32.size(), 0);
    Y26ConvOutputQuantizeParams output_q_params{reference_q.size(),
                                                cfg.model4_cv2.params.output_c,
                                                cfg.model4_cv2.input_scale,
                                                cfg.model4_cv2.weight_scales,
                                                cfg.model4_cv2.output_scale,
                                                cfg.model4_cv2.output_zero_point_u8};
    status = y26_conv_output_quantize_i32_to_u8_scalar_unrolled(&output_q_params, reference_i32.data(), reference_q.data());
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage23 reference output quantize failed status=" << status << "\n";
        y26_stage16_model4_c2f_release(&cut_ws);
        y26_stage16_model4_c2f_release(&reference_ws);
        return 1;
    }

    Y26Stage16TimingUs cut_timing {};
    status = y26_stage16_model4_c2f_run_cut_u8_output(&cfg,
                                                      &cut_ws,
                                                      cut_input.data(),
                                                      cut_q.data(),
                                                      0,
                                                      0,
                                                      0,
                                                      &cut_timing);
    const std::size_t output_mismatches = mismatches_u8(cut_q, reference_q);
    const std::size_t concat_mismatches =
        mismatches_i8_stage16(y26_stage16_model4_c2f_concat_s8(&cut_ws),
                              y26_stage16_model4_c2f_concat_s8(&reference_ws),
                              fixture.expected_concat_count);
    std::cout << "stage23_runner_api_cut fixture=" << fixture.label
              << " status=" << status
              << " output_mismatches=" << output_mismatches
              << " concat_mismatches=" << concat_mismatches
              << " reference_checksum=" << checksum_u8(reference_q)
              << " cut_checksum=" << checksum_u8(cut_q)
              << " output_quantize_us=" << cut_timing.output_quantize_us
              << "\n";
    y26_stage16_model4_c2f_release(&cut_ws);
    y26_stage16_model4_c2f_release(&reference_ws);
    return status == Y26_CONV_STATUS_SUCCESS && output_mismatches == 0 && concat_mismatches == 0 ? 0 : 1;
}

}  // namespace

int main() {
    int failures = 0;
    failures += verify_cut_api_from_compact_fixture(y26_stage16_model4_c2f_fixture::kSyntheticSeededFixture);
    return failures == 0 ? 0 : 1;
}
