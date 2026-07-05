#define Y26_STAGE14_NO_TEST_MAIN 1
#include "../tests/test_stage14_next_c2f_runner.cpp"

#include <cstdlib>

namespace {

void accumulate_timing(Y26Stage14TimingUs& dst, const Y26Stage14TimingUs& src) {
    dst.conv_us += src.conv_us;
    dst.activation_requant_us += src.activation_requant_us;
    dst.split_copy_us += src.split_copy_us;
    dst.merge_us += src.merge_us;
    dst.post_qdq_us += src.post_qdq_us;
    dst.pack_layout_us += src.pack_layout_us;
    dst.view_span_us += src.view_span_us;
    dst.add_us += src.add_us;
    dst.concat_us += src.concat_us;
    dst.correction_us += src.correction_us;
    dst.model3_conv_us += src.model3_conv_us;
    dst.model3_correction_us += src.model3_correction_us;
    dst.model4_cv1_conv_us += src.model4_cv1_conv_us;
    dst.model4_cv1_correction_us += src.model4_cv1_correction_us;
    dst.total_us += src.total_us;
}

void divide_timing(Y26Stage14TimingUs& timing, double denom) {
    timing.conv_us /= denom;
    timing.activation_requant_us /= denom;
    timing.split_copy_us /= denom;
    timing.merge_us /= denom;
    timing.post_qdq_us /= denom;
    timing.pack_layout_us /= denom;
    timing.view_span_us /= denom;
    timing.add_us /= denom;
    timing.concat_us /= denom;
    timing.correction_us /= denom;
    timing.model3_conv_us /= denom;
    timing.model3_correction_us /= denom;
    timing.model4_cv1_conv_us /= denom;
    timing.model4_cv1_correction_us /= denom;
    timing.total_us /= denom;
    if (timing.total_us > 0.0) {
        timing.conv_share_pct = 100.0 * timing.conv_us / timing.total_us;
        timing.activation_share_pct = 100.0 * timing.activation_requant_us / timing.total_us;
        timing.merge_share_pct = 100.0 * timing.merge_us / timing.total_us;
        timing.pack_layout_share_pct = 100.0 * timing.pack_layout_us / timing.total_us;
        timing.split_branch_share_pct =
            100.0 * (timing.split_copy_us + timing.add_us + timing.concat_us) / timing.total_us;
    }
}

long long checksum_i32(const std::int32_t* values, std::size_t count) {
    long long sum = 0;
    for (std::size_t i = 0; i < count; ++i) {
        sum += values[i];
    }
    return sum;
}

int run_candidate(const y26_stage14_next_c2f_fixture::NextC2fFixture& fixture,
                  const char* candidate,
                  int activation_mode,
                  bool use_ime,
                  int iterations) {
    Y26Stage14NextC2fConfig cfg = config_from_fixture(fixture, activation_mode);
    Y26Stage14NextC2fWorkspace ws {};
    int status = y26_stage14_next_c2f_prepare(&cfg, &ws);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cout << "candidate=" << candidate << " correctness_status=prepare_fail status=" << status << "\n";
        return 1;
    }
    const std::int8_t* input = fixture.stage12_fixture->stage11_fixture->stage10_fixture->stage9_fixture->input_nhwc_s8;
    std::vector<std::int32_t> output(y26_stage14_next_c2f_output_count(&cfg), 0);
    Y26Stage14TimingUs sum {};
    std::size_t total_mismatches = 0;
    int last_status = Y26_CONV_STATUS_SUCCESS;
    for (int i = 0; i < iterations; ++i) {
        Y26Stage14TimingUs timing {};
        last_status =
            use_ime ? y26_stage14_next_c2f_run_ime_cluster0_hotpath(&cfg, &ws, input, output.data(), &timing)
                    : y26_stage14_next_c2f_run_scalar(&cfg, &ws, input, output.data(), &timing);
        total_mismatches +=
            mismatches_i32(output.data(), fixture.expected_model4_cv1_i32_nhwc, fixture.expected_model4_cv1_count);
        accumulate_timing(sum, timing);
    }
    divide_timing(sum, static_cast<double>(iterations));
    const char* correctness = last_status == Y26_CONV_STATUS_SUCCESS && total_mismatches == 0 ? "pass" : "fail";
    std::cout << "candidate=" << candidate
              << " fixture=" << fixture.label
              << " correctness_status=" << correctness
              << " status=" << last_status
              << " mismatches=" << total_mismatches
              << " checksum=" << checksum_i32(output.data(), output.size())
              << " total_us=" << sum.total_us
              << " conv_us=" << sum.conv_us
              << " activation_requant_us=" << sum.activation_requant_us
              << " split_copy_us=" << sum.split_copy_us
              << " merge_us=" << sum.merge_us
              << " post_qdq_us=" << sum.post_qdq_us
              << " pack_layout_us=" << sum.pack_layout_us
              << " add_us=" << sum.add_us
              << " concat_us=" << sum.concat_us
              << " correction_us=" << sum.correction_us
              << " model3_conv_us=" << sum.model3_conv_us
              << " model3_correction_us=" << sum.model3_correction_us
              << " model4_cv1_conv_us=" << sum.model4_cv1_conv_us
              << " model4_cv1_correction_us=" << sum.model4_cv1_correction_us
              << " conv_share_pct=" << sum.conv_share_pct
              << " activation_share_pct=" << sum.activation_share_pct
              << " merge_share_pct=" << sum.merge_share_pct
              << " pack_layout_share_pct=" << sum.pack_layout_share_pct
              << " split_branch_share_pct=" << sum.split_branch_share_pct
              << "\n";
    y26_stage14_next_c2f_release(&ws);
    return correctness[0] == 'p' ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    int iterations = 3;
    if (argc > 1) {
        iterations = std::max(1, std::atoi(argv[1]));
    }
    const auto& fixture = y26_stage14_next_c2f_fixture::kSyntheticSeededFixture;
    std::cout << "subset=candidate_H3_model2_act_model3_act_model4_cv1_conv iterations=" << iterations << "\n";
    int failures = 0;
    failures += run_candidate(fixture, "scalar_reference_int8_lut", Y26_ACTIVATION_MODE_INT8_LUT, false, iterations);
    if (y26_vmadot_4x4x8_ime_available_buildtime()) {
        (void)y26_k1x_ime_probe_once();
        failures += run_candidate(fixture,
                                  "stage14_IME_A2_rvv_f32_lut",
                                  Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT,
                                  true,
                                  iterations);
    } else {
        std::cout << "candidate=stage14_IME_A2_rvv_f32_lut correctness_status=not_built\n";
    }
    return failures == 0 ? 0 : 1;
}
