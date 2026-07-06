#define Y26_STAGE16_NO_TEST_MAIN 1
#include "test_stage16_model4_c2f_runner.cpp"

#include <iostream>
#include <vector>

namespace {

long long checksum_i32_stage19(const std::int32_t* values, std::size_t count) {
    long long sum = 0;
    for (std::size_t i = 0; i < count; ++i) {
        sum += values[i];
    }
    return sum;
}

int verify_threaded_stage16_mode(const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture,
                                 int thread_count,
                                 int thread_activation) {
    Y26Stage16Model4C2fConfig cfg =
        stage16_config_from_fixture(fixture, Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT);
    Y26Stage16Model4C2fWorkspace ws {};
    int status = y26_stage16_model4_c2f_prepare(&cfg, &ws);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage19 prepare failed status=" << status << "\n";
        return 1;
    }
    status = y26_stage16_model4_c2f_prepare_threaded_branch0(&cfg, &ws, thread_count);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage19 threaded prepare failed thread_count=" << thread_count
                  << " status=" << status << "\n";
        y26_stage16_model4_c2f_release(&ws);
        return 1;
    }

    const std::int8_t* input = fixture.stage15_fixture->stage14_fixture->stage12_fixture->stage11_fixture
                                   ->stage10_fixture->stage9_fixture->input_nhwc_s8;
    std::vector<std::int32_t> output(y26_stage16_model4_c2f_output_count(&cfg), 0);
    Y26Stage16TimingUs timing {};
    status = y26_stage16_model4_c2f_run_ime_threaded_branch0_cluster0_hotpath(
        &cfg, &ws, input, output.data(), thread_activation, &timing);
    const std::size_t branch1_mismatches = mismatches_i32_stage16(
        y26_stage16_model4_c2f_branch1_i32(&ws), fixture.expected_branch1_i32_nhwc, fixture.expected_branch1_count);
    const std::size_t concat_mismatches = mismatches_i8_stage16(
        y26_stage16_model4_c2f_concat_s8(&ws), fixture.expected_concat_s8_nhwc, fixture.expected_concat_count);
    const std::size_t model4_cv2_mismatches = mismatches_i32_stage16(
        output.data(), fixture.expected_model4_cv2_i32_nhwc, fixture.expected_model4_cv2_count);
    const int affinity_ok = y26_stage16_model4_c2f_threaded_worker_affinity_ok(&ws);
    std::cout << "stage19_threaded_c2f_test thread_count=" << thread_count
              << " thread_activation=" << thread_activation
              << " status=" << status
              << " branch1_mismatches=" << branch1_mismatches
              << " concat_mismatches=" << concat_mismatches
              << " model4_cv2_mismatches=" << model4_cv2_mismatches
              << " checksum=" << checksum_i32_stage19(output.data(), output.size())
              << " affinity_ok=" << affinity_ok
              << " total_us=" << timing.total_us
              << " conv_us=" << timing.conv_us
              << " activation_requant_us=" << timing.activation_requant_us
              << " thread_overhead_us=" << timing.thread_overhead_us << "\n";
    y26_stage16_model4_c2f_release(&ws);
    return status == Y26_CONV_STATUS_SUCCESS && branch1_mismatches == 0 && concat_mismatches == 0 &&
                   model4_cv2_mismatches == 0 && affinity_ok == 1
               ? 0
               : 1;
}

}  // namespace

int main() {
    const auto& fixture = y26_stage16_model4_c2f_fixture::kSyntheticSeededFixture;
    Y26Stage16Model4C2fConfig cfg =
        stage16_config_from_fixture(fixture, Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT);
    Y26Stage16Model4C2fWorkspace ws {};
    int failures = 0;
    const int prepare_status = y26_stage16_model4_c2f_prepare(&cfg, &ws);
    if (prepare_status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage19 base prepare failed status=" << prepare_status << "\n";
        return 1;
    }
    const int threaded_prepare = y26_stage16_model4_c2f_prepare_threaded_branch0(&cfg, &ws, 4);
    std::cout << "stage19_threaded_c2f_test prepare_threaded_4t_status=" << threaded_prepare
              << " prepared_thread_count=" << y26_stage16_model4_c2f_threaded_thread_count(&ws) << "\n";
    y26_stage16_model4_c2f_release(&ws);
    if (threaded_prepare != Y26_CONV_STATUS_SUCCESS) {
        return 1;
    }
    if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
        std::cout << "stage19_threaded_c2f_test skipped_no_ime_build\n";
        return 0;
    }
    (void)y26_k1x_ime_probe_once();
    failures += verify_threaded_stage16_mode(fixture, 1, 0);
    failures += verify_threaded_stage16_mode(fixture, 2, 0);
    failures += verify_threaded_stage16_mode(fixture, 3, 0);
    failures += verify_threaded_stage16_mode(fixture, 4, 0);
    failures += verify_threaded_stage16_mode(fixture, 4, 1);
    return failures == 0 ? 0 : 1;
}
