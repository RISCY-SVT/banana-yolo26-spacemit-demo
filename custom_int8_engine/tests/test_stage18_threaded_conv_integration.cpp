#define main y26_stage16_fullshape_gate_embedded_main
#include "../tools/bench_stage16_fullshape_gate.cpp"
#undef main

#include "y26_k1x_threaded_conv.h"
#include "y26_k1x_vmadot.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

int verify_plan(const Y26Stage7ConvNodeConfig& cfg, int thread_count) {
    Y26ThreadedConvWorkspace* workspace = y26_threaded_conv_create_spatial_rows(&cfg, thread_count);
    if (workspace == nullptr) {
        std::cerr << "threaded conv workspace create failed thread_count=" << thread_count << "\n";
        return 1;
    }
    Y26ThreadedConvPlan plan {};
    int status = y26_threaded_conv_get_plan(workspace, &plan);
    int failures = 0;
    failures += status == Y26_CONV_STATUS_SUCCESS ? 0 : 1;
    failures += plan.thread_count == thread_count ? 0 : 1;
    failures += plan.output_h == 80 && plan.output_w == 80 && plan.output_c == 16 ? 0 : 1;
    failures += plan.kernel_h == 3 && plan.kernel_w == 3 && plan.input_c == 32 ? 0 : 1;
    failures += plan.total_discarded_rows == (thread_count == 1 ? 0 : 2) ? 0 : 1;
    failures += plan.estimated_extra_macs ==
                        (thread_count == 1 ? 0LL : 2LL * 80LL * 16LL * 3LL * 3LL * 32LL)
                    ? 0
                    : 1;
    for (int i = 0; i < thread_count; ++i) {
        const Y26ThreadedConvWorkerPlan& worker = plan.workers[i];
        failures += worker.cpu == i ? 0 : 1;
        failures += worker.row_begin < worker.row_end ? 0 : 1;
        failures += worker.output_rows_written == worker.row_end - worker.row_begin ? 0 : 1;
        failures += worker.prepacked_bytes > 0 && worker.workspace_bytes > 0 ? 0 : 1;
    }
    y26_threaded_conv_destroy(workspace);
    return failures;
}

int verify_threaded_run(const Y26Stage7ConvNodeConfig& cfg,
                        const std::vector<std::int8_t>& split1,
                        const std::vector<std::int32_t>& expected,
                        int thread_count) {
    Y26ThreadedConvWorkspace* workspace = y26_threaded_conv_create_spatial_rows(&cfg, thread_count);
    if (workspace == nullptr) {
        std::cerr << "threaded conv workspace create failed for run thread_count=" << thread_count << "\n";
        return 1;
    }
    std::vector<std::int32_t> actual(expected.size(), 0);
    Y26ThreadedConvTimingUs timing {};
    const int status =
        y26_threaded_conv_run_ime_cluster0(workspace, split1.data(), actual.data(), &timing);
    std::size_t mismatches = 0;
    long long checksum = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        mismatches += actual[i] != expected[i] ? 1U : 0U;
        checksum += actual[i];
    }
    const int affinity_ok = y26_threaded_conv_worker_affinity_ok(workspace);
    std::cout << "stage18_threaded_conv_test thread_count=" << thread_count << " status=" << status
              << " mismatches=" << mismatches << " checksum=" << checksum
              << " total_us=" << timing.total_us << " worker_affinity_ok=" << affinity_ok << "\n";
    y26_threaded_conv_destroy(workspace);
    return status == Y26_CONV_STATUS_SUCCESS && mismatches == 0 && affinity_ok == 1 ? 0 : 1;
}

}  // namespace

int main() {
    const auto& fixture = y26_stage15_model4_branch_fixture::kSyntheticSeededFixture;
    Y26Stage7ConvNodeConfig branch0 = fullshape_branch0_config(fixture);
    int failures = 0;
    for (int threads = 1; threads <= 4; ++threads) {
        failures += verify_plan(branch0, threads);
    }

    constexpr int model4_count = kFullH * kFullW * kModel4Cv1C;
    constexpr int split_count = kFullH * kFullW * (kModel4Cv1C / 2);
    constexpr int branch_count = kFullH * kFullW * 16;
    std::vector<std::int32_t> model4_cv1_i32(model4_count, 0);
    std::vector<std::int8_t> expected_split1(split_count, 0);
    std::vector<std::int32_t> expected_branch0(branch_count, 0);
    std::vector<std::int8_t> expected_branch0_act(branch_count, 0);
    fill_model4_cv1_i32(fixture, model4_cv1_i32);
    GateTiming reference_timing {};
    const int reference_status = run_once(fixture,
                                          Y26_ACTIVATION_MODE_INT8_LUT,
                                          false,
                                          model4_cv1_i32,
                                          expected_split1,
                                          expected_branch0,
                                          expected_branch0_act,
                                          reference_timing);
    if (reference_status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage18 reference generation failed status=" << reference_status << "\n";
        return 1;
    }
    if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
        std::cout << "stage18_threaded_conv_test skipped_no_ime_build\n";
        return failures == 0 ? 0 : 1;
    }
    (void)y26_k1x_ime_probe_once();
    for (int threads = 1; threads <= 4; ++threads) {
        failures += verify_threaded_run(branch0, expected_split1, expected_branch0, threads);
    }
    return failures == 0 ? 0 : 1;
}
