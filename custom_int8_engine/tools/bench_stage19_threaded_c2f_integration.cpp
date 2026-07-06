#define Y26_STAGE16_NO_TEST_MAIN 1
#include "../tests/test_stage16_model4_c2f_runner.cpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

struct Protocol {
    int warmup = 10;
    int runs = 100;
    int repeats = 5;
};

struct MetricStats {
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;
    double cv_pct = 0.0;
};

struct CandidateSummary {
    Y26Stage16TimingUs timing {};
    MetricStats total {};
    MetricStats conv {};
    MetricStats activation {};
    std::size_t mismatches = 0;
    long long checksum = 0;
    int status = Y26_CONV_STATUS_SUCCESS;
    int affinity_ok = 1;
};

Protocol parse_protocol(int argc, char** argv) {
    Protocol protocol {};
    if (argc > 1) {
        protocol.warmup = std::max(0, std::atoi(argv[1]));
    }
    if (argc > 2) {
        protocol.runs = std::max(1, std::atoi(argv[2]));
    }
    if (argc > 3) {
        protocol.repeats = std::max(1, std::atoi(argv[3]));
    }
    return protocol;
}

MetricStats stats_from_values(const std::vector<double>& values) {
    MetricStats stats {};
    if (values.empty()) {
        return stats;
    }
    stats.min = *std::min_element(values.begin(), values.end());
    stats.max = *std::max_element(values.begin(), values.end());
    stats.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    double sum_sq = 0.0;
    for (double value : values) {
        const double delta = value - stats.mean;
        sum_sq += delta * delta;
    }
    stats.stddev = values.size() > 1 ? std::sqrt(sum_sq / static_cast<double>(values.size() - 1)) : 0.0;
    stats.cv_pct = stats.mean > 0.0 ? 100.0 * stats.stddev / stats.mean : 0.0;
    return stats;
}

void add_timing(Y26Stage16TimingUs& dst, const Y26Stage16TimingUs& src) {
    dst.conv_us += src.conv_us;
    dst.activation_requant_us += src.activation_requant_us;
    dst.split_us += src.split_us;
    dst.merge_us += src.merge_us;
    dst.add_us += src.add_us;
    dst.concat_us += src.concat_us;
    dst.post_qdq_us += src.post_qdq_us;
    dst.pack_layout_us += src.pack_layout_us;
    dst.correction_us += src.correction_us;
    dst.copy_us += src.copy_us;
    dst.branch1_conv_us += src.branch1_conv_us;
    dst.branch1_correction_us += src.branch1_correction_us;
    dst.branch1_activation_us += src.branch1_activation_us;
    dst.model4_cv2_conv_us += src.model4_cv2_conv_us;
    dst.model4_cv2_correction_us += src.model4_cv2_correction_us;
    dst.thread_overhead_us += src.thread_overhead_us;
    dst.total_us += src.total_us;
    dst.stage15_timing_us.conv_us += src.stage15_timing_us.conv_us;
    dst.stage15_timing_us.activation_requant_us += src.stage15_timing_us.activation_requant_us;
    dst.stage15_timing_us.split_us += src.stage15_timing_us.split_us;
    dst.stage15_timing_us.merge_us += src.stage15_timing_us.merge_us;
    dst.stage15_timing_us.correction_us += src.stage15_timing_us.correction_us;
    dst.stage15_timing_us.branch0_conv_us += src.stage15_timing_us.branch0_conv_us;
    dst.stage15_timing_us.branch0_correction_us += src.stage15_timing_us.branch0_correction_us;
    dst.stage15_timing_us.branch0_activation_us += src.stage15_timing_us.branch0_activation_us;
    dst.stage15_timing_us.thread_overhead_us += src.stage15_timing_us.thread_overhead_us;
}

void divide_timing(Y26Stage16TimingUs& timing, double denom) {
    if (denom <= 0.0) {
        return;
    }
    timing.conv_us /= denom;
    timing.activation_requant_us /= denom;
    timing.split_us /= denom;
    timing.merge_us /= denom;
    timing.add_us /= denom;
    timing.concat_us /= denom;
    timing.post_qdq_us /= denom;
    timing.pack_layout_us /= denom;
    timing.correction_us /= denom;
    timing.copy_us /= denom;
    timing.branch1_conv_us /= denom;
    timing.branch1_correction_us /= denom;
    timing.branch1_activation_us /= denom;
    timing.model4_cv2_conv_us /= denom;
    timing.model4_cv2_correction_us /= denom;
    timing.thread_overhead_us /= denom;
    timing.total_us /= denom;
    timing.stage15_timing_us.conv_us /= denom;
    timing.stage15_timing_us.activation_requant_us /= denom;
    timing.stage15_timing_us.split_us /= denom;
    timing.stage15_timing_us.merge_us /= denom;
    timing.stage15_timing_us.correction_us /= denom;
    timing.stage15_timing_us.branch0_conv_us /= denom;
    timing.stage15_timing_us.branch0_correction_us /= denom;
    timing.stage15_timing_us.branch0_activation_us /= denom;
    timing.stage15_timing_us.thread_overhead_us /= denom;
    if (timing.total_us > 0.0) {
        timing.activation_share_pct = 100.0 * timing.activation_requant_us / timing.total_us;
        timing.conv_share_pct = 100.0 * timing.conv_us / timing.total_us;
        timing.merge_share_pct = 100.0 * timing.merge_us / timing.total_us;
        timing.pack_layout_share_pct = 100.0 * timing.pack_layout_us / timing.total_us;
    }
}

long long checksum_i32_stage19(const std::int32_t* values, std::size_t count) {
    long long sum = 0;
    for (std::size_t i = 0; i < count; ++i) {
        sum += values[i];
    }
    return sum;
}

std::size_t output_mismatches(const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture,
                              const Y26Stage16Model4C2fWorkspace& ws,
                              const std::vector<std::int32_t>& output) {
    return mismatches_i32_stage16(
               y26_stage16_model4_c2f_branch1_i32(&ws), fixture.expected_branch1_i32_nhwc, fixture.expected_branch1_count) +
           mismatches_i8_stage16(
               y26_stage16_model4_c2f_concat_s8(&ws), fixture.expected_concat_s8_nhwc, fixture.expected_concat_count) +
           mismatches_i32_stage16(output.data(), fixture.expected_model4_cv2_i32_nhwc, fixture.expected_model4_cv2_count);
}

CandidateSummary run_candidate(const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture,
                               const Protocol& protocol,
                               int thread_count,
                               int thread_activation,
                               bool threaded) {
    CandidateSummary summary {};
    Y26Stage16Model4C2fConfig cfg =
        stage16_config_from_fixture(fixture, Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT);
    Y26Stage16Model4C2fWorkspace ws {};
    summary.status = y26_stage16_model4_c2f_prepare(&cfg, &ws);
    if (summary.status != Y26_CONV_STATUS_SUCCESS) {
        return summary;
    }
    if (threaded) {
        summary.status = y26_stage16_model4_c2f_prepare_threaded_branch0(&cfg, &ws, thread_count);
        if (summary.status != Y26_CONV_STATUS_SUCCESS) {
            y26_stage16_model4_c2f_release(&ws);
            return summary;
        }
        summary.affinity_ok = y26_stage16_model4_c2f_threaded_worker_affinity_ok(&ws);
    }
    const std::int8_t* input = fixture.stage15_fixture->stage14_fixture->stage12_fixture->stage11_fixture
                                   ->stage10_fixture->stage9_fixture->input_nhwc_s8;
    std::vector<std::int32_t> output(y26_stage16_model4_c2f_output_count(&cfg), 0);
    std::vector<double> repeat_total;
    std::vector<double> repeat_conv;
    std::vector<double> repeat_activation;
    for (int repeat = 0; repeat < protocol.repeats; ++repeat) {
        for (int i = 0; i < protocol.warmup; ++i) {
            Y26Stage16TimingUs timing {};
            summary.status = threaded
                                 ? y26_stage16_model4_c2f_run_ime_threaded_branch0_cluster0_hotpath(
                                       &cfg, &ws, input, output.data(), thread_activation, &timing)
                                 : y26_stage16_model4_c2f_run_ime_cluster0_hotpath(
                                       &cfg, &ws, input, output.data(), &timing);
        }
        Y26Stage16TimingUs repeat_sum {};
        for (int i = 0; i < protocol.runs && summary.status == Y26_CONV_STATUS_SUCCESS; ++i) {
            Y26Stage16TimingUs timing {};
            summary.status = threaded
                                 ? y26_stage16_model4_c2f_run_ime_threaded_branch0_cluster0_hotpath(
                                       &cfg, &ws, input, output.data(), thread_activation, &timing)
                                 : y26_stage16_model4_c2f_run_ime_cluster0_hotpath(
                                       &cfg, &ws, input, output.data(), &timing);
            if (summary.status != Y26_CONV_STATUS_SUCCESS) {
                break;
            }
            add_timing(repeat_sum, timing);
        }
        if (summary.status != Y26_CONV_STATUS_SUCCESS) {
            summary.mismatches += 1;
            break;
        }
        divide_timing(repeat_sum, static_cast<double>(protocol.runs));
        add_timing(summary.timing, repeat_sum);
        repeat_total.push_back(repeat_sum.total_us);
        repeat_conv.push_back(repeat_sum.conv_us);
        repeat_activation.push_back(repeat_sum.activation_requant_us);
        summary.mismatches += output_mismatches(fixture, ws, output);
        summary.checksum = checksum_i32_stage19(output.data(), output.size());
    }
    if (!repeat_total.empty()) {
        divide_timing(summary.timing, static_cast<double>(repeat_total.size()));
    }
    summary.total = stats_from_values(repeat_total);
    summary.conv = stats_from_values(repeat_conv);
    summary.activation = stats_from_values(repeat_activation);
    y26_stage16_model4_c2f_release(&ws);
    return summary;
}

void print_summary(const char* candidate,
                   int thread_count,
                   const char* cpus,
                   int thread_activation,
                   const CandidateSummary& summary,
                   double baseline_total,
                   double baseline_branch0_conv) {
    const char* correctness =
        summary.status == Y26_CONV_STATUS_SUCCESS && summary.mismatches == 0 ? "pass" : "fail";
    const double total_speedup = summary.total.mean > 0.0 && baseline_total > 0.0
                                     ? baseline_total / summary.total.mean
                                     : 1.0;
    const double branch0_speedup = summary.timing.stage15_timing_us.branch0_conv_us > 0.0 &&
                                           baseline_branch0_conv > 0.0
                                       ? baseline_branch0_conv / summary.timing.stage15_timing_us.branch0_conv_us
                                       : 1.0;
    std::cout << "stage19_result candidate=" << candidate
              << " shape_class=compact_oracle_scope"
              << " thread_count=" << thread_count
              << " cpus=" << cpus
              << " thread_activation=" << thread_activation
              << " correctness_status=" << correctness
              << " status=" << summary.status
              << " mismatches=" << summary.mismatches
              << " checksum=" << summary.checksum
              << " affinity_ok=" << summary.affinity_ok
              << " mean_total_us=" << summary.total.mean
              << " stddev_total_us=" << summary.total.stddev
              << " cv_total_pct=" << summary.total.cv_pct
              << " mean_conv_us=" << summary.conv.mean
              << " stddev_conv_us=" << summary.conv.stddev
              << " mean_activation_requant_us=" << summary.activation.mean
              << " stddev_activation_requant_us=" << summary.activation.stddev
              << " branch0_conv_us=" << summary.timing.stage15_timing_us.branch0_conv_us
              << " branch0_activation_us=" << summary.timing.stage15_timing_us.branch0_activation_us
              << " branch1_conv_us=" << summary.timing.branch1_conv_us
              << " model4_cv2_conv_us=" << summary.timing.model4_cv2_conv_us
              << " split_us=" << summary.timing.split_us
              << " add_us=" << summary.timing.add_us
              << " concat_us=" << summary.timing.concat_us
              << " post_concat_qdq_us=" << summary.timing.post_qdq_us
              << " pack_layout_us=" << summary.timing.pack_layout_us
              << " correction_us=" << summary.timing.correction_us
              << " thread_overhead_us=" << summary.timing.thread_overhead_us
              << " conv_share_pct=" << summary.timing.conv_share_pct
              << " activation_share_pct=" << summary.timing.activation_share_pct
              << " merge_share_pct=" << summary.timing.merge_share_pct
              << " pack_layout_share_pct=" << summary.timing.pack_layout_share_pct
              << " total_speedup_vs_A0=" << total_speedup
              << " branch0_conv_speedup_vs_A0=" << branch0_speedup << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::cout << std::fixed << std::setprecision(6);
    const Protocol protocol = parse_protocol(argc, argv);
    std::cout << "protocol warmup=" << protocol.warmup
              << " runs=" << protocol.runs
              << " repeats=" << protocol.repeats
              << " pin=taskset_cpu0_3_required\n";
    std::cout << "subset=candidate_K_model4_threaded_c2f_compact_oracle_scope\n";
    const auto& fixture = y26_stage16_model4_c2f_fixture::kSyntheticSeededFixture;
    if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
        std::cout << "stage19_result candidate=A0_single_thread_c2f correctness_status=not_built\n";
        return 0;
    }
    (void)y26_k1x_ime_probe_once();
    CandidateSummary baseline = run_candidate(fixture, protocol, 1, 0, false);
    int failures = baseline.status == Y26_CONV_STATUS_SUCCESS && baseline.mismatches == 0 ? 0 : 1;
    print_summary("A0_single_thread_c2f",
                  1,
                  "0",
                  0,
                  baseline,
                  baseline.total.mean,
                  baseline.timing.stage15_timing_us.branch0_conv_us);
    for (int threads = 1; threads <= 4; ++threads) {
        CandidateSummary summary = run_candidate(fixture, protocol, threads, 0, true);
        const char* cpus = threads == 1 ? "0" : (threads == 2 ? "0-1" : (threads == 3 ? "0-2" : "0-3"));
        print_summary(threads == 1   ? "A1_threaded_conv_1t"
                      : threads == 2 ? "A2_threaded_conv_2t"
                      : threads == 3 ? "A3_threaded_conv_3t"
                                     : "A4_threaded_conv_4t",
                      threads,
                      cpus,
                      0,
                      summary,
                      baseline.total.mean,
                      baseline.timing.stage15_timing_us.branch0_conv_us);
        if (summary.status != Y26_CONV_STATUS_SUCCESS || summary.mismatches != 0 || summary.affinity_ok != 1) {
            failures += 1;
        }
    }
    CandidateSummary activation_summary = run_candidate(fixture, protocol, 4, 1, true);
    print_summary("A5_threaded_conv_threaded_activation_4t",
                  4,
                  "0-3",
                  1,
                  activation_summary,
                  baseline.total.mean,
                  baseline.timing.stage15_timing_us.branch0_conv_us);
    if (activation_summary.status != Y26_CONV_STATUS_SUCCESS || activation_summary.mismatches != 0 ||
        activation_summary.affinity_ok != 1) {
        failures += 1;
    }
    return failures == 0 ? 0 : 1;
}
