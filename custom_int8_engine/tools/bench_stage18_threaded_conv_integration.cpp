#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#endif

#define main y26_stage16_fullshape_gate_embedded_main
#include "bench_stage16_fullshape_gate.cpp"
#undef main

#include "y26_k1x_threaded_conv.h"

namespace stage18 {

constexpr int kDefaultWarmup = 10;
constexpr int kDefaultRuns = 100;
constexpr int kDefaultRepeats = 5;

struct Protocol {
    int warmup = kDefaultWarmup;
    int runs = kDefaultRuns;
    int repeats = kDefaultRepeats;
};

struct MetricStats {
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;
    double cv_pct = 0.0;
};

struct TimingSummary {
    GateTiming mean_timing {};
    MetricStats total_stats {};
    MetricStats conv_stats {};
    std::size_t mismatches = 0;
    long long checksum = 0;
    int status = Y26_CONV_STATUS_SUCCESS;
    int affinity_ok = 1;
};

struct ThreadedContext {
    explicit ThreadedContext(const y26_stage15_model4_branch_fixture::Model4BranchFixture& source_fixture)
        : fixture(source_fixture),
          producer(fullshape_model4_cv1_producer(source_fixture)),
          branch0(fullshape_branch0_config(source_fixture)) {}

    const y26_stage15_model4_branch_fixture::Model4BranchFixture& fixture;
    Y26Stage7ConvNodeConfig producer;
    Y26Stage7ConvNodeConfig branch0;
    Y26ThreadedConvWorkspace* threaded_workspace = nullptr;
    std::vector<std::int8_t> model4_cv1_act;
    std::vector<std::int8_t> split1;
    std::vector<std::int32_t> branch0_i32;
    std::vector<std::int8_t> branch0_act;
    std::int8_t split_lut[256] {};
    std::int8_t branch_lut[256] {};
};

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

void add_timing(GateTiming& dst, const GateTiming& src) {
    dst.conv_us += src.conv_us;
    dst.activation_requant_us += src.activation_requant_us;
    dst.split_us += src.split_us;
    dst.merge_us += src.merge_us;
    dst.post_qdq_us += src.post_qdq_us;
    dst.pack_layout_us += src.pack_layout_us;
    dst.correction_us += src.correction_us;
    dst.copy_us += src.copy_us;
    dst.total_us += src.total_us;
}

void scale_timing(GateTiming& timing, double denom) {
    if (denom <= 0.0) {
        return;
    }
    timing.conv_us /= denom;
    timing.activation_requant_us /= denom;
    timing.split_us /= denom;
    timing.merge_us /= denom;
    timing.post_qdq_us /= denom;
    timing.pack_layout_us /= denom;
    timing.correction_us /= denom;
    timing.copy_us /= denom;
    timing.total_us /= denom;
    if (timing.total_us > 0.0) {
        timing.conv_share_pct = 100.0 * timing.conv_us / timing.total_us;
        timing.activation_share_pct = 100.0 * timing.activation_requant_us / timing.total_us;
        timing.merge_share_pct = 100.0 * timing.merge_us / timing.total_us;
        timing.pack_layout_share_pct = 100.0 * timing.pack_layout_us / timing.total_us;
    }
}

bool pin_main_to_cpu0() {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(0, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    return false;
#endif
}

ThreadedContext make_context(const y26_stage15_model4_branch_fixture::Model4BranchFixture& fixture,
                             int thread_count) {
    ThreadedContext context(fixture);
    constexpr int model4_count = kFullH * kFullW * kModel4Cv1C;
    constexpr int split_count = kFullH * kFullW * (kModel4Cv1C / 2);
    constexpr int branch_count = kFullH * kFullW * 16;
    context.model4_cv1_act.resize(model4_count);
    context.split1.resize(split_count);
    context.branch0_i32.resize(branch_count);
    context.branch0_act.resize(branch_count);
    (void)y26_build_silu_u8_to_s8_lut(context.producer.output_scale,
                                      context.producer.output_zero_point_u8,
                                      fixture.split1_output_scale,
                                      fixture.split1_output_zero_point_u8,
                                      context.split_lut);
    (void)y26_build_silu_u8_to_s8_lut(context.branch0.output_scale,
                                      context.branch0.output_zero_point_u8,
                                      fixture.branch0_act_output_scale,
                                      fixture.branch0_act_output_zero_point_u8,
                                      context.branch_lut);
    context.threaded_workspace = y26_threaded_conv_create_spatial_rows(&context.branch0, thread_count);
    return context;
}

void destroy_context(ThreadedContext& context) {
    y26_threaded_conv_destroy(context.threaded_workspace);
    context.threaded_workspace = nullptr;
}

int run_integrated_once(ThreadedContext& context,
                        const std::vector<std::int32_t>& model4_cv1_i32,
                        GateTiming& timing) {
    const auto begin = Clock::now();
    const auto act0_begin = Clock::now();
    Y26ActivationRequantParams split_params = activation_params(context.producer,
                                                                model4_cv1_i32.size(),
                                                                context.fixture.split1_output_scale,
                                                                context.fixture.split1_output_zero_point_u8);
    int status = apply_activation(Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT,
                                  split_params,
                                  context.split_lut,
                                  model4_cv1_i32.data(),
                                  context.model4_cv1_act.data());
    const auto act0_end = Clock::now();
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    const auto split_begin = Clock::now();
    split_second_half(context.model4_cv1_act.data(), context.split1.data());
    const auto split_end = Clock::now();

    Y26ThreadedConvTimingUs conv_timing {};
    status = y26_threaded_conv_run_ime_cluster0(
        context.threaded_workspace, context.split1.data(), context.branch0_i32.data(), &conv_timing);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }

    const auto act1_begin = Clock::now();
    Y26ActivationRequantParams branch_params = activation_params(context.branch0,
                                                                 context.branch0_i32.size(),
                                                                 context.fixture.branch0_act_output_scale,
                                                                 context.fixture.branch0_act_output_zero_point_u8);
    status = apply_activation(Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT,
                              branch_params,
                              context.branch_lut,
                              context.branch0_i32.data(),
                              context.branch0_act.data());
    const auto act1_end = Clock::now();
    const auto end = Clock::now();
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    timing.activation_requant_us = elapsed_us(act0_begin, act0_end) + elapsed_us(act1_begin, act1_end);
    timing.split_us = elapsed_us(split_begin, split_end);
    timing.merge_us = timing.split_us;
    timing.post_qdq_us = elapsed_us(act0_begin, act0_end);
    timing.conv_us = conv_timing.total_us;
    timing.correction_us = conv_timing.correction_us;
    timing.total_us = elapsed_us(begin, end);
    if (timing.total_us > 0.0) {
        timing.conv_share_pct = 100.0 * timing.conv_us / timing.total_us;
        timing.activation_share_pct = 100.0 * timing.activation_requant_us / timing.total_us;
        timing.merge_share_pct = 100.0 * timing.merge_us / timing.total_us;
    }
    return Y26_CONV_STATUS_SUCCESS;
}

TimingSummary run_baseline_replay(const y26_stage15_model4_branch_fixture::Model4BranchFixture& fixture,
                                  const Protocol& protocol,
                                  const std::vector<std::int8_t>& expected_split1,
                                  const std::vector<std::int32_t>& expected_branch0,
                                  const std::vector<std::int8_t>& expected_branch0_act,
                                  const std::vector<std::int32_t>& model4_cv1_i32) {
    TimingSummary summary {};
    std::vector<double> repeat_total;
    std::vector<double> repeat_conv;
    constexpr int split_count = kFullH * kFullW * (kModel4Cv1C / 2);
    constexpr int branch_count = kFullH * kFullW * 16;
    for (int repeat = 0; repeat < protocol.repeats; ++repeat) {
        std::vector<std::int8_t> split1(split_count, 0);
        std::vector<std::int32_t> branch0(branch_count, 0);
        std::vector<std::int8_t> branch0_act(branch_count, 0);
        for (int i = 0; i < protocol.warmup; ++i) {
            GateTiming timing {};
            summary.status = run_once(fixture,
                                      Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT,
                                      true,
                                      model4_cv1_i32,
                                      split1,
                                      branch0,
                                      branch0_act,
                                      timing);
        }
        GateTiming repeat_sum {};
        for (int i = 0; i < protocol.runs && summary.status == Y26_CONV_STATUS_SUCCESS; ++i) {
            GateTiming timing {};
            summary.status = run_once(fixture,
                                      Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT,
                                      true,
                                      model4_cv1_i32,
                                      split1,
                                      branch0,
                                      branch0_act,
                                      timing);
            if (summary.status != Y26_CONV_STATUS_SUCCESS) {
                break;
            }
            add_timing(repeat_sum, timing);
        }
        if (summary.status != Y26_CONV_STATUS_SUCCESS) {
            summary.mismatches += 1;
            break;
        }
        scale_timing(repeat_sum, static_cast<double>(protocol.runs));
        add_timing(summary.mean_timing, repeat_sum);
        repeat_total.push_back(repeat_sum.total_us);
        repeat_conv.push_back(repeat_sum.conv_us);
        summary.mismatches += mismatches_i8(split1, expected_split1);
        summary.mismatches += mismatches_i32(branch0, expected_branch0);
        summary.mismatches += mismatches_i8(branch0_act, expected_branch0_act);
        summary.checksum = checksum_i32(branch0);
    }
    if (!repeat_total.empty()) {
        scale_timing(summary.mean_timing, static_cast<double>(repeat_total.size()));
    }
    summary.total_stats = stats_from_values(repeat_total);
    summary.conv_stats = stats_from_values(repeat_conv);
    return summary;
}

TimingSummary run_threaded_mode(const y26_stage15_model4_branch_fixture::Model4BranchFixture& fixture,
                                int thread_count,
                                const Protocol& protocol,
                                const std::vector<std::int8_t>& expected_split1,
                                const std::vector<std::int32_t>& expected_branch0,
                                const std::vector<std::int8_t>& expected_branch0_act,
                                const std::vector<std::int32_t>& model4_cv1_i32) {
    TimingSummary summary {};
    ThreadedContext context = make_context(fixture, thread_count);
    if (context.threaded_workspace == nullptr) {
        summary.status = Y26_CONV_STATUS_INVALID_ARGUMENT;
        return summary;
    }
    summary.affinity_ok = y26_threaded_conv_worker_affinity_ok(context.threaded_workspace);
    std::vector<double> repeat_total;
    std::vector<double> repeat_conv;
    for (int repeat = 0; repeat < protocol.repeats; ++repeat) {
        for (int i = 0; i < protocol.warmup; ++i) {
            GateTiming timing {};
            summary.status = run_integrated_once(context, model4_cv1_i32, timing);
        }
        GateTiming repeat_sum {};
        for (int i = 0; i < protocol.runs && summary.status == Y26_CONV_STATUS_SUCCESS; ++i) {
            GateTiming timing {};
            summary.status = run_integrated_once(context, model4_cv1_i32, timing);
            if (summary.status != Y26_CONV_STATUS_SUCCESS) {
                break;
            }
            add_timing(repeat_sum, timing);
        }
        if (summary.status != Y26_CONV_STATUS_SUCCESS) {
            summary.mismatches += 1;
            break;
        }
        scale_timing(repeat_sum, static_cast<double>(protocol.runs));
        add_timing(summary.mean_timing, repeat_sum);
        repeat_total.push_back(repeat_sum.total_us);
        repeat_conv.push_back(repeat_sum.conv_us);
        summary.mismatches += mismatches_i8(context.split1, expected_split1);
        summary.mismatches += mismatches_i32(context.branch0_i32, expected_branch0);
        summary.mismatches += mismatches_i8(context.branch0_act, expected_branch0_act);
        summary.checksum = checksum_i32(context.branch0_i32);
    }
    if (!repeat_total.empty()) {
        scale_timing(summary.mean_timing, static_cast<double>(repeat_total.size()));
    }
    summary.total_stats = stats_from_values(repeat_total);
    summary.conv_stats = stats_from_values(repeat_conv);
    destroy_context(context);
    return summary;
}

void print_summary(const char* candidate,
                   int thread_count,
                   const char* cpu_list,
                   const TimingSummary& summary,
                   double baseline_total,
                   double baseline_conv) {
    const char* correctness =
        summary.status == Y26_CONV_STATUS_SUCCESS && summary.mismatches == 0 ? "pass" : "fail";
    const double total_speedup = summary.total_stats.mean > 0.0 && baseline_total > 0.0
                                     ? baseline_total / summary.total_stats.mean
                                     : 1.0;
    const double conv_speedup = summary.conv_stats.mean > 0.0 && baseline_conv > 0.0
                                    ? baseline_conv / summary.conv_stats.mean
                                    : 1.0;
    std::cout << "stage18_result candidate=" << candidate << " thread_count=" << thread_count
              << " cpus=" << cpu_list << " correctness_status=" << correctness
              << " status=" << summary.status << " mismatches=" << summary.mismatches
              << " checksum=" << summary.checksum << " worker_affinity_ok=" << summary.affinity_ok
              << " mean_total_us=" << summary.total_stats.mean
              << " stddev_total_us=" << summary.total_stats.stddev
              << " min_total_us=" << summary.total_stats.min
              << " max_total_us=" << summary.total_stats.max
              << " cv_total_pct=" << summary.total_stats.cv_pct
              << " mean_conv_us=" << summary.conv_stats.mean
              << " stddev_conv_us=" << summary.conv_stats.stddev
              << " mean_activation_requant_us=" << summary.mean_timing.activation_requant_us
              << " mean_split_us=" << summary.mean_timing.split_us
              << " mean_correction_us=" << summary.mean_timing.correction_us
              << " conv_share_pct=" << summary.mean_timing.conv_share_pct
              << " activation_share_pct=" << summary.mean_timing.activation_share_pct
              << " speedup_total_vs_A0=" << total_speedup
              << " speedup_conv_vs_A0=" << conv_speedup << "\n";
}

void print_plan(const Y26Stage7ConvNodeConfig& cfg, int thread_count) {
    Y26ThreadedConvWorkspace* workspace = y26_threaded_conv_create_spatial_rows(&cfg, thread_count);
    if (workspace == nullptr) {
        std::cout << "halo_plan thread_count=" << thread_count << " status=create_failed\n";
        return;
    }
    Y26ThreadedConvPlan plan {};
    const int status = y26_threaded_conv_get_plan(workspace, &plan);
    std::cout << "halo_plan thread_count=" << thread_count << " status=" << status
              << " total_overcomputed_rows=" << plan.total_overcomputed_rows
              << " total_discarded_rows=" << plan.total_discarded_rows
              << " estimated_extra_macs=" << plan.estimated_extra_macs
              << " estimated_extra_mac_pct=" << plan.estimated_extra_mac_pct << "\n";
    for (int i = 0; i < thread_count; ++i) {
        const Y26ThreadedConvWorkerPlan& worker = plan.workers[i];
        std::cout << "halo_worker thread_count=" << thread_count << " worker=" << i
                  << " cpu=" << worker.cpu << " rows=" << worker.row_begin << ":" << worker.row_end
                  << " input_rows=" << worker.input_row_begin << ":" << worker.input_row_end
                  << " local_output_h=" << worker.local_output_h
                  << " local_output_offset=" << worker.local_output_offset
                  << " output_rows_written=" << worker.output_rows_written
                  << " overcomputed_rows=" << worker.overcomputed_rows
                  << " discarded_rows=" << worker.discarded_rows
                  << " workspace_bytes=" << worker.workspace_bytes
                  << " prepacked_bytes=" << worker.prepacked_bytes << "\n";
    }
    y26_threaded_conv_destroy(workspace);
}

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

}  // namespace stage18

int main(int argc, char** argv) {
    using namespace stage18;
    std::cout << std::fixed << std::setprecision(6);
    const Protocol protocol = parse_protocol(argc, argv);
    std::cout << "protocol warmup=" << protocol.warmup << " runs=" << protocol.runs
              << " repeats=" << protocol.repeats << " pin=taskset_cpu0_3_required\n";
    const bool main_pin = pin_main_to_cpu0();
    std::cout << "main_thread_pin_cpu=0 status=" << (main_pin ? "pass" : "not_available") << "\n";
    const auto& fixture = y26_stage15_model4_branch_fixture::kSyntheticSeededFixture;
    constexpr int model4_count = kFullH * kFullW * kModel4Cv1C;
    constexpr int split_count = kFullH * kFullW * (kModel4Cv1C / 2);
    constexpr int branch_count = kFullH * kFullW * 16;
    std::vector<std::int32_t> model4_cv1_i32(model4_count, 0);
    std::vector<std::int8_t> expected_split1(split_count, 0);
    std::vector<std::int32_t> expected_branch0(branch_count, 0);
    std::vector<std::int8_t> expected_branch0_act(branch_count, 0);
    fill_model4_cv1_i32(fixture, model4_cv1_i32);
    GateTiming reference_timing {};
    int status = run_once(fixture,
                          Y26_ACTIVATION_MODE_INT8_LUT,
                          false,
                          model4_cv1_i32,
                          expected_split1,
                          expected_branch0,
                          expected_branch0_act,
                          reference_timing);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cout << "reference_status=fail status=" << status << "\n";
        return 1;
    }
    std::cout << "subset=candidate_I_model4_split_first_branch_threaded_sidecar"
              << " node=/model.4/m.0/cv1/conv/Conv"
              << " shape_class=representative_full_shape_model4_branch_entry"
              << " reference_checksum=" << checksum_i32(expected_branch0) << "\n";
    Y26Stage7ConvNodeConfig branch0 = fullshape_branch0_config(fixture);
    for (int threads = 1; threads <= 4; ++threads) {
        print_plan(branch0, threads);
    }
    if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
        std::cout << "stage18_result candidate=A0_stage17_single_thread_replay correctness_status=not_built\n";
        return 0;
    }
    (void)y26_k1x_ime_probe_once();
    TimingSummary baseline = run_baseline_replay(
        fixture, protocol, expected_split1, expected_branch0, expected_branch0_act, model4_cv1_i32);
    int failures = baseline.status == Y26_CONV_STATUS_SUCCESS && baseline.mismatches == 0 ? 0 : 1;
    print_summary("A0_stage17_single_thread_replay", 1, "0", baseline, baseline.total_stats.mean, baseline.conv_stats.mean);
    for (int threads = 1; threads <= 4; ++threads) {
        TimingSummary threaded =
            run_threaded_mode(fixture, threads, protocol, expected_split1, expected_branch0, expected_branch0_act,
                              model4_cv1_i32);
        const char* cpus = threads == 1 ? "0" : (threads == 2 ? "0-1" : (threads == 3 ? "0-2" : "0-3"));
        print_summary(threads == 1   ? "A1_integrated_threaded_conv_1t"
                      : threads == 2 ? "A2_integrated_threaded_conv_2t"
                      : threads == 3 ? "A3_integrated_threaded_conv_3t"
                                     : "A4_integrated_threaded_conv_4t",
                      threads,
                      cpus,
                      threaded,
                      baseline.total_stats.mean,
                      baseline.conv_stats.mean);
        if (threaded.status != Y26_CONV_STATUS_SUCCESS || threaded.mismatches != 0 || threaded.affinity_ok != 1) {
            failures += 1;
        }
    }
    return failures == 0 ? 0 : 1;
}
