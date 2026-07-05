#include <algorithm>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#define main y26_stage16_fullshape_gate_embedded_main
#include "bench_stage16_fullshape_gate.cpp"
#undef main

namespace stage17 {

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

struct ReplaySummary {
    GateTiming mean_timing {};
    MetricStats total_stats {};
    std::size_t mismatches = 0;
    long long checksum = 0;
    int status = Y26_CONV_STATUS_SUCCESS;
};

struct ThreadChunk {
    int cpu = 0;
    int row_begin = 0;
    int row_end = 0;
    int input_row_begin = 0;
    int local_output_offset = 0;
    int local_output_h = 0;
    Y26Stage7ConvNodeConfig cfg {};
    Y26PrepackedConvWeights* weights = nullptr;
    Y26ConvWorkspace* workspace = nullptr;
    std::vector<std::int32_t> raw;
    std::vector<std::int32_t> corrected;
};

struct ThreadSummary {
    int thread_count = 0;
    MetricStats total_stats {};
    double speedup_vs_1thread = 1.0;
    std::size_t mismatches = 0;
    long long checksum = 0;
    int status = Y26_CONV_STATUS_SUCCESS;
};

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

void print_protocol(const Protocol& protocol) {
    std::cout << "protocol warmup=" << protocol.warmup << " runs=" << protocol.runs
              << " repeats=" << protocol.repeats << " pin=external_taskset_cpu0_or_cpu0_3\n";
}

ReplaySummary run_stable_replay(const y26_stage15_model4_branch_fixture::Model4BranchFixture& fixture,
                                const char* candidate,
                                int activation_mode,
                                bool use_ime,
                                const Protocol& protocol,
                                const std::vector<std::int8_t>& expected_split1,
                                const std::vector<std::int32_t>& expected_branch0,
                                const std::vector<std::int8_t>& expected_branch0_act,
                                const std::vector<std::int32_t>& model4_cv1_i32) {
    constexpr int split_count = kFullH * kFullW * (kModel4Cv1C / 2);
    constexpr int branch_count = kFullH * kFullW * 16;
    std::vector<double> repeat_total_us;
    ReplaySummary summary {};
    for (int repeat = 0; repeat < protocol.repeats; ++repeat) {
        std::vector<std::int8_t> split1(split_count, 0);
        std::vector<std::int32_t> branch0(branch_count, 0);
        std::vector<std::int8_t> branch0_act(branch_count, 0);
        for (int i = 0; i < protocol.warmup; ++i) {
            GateTiming warmup_timing {};
            summary.status = run_once(
                fixture, activation_mode, use_ime, model4_cv1_i32, split1, branch0, branch0_act, warmup_timing);
            if (summary.status != Y26_CONV_STATUS_SUCCESS) {
                break;
            }
        }
        GateTiming repeat_sum {};
        for (int i = 0; i < protocol.runs && summary.status == Y26_CONV_STATUS_SUCCESS; ++i) {
            GateTiming timing {};
            summary.status = run_once(
                fixture, activation_mode, use_ime, model4_cv1_i32, split1, branch0, branch0_act, timing);
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
        repeat_total_us.push_back(repeat_sum.total_us);
        add_timing(summary.mean_timing, repeat_sum);
        const std::size_t split_mismatches = mismatches_i8(split1, expected_split1);
        const std::size_t branch_mismatches = mismatches_i32(branch0, expected_branch0);
        const std::size_t branch_act_mismatches = mismatches_i8(branch0_act, expected_branch0_act);
        summary.mismatches += split_mismatches + branch_mismatches + branch_act_mismatches;
        summary.checksum = checksum_i32(branch0);
    }
    if (!repeat_total_us.empty()) {
        scale_timing(summary.mean_timing, static_cast<double>(repeat_total_us.size()));
    }
    summary.total_stats = stats_from_values(repeat_total_us);
    const char* correctness = summary.status == Y26_CONV_STATUS_SUCCESS && summary.mismatches == 0 ? "pass" : "fail";
    std::cout << "stable_replay candidate=" << candidate << " correctness_status=" << correctness
              << " status=" << summary.status << " mismatches=" << summary.mismatches
              << " checksum=" << summary.checksum << " mean_total_us=" << summary.total_stats.mean
              << " stddev_total_us=" << summary.total_stats.stddev << " min_total_us=" << summary.total_stats.min
              << " max_total_us=" << summary.total_stats.max << " cv_total_pct=" << summary.total_stats.cv_pct
              << " mean_conv_us=" << summary.mean_timing.conv_us
              << " mean_activation_requant_us=" << summary.mean_timing.activation_requant_us
              << " mean_split_us=" << summary.mean_timing.split_us
              << " mean_correction_us=" << summary.mean_timing.correction_us
              << " conv_share_pct=" << summary.mean_timing.conv_share_pct
              << " activation_share_pct=" << summary.mean_timing.activation_share_pct
              << "\n";
    return summary;
}

bool pin_current_thread_to_cpu(int cpu) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    (void)cpu;
    return false;
#endif
}

void destroy_chunks(std::vector<ThreadChunk>& chunks) {
    for (ThreadChunk& chunk : chunks) {
        y26_prepacked_conv_weights_destroy(chunk.weights);
        y26_conv_workspace_destroy(chunk.workspace);
        chunk.weights = nullptr;
        chunk.workspace = nullptr;
    }
}

bool make_spatial_chunks(const Y26Stage7ConvNodeConfig& branch0,
                         int thread_count,
                         std::vector<ThreadChunk>& chunks) {
    chunks.clear();
    if (thread_count <= 0 || thread_count > 4 || branch0.kernel_h != 3 || branch0.kernel_w != 3 ||
        branch0.params.stride_h != 1 || branch0.params.stride_w != 1 || branch0.params.pad_h != 1 ||
        branch0.params.pad_w != 1) {
        return false;
    }
    constexpr int output_h = kFullH;
    constexpr int output_w = kFullW;
    for (int tid = 0; tid < thread_count; ++tid) {
        const int row_begin = (output_h * tid) / thread_count;
        const int row_end = (output_h * (tid + 1)) / thread_count;
        const bool top_chunk = row_begin == 0;
        const bool bottom_chunk = row_end == output_h;
        const int input_row_begin = thread_count == 1 ? 0 : std::max(0, row_begin - 1);
        const int input_row_end = thread_count == 1 ? output_h : std::min(output_h, row_end + 1);
        const int symmetric_pad_h = top_chunk || bottom_chunk ? 1 : 0;
        ThreadChunk chunk {};
        chunk.cpu = tid;
        chunk.row_begin = row_begin;
        chunk.row_end = row_end;
        chunk.input_row_begin = input_row_begin;
        chunk.local_output_offset = (!top_chunk && bottom_chunk) ? 1 : 0;
        chunk.cfg = branch0;
        chunk.cfg.params.input_h = input_row_end - input_row_begin;
        chunk.cfg.params.input_w = output_w;
        chunk.cfg.params.pad_h = symmetric_pad_h;
        chunk.cfg.params.pad_w = 1;
        const int local_output_h = output_h_for_kernel(chunk.cfg.params, chunk.cfg.kernel_h);
        chunk.local_output_h = local_output_h;
        const int wanted_output_h = row_end - row_begin;
        if (local_output_h < wanted_output_h + chunk.local_output_offset ||
            output_w_for_kernel(chunk.cfg.params, chunk.cfg.kernel_w) != output_w) {
            destroy_chunks(chunks);
            return false;
        }
        chunk.weights = y26_prepacked_conv_weights_create_mmt4d_s8(chunk.cfg.weights_ohwi_s8,
                                                                   &chunk.cfg.params,
                                                                   chunk.cfg.kernel_h,
                                                                   chunk.cfg.kernel_w,
                                                                   chunk.cfg.node_name,
                                                                   chunk.cfg.weight_scales);
        chunk.workspace = y26_conv_workspace_create(&chunk.cfg.params, chunk.cfg.kernel_h, chunk.cfg.kernel_w);
        chunk.raw.resize(static_cast<std::size_t>(local_output_h) * output_w * branch0.params.output_c);
        chunk.corrected.resize(chunk.raw.size());
        if (chunk.weights == nullptr || chunk.workspace == nullptr) {
            destroy_chunks(chunks);
            return false;
        }
        chunks.push_back(std::move(chunk));
    }
    return true;
}

int run_chunk_ime(ThreadChunk& chunk,
                  const std::int8_t* split1_s8,
                  std::int32_t* output_i32) {
    const std::int8_t* local_input =
        split1_s8 + static_cast<std::size_t>(chunk.input_row_begin) * kFullW * chunk.cfg.params.input_c;
    const int status = y26_conv2d_i8s8s32_nhwc_ime_prepacked_v1(local_input,
                                                                chunk.weights,
                                                                const_cast<std::int32_t*>(chunk.raw.data()),
                                                                chunk.cfg.input_storage_zero_point_s8,
                                                                chunk.workspace,
                                                                Y26_CONV_LOOP_ORDER_M_MAJOR);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    const int local_output_m = chunk.local_output_h * kFullW;
    const int correction_status = y26_conv2d_apply_u8_as_s8_correction_nhwc(chunk.raw.data(),
                                                                            chunk.cfg.bias_i32,
                                                                            y26_prepacked_conv_weights_sums(chunk.weights),
                                                                            chunk.corrected.data(),
                                                                            local_output_m,
                                                                            chunk.cfg.params.output_c,
                                                                            chunk.cfg.activation_zero_point_u8);
    if (correction_status != Y26_CONV_STATUS_SUCCESS) {
        return correction_status;
    }
    const int wanted_rows = chunk.row_end - chunk.row_begin;
    const std::size_t row_bytes = static_cast<std::size_t>(kFullW) * chunk.cfg.params.output_c * sizeof(std::int32_t);
    const std::int32_t* src =
        chunk.corrected.data() + static_cast<std::size_t>(chunk.local_output_offset) * kFullW * chunk.cfg.params.output_c;
    std::int32_t* dst = output_i32 + static_cast<std::size_t>(chunk.row_begin) * kFullW * chunk.cfg.params.output_c;
    for (int row = 0; row < wanted_rows; ++row) {
        std::memcpy(dst + static_cast<std::size_t>(row) * kFullW * chunk.cfg.params.output_c,
                    src + static_cast<std::size_t>(row) * kFullW * chunk.cfg.params.output_c,
                    row_bytes);
    }
    return Y26_CONV_STATUS_SUCCESS;
}

ThreadSummary run_threading_candidate(const Y26Stage7ConvNodeConfig& branch0,
                                      const std::int8_t* split1_s8,
                                      const std::vector<std::int32_t>& expected,
                                      int thread_count,
                                      const Protocol& protocol,
                                      double baseline_mean_us) {
    ThreadSummary summary {};
    summary.thread_count = thread_count;
    if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
        summary.status = Y26_CONV_STATUS_NOT_BUILT_WITH_IME;
        return summary;
    }
    std::vector<ThreadChunk> chunks;
    if (!make_spatial_chunks(branch0, thread_count, chunks)) {
        summary.status = Y26_CONV_STATUS_INVALID_ARGUMENT;
        return summary;
    }
    std::vector<std::int32_t> output(expected.size(), 0);
    std::vector<int> statuses(static_cast<std::size_t>(thread_count), Y26_CONV_STATUS_SUCCESS);
    std::barrier start_barrier(thread_count + 1);
    std::barrier done_barrier(thread_count + 1);
    bool stop = false;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(thread_count));
    for (int tid = 0; tid < thread_count; ++tid) {
        workers.emplace_back([&, tid]() {
            (void)pin_current_thread_to_cpu(tid);
            for (;;) {
                start_barrier.arrive_and_wait();
                if (stop) {
                    break;
                }
                statuses[static_cast<std::size_t>(tid)] = run_chunk_ime(chunks[static_cast<std::size_t>(tid)],
                                                                        split1_s8,
                                                                        output.data());
                done_barrier.arrive_and_wait();
            }
        });
    }
    auto run_once_threaded = [&]() -> double {
        std::fill(statuses.begin(), statuses.end(), Y26_CONV_STATUS_SUCCESS);
        const auto begin = Clock::now();
        start_barrier.arrive_and_wait();
        done_barrier.arrive_and_wait();
        const auto end = Clock::now();
        for (int status : statuses) {
            if (status != Y26_CONV_STATUS_SUCCESS) {
                summary.status = status;
            }
        }
        return elapsed_us(begin, end);
    };
    for (int i = 0; i < protocol.warmup && summary.status == Y26_CONV_STATUS_SUCCESS; ++i) {
        (void)run_once_threaded();
    }
    std::vector<double> repeat_means;
    for (int repeat = 0; repeat < protocol.repeats && summary.status == Y26_CONV_STATUS_SUCCESS; ++repeat) {
        double repeat_sum = 0.0;
        for (int run = 0; run < protocol.runs && summary.status == Y26_CONV_STATUS_SUCCESS; ++run) {
            repeat_sum += run_once_threaded();
        }
        repeat_means.push_back(repeat_sum / static_cast<double>(protocol.runs));
        summary.mismatches += mismatches_i32(output, expected);
        summary.checksum = checksum_i32(output);
    }
    stop = true;
    start_barrier.arrive_and_wait();
    for (std::thread& worker : workers) {
        worker.join();
    }
    destroy_chunks(chunks);
    summary.total_stats = stats_from_values(repeat_means);
    summary.speedup_vs_1thread = summary.total_stats.mean > 0.0 && baseline_mean_us > 0.0
                                     ? baseline_mean_us / summary.total_stats.mean
                                     : 1.0;
    const char* correctness = summary.status == Y26_CONV_STATUS_SUCCESS && summary.mismatches == 0 ? "pass" : "fail";
    std::cout << "threading candidate=spatial_row_split thread_count=" << thread_count << " cpus=0-"
              << (thread_count - 1) << " correctness_status=" << correctness << " status=" << summary.status
              << " mismatches=" << summary.mismatches << " checksum=" << summary.checksum
              << " mean_us=" << summary.total_stats.mean << " stddev_us=" << summary.total_stats.stddev
              << " min_us=" << summary.total_stats.min << " max_us=" << summary.total_stats.max
              << " cv_pct=" << summary.total_stats.cv_pct
              << " speedup_vs_1thread=" << summary.speedup_vs_1thread << "\n";
    return summary;
}

void print_roofline(const char* node_name,
                    const char* shape_class,
                    int input_h,
                    int input_w,
                    int input_c,
                    int output_c,
                    int kernel_h,
                    int kernel_w,
                    int stride,
                    int pad,
                    double ime_us) {
    const int output_h = (input_h + 2 * pad - kernel_h) / stride + 1;
    const int output_w = (input_w + 2 * pad - kernel_w) / stride + 1;
    const long long macs = static_cast<long long>(output_h) * output_w * output_c * kernel_h * kernel_w * input_c;
    const long long weight_bytes = static_cast<long long>(output_c) * kernel_h * kernel_w * input_c;
    const long long activation_bytes = static_cast<long long>(input_h) * input_w * input_c;
    const long long output_bytes = static_cast<long long>(output_h) * output_w * output_c * sizeof(std::int32_t);
    const double gmac_s = ime_us > 0.0 ? static_cast<double>(macs) / (ime_us * 1000.0) : 0.0;
    const double tops = gmac_s / 1000.0;
    const double pct_2tops = 100.0 * tops / 2.0;
    const char* bottleneck = kernel_h == 3 && input_c <= 32 ? "structural_low_K_or_packing" : "unclear";
    std::cout << "roofline node=" << node_name << " shape_class=" << shape_class
              << " input_shape=1x" << input_h << "x" << input_w << "x" << input_c
              << " output_shape=1x" << output_h << "x" << output_w << "x" << output_c
              << " kernel=" << kernel_h << "x" << kernel_w << " stride=" << stride << " pad=" << pad
              << " mac_count=" << macs << " weight_bytes=" << weight_bytes
              << " activation_read_bytes=" << activation_bytes << " output_write_bytes=" << output_bytes
              << " ime_single_thread_us=" << ime_us << " effective_GMAC_s=" << gmac_s
              << " effective_TOPS=" << tops << " percent_of_2TOPS=" << pct_2tops
              << " bottleneck_class=" << bottleneck << "\n";
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

}  // namespace stage17

int main(int argc, char** argv) {
    using namespace stage17;
    std::cout << std::fixed << std::setprecision(6);
    const Protocol protocol = parse_protocol(argc, argv);
    print_protocol(protocol);
    const bool main_pinned = pin_current_thread_to_cpu(0);
    std::cout << "main_thread_pin_cpu=0 status=" << (main_pinned ? "pass" : "not_available") << "\n";

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
    std::cout << "subset=candidate_I_model4_split_first_branch shape_class=full_shape_model4_branch_entry"
              << " reference_checksum=" << checksum_i32(expected_branch0) << "\n";

    ReplaySummary scalar = run_stable_replay(fixture,
                                             "scalar_reference_int8_lut",
                                             Y26_ACTIVATION_MODE_INT8_LUT,
                                             false,
                                             protocol,
                                             expected_split1,
                                             expected_branch0,
                                             expected_branch0_act,
                                             model4_cv1_i32);
    int failures = scalar.status == Y26_CONV_STATUS_SUCCESS && scalar.mismatches == 0 ? 0 : 1;
    ReplaySummary ime {};
    if (y26_vmadot_4x4x8_ime_available_buildtime()) {
        (void)y26_k1x_ime_probe_once();
        ime = run_stable_replay(fixture,
                                "stage17_IME_A2_rvv_f32_lut",
                                Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT,
                                true,
                                protocol,
                                expected_split1,
                                expected_branch0,
                                expected_branch0_act,
                                model4_cv1_i32);
        if (ime.status != Y26_CONV_STATUS_SUCCESS || ime.mismatches != 0) {
            failures += 1;
        }
        print_roofline("/model.4/m.0/cv1/conv/Conv",
                       "full_shape_model4_branch_entry",
                       80,
                       80,
                       32,
                       16,
                       3,
                       3,
                       1,
                       1,
                       ime.mean_timing.conv_us);
        print_roofline("/model.4/cv1/conv/Conv",
                       "upstream_full_shape_metadata_only",
                       80,
                       80,
                       64,
                       64,
                       1,
                       1,
                       1,
                       0,
                       0.0);
        print_roofline("/model.4/m.0/cv2/conv/Conv",
                       "compact_only_metadata_not_compared",
                       80,
                       80,
                       16,
                       32,
                       3,
                       3,
                       1,
                       1,
                       0.0);

        Y26Stage7ConvNodeConfig branch0 = fullshape_branch0_config(fixture);
        std::vector<ThreadSummary> thread_results;
        double one_thread_mean = 0.0;
        for (int threads = 1; threads <= 4; ++threads) {
            ThreadSummary result = run_threading_candidate(
                branch0, expected_split1.data(), expected_branch0, threads, protocol, one_thread_mean);
            if (threads == 1) {
                one_thread_mean = result.total_stats.mean;
                result.speedup_vs_1thread = 1.0;
            }
            if (result.status != Y26_CONV_STATUS_SUCCESS || result.mismatches != 0) {
                failures += 1;
            }
            thread_results.push_back(result);
        }
        const double four_thread_speedup = thread_results.size() >= 4 ? thread_results[3].speedup_vs_1thread : 0.0;
        const char* threading_feasibility = "rejected_for_now";
        if (four_thread_speedup >= 3.0) {
            threading_feasibility = "strong_positive";
        } else if (four_thread_speedup >= 2.0) {
            threading_feasibility = "positive";
        } else if (four_thread_speedup >= 1.5) {
            threading_feasibility = "weak_positive";
        }
        std::cout << "threading_feasibility=" << threading_feasibility
                  << " four_thread_speedup=" << four_thread_speedup << "\n";
    } else {
        std::cout << "stable_replay candidate=stage17_IME_A2_rvv_f32_lut correctness_status=not_built\n";
        std::cout << "threading_feasibility=not_built\n";
    }
    return failures == 0 ? 0 : 1;
}
