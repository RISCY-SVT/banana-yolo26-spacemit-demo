#define Y26_STAGE16_NO_TEST_MAIN 1
#include "../tests/test_stage16_model4_c2f_runner.cpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kFullH = 80;
constexpr int kFullW = 80;

struct Protocol {
    int warmup = 10;
    int runs = 100;
    int repeats = 5;
};

struct Options {
    std::string fixture_dir;
    std::string mode = "scalar";
    std::string output_quantize = "scalar";
    std::string merge_repair = "baseline";
    std::string dump_actual;
    Protocol protocol {};
    bool frm_sweep = false;
    int thread_branch0 = 4;
    int thread_branch1 = 0;
    int thread_model4_cv2 = 0;
};

struct MetricStats {
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;
    double cv_pct = 0.0;
};

struct Summary {
    MetricStats total {};
    Y26Stage16TimingUs mean_timing {};
    double mean_attributed_us = 0.0;
    double mean_attribution_pct = 0.0;
    double mean_other_us = 0.0;
    std::size_t mismatches = 0;
    int max_abs_diff = 0;
    unsigned long long checksum = 0;
    int status = Y26_CONV_STATUS_SUCCESS;
    int affinity_ok = 1;
};

#if defined(__riscv)
unsigned read_frm() {
    unsigned frm = 0;
    asm volatile("frrm %0" : "=r"(frm));
    return frm & 7U;
}

void set_frm(unsigned frm) {
    switch (frm) {
        case 0:
            asm volatile("fsrmi 0" ::: "memory");
            break;
        case 1:
            asm volatile("fsrmi 1" ::: "memory");
            break;
        case 2:
            asm volatile("fsrmi 2" ::: "memory");
            break;
        case 3:
            asm volatile("fsrmi 3" ::: "memory");
            break;
        case 4:
            asm volatile("fsrmi 4" ::: "memory");
            break;
        default:
            asm volatile("fsrmi 0" ::: "memory");
            break;
    }
}
#endif

Y26Stage16Model4C2fConfig fullshape_config_from_fixture(
    const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture,
    int activation_mode,
    int merge_mode) {
    Y26Stage16Model4C2fConfig cfg =
        stage16_config_from_fixture(fixture, activation_mode, merge_mode);
    cfg.stage15.stage14.model4_cv1.params.input_h = kFullH;
    cfg.stage15.stage14.model4_cv1.params.input_w = kFullW;
    cfg.stage15.branch0.params.input_h = kFullH;
    cfg.stage15.branch0.params.input_w = kFullW;
    cfg.branch1.params.input_h = kFullH;
    cfg.branch1.params.input_w = kFullW;
    cfg.model4_cv2.params.input_h = kFullH;
    cfg.model4_cv2.params.input_w = kFullW;
    return cfg;
}

std::vector<std::uint8_t> read_u8_file(const std::string& path, std::size_t expected_count) {
    std::vector<std::uint8_t> values(expected_count);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size()));
    if (in.gcount() != static_cast<std::streamsize>(values.size())) {
        return {};
    }
    char extra = 0;
    if (in.read(&extra, 1)) {
        return {};
    }
    return values;
}

bool write_u8_file(const std::string& path, const std::vector<std::uint8_t>& values) {
    if (path.empty()) {
        return true;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size()));
    return static_cast<bool>(out);
}

unsigned long long checksum_u8(const std::vector<std::uint8_t>& values) {
    unsigned long long sum = 0;
    for (std::uint8_t value : values) {
        sum += value;
    }
    return sum;
}

void compare_u8(const std::vector<std::uint8_t>& actual,
                const std::vector<std::uint8_t>& expected,
                std::size_t& mismatches,
                int& max_abs_diff) {
    mismatches = 0;
    max_abs_diff = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const int diff = std::abs(static_cast<int>(actual[i]) - static_cast<int>(expected[i]));
        if (diff != 0) {
            ++mismatches;
            max_abs_diff = std::max(max_abs_diff, diff);
        }
    }
}

void add_timing(Y26Stage16TimingUs& dst, const Y26Stage16TimingUs& src) {
    dst.input_adapter_us += src.input_adapter_us;
    dst.conv_us += src.conv_us;
    dst.activation_requant_us += src.activation_requant_us;
    dst.split_us += src.split_us;
    dst.merge_us += src.merge_us;
    dst.add_us += src.add_us;
    dst.concat_us += src.concat_us;
    dst.post_qdq_us += src.post_qdq_us;
    dst.output_quantize_us += src.output_quantize_us;
    dst.copy_layout_us += src.copy_layout_us;
    dst.pack_layout_us += src.pack_layout_us;
    dst.correction_us += src.correction_us;
    dst.thread_overhead_us += src.thread_overhead_us;
    dst.branch1_conv_us += src.branch1_conv_us;
    dst.branch1_activation_us += src.branch1_activation_us;
    dst.model4_cv2_conv_us += src.model4_cv2_conv_us;
    dst.total_us += src.total_us;
    dst.stage15_timing_us.branch0_conv_us += src.stage15_timing_us.branch0_conv_us;
    dst.stage15_timing_us.branch0_activation_us += src.stage15_timing_us.branch0_activation_us;
}

void divide_timing(Y26Stage16TimingUs& timing, double denom) {
    if (denom <= 0.0) {
        return;
    }
    timing.input_adapter_us /= denom;
    timing.conv_us /= denom;
    timing.activation_requant_us /= denom;
    timing.split_us /= denom;
    timing.merge_us /= denom;
    timing.add_us /= denom;
    timing.concat_us /= denom;
    timing.post_qdq_us /= denom;
    timing.output_quantize_us /= denom;
    timing.copy_layout_us /= denom;
    timing.pack_layout_us /= denom;
    timing.correction_us /= denom;
    timing.thread_overhead_us /= denom;
    timing.branch1_conv_us /= denom;
    timing.branch1_activation_us /= denom;
    timing.model4_cv2_conv_us /= denom;
    timing.total_us /= denom;
    timing.stage15_timing_us.branch0_conv_us /= denom;
    timing.stage15_timing_us.branch0_activation_us /= denom;
}

double attributed_nonoverlap_us(const Y26Stage16TimingUs& timing) {
    return timing.input_adapter_us + timing.conv_us + timing.activation_requant_us + timing.merge_us +
           timing.output_quantize_us + timing.copy_layout_us + timing.pack_layout_us;
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

int run_protocol(const Y26Stage16Model4C2fConfig& cfg,
                 Y26Stage16Model4C2fWorkspace& ws,
                 const Options& options,
                 const std::vector<std::uint8_t>& input,
                 const std::vector<std::uint8_t>& expected,
                 std::vector<std::uint8_t>& actual,
                 Summary& summary) {
    const bool use_ime = options.mode == "ime" || options.mode == "ime_threaded";
    const bool use_threaded = options.mode == "ime_threaded";
    const bool optimized_output_quantize = options.output_quantize == "rvv";
    if (use_ime) {
        (void)y26_k1x_ime_probe_once();
    }
    for (int i = 0; i < options.protocol.warmup; ++i) {
        Y26Stage16TimingUs timing {};
        const int status = y26_stage16_model4_c2f_run_cut_u8_output(&cfg,
                                                                    &ws,
                                                                    input.data(),
                                                                    actual.data(),
                                                                    use_ime ? 1 : 0,
                                                                    use_threaded ? 1 : 0,
                                                                    optimized_output_quantize ? 1 : 0,
                                                                    &timing);
        if (status != Y26_CONV_STATUS_SUCCESS) {
            summary.status = status;
            return status;
        }
    }

    std::vector<double> repeat_totals;
    Y26Stage16TimingUs timing_sum {};
    int status = Y26_CONV_STATUS_SUCCESS;
    for (int repeat = 0; repeat < options.protocol.repeats; ++repeat) {
        Y26Stage16TimingUs repeat_sum {};
        for (int run = 0; run < options.protocol.runs; ++run) {
            Y26Stage16TimingUs timing {};
            status = y26_stage16_model4_c2f_run_cut_u8_output(&cfg,
                                                              &ws,
                                                              input.data(),
                                                              actual.data(),
                                                              use_ime ? 1 : 0,
                                                              use_threaded ? 1 : 0,
                                                              optimized_output_quantize ? 1 : 0,
                                                              &timing);
            if (status != Y26_CONV_STATUS_SUCCESS) {
                break;
            }
            add_timing(repeat_sum, timing);
        }
        if (status != Y26_CONV_STATUS_SUCCESS) {
            break;
        }
        divide_timing(repeat_sum, static_cast<double>(options.protocol.runs));
        repeat_totals.push_back(repeat_sum.total_us);
        add_timing(timing_sum, repeat_sum);
    }
    if (!repeat_totals.empty()) {
        divide_timing(timing_sum, static_cast<double>(repeat_totals.size()));
    }
    std::size_t mismatches = 0;
    int max_abs_diff = 0;
    compare_u8(actual, expected, mismatches, max_abs_diff);
    summary.total = stats_from_values(repeat_totals);
    summary.mean_timing = timing_sum;
    summary.mismatches = mismatches;
    summary.max_abs_diff = max_abs_diff;
    summary.checksum = checksum_u8(actual);
    summary.status = status;
    summary.affinity_ok = !use_threaded ? 1 : y26_stage16_model4_c2f_threaded_worker_affinity_ok(&ws);
    summary.mean_attributed_us = attributed_nonoverlap_us(timing_sum);
    summary.mean_other_us = timing_sum.total_us - summary.mean_attributed_us;
    summary.mean_attribution_pct =
        timing_sum.total_us > 0.0 ? 100.0 * summary.mean_attributed_us / timing_sum.total_us : 0.0;
    return status == Y26_CONV_STATUS_SUCCESS && mismatches == 0 && summary.affinity_ok == 1 ? 0 : 1;
}

Options parse_options(int argc, char** argv) {
    Options options {};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--fixture-dir") {
            options.fixture_dir = require_value("--fixture-dir");
        } else if (arg == "--mode") {
            options.mode = require_value("--mode");
        } else if (arg == "--output-quantize") {
            options.output_quantize = require_value("--output-quantize");
        } else if (arg == "--merge-repair") {
            options.merge_repair = require_value("--merge-repair");
        } else if (arg == "--warmup") {
            options.protocol.warmup = std::max(0, std::atoi(require_value("--warmup").c_str()));
        } else if (arg == "--runs") {
            options.protocol.runs = std::max(1, std::atoi(require_value("--runs").c_str()));
        } else if (arg == "--repeats") {
            options.protocol.repeats = std::max(1, std::atoi(require_value("--repeats").c_str()));
        } else if (arg == "--dump-actual") {
            options.dump_actual = require_value("--dump-actual");
        } else if (arg == "--frm-sweep") {
            options.frm_sweep = true;
        } else if (arg == "--thread-branch0") {
            options.thread_branch0 = std::max(1, std::atoi(require_value("--thread-branch0").c_str()));
        } else if (arg == "--thread-branch1") {
            options.thread_branch1 = std::max(0, std::atoi(require_value("--thread-branch1").c_str()));
        } else if (arg == "--thread-model4-cv2") {
            options.thread_model4_cv2 = std::max(0, std::atoi(require_value("--thread-model4-cv2").c_str()));
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            std::exit(2);
        }
    }
    return options;
}

int run_frm_sweep(const Y26Stage16Model4C2fConfig& cfg,
                  Y26Stage16Model4C2fWorkspace& ws,
                  const Options& options,
                  const std::vector<std::uint8_t>& input,
                  const std::vector<std::uint8_t>& expected,
                  std::vector<std::uint8_t>& actual) {
#if defined(__riscv)
    const unsigned saved = read_frm();
    int failures = 0;
    for (unsigned frm : {0U, 1U, 2U, 3U, 4U}) {
        set_frm(frm);
        Options one = options;
        one.protocol = Protocol{0, 1, 1};
        Summary summary {};
        const int status = run_protocol(cfg, ws, one, input, expected, actual, summary);
        const unsigned after = read_frm();
        std::cout << "stage23_frm"
                  << " ambient_frm=" << frm
                  << " status=" << status
                  << " mismatches=" << summary.mismatches
                  << " max_abs_diff=" << summary.max_abs_diff
                  << " after_frm=" << after
                  << " checksum=" << summary.checksum
                  << "\n";
        failures += status == 0 && after == frm ? 0 : 1;
    }
    set_frm(saved);
    return failures == 0 ? 0 : 1;
#else
    (void)cfg;
    (void)ws;
    (void)options;
    (void)input;
    (void)expected;
    (void)actual;
    std::cout << "stage23_frm skipped_non_riscv\n";
    return 0;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    if (options.fixture_dir.empty()) {
        std::cerr << "usage: bench_stage23_model4_runner_cut --fixture-dir <dir>"
                  << " [--mode scalar|ime|ime_threaded] [--output-quantize scalar|rvv]"
                  << " [--merge-repair baseline|split1_lut]"
                  << " [--thread-branch0 1|2|3|4] [--thread-branch1 0|1|2|3|4]"
                  << " [--thread-model4-cv2 0|1|2|3|4]\n";
        return 2;
    }
    if (options.merge_repair != "baseline" && options.merge_repair != "split1_lut" &&
        options.merge_repair != "branch1_add_lut") {
        std::cerr << "unsupported --merge-repair " << options.merge_repair << "\n";
        return 2;
    }
    const int merge_mode = options.merge_repair == "branch1_add_lut"
                               ? Y26_STAGE16_MERGE_MODE_STAGE26_BRANCH1_ADD_LUT
                               : (options.merge_repair == "split1_lut"
                                      ? Y26_STAGE16_MERGE_MODE_STAGE24_B3_SPLIT1_LUT
                                      : Y26_STAGE16_MERGE_MODE_C2_SPLIT0_CONCAT_LUT);
    const bool use_threaded = options.mode == "ime_threaded";
    const bool use_ime = options.mode == "ime" || use_threaded;
    const int activation_mode = use_ime ? Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT : Y26_ACTIVATION_MODE_INT8_LUT;
    const auto& fixture = y26_stage16_model4_c2f_fixture::kSyntheticSeededFixture;
    Y26Stage16Model4C2fConfig cfg = fullshape_config_from_fixture(fixture, activation_mode, merge_mode);
    Y26Stage16Model4C2fWorkspace ws {};
    int status = y26_stage16_model4_c2f_prepare_cut(&cfg, &ws);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "prepare_cut failed status=" << status << "\n";
        return 1;
    }
    if (use_threaded) {
        status = y26_stage16_model4_c2f_prepare_cut_threaded_branch0(&cfg, &ws, options.thread_branch0);
        if (status != Y26_CONV_STATUS_SUCCESS) {
            std::cerr << "prepare_cut_threaded_branch0 failed status=" << status << "\n";
            y26_stage16_model4_c2f_release(&ws);
            return 1;
        }
        if (options.thread_branch1 > 0) {
            status = y26_stage16_model4_c2f_prepare_cut_threaded_branch1(&cfg, &ws, options.thread_branch1);
            if (status != Y26_CONV_STATUS_SUCCESS) {
                std::cerr << "prepare_cut_threaded_branch1 failed status=" << status << "\n";
                y26_stage16_model4_c2f_release(&ws);
                return 1;
            }
        }
        if (options.thread_model4_cv2 > 0) {
            status = y26_stage16_model4_c2f_prepare_cut_threaded_model4_cv2(&cfg, &ws, options.thread_model4_cv2);
            if (status != Y26_CONV_STATUS_SUCCESS) {
                std::cerr << "prepare_cut_threaded_model4_cv2 failed status=" << status << "\n";
                y26_stage16_model4_c2f_release(&ws);
                return 1;
            }
        }
    }

    std::vector<std::uint8_t> input =
        read_u8_file(options.fixture_dir + "/model4_cv1_conv_q_u8_nhwc.bin",
                     y26_stage16_model4_c2f_cut_input_count(&cfg));
    std::vector<std::uint8_t> expected =
        read_u8_file(options.fixture_dir + "/model4_cv2_conv_q_u8_expected_nhwc.bin",
                     y26_stage16_model4_c2f_output_count(&cfg));
    std::vector<std::uint8_t> actual(y26_stage16_model4_c2f_output_count(&cfg));
    if (input.empty() || expected.empty()) {
        std::cerr << "failed to read Stage22 cut fixture files from " << options.fixture_dir << "\n";
        y26_stage16_model4_c2f_release(&ws);
        return 1;
    }

    Summary summary {};
    status = run_protocol(cfg, ws, options, input, expected, actual, summary);
    const bool dump_ok = write_u8_file(options.dump_actual, actual);
    const Y26Stage16TimingUs& t = summary.mean_timing;
    std::cout << "stage23_runner_cut"
              << " mode=" << options.mode
              << " output_quantize=" << options.output_quantize
              << " merge_repair=" << options.merge_repair
              << " warmup=" << options.protocol.warmup
              << " runs=" << options.protocol.runs
              << " repeats=" << options.protocol.repeats
              << " thread_branch0=" << (use_threaded ? options.thread_branch0 : 0)
              << " thread_branch1=" << (use_threaded ? options.thread_branch1 : 0)
              << " thread_model4_cv2=" << (use_threaded ? options.thread_model4_cv2 : 0)
              << " status=" << summary.status
              << " mismatches=" << summary.mismatches
              << " max_abs_diff=" << summary.max_abs_diff
              << " checksum=" << summary.checksum
              << " expected_checksum=" << checksum_u8(expected)
              << " affinity_ok=" << summary.affinity_ok
              << " mean_total_us=" << summary.total.mean
              << " stddev_total_us=" << summary.total.stddev
              << " min_total_us=" << summary.total.min
              << " max_total_us=" << summary.total.max
              << " cv_total_pct=" << summary.total.cv_pct
              << " mean_input_adapter_us=" << t.input_adapter_us
              << " mean_conv_us=" << t.conv_us
              << " mean_activation_requant_us=" << t.activation_requant_us
              << " mean_merge_us=" << t.merge_us
              << " mean_post_concat_qdq_us=" << t.post_qdq_us
              << " mean_output_quantize_us=" << t.output_quantize_us
              << " mean_copy_layout_us=" << t.copy_layout_us
              << " mean_pack_layout_us=" << t.pack_layout_us
              << " mean_thread_overhead_us=" << t.thread_overhead_us
              << " mean_correction_us=" << t.correction_us
              << " mean_branch0_conv_us=" << t.stage15_timing_us.branch0_conv_us
              << " mean_branch0_activation_us=" << t.stage15_timing_us.branch0_activation_us
              << " mean_branch1_conv_us=" << t.branch1_conv_us
              << " mean_branch1_activation_us=" << t.branch1_activation_us
              << " mean_model4_cv2_conv_us=" << t.model4_cv2_conv_us
              << " mean_attributed_us=" << summary.mean_attributed_us
              << " mean_attribution_pct=" << summary.mean_attribution_pct
              << " mean_other_us=" << summary.mean_other_us
              << " conv_share_pct=" << (t.total_us > 0.0 ? 100.0 * t.conv_us / t.total_us : 0.0)
              << " activation_share_pct=" << (t.total_us > 0.0 ? 100.0 * t.activation_requant_us / t.total_us : 0.0)
              << " merge_share_pct=" << (t.total_us > 0.0 ? 100.0 * t.merge_us / t.total_us : 0.0)
              << " output_quantize_share_pct="
              << (t.total_us > 0.0 ? 100.0 * t.output_quantize_us / t.total_us : 0.0)
              << " dump_actual_ok=" << (dump_ok ? 1 : 0)
              << " note=selected-model4-cut-not-model-fps"
              << "\n";
    int failures = status == 0 && dump_ok ? 0 : 1;
    if (options.frm_sweep) {
        failures += run_frm_sweep(cfg, ws, options, input, expected, actual);
    }
    y26_stage16_model4_c2f_release(&ws);
    return failures == 0 ? 0 : 1;
}
