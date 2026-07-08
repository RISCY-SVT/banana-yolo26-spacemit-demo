#define main y26_stage16_fullshape_gate_embedded_main
#include "bench_stage16_fullshape_gate.cpp"
#undef main

#include "y26_k1x_threaded_conv.h"
#include "y26_k1x_vmadot123_direct_conv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct Protocol {
    int warmup = 10;
    int runs = 100;
    int repeats = 5;
};

struct Summary {
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;
    double cv_pct = 0.0;
};

Summary summarize(const std::vector<double>& values) {
    Summary s {};
    if (values.empty()) {
        return s;
    }
    s.min = *std::min_element(values.begin(), values.end());
    s.max = *std::max_element(values.begin(), values.end());
    for (double v : values) {
        s.mean += v;
    }
    s.mean /= static_cast<double>(values.size());
    for (double v : values) {
        const double d = v - s.mean;
        s.stddev += d * d;
    }
    s.stddev = std::sqrt(s.stddev / static_cast<double>(values.size()));
    s.cv_pct = s.mean != 0.0 ? 100.0 * s.stddev / s.mean : 0.0;
    return s;
}

std::size_t mismatch_count_i32(const std::vector<std::int32_t>& actual,
                               const std::vector<std::int32_t>& expected,
                               int& max_abs_diff) {
    std::size_t mismatches = 0;
    max_abs_diff = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const int diff = std::abs(actual[i] - expected[i]);
        if (diff != 0) {
            ++mismatches;
            max_abs_diff = std::max(max_abs_diff, diff);
        }
    }
    return mismatches;
}

long long checksum_i32_vec(const std::vector<std::int32_t>& values) {
    long long checksum = 0;
    for (std::int32_t v : values) {
        checksum += v;
    }
    return checksum;
}

struct DirectRunResult {
    int status = Y26_CONV_STATUS_SUCCESS;
    Y26Vmadot123DirectConvTimingUs timing {};
};

DirectRunResult run_direct_once(const Y26Stage7ConvNodeConfig& cfg,
                                const std::vector<std::int8_t>& input,
                                Y26PrepackedConvWeights* weights,
                                Y26Vmadot123DirectConvWorkspace* workspace,
                                std::vector<std::int32_t>& output) {
    DirectRunResult result {};
    result.status = y26_vmadot123_direct_conv3x3_i8s8s32_nhwc_single_thread(input.data(),
                                                                             weights,
                                                                             cfg.bias_i32,
                                                                             output.data(),
                                                                             &cfg.params,
                                                                             cfg.input_storage_zero_point_s8,
                                                                             cfg.activation_zero_point_u8,
                                                                             workspace,
                                                                             &result.timing);
    return result;
}

int run_threaded_once(Y26ThreadedConvWorkspace* workspace,
                      const std::vector<std::int8_t>& input,
                      std::vector<std::int32_t>& output,
                      Y26ThreadedConvTimingUs& timing) {
    return y26_threaded_conv_run_ime_cluster0(workspace, input.data(), output.data(), &timing);
}

std::vector<std::int8_t> make_primary_input(const y26_stage15_model4_branch_fixture::Model4BranchFixture& fixture,
                                            std::vector<std::int32_t>& expected_branch0,
                                            std::vector<std::int8_t>& expected_branch0_act) {
    constexpr int model4_count = kFullH * kFullW * kModel4Cv1C;
    constexpr int split_count = kFullH * kFullW * (kModel4Cv1C / 2);
    constexpr int branch_count = kFullH * kFullW * 16;
    std::vector<std::int32_t> model4_cv1_i32(model4_count, 0);
    std::vector<std::int8_t> split1(split_count, 0);
    expected_branch0.assign(branch_count, 0);
    expected_branch0_act.assign(branch_count, 0);
    fill_model4_cv1_i32(fixture, model4_cv1_i32);
    GateTiming timing {};
    const int status = run_once(fixture,
                                Y26_ACTIVATION_MODE_INT8_LUT,
                                false,
                                model4_cv1_i32,
                                split1,
                                expected_branch0,
                                expected_branch0_act,
                                timing);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return {};
    }
    return split1;
}

}  // namespace

int main(int argc, char** argv) {
    Protocol protocol {};
    bool correctness_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--correctness-only") == 0) {
            correctness_only = true;
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            protocol.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            protocol.runs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            protocol.repeats = std::atoi(argv[++i]);
        } else {
            std::cerr << "unsupported argument: " << argv[i] << "\n";
            return 2;
        }
    }

    const auto& fixture = y26_stage15_model4_branch_fixture::kSyntheticSeededFixture;
    Y26Stage7ConvNodeConfig cfg = fullshape_branch0_config(fixture);
    std::vector<std::int32_t> expected_branch0;
    std::vector<std::int8_t> expected_branch0_act;
    std::vector<std::int8_t> input = make_primary_input(fixture, expected_branch0, expected_branch0_act);
    if (input.empty()) {
        std::cerr << "failed to create primary real-node input\n";
        return 1;
    }
    std::vector<std::int32_t> direct_output(expected_branch0.size(), 0);
    std::vector<std::int32_t> mmt4d_1t_output(expected_branch0.size(), 0);
    std::vector<std::int32_t> mmt4d_4t_output(expected_branch0.size(), 0);

    Y26PrepackedConvWeights* weights = y26_prepacked_conv_weights_create_mmt4d_s8(cfg.weights_ohwi_s8,
                                                                                   &cfg.params,
                                                                                   cfg.kernel_h,
                                                                                   cfg.kernel_w,
                                                                                   cfg.node_name,
                                                                                   cfg.weight_scales);
    Y26Vmadot123DirectConvWorkspace* direct_ws =
        y26_vmadot123_direct_conv3x3_workspace_create(&cfg.params);
    Y26ThreadedConvWorkspace* threaded_1t = y26_threaded_conv_create_spatial_rows(&cfg, 1);
    Y26ThreadedConvWorkspace* threaded_4t = y26_threaded_conv_create_spatial_rows(&cfg, 4);
    if (weights == nullptr || direct_ws == nullptr || threaded_1t == nullptr || threaded_4t == nullptr) {
        std::cerr << "workspace prepare failed\n";
        y26_vmadot123_direct_conv3x3_workspace_destroy(direct_ws);
        y26_threaded_conv_destroy(threaded_1t);
        y26_threaded_conv_destroy(threaded_4t);
        y26_prepacked_conv_weights_destroy(weights);
        return 1;
    }

    DirectRunResult direct_check = run_direct_once(cfg, input, weights, direct_ws, direct_output);
    Y26ThreadedConvTimingUs mmt4d_1t_timing {};
    Y26ThreadedConvTimingUs mmt4d_4t_timing {};
    const int mmt4d_1t_status = run_threaded_once(threaded_1t, input, mmt4d_1t_output, mmt4d_1t_timing);
    const int mmt4d_4t_status = run_threaded_once(threaded_4t, input, mmt4d_4t_output, mmt4d_4t_timing);

    int max_abs_direct = 0;
    int max_abs_mmt4d_1t = 0;
    int max_abs_mmt4d_4t = 0;
    const std::size_t direct_mismatches =
        direct_check.status == Y26_CONV_STATUS_SUCCESS
            ? mismatch_count_i32(direct_output, expected_branch0, max_abs_direct)
            : expected_branch0.size();
    const std::size_t mmt4d_1t_mismatches =
        mmt4d_1t_status == Y26_CONV_STATUS_SUCCESS
            ? mismatch_count_i32(mmt4d_1t_output, expected_branch0, max_abs_mmt4d_1t)
            : expected_branch0.size();
    const std::size_t mmt4d_4t_mismatches =
        mmt4d_4t_status == Y26_CONV_STATUS_SUCCESS
            ? mismatch_count_i32(mmt4d_4t_output, expected_branch0, max_abs_mmt4d_4t)
            : expected_branch0.size();

    std::cout << "stage31_correctness"
              << " node=/model.4/m.0/cv1/conv/Conv"
              << " direct_status=" << direct_check.status
              << " direct_mismatches=" << direct_mismatches
              << " direct_max_abs_diff=" << max_abs_direct
              << " mmt4d_1t_status=" << mmt4d_1t_status
              << " mmt4d_1t_mismatches=" << mmt4d_1t_mismatches
              << " mmt4d_4t_status=" << mmt4d_4t_status
              << " mmt4d_4t_mismatches=" << mmt4d_4t_mismatches
              << " checksum_direct=" << checksum_i32_vec(direct_output)
              << " checksum_expected=" << checksum_i32_vec(expected_branch0)
              << " workspace_bytes=" << y26_vmadot123_direct_conv3x3_workspace_bytes(direct_ws)
              << " affinity_1t=" << y26_threaded_conv_worker_affinity_ok(threaded_1t)
              << " affinity_4t=" << y26_threaded_conv_worker_affinity_ok(threaded_4t)
              << "\n";

    if (direct_check.status != Y26_CONV_STATUS_SUCCESS || direct_mismatches != 0 ||
        mmt4d_1t_mismatches != 0 || mmt4d_4t_mismatches != 0 || correctness_only) {
        y26_vmadot123_direct_conv3x3_workspace_destroy(direct_ws);
        y26_threaded_conv_destroy(threaded_1t);
        y26_threaded_conv_destroy(threaded_4t);
        y26_prepacked_conv_weights_destroy(weights);
        return direct_check.status == Y26_CONV_STATUS_SUCCESS && direct_mismatches == 0 &&
                       mmt4d_1t_mismatches == 0 && mmt4d_4t_mismatches == 0
                   ? 0
                   : 1;
    }

    std::vector<double> direct_totals;
    std::vector<double> direct_panel;
    std::vector<double> direct_compute;
    std::vector<double> direct_correction;
    std::vector<double> direct_writeback;
    std::vector<double> mmt4d_1t_totals;
    std::vector<double> mmt4d_4t_totals;
    for (int repeat = 0; repeat < protocol.repeats; ++repeat) {
        for (int i = 0; i < protocol.warmup; ++i) {
            (void)run_direct_once(cfg, input, weights, direct_ws, direct_output);
            (void)run_threaded_once(threaded_1t, input, mmt4d_1t_output, mmt4d_1t_timing);
            (void)run_threaded_once(threaded_4t, input, mmt4d_4t_output, mmt4d_4t_timing);
        }
        double direct_acc = 0.0;
        double panel_acc = 0.0;
        double compute_acc = 0.0;
        double correction_acc = 0.0;
        double writeback_acc = 0.0;
        double mmt4d_1t_acc = 0.0;
        double mmt4d_4t_acc = 0.0;
        for (int run = 0; run < protocol.runs; ++run) {
            DirectRunResult direct = run_direct_once(cfg, input, weights, direct_ws, direct_output);
            if (direct.status != Y26_CONV_STATUS_SUCCESS) {
                std::cerr << "direct timing run failed status=" << direct.status << "\n";
                return 1;
            }
            direct_acc += direct.timing.total_us;
            panel_acc += direct.timing.panel_build_us;
            compute_acc += direct.timing.kernel_compute_us;
            correction_acc += direct.timing.correction_us;
            writeback_acc += direct.timing.writeback_us;
            if (run_threaded_once(threaded_1t, input, mmt4d_1t_output, mmt4d_1t_timing) !=
                Y26_CONV_STATUS_SUCCESS) {
                std::cerr << "mmt4d 1t timing run failed\n";
                return 1;
            }
            if (run_threaded_once(threaded_4t, input, mmt4d_4t_output, mmt4d_4t_timing) !=
                Y26_CONV_STATUS_SUCCESS) {
                std::cerr << "mmt4d 4t timing run failed\n";
                return 1;
            }
            mmt4d_1t_acc += mmt4d_1t_timing.total_us;
            mmt4d_4t_acc += mmt4d_4t_timing.total_us;
        }
        direct_totals.push_back(direct_acc / protocol.runs);
        direct_panel.push_back(panel_acc / protocol.runs);
        direct_compute.push_back(compute_acc / protocol.runs);
        direct_correction.push_back(correction_acc / protocol.runs);
        direct_writeback.push_back(writeback_acc / protocol.runs);
        mmt4d_1t_totals.push_back(mmt4d_1t_acc / protocol.runs);
        mmt4d_4t_totals.push_back(mmt4d_4t_acc / protocol.runs);
    }

    const Summary direct_summary = summarize(direct_totals);
    const Summary panel_summary = summarize(direct_panel);
    const Summary compute_summary = summarize(direct_compute);
    const Summary correction_summary = summarize(direct_correction);
    const Summary writeback_summary = summarize(direct_writeback);
    const Summary mmt4d_1t_summary = summarize(mmt4d_1t_totals);
    const Summary mmt4d_4t_summary = summarize(mmt4d_4t_totals);
    const double speedup_vs_1t = direct_summary.mean > 0.0 ? mmt4d_1t_summary.mean / direct_summary.mean : 0.0;
    const double speedup_vs_4t = direct_summary.mean > 0.0 ? mmt4d_4t_summary.mean / direct_summary.mean : 0.0;

    std::cout << "stage31_benchmark"
              << " node=/model.4/m.0/cv1/conv/Conv"
              << " warmup=" << protocol.warmup
              << " runs=" << protocol.runs
              << " repeats=" << protocol.repeats
              << " direct_mean_us=" << direct_summary.mean
              << " direct_stddev_us=" << direct_summary.stddev
              << " direct_cv_pct=" << direct_summary.cv_pct
              << " panel_build_mean_us=" << panel_summary.mean
              << " kernel_compute_mean_us=" << compute_summary.mean
              << " correction_mean_us=" << correction_summary.mean
              << " writeback_mean_us=" << writeback_summary.mean
              << " mmt4d_1t_mean_us=" << mmt4d_1t_summary.mean
              << " mmt4d_1t_stddev_us=" << mmt4d_1t_summary.stddev
              << " mmt4d_4t_mean_us=" << mmt4d_4t_summary.mean
              << " mmt4d_4t_stddev_us=" << mmt4d_4t_summary.stddev
              << " speedup_vs_1t=" << speedup_vs_1t
              << " speedup_vs_4t=" << speedup_vs_4t
              << " direct_min_us=" << direct_summary.min
              << " direct_max_us=" << direct_summary.max
              << "\n";

    y26_vmadot123_direct_conv3x3_workspace_destroy(direct_ws);
    y26_threaded_conv_destroy(threaded_1t);
    y26_threaded_conv_destroy(threaded_4t);
    y26_prepacked_conv_weights_destroy(weights);
    return 0;
}
