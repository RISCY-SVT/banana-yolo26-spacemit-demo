#define main y26_stage16_fullshape_gate_embedded_main
#include "bench_stage16_fullshape_gate.cpp"
#undef main

#include "y26_k1x_threaded_conv.h"
#include "y26_k1x_vmadot.h"
#include "y26_k1x_vmadot123_direct_conv.h"
#include "y26_k1x_vmadot123_probe.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <csetjmp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kKernelH = 3;
constexpr int kKernelW = 3;
constexpr int kOutputMStep = 7;
constexpr double kPanelGateUs = 7800.0;

struct Protocol {
    int warmup = 10;
    int runs = 100;
    int repeats = 5;
};

struct Stats {
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;
    double cv_pct = 0.0;
};

struct LayoutResult {
    std::string candidate;
    int attachable = 0;
    Stats stats {};
    long long checksum = 0;
};

double elapsed_stage32_us(Clock::time_point begin, Clock::time_point end) {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000.0;
}

Stats summarize(const std::vector<double>& values) {
    Stats s {};
    if (values.empty()) {
        return s;
    }
    s.min = *std::min_element(values.begin(), values.end());
    s.max = *std::max_element(values.begin(), values.end());
    s.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    double sq = 0.0;
    for (double v : values) {
        const double d = v - s.mean;
        sq += d * d;
    }
    s.stddev = std::sqrt(sq / static_cast<double>(values.size()));
    s.cv_pct = s.mean != 0.0 ? 100.0 * s.stddev / s.mean : 0.0;
    return s;
}

int align_up_local(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

std::int8_t input_or_pad_local(const std::int8_t* input,
                               const Y26Conv2DParams& params,
                               int ih,
                               int iw,
                               int ic,
                               std::int8_t pad_value) {
    if (ih < 0 || ih >= params.input_h || iw < 0 || iw >= params.input_w) {
        return pad_value;
    }
    return input[(static_cast<std::size_t>(ih) * params.input_w + static_cast<std::size_t>(iw)) *
                     static_cast<std::size_t>(params.input_c) +
                 static_cast<std::size_t>(ic)];
}

void pack_stage31_panel(const std::int8_t* input,
                        const Y26Conv2DParams& params,
                        int output_w,
                        int output_m,
                        int m0,
                        int k_padded,
                        std::int8_t pad_value,
                        std::int8_t* panel) {
    std::fill(panel, panel + static_cast<std::size_t>(8 * k_padded), std::int8_t{0});
    for (int m = 0; m < 8; ++m) {
        const int flat_m = m0 + m;
        if (flat_m >= output_m) {
            continue;
        }
        const int oh = flat_m / output_w;
        const int ow = flat_m - oh * output_w;
        int flat_k = 0;
        for (int kh = 0; kh < kKernelH; ++kh) {
            const int ih = oh + kh - params.pad_h;
            for (int kw = 0; kw < kKernelW; ++kw) {
                const int iw = ow + kw - params.pad_w;
                for (int ic = 0; ic < params.input_c; ++ic, ++flat_k) {
                    const int k_tile = flat_k / 8;
                    const int k_lane = flat_k - k_tile * 8;
                    panel[static_cast<std::size_t>(k_tile) * 64U + static_cast<std::size_t>(m) * 8U +
                          static_cast<std::size_t>(k_lane)] =
                        input_or_pad_local(input, params, ih, iw, ic, pad_value);
                }
            }
        }
    }
}

void pack_interior_fast_panel(const std::int8_t* input,
                              const Y26Conv2DParams& params,
                              int output_w,
                              int output_m,
                              int m0,
                              int k_padded,
                              std::int8_t pad_value,
                              std::int8_t* panel) {
    std::fill(panel, panel + static_cast<std::size_t>(8 * k_padded), std::int8_t{0});
    for (int m = 0; m < 8; ++m) {
        const int flat_m = m0 + m;
        if (flat_m >= output_m) {
            continue;
        }
        const int oh = flat_m / output_w;
        const int ow = flat_m - oh * output_w;
        int flat_k = 0;
        const bool interior = oh > 0 && ow > 0 && oh + 1 < params.input_h && ow + 1 < params.input_w;
        if (interior) {
            for (int kh = 0; kh < kKernelH; ++kh) {
                const int ih = oh + kh - params.pad_h;
                for (int kw = 0; kw < kKernelW; ++kw) {
                    const int iw = ow + kw - params.pad_w;
                    const std::int8_t* src =
                        input + (static_cast<std::size_t>(ih) * params.input_w + static_cast<std::size_t>(iw)) *
                                    static_cast<std::size_t>(params.input_c);
                    for (int ic = 0; ic < params.input_c; ++ic, ++flat_k) {
                        const int k_tile = flat_k / 8;
                        const int k_lane = flat_k - k_tile * 8;
                        panel[static_cast<std::size_t>(k_tile) * 64U + static_cast<std::size_t>(m) * 8U +
                              static_cast<std::size_t>(k_lane)] = src[ic];
                    }
                }
            }
        } else {
            int edge_k = 0;
            for (int kh = 0; kh < kKernelH; ++kh) {
                const int ih = oh + kh - params.pad_h;
                for (int kw = 0; kw < kKernelW; ++kw) {
                    const int iw = ow + kw - params.pad_w;
                    for (int ic = 0; ic < params.input_c; ++ic, ++edge_k) {
                        const int k_tile = edge_k / 8;
                        const int k_lane = edge_k - k_tile * 8;
                        panel[static_cast<std::size_t>(k_tile) * 64U + static_cast<std::size_t>(m) * 8U +
                              static_cast<std::size_t>(k_lane)] =
                            input_or_pad_local(input, params, ih, iw, ic, pad_value);
                    }
                }
            }
        }
    }
}

void build_row_cache(const std::int8_t* input,
                     const Y26Conv2DParams& params,
                     int output_w,
                     int output_m,
                     int k_padded,
                     std::int8_t pad_value,
                     std::int8_t* row_cache) {
    std::fill(row_cache, row_cache + static_cast<std::size_t>(output_m) * k_padded, std::int8_t{0});
    for (int flat_m = 0; flat_m < output_m; ++flat_m) {
        const int oh = flat_m / output_w;
        const int ow = flat_m - oh * output_w;
        std::int8_t* dst = row_cache + static_cast<std::size_t>(flat_m) * static_cast<std::size_t>(k_padded);
        int flat_k = 0;
        for (int kh = 0; kh < kKernelH; ++kh) {
            const int ih = oh + kh - params.pad_h;
            for (int kw = 0; kw < kKernelW; ++kw) {
                const int iw = ow + kw - params.pad_w;
                for (int ic = 0; ic < params.input_c; ++ic, ++flat_k) {
                    dst[flat_k] = input_or_pad_local(input, params, ih, iw, ic, pad_value);
                }
            }
        }
    }
}

void assemble_panel_from_row_cache(const std::int8_t* row_cache,
                                   int output_m,
                                   int m0,
                                   int k_padded,
                                   std::int8_t* panel) {
    std::fill(panel, panel + static_cast<std::size_t>(8 * k_padded), std::int8_t{0});
    for (int m = 0; m < 8; ++m) {
        const int flat_m = m0 + m;
        if (flat_m >= output_m) {
            continue;
        }
        const std::int8_t* src = row_cache + static_cast<std::size_t>(flat_m) * static_cast<std::size_t>(k_padded);
        for (int flat_k = 0; flat_k < k_padded; ++flat_k) {
            const int k_tile = flat_k / 8;
            const int k_lane = flat_k - k_tile * 8;
            panel[static_cast<std::size_t>(k_tile) * 64U + static_cast<std::size_t>(m) * 8U +
                  static_cast<std::size_t>(k_lane)] = src[flat_k];
        }
    }
}

long long checksum_panel(const std::int8_t* panel, int bytes) {
    long long checksum = 0;
    for (int i = 0; i < bytes; ++i) {
        checksum += static_cast<int>(panel[i]);
    }
    return checksum;
}

long long checksum_descriptors(const Y26Conv2DParams& params, int output_w, int output_m) {
    long long checksum = 0;
    for (int m0 = 0; m0 < output_m; m0 += kOutputMStep) {
        for (int m = 0; m < 8; ++m) {
            const int flat_m = m0 + m;
            if (flat_m >= output_m) {
                continue;
            }
            const int oh = flat_m / output_w;
            const int ow = flat_m - oh * output_w;
            for (int kh = 0; kh < kKernelH; ++kh) {
                for (int kw = 0; kw < kKernelW; ++kw) {
                    const int ih = oh + kh - params.pad_h;
                    const int iw = ow + kw - params.pad_w;
                    checksum += static_cast<long long>((ih + 11) * 131 + (iw + 17) * 17 + flat_m);
                }
            }
        }
    }
    return checksum;
}

LayoutResult measure_layout_candidate(const std::string& name,
                                      int attachable,
                                      const Protocol& protocol,
                                      const Y26Conv2DParams& params,
                                      const std::vector<std::int8_t>& input,
                                      int output_w,
                                      int output_m,
                                      int k_padded,
                                      std::int8_t pad_value) {
    std::vector<double> repeats;
    long long checksum = 0;
    std::vector<std::int8_t> panel(static_cast<std::size_t>(8 * k_padded), 0);
    std::vector<std::int8_t> row_cache;
    if (name == "B1_row_cache_materialized" || name == "B4_row_cache_descriptor_model") {
        row_cache.resize(static_cast<std::size_t>(output_m) * static_cast<std::size_t>(k_padded));
    }
    auto run_once = [&]() {
        const auto begin = Clock::now();
        long long local_checksum = 0;
        if (name == "B0_stage31_full_panel") {
            for (int m0 = 0; m0 < output_m; m0 += kOutputMStep) {
                pack_stage31_panel(input.data(), params, output_w, output_m, m0, k_padded, pad_value, panel.data());
                local_checksum += checksum_panel(panel.data(), static_cast<int>(panel.size()));
            }
        } else if (name == "B1_row_cache_materialized") {
            build_row_cache(input.data(), params, output_w, output_m, k_padded, pad_value, row_cache.data());
            for (int m0 = 0; m0 < output_m; m0 += kOutputMStep) {
                assemble_panel_from_row_cache(row_cache.data(), output_m, m0, k_padded, panel.data());
                local_checksum += checksum_panel(panel.data(), static_cast<int>(panel.size()));
            }
        } else if (name == "B2_descriptor_only") {
            local_checksum += checksum_descriptors(params, output_w, output_m);
        } else if (name == "B3_interior_fast_path") {
            for (int m0 = 0; m0 < output_m; m0 += kOutputMStep) {
                pack_interior_fast_panel(input.data(), params, output_w, output_m, m0, k_padded, pad_value, panel.data());
                local_checksum += checksum_panel(panel.data(), static_cast<int>(panel.size()));
            }
        } else if (name == "B4_row_cache_descriptor_model") {
            build_row_cache(input.data(), params, output_w, output_m, k_padded, pad_value, row_cache.data());
            local_checksum += checksum_panel(row_cache.data(), static_cast<int>(row_cache.size()));
            local_checksum += checksum_descriptors(params, output_w, output_m);
        }
        const auto end = Clock::now();
        checksum = local_checksum;
        return elapsed_stage32_us(begin, end);
    };

    for (int i = 0; i < protocol.warmup; ++i) {
        (void)run_once();
    }
    for (int repeat = 0; repeat < protocol.repeats; ++repeat) {
        double acc = 0.0;
        for (int run = 0; run < protocol.runs; ++run) {
            acc += run_once();
        }
        repeats.push_back(acc / static_cast<double>(protocol.runs));
    }
    LayoutResult result {};
    result.candidate = name;
    result.attachable = attachable;
    result.stats = summarize(repeats);
    result.checksum = checksum;
    return result;
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
    for (std::int32_t value : values) {
        checksum += value;
    }
    return checksum;
}

int conv_status_from_vmadot_status_local(int status) {
    switch (status) {
    case Y26_VMADOT_STATUS_SUCCESS:
        return Y26_CONV_STATUS_SUCCESS;
    case Y26_VMADOT_STATUS_NOT_BUILT_WITH_IME:
        return Y26_CONV_STATUS_NOT_BUILT_WITH_IME;
    case Y26_VMADOT_STATUS_RUNTIME_SAFETY_FAILED:
        return Y26_CONV_STATUS_RUNTIME_SAFETY_FAILED;
    case Y26_VMADOT_STATUS_SIGILL_CAUGHT:
        return Y26_CONV_STATUS_SIGILL_CAUGHT;
    default:
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
}

void store_c_rows_local(const std::array<std::int32_t, 16>& c_tile,
                        int c_row,
                        int flat_m,
                        int n0,
                        int output_m,
                        int output_c,
                        std::int32_t* raw_output) {
    if (flat_m >= output_m) {
        return;
    }
    for (int n = 0; n < 4; ++n) {
        const int oc = n0 + n;
        if (oc < output_c) {
            raw_output[static_cast<std::size_t>(flat_m) * output_c + static_cast<std::size_t>(oc)] =
                c_tile[static_cast<std::size_t>(c_row) * 4U + static_cast<std::size_t>(n)];
        }
    }
}

int run_one_c_group_local(const std::int8_t* a_panel,
                          const std::int8_t* packed_b,
                          int n_tile,
                          int k_tiles,
                          std::array<std::int32_t, 16>& c0,
                          std::array<std::int32_t, 16>& c1,
                          std::array<std::int32_t, 16>& c2,
                          std::array<std::int32_t, 16>& c3) {
    c0.fill(0);
    c1.fill(0);
    c2.fill(0);
    c3.fill(0);
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const std::int8_t* a_tile = a_panel + static_cast<std::size_t>(k_tile) * 64U;
        const std::int8_t* b_tile = packed_b + (static_cast<std::size_t>(n_tile) *
                                                    static_cast<std::size_t>(k_tiles) +
                                                static_cast<std::size_t>(k_tile)) *
                                                   32U;
        const bool accumulate = k_tile != 0;
        int status = y26_k1x_vmadot_4x4x8_unsafe_cluster0_s8s8s32(a_tile, b_tile, c0.data(), accumulate);
        if (status != Y26_VMADOT_STATUS_SUCCESS) {
            return conv_status_from_vmadot_status_local(status);
        }
        status = y26_k1x_vmadot123_unsafe_cluster0_s8s8s32(
            Y26_VMADOT123_VARIANT_1, a_tile, b_tile, c1.data(), accumulate);
        if (status != Y26_VMADOT_STATUS_SUCCESS) {
            return conv_status_from_vmadot_status_local(status);
        }
        status = y26_k1x_vmadot123_unsafe_cluster0_s8s8s32(
            Y26_VMADOT123_VARIANT_2, a_tile, b_tile, c2.data(), accumulate);
        if (status != Y26_VMADOT_STATUS_SUCCESS) {
            return conv_status_from_vmadot_status_local(status);
        }
        status = y26_k1x_vmadot123_unsafe_cluster0_s8s8s32(
            Y26_VMADOT123_VARIANT_3, a_tile, b_tile, c3.data(), accumulate);
        if (status != Y26_VMADOT_STATUS_SUCCESS) {
            return conv_status_from_vmadot_status_local(status);
        }
    }
    return Y26_CONV_STATUS_SUCCESS;
}

struct DirectCachedResult {
    int status = Y26_CONV_STATUS_SUCCESS;
    double total_us = 0.0;
    double panel_build_us = 0.0;
    double kernel_compute_us = 0.0;
    double correction_us = 0.0;
    double writeback_us = 0.0;
};

DirectCachedResult run_direct_row_cache_once(const Y26Stage7ConvNodeConfig& cfg,
                                             const std::vector<std::int8_t>& input,
                                             Y26PrepackedConvWeights* weights,
                                             std::vector<std::int8_t>& row_cache,
                                             std::vector<std::int8_t>& panel,
                                             std::vector<std::int32_t>& raw_output,
                                             std::vector<std::int32_t>& corrected_output) {
    DirectCachedResult result {};
    const int output_h = y26_conv3x3_output_h(&cfg.params);
    const int output_w = y26_conv3x3_output_w(&cfg.params);
    const int output_m = output_h * output_w;
    const int kernel_k = kKernelH * kKernelW * cfg.params.input_c;
    const int k_padded = align_up_local(kernel_k, 8);
    const int k_tiles = k_padded / 8;
    const int n_tiles = align_up_local(cfg.params.output_c, 4) / 4;
    const auto begin = Clock::now();
    const auto panel_begin = Clock::now();
    build_row_cache(input.data(),
                    cfg.params,
                    output_w,
                    output_m,
                    k_padded,
                    static_cast<std::int8_t>(cfg.input_storage_zero_point_s8),
                    row_cache.data());
    const auto panel_end = Clock::now();
    result.panel_build_us += elapsed_stage32_us(panel_begin, panel_end);

    std::fill(raw_output.begin(), raw_output.end(), 0);
    std::array<std::int32_t, 16> c0 {};
    std::array<std::int32_t, 16> c1 {};
    std::array<std::int32_t, 16> c2 {};
    std::array<std::int32_t, 16> c3 {};
    const std::int8_t* packed_b = y26_prepacked_conv_weights_packed_b(weights);
    const std::int32_t* weight_sums = y26_prepacked_conv_weights_sums(weights);
    if (packed_b == nullptr || weight_sums == nullptr) {
        result.status = Y26_CONV_STATUS_INVALID_ARGUMENT;
        return result;
    }
    for (int m0 = 0; m0 < output_m; m0 += kOutputMStep) {
        const auto assemble_begin = Clock::now();
        assemble_panel_from_row_cache(row_cache.data(), output_m, m0, k_padded, panel.data());
        const auto assemble_end = Clock::now();
        result.panel_build_us += elapsed_stage32_us(assemble_begin, assemble_end);
        for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
            const int n0 = n_tile * 4;
            const auto compute_begin = Clock::now();
            const int status =
                run_one_c_group_local(panel.data(), packed_b, n_tile, k_tiles, c0, c1, c2, c3);
            const auto compute_end = Clock::now();
            result.kernel_compute_us += elapsed_stage32_us(compute_begin, compute_end);
            if (status != Y26_CONV_STATUS_SUCCESS) {
                result.status = status;
                return result;
            }
            const auto write_begin = Clock::now();
            store_c_rows_local(c0, 0, m0 + 0, n0, output_m, cfg.params.output_c, raw_output.data());
            store_c_rows_local(c0, 1, m0 + 1, n0, output_m, cfg.params.output_c, raw_output.data());
            store_c_rows_local(c0, 2, m0 + 2, n0, output_m, cfg.params.output_c, raw_output.data());
            store_c_rows_local(c0, 3, m0 + 3, n0, output_m, cfg.params.output_c, raw_output.data());
            store_c_rows_local(c1, 3, m0 + 4, n0, output_m, cfg.params.output_c, raw_output.data());
            store_c_rows_local(c2, 3, m0 + 5, n0, output_m, cfg.params.output_c, raw_output.data());
            store_c_rows_local(c3, 3, m0 + 6, n0, output_m, cfg.params.output_c, raw_output.data());
            const auto write_end = Clock::now();
            result.writeback_us += elapsed_stage32_us(write_begin, write_end);
        }
    }
    const auto correction_begin = Clock::now();
    result.status = y26_conv2d_apply_u8_as_s8_correction_nhwc(raw_output.data(),
                                                              cfg.bias_i32,
                                                              weight_sums,
                                                              corrected_output.data(),
                                                              output_m,
                                                              cfg.params.output_c,
                                                              cfg.activation_zero_point_u8);
    const auto correction_end = Clock::now();
    const auto end = Clock::now();
    result.correction_us = elapsed_stage32_us(correction_begin, correction_end);
    result.total_us = elapsed_stage32_us(begin, end);
    return result;
}

void run_layout_gate(const Protocol& protocol) {
    const auto& fixture = y26_stage15_model4_branch_fixture::kSyntheticSeededFixture;
    Y26Stage7ConvNodeConfig cfg = fullshape_branch0_config(fixture);
    std::vector<std::int32_t> expected_branch0;
    std::vector<std::int8_t> expected_branch0_act;
    std::vector<std::int8_t> input = make_primary_input(fixture, expected_branch0, expected_branch0_act);
    if (input.empty()) {
        std::cout << "stage32_layout status=input_prepare_failed\n";
        return;
    }
    const int output_h = y26_conv3x3_output_h(&cfg.params);
    const int output_w = y26_conv3x3_output_w(&cfg.params);
    const int output_m = output_h * output_w;
    const int k_padded = align_up_local(kKernelH * kKernelW * cfg.params.input_c, 8);
    const std::int8_t pad_value = static_cast<std::int8_t>(cfg.input_storage_zero_point_s8);
    const LayoutResult b0 = measure_layout_candidate(
        "B0_stage31_full_panel", 1, protocol, cfg.params, input, output_w, output_m, k_padded, pad_value);
    const LayoutResult b1 = measure_layout_candidate(
        "B1_row_cache_materialized", 1, protocol, cfg.params, input, output_w, output_m, k_padded, pad_value);
    const LayoutResult b2 = measure_layout_candidate(
        "B2_descriptor_only", 0, protocol, cfg.params, input, output_w, output_m, k_padded, pad_value);
    const LayoutResult b3 = measure_layout_candidate(
        "B3_interior_fast_path", 1, protocol, cfg.params, input, output_w, output_m, k_padded, pad_value);
    const LayoutResult b4 = measure_layout_candidate(
        "B4_row_cache_descriptor_model", 0, protocol, cfg.params, input, output_w, output_m, k_padded, pad_value);
    for (const LayoutResult& r : {b0, b1, b2, b3, b4}) {
        std::cout << "stage32_layout"
                  << " candidate=" << r.candidate
                  << " attachable_to_current_kernel=" << r.attachable
                  << " warmup=" << protocol.warmup
                  << " runs=" << protocol.runs
                  << " repeats=" << protocol.repeats
                  << " mean_us=" << r.stats.mean
                  << " stddev_us=" << r.stats.stddev
                  << " min_us=" << r.stats.min
                  << " max_us=" << r.stats.max
                  << " cv_pct=" << r.stats.cv_pct
                  << " checksum=" << r.checksum
                  << " gate_us=" << kPanelGateUs
                  << " gate_status=" << (r.stats.mean <= kPanelGateUs ? "pass" : "fail")
                  << "\n";
    }

    if (b1.stats.mean <= kPanelGateUs && y26_vmadot_4x4x8_ime_available_buildtime()) {
        (void)y26_k1x_ime_probe_once();
        Y26PrepackedConvWeights* weights = y26_prepacked_conv_weights_create_mmt4d_s8(cfg.weights_ohwi_s8,
                                                                                       &cfg.params,
                                                                                       cfg.kernel_h,
                                                                                       cfg.kernel_w,
                                                                                       cfg.node_name,
                                                                                       cfg.weight_scales);
        Y26ThreadedConvWorkspace* threaded_1t = y26_threaded_conv_create_spatial_rows(&cfg, 1);
        Y26ThreadedConvWorkspace* threaded_4t = y26_threaded_conv_create_spatial_rows(&cfg, 4);
        if (weights == nullptr || threaded_1t == nullptr || threaded_4t == nullptr) {
            std::cout << "stage32_direct_row_cache status=prepare_failed\n";
            y26_prepacked_conv_weights_destroy(weights);
            y26_threaded_conv_destroy(threaded_1t);
            y26_threaded_conv_destroy(threaded_4t);
            return;
        }
        std::vector<std::int8_t> row_cache(static_cast<std::size_t>(output_m) * static_cast<std::size_t>(k_padded));
        std::vector<std::int8_t> panel(static_cast<std::size_t>(8 * k_padded));
        std::vector<std::int32_t> raw(expected_branch0.size(), 0);
        std::vector<std::int32_t> direct(expected_branch0.size(), 0);
        std::vector<std::int32_t> mmt4d_1t(expected_branch0.size(), 0);
        std::vector<std::int32_t> mmt4d_4t(expected_branch0.size(), 0);
        std::vector<double> direct_totals;
        std::vector<double> panel_totals;
        std::vector<double> compute_totals;
        std::vector<double> correction_totals;
        std::vector<double> writeback_totals;
        std::vector<double> mmt4d_1t_totals;
        std::vector<double> mmt4d_4t_totals;
        Y26ThreadedConvTimingUs t1 {};
        Y26ThreadedConvTimingUs t4 {};
        for (int i = 0; i < protocol.warmup; ++i) {
            (void)run_direct_row_cache_once(cfg, input, weights, row_cache, panel, raw, direct);
            (void)y26_threaded_conv_run_ime_cluster0(threaded_1t, input.data(), mmt4d_1t.data(), &t1);
            (void)y26_threaded_conv_run_ime_cluster0(threaded_4t, input.data(), mmt4d_4t.data(), &t4);
        }
        int last_status = Y26_CONV_STATUS_SUCCESS;
        for (int repeat = 0; repeat < protocol.repeats; ++repeat) {
            double direct_acc = 0.0;
            double panel_acc = 0.0;
            double compute_acc = 0.0;
            double correction_acc = 0.0;
            double writeback_acc = 0.0;
            double m1_acc = 0.0;
            double m4_acc = 0.0;
            for (int run = 0; run < protocol.runs; ++run) {
                DirectCachedResult dc = run_direct_row_cache_once(cfg, input, weights, row_cache, panel, raw, direct);
                last_status = dc.status;
                if (last_status != Y26_CONV_STATUS_SUCCESS) {
                    break;
                }
                direct_acc += dc.total_us;
                panel_acc += dc.panel_build_us;
                compute_acc += dc.kernel_compute_us;
                correction_acc += dc.correction_us;
                writeback_acc += dc.writeback_us;
                last_status = y26_threaded_conv_run_ime_cluster0(threaded_1t, input.data(), mmt4d_1t.data(), &t1);
                if (last_status != Y26_CONV_STATUS_SUCCESS) {
                    break;
                }
                last_status = y26_threaded_conv_run_ime_cluster0(threaded_4t, input.data(), mmt4d_4t.data(), &t4);
                if (last_status != Y26_CONV_STATUS_SUCCESS) {
                    break;
                }
                m1_acc += t1.total_us;
                m4_acc += t4.total_us;
            }
            if (last_status != Y26_CONV_STATUS_SUCCESS) {
                break;
            }
            direct_totals.push_back(direct_acc / protocol.runs);
            panel_totals.push_back(panel_acc / protocol.runs);
            compute_totals.push_back(compute_acc / protocol.runs);
            correction_totals.push_back(correction_acc / protocol.runs);
            writeback_totals.push_back(writeback_acc / protocol.runs);
            mmt4d_1t_totals.push_back(m1_acc / protocol.runs);
            mmt4d_4t_totals.push_back(m4_acc / protocol.runs);
        }
        int max_abs = 0;
        const std::size_t mismatches = mismatch_count_i32(direct, expected_branch0, max_abs);
        const Stats ds = summarize(direct_totals);
        const Stats ps = summarize(panel_totals);
        const Stats cs = summarize(compute_totals);
        const Stats rs = summarize(correction_totals);
        const Stats ws = summarize(writeback_totals);
        const Stats m1 = summarize(mmt4d_1t_totals);
        const Stats m4 = summarize(mmt4d_4t_totals);
        std::cout << "stage32_direct_row_cache"
                  << " status=" << last_status
                  << " mismatches=" << mismatches
                  << " max_abs_diff=" << max_abs
                  << " checksum=" << checksum_i32_vec(direct)
                  << " expected_checksum=" << checksum_i32_vec(expected_branch0)
                  << " mean_us=" << ds.mean
                  << " stddev_us=" << ds.stddev
                  << " panel_build_mean_us=" << ps.mean
                  << " kernel_compute_mean_us=" << cs.mean
                  << " correction_mean_us=" << rs.mean
                  << " writeback_mean_us=" << ws.mean
                  << " mmt4d_1t_mean_us=" << m1.mean
                  << " mmt4d_4t_mean_us=" << m4.mean
                  << " speedup_vs_1t=" << (ds.mean > 0.0 ? m1.mean / ds.mean : 0.0)
                  << " speedup_vs_4t=" << (ds.mean > 0.0 ? m4.mean / ds.mean : 0.0)
                  << " single_thread_gate="
                  << ((ds.mean > 0.0 && m1.mean / ds.mean >= 1.20 && mismatches == 0) ? "pass" : "fail")
                  << " threaded_gate="
                  << ((ds.mean > 0.0 && m4.mean / ds.mean >= 1.15 && mismatches == 0) ? "pass" : "fail")
                  << "\n";
        y26_prepacked_conv_weights_destroy(weights);
        y26_threaded_conv_destroy(threaded_1t);
        y26_threaded_conv_destroy(threaded_4t);
    } else {
        std::cout << "stage32_direct_row_cache status=not_attempted layout_gate="
                  << (b1.stats.mean <= kPanelGateUs ? "pass" : "fail")
                  << " buildtime_ime=" << (y26_vmadot_4x4x8_ime_available_buildtime() ? 1 : 0)
                  << "\n";
    }
}

using SignedFunc = void (*)(const std::int8_t*, const std::int8_t*, std::int32_t*);

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
sigjmp_buf g_stage32_sigill_jump;
volatile sig_atomic_t g_stage32_sigill_seen = 0;

void stage32_sigill_handler(int, siginfo_t*, void*) {
    g_stage32_sigill_seen = 1;
    siglongjmp(g_stage32_sigill_jump, 1);
}

__attribute__((noinline)) void run_dot_named(const char* variant,
                                             const std::int8_t* a,
                                             const std::int8_t* b,
                                             std::int32_t* c) {
    if (std::strcmp(variant, "smt.vmadot") == 0) {
        __asm__ volatile(
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vxor.vv      v28, v28, v28           \n\t"
            "vsetvli      t0, zero, e8, m1        \n\t"
            "vle8.v       v0, (%[A])              \n\t"
            "vle8.v       v1, (%[B])              \n\t"
            "smt.vmadot   v28, v0, v1             \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vse32.v      v28, (%[C])             \n\t"
            :
            : [A] "r"(a), [B] "r"(b), [C] "r"(c)
            : "cc", "memory", "t0", "v0", "v1", "v28", "v29");
    } else if (std::strcmp(variant, "smt.vmadotu") == 0) {
        __asm__ volatile(
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vxor.vv      v28, v28, v28           \n\t"
            "vsetvli      t0, zero, e8, m1        \n\t"
            "vle8.v       v0, (%[A])              \n\t"
            "vle8.v       v1, (%[B])              \n\t"
            "smt.vmadotu  v28, v0, v1             \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vse32.v      v28, (%[C])             \n\t"
            :
            : [A] "r"(a), [B] "r"(b), [C] "r"(c)
            : "cc", "memory", "t0", "v0", "v1", "v28", "v29");
    } else if (std::strcmp(variant, "smt.vmadotsu") == 0) {
        __asm__ volatile(
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vxor.vv      v28, v28, v28           \n\t"
            "vsetvli      t0, zero, e8, m1        \n\t"
            "vle8.v       v0, (%[A])              \n\t"
            "vle8.v       v1, (%[B])              \n\t"
            "smt.vmadotsu v28, v0, v1             \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vse32.v      v28, (%[C])             \n\t"
            :
            : [A] "r"(a), [B] "r"(b), [C] "r"(c)
            : "cc", "memory", "t0", "v0", "v1", "v28", "v29");
    } else if (std::strcmp(variant, "smt.vmadotus") == 0) {
        __asm__ volatile(
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vxor.vv      v28, v28, v28           \n\t"
            "vsetvli      t0, zero, e8, m1        \n\t"
            "vle8.v       v0, (%[A])              \n\t"
            "vle8.v       v1, (%[B])              \n\t"
            "smt.vmadotus v28, v0, v1             \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vse32.v      v28, (%[C])             \n\t"
            :
            : [A] "r"(a), [B] "r"(b), [C] "r"(c)
            : "cc", "memory", "t0", "v0", "v1", "v28", "v29");
    }
}
#endif

int scalar_value(std::int8_t v, bool unsigned_domain) {
    return unsigned_domain ? static_cast<int>(static_cast<std::uint8_t>(v)) : static_cast<int>(v);
}

void scalar_dot_hypothesis(const std::array<std::int8_t, 32>& a,
                           const std::array<std::int8_t, 32>& b,
                           bool a_unsigned,
                           bool b_unsigned,
                           std::array<std::int32_t, 16>& c) {
    c.fill(0);
    for (int m = 0; m < 4; ++m) {
        for (int n = 0; n < 4; ++n) {
            std::int32_t acc = 0;
            for (int k = 0; k < 8; ++k) {
                acc += scalar_value(a[static_cast<std::size_t>(m) * 8U + static_cast<std::size_t>(k)], a_unsigned) *
                       scalar_value(b[static_cast<std::size_t>(n) * 8U + static_cast<std::size_t>(k)], b_unsigned);
            }
            c[static_cast<std::size_t>(m) * 4U + static_cast<std::size_t>(n)] = acc;
        }
    }
}

int mismatch_i32_array(const std::array<std::int32_t, 16>& actual,
                       const std::array<std::int32_t, 16>& expected) {
    int mismatches = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        mismatches += actual[i] == expected[i] ? 0 : 1;
    }
    return mismatches;
}

void make_signedness_fixture(int fixture_id, std::array<std::int8_t, 32>& a, std::array<std::int8_t, 32>& b) {
    a.fill(0);
    b.fill(0);
    if (fixture_id == 0) {
        return;
    }
    if (fixture_id == 1) {
        a[0] = 1;
        b[0] = 1;
        return;
    }
    if (fixture_id == 2) {
        for (int i = 0; i < 32; ++i) {
            a[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(i * 7 - 100);
            b[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(80 - i * 5);
        }
        return;
    }
    if (fixture_id == 3) {
        for (int i = 0; i < 32; ++i) {
            a[static_cast<std::size_t>(i)] = static_cast<std::int8_t>((i & 1) ? -128 : 127);
            b[static_cast<std::size_t>(i)] = static_cast<std::int8_t>((i & 2) ? -127 : 126);
        }
        return;
    }
    std::uint32_t state = 0x31415926U + static_cast<std::uint32_t>(fixture_id);
    for (int i = 0; i < 32; ++i) {
        state = state * 1664525U + 1013904223U;
        a[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(state >> 24);
        state = state * 1664525U + 1013904223U;
        b[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(state >> 24);
    }
}

void run_signedness_family() {
    const char* variants[] = {"smt.vmadot", "smt.vmadotu", "smt.vmadotsu", "smt.vmadotus"};
    const char* hypotheses[] = {"s8xs8", "u8xu8", "s8xu8", "u8xs8"};
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    struct sigaction old_action {};
    struct sigaction new_action {};
    new_action.sa_sigaction = stage32_sigill_handler;
    new_action.sa_flags = SA_SIGINFO;
    sigemptyset(&new_action.sa_mask);
    if (sigaction(SIGILL, &new_action, &old_action) != 0) {
        std::cout << "stage32_signedness status=sigaction_failed\n";
        return;
    }
    for (const char* variant : variants) {
        int traps = 0;
        int total_mismatches[4] = {0, 0, 0, 0};
        long long checksum = 0;
        for (int fixture_id = 0; fixture_id < 6; ++fixture_id) {
            std::array<std::int8_t, 32> a {};
            std::array<std::int8_t, 32> b {};
            std::array<std::int32_t, 16> actual {};
            make_signedness_fixture(fixture_id, a, b);
            g_stage32_sigill_seen = 0;
            if (sigsetjmp(g_stage32_sigill_jump, 1) == 0) {
                run_dot_named(variant, a.data(), b.data(), actual.data());
            } else {
                ++traps;
                continue;
            }
            for (std::int32_t value : actual) {
                checksum += value;
            }
            for (int h = 0; h < 4; ++h) {
                std::array<std::int32_t, 16> expected {};
                scalar_dot_hypothesis(a, b, h == 1 || h == 3, h == 1 || h == 2, expected);
                total_mismatches[h] += mismatch_i32_array(actual, expected);
            }
        }
        int best_index = 0;
        for (int h = 1; h < 4; ++h) {
            if (total_mismatches[h] < total_mismatches[best_index]) {
                best_index = h;
            }
        }
        std::cout << "stage32_signedness"
                  << " variant=" << variant
                  << " traps=" << traps
                  << " best_hypothesis=" << hypotheses[best_index]
                  << " best_mismatches=" << total_mismatches[best_index]
                  << " mismatches_s8xs8=" << total_mismatches[0]
                  << " mismatches_u8xu8=" << total_mismatches[1]
                  << " mismatches_s8xu8=" << total_mismatches[2]
                  << " mismatches_u8xs8=" << total_mismatches[3]
                  << " checksum=" << checksum
                  << " oracle_status=" << (traps == 0 && total_mismatches[best_index] == 0 ? "pass" : "fail")
                  << "\n";
    }
    sigaction(SIGILL, &old_action, nullptr);
#else
    for (const char* variant : variants) {
        std::cout << "stage32_signedness"
                  << " variant=" << variant
                  << " status=not_built_or_not_riscv"
                  << "\n";
    }
#endif
}

Protocol parse_protocol(int argc, char** argv) {
    Protocol protocol {};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--warmup") {
            protocol.warmup = std::max(0, std::atoi(require_value("--warmup")));
        } else if (arg == "--runs") {
            protocol.runs = std::max(1, std::atoi(require_value("--runs")));
        } else if (arg == "--repeats") {
            protocol.repeats = std::max(1, std::atoi(require_value("--repeats")));
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            std::exit(2);
        }
    }
    return protocol;
}

}  // namespace

int main(int argc, char** argv) {
    const Protocol protocol = parse_protocol(argc, argv);
    std::cout << "stage32_begin"
              << " warmup=" << protocol.warmup
              << " runs=" << protocol.runs
              << " repeats=" << protocol.repeats
              << " note=selected-node-layout-and-signedness-proof-not-model-fps"
              << "\n";
    run_layout_gate(protocol);
    run_signedness_family();
    std::cout << "stage32_end\n";
    return 0;
}
