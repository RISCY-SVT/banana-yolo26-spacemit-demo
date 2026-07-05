#define Y26_STAGE15_NO_TEST_MAIN 1
#include "../tests/test_stage15_model4_branch_runner.cpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kFullH = 80;
constexpr int kFullW = 80;
constexpr int kModel4Cv1C = 64;

struct GateTiming {
    double conv_us = 0.0;
    double activation_requant_us = 0.0;
    double split_us = 0.0;
    double merge_us = 0.0;
    double post_qdq_us = 0.0;
    double pack_layout_us = 0.0;
    double correction_us = 0.0;
    double copy_us = 0.0;
    double total_us = 0.0;
    double conv_share_pct = 0.0;
    double activation_share_pct = 0.0;
    double merge_share_pct = 0.0;
    double pack_layout_share_pct = 0.0;
};

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000.0;
}

int output_h_for_kernel(const Y26Conv2DParams& params, int kernel_h) {
    return kernel_h == 1 ? y26_conv1x1_output_h(&params) : y26_conv3x3_output_h(&params);
}

int output_w_for_kernel(const Y26Conv2DParams& params, int kernel_w) {
    return kernel_w == 1 ? y26_conv1x1_output_w(&params) : y26_conv3x3_output_w(&params);
}

std::int8_t weight_at(const Y26Stage7ConvNodeConfig& cfg, int oc, int kh, int kw, int ic) {
    const int index = ((oc * cfg.kernel_h + kh) * cfg.kernel_w + kw) * cfg.params.input_c + ic;
    return cfg.weights_ohwi_s8[index];
}

int scalar_raw_dot(const Y26Stage7ConvNodeConfig& cfg,
                   const std::int8_t* input_nhwc_s8,
                   std::int32_t* raw_i32_nhwc) {
    const int output_h = output_h_for_kernel(cfg.params, cfg.kernel_h);
    const int output_w = output_w_for_kernel(cfg.params, cfg.kernel_w);
    const std::int8_t pad = static_cast<std::int8_t>(cfg.input_storage_zero_point_s8);
    for (int oh = 0; oh < output_h; ++oh) {
        for (int ow = 0; ow < output_w; ++ow) {
            for (int oc = 0; oc < cfg.params.output_c; ++oc) {
                std::int32_t acc = 0;
                for (int kh = 0; kh < cfg.kernel_h; ++kh) {
                    const int ih = oh * cfg.params.stride_h + kh - cfg.params.pad_h;
                    const bool valid_h = ih >= 0 && ih < cfg.params.input_h;
                    for (int kw = 0; kw < cfg.kernel_w; ++kw) {
                        const int iw = ow * cfg.params.stride_w + kw - cfg.params.pad_w;
                        const bool inside = valid_h && iw >= 0 && iw < cfg.params.input_w;
                        const std::int8_t* src =
                            inside ? input_nhwc_s8 + (ih * cfg.params.input_w + iw) * cfg.params.input_c : nullptr;
                        for (int ic = 0; ic < cfg.params.input_c; ++ic) {
                            const std::int8_t a = inside ? src[ic] : pad;
                            acc += static_cast<std::int32_t>(a) *
                                   static_cast<std::int32_t>(weight_at(cfg, oc, kh, kw, ic));
                        }
                    }
                }
                raw_i32_nhwc[(oh * output_w + ow) * cfg.params.output_c + oc] = acc;
            }
        }
    }
    return Y26_CONV_STATUS_SUCCESS;
}

int apply_correction(const Y26Stage7ConvNodeConfig& cfg,
                     const Y26PrepackedConvWeights* weights,
                     const std::int32_t* raw_i32,
                     std::int32_t* corrected_i32) {
    return y26_conv2d_apply_u8_as_s8_correction_nhwc(raw_i32,
                                                     cfg.bias_i32,
                                                     y26_prepacked_conv_weights_sums(weights),
                                                     corrected_i32,
                                                     output_h_for_kernel(cfg.params, cfg.kernel_h) *
                                                         output_w_for_kernel(cfg.params, cfg.kernel_w),
                                                     cfg.params.output_c,
                                                     cfg.activation_zero_point_u8);
}

Y26ActivationRequantParams activation_params(const Y26Stage7ConvNodeConfig& producer,
                                             std::size_t output_count,
                                             float act_output_scale,
                                             int act_output_zero_point_u8) {
    return Y26ActivationRequantParams{output_count,
                                      producer.params.output_c,
                                      producer.input_scale,
                                      producer.weight_scales,
                                      producer.output_scale,
                                      producer.output_zero_point_u8,
                                      act_output_scale,
                                      act_output_zero_point_u8};
}

int apply_activation(int mode,
                     const Y26ActivationRequantParams& params,
                     const std::int8_t* lut_s8,
                     const std::int32_t* input_i32,
                     std::int8_t* output_s8) {
    if (mode == Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT) {
        return y26_activation_requant_silu_int8_lut_rvv_f32(&params, input_i32, lut_s8, output_s8);
    }
    return y26_activation_requant_silu_int8_lut(&params, input_i32, lut_s8, output_s8);
}

void fill_model4_cv1_i32(const y26_stage15_model4_branch_fixture::Model4BranchFixture& fixture,
                         std::vector<std::int32_t>& values) {
    const std::int32_t* pattern = fixture.stage14_fixture->expected_model4_cv1_i32_nhwc;
    const std::size_t pattern_count = fixture.stage14_fixture->expected_model4_cv1_count;
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = pattern[i % pattern_count];
    }
}

void split_second_half(const std::int8_t* full_s8, std::int8_t* split1_s8) {
    constexpr int split_c = kModel4Cv1C / 2;
    constexpr int spatial = kFullH * kFullW;
    for (int m = 0; m < spatial; ++m) {
        std::memcpy(split1_s8 + m * split_c, full_s8 + m * kModel4Cv1C + split_c, split_c);
    }
}

std::size_t mismatches_i32(const std::vector<std::int32_t>& actual, const std::vector<std::int32_t>& expected) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        mismatches += actual[i] != expected[i] ? 1U : 0U;
    }
    return mismatches;
}

std::size_t mismatches_i8(const std::vector<std::int8_t>& actual, const std::vector<std::int8_t>& expected) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        mismatches += actual[i] != expected[i] ? 1U : 0U;
    }
    return mismatches;
}

long long checksum_i32(const std::vector<std::int32_t>& values) {
    long long sum = 0;
    for (std::int32_t value : values) {
        sum += value;
    }
    return sum;
}

Y26Stage7ConvNodeConfig fullshape_branch0_config(
    const y26_stage15_model4_branch_fixture::Model4BranchFixture& fixture) {
    Y26Stage7ConvNodeConfig cfg = stage15_branch0_config_from_fixture(fixture);
    cfg.params.input_h = kFullH;
    cfg.params.input_w = kFullW;
    return cfg;
}

Y26Stage7ConvNodeConfig fullshape_model4_cv1_producer(
    const y26_stage15_model4_branch_fixture::Model4BranchFixture& fixture) {
    Y26Stage7ConvNodeConfig cfg =
        config_from_fixture(*fixture.stage14_fixture, Y26_ACTIVATION_MODE_INT8_LUT).model4_cv1;
    cfg.params.input_h = kFullH;
    cfg.params.input_w = kFullW;
    return cfg;
}

int run_once(const y26_stage15_model4_branch_fixture::Model4BranchFixture& fixture,
             int activation_mode,
             bool use_ime,
             const std::vector<std::int32_t>& model4_cv1_i32,
             std::vector<std::int8_t>& split1_s8,
             std::vector<std::int32_t>& branch0_i32,
             std::vector<std::int8_t>& branch0_act_s8,
             GateTiming& timing) {
    Y26Stage7ConvNodeConfig producer = fullshape_model4_cv1_producer(fixture);
    Y26Stage7ConvNodeConfig branch0 = fullshape_branch0_config(fixture);
    Y26PrepackedConvWeights* weights = y26_prepacked_conv_weights_create_mmt4d_s8(branch0.weights_ohwi_s8,
                                                                                  &branch0.params,
                                                                                  branch0.kernel_h,
                                                                                  branch0.kernel_w,
                                                                                  branch0.node_name,
                                                                                  branch0.weight_scales);
    Y26ConvWorkspace* workspace = y26_conv_workspace_create(&branch0.params, branch0.kernel_h, branch0.kernel_w);
    std::vector<std::int8_t> model4_cv1_act(model4_cv1_i32.size(), 0);
    std::vector<std::int32_t> raw(branch0_i32.size(), 0);
    std::int8_t split_lut[256] {};
    std::int8_t branch_lut[256] {};
    y26_build_silu_u8_to_s8_lut(producer.output_scale,
                                producer.output_zero_point_u8,
                                fixture.split1_output_scale,
                                fixture.split1_output_zero_point_u8,
                                split_lut);
    y26_build_silu_u8_to_s8_lut(branch0.output_scale,
                                branch0.output_zero_point_u8,
                                fixture.branch0_act_output_scale,
                                fixture.branch0_act_output_zero_point_u8,
                                branch_lut);

    const auto begin = Clock::now();
    const auto act0_begin = Clock::now();
    Y26ActivationRequantParams split_params = activation_params(
        producer, model4_cv1_i32.size(), fixture.split1_output_scale, fixture.split1_output_zero_point_u8);
    int status = apply_activation(activation_mode, split_params, split_lut, model4_cv1_i32.data(), model4_cv1_act.data());
    const auto act0_end = Clock::now();
    if (status != Y26_CONV_STATUS_SUCCESS) {
        y26_prepacked_conv_weights_destroy(weights);
        y26_conv_workspace_destroy(workspace);
        return status;
    }
    const auto split_begin = Clock::now();
    split_second_half(model4_cv1_act.data(), split1_s8.data());
    const auto split_end = Clock::now();

    double conv_us = 0.0;
    double correction_us = 0.0;
    const auto conv_begin = Clock::now();
    if (use_ime) {
        status = y26_conv2d_i8s8s32_nhwc_ime_prepacked_v1(split1_s8.data(),
                                                          weights,
                                                          raw.data(),
                                                          branch0.input_storage_zero_point_s8,
                                                          workspace,
                                                          Y26_CONV_LOOP_ORDER_M_MAJOR);
    } else {
        status = scalar_raw_dot(branch0, split1_s8.data(), raw.data());
    }
    const auto correction_begin = Clock::now();
    if (status == Y26_CONV_STATUS_SUCCESS) {
        status = apply_correction(branch0, weights, raw.data(), branch0_i32.data());
    }
    const auto conv_end = Clock::now();
    conv_us = elapsed_us(conv_begin, conv_end);
    correction_us = elapsed_us(correction_begin, conv_end);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        y26_prepacked_conv_weights_destroy(weights);
        y26_conv_workspace_destroy(workspace);
        return status;
    }

    const auto act1_begin = Clock::now();
    Y26ActivationRequantParams branch_params = activation_params(
        branch0, branch0_i32.size(), fixture.branch0_act_output_scale, fixture.branch0_act_output_zero_point_u8);
    status = apply_activation(activation_mode, branch_params, branch_lut, branch0_i32.data(), branch0_act_s8.data());
    const auto act1_end = Clock::now();
    const auto end = Clock::now();

    timing.activation_requant_us = elapsed_us(act0_begin, act0_end) + elapsed_us(act1_begin, act1_end);
    timing.split_us = elapsed_us(split_begin, split_end);
    timing.merge_us = timing.split_us;
    timing.post_qdq_us = elapsed_us(act0_begin, act0_end);
    timing.conv_us = conv_us;
    timing.correction_us = correction_us;
    timing.total_us = elapsed_us(begin, end);
    if (timing.total_us > 0.0) {
        timing.conv_share_pct = 100.0 * timing.conv_us / timing.total_us;
        timing.activation_share_pct = 100.0 * timing.activation_requant_us / timing.total_us;
        timing.merge_share_pct = 100.0 * timing.merge_us / timing.total_us;
        timing.pack_layout_share_pct = 100.0 * timing.pack_layout_us / timing.total_us;
    }
    y26_prepacked_conv_weights_destroy(weights);
    y26_conv_workspace_destroy(workspace);
    return status;
}

void accumulate(GateTiming& dst, const GateTiming& src) {
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

void divide(GateTiming& timing, double denom) {
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

int run_candidate(const y26_stage15_model4_branch_fixture::Model4BranchFixture& fixture,
                  const char* candidate,
                  int activation_mode,
                  bool use_ime,
                  int iterations,
                  const std::vector<std::int8_t>& expected_split1,
                  const std::vector<std::int32_t>& expected_branch0,
                  const std::vector<std::int8_t>& expected_branch0_act,
                  const std::vector<std::int32_t>& model4_cv1_i32) {
    constexpr int split_count = kFullH * kFullW * (kModel4Cv1C / 2);
    constexpr int branch_count = kFullH * kFullW * 16;
    std::vector<std::int8_t> split1(split_count, 0);
    std::vector<std::int32_t> branch0(branch_count, 0);
    std::vector<std::int8_t> branch0_act(branch_count, 0);
    GateTiming sum {};
    std::size_t total_mismatches = 0;
    std::size_t split_mismatches = 0;
    std::size_t branch_mismatches = 0;
    std::size_t branch_act_mismatches = 0;
    int last_status = Y26_CONV_STATUS_SUCCESS;
    for (int i = 0; i < iterations; ++i) {
        GateTiming timing {};
        last_status = run_once(fixture, activation_mode, use_ime, model4_cv1_i32, split1, branch0, branch0_act, timing);
        if (last_status != Y26_CONV_STATUS_SUCCESS) {
            total_mismatches += 1;
            break;
        }
        const std::size_t split_iter = mismatches_i8(split1, expected_split1);
        const std::size_t branch_iter = mismatches_i32(branch0, expected_branch0);
        const std::size_t branch_act_iter = mismatches_i8(branch0_act, expected_branch0_act);
        split_mismatches += split_iter;
        branch_mismatches += branch_iter;
        branch_act_mismatches += branch_act_iter;
        total_mismatches += split_iter + branch_iter + branch_act_iter;
        accumulate(sum, timing);
    }
    divide(sum, static_cast<double>(iterations));
    const char* correctness = last_status == Y26_CONV_STATUS_SUCCESS && total_mismatches == 0 ? "pass" : "fail";
    std::cout << "candidate=" << candidate
              << " fixture=representative_full_shape_synthetic"
              << " shape_class=full_shape_model4_branch_entry"
              << " h=" << kFullH
              << " w=" << kFullW
              << " model4_cv1_c=" << kModel4Cv1C
              << " correctness_status=" << correctness
              << " status=" << last_status
              << " mismatches=" << total_mismatches
              << " split_mismatches=" << split_mismatches
              << " branch_mismatches=" << branch_mismatches
              << " branch_act_mismatches=" << branch_act_mismatches
              << " checksum=" << checksum_i32(branch0)
              << " total_us=" << sum.total_us
              << " conv_us=" << sum.conv_us
              << " activation_requant_us=" << sum.activation_requant_us
              << " split_us=" << sum.split_us
              << " merge_us=" << sum.merge_us
              << " post_qdq_us=" << sum.post_qdq_us
              << " pack_layout_us=" << sum.pack_layout_us
              << " correction_us=" << sum.correction_us
              << " copy_us=" << sum.copy_us
              << " conv_share_pct=" << sum.conv_share_pct
              << " activation_share_pct=" << sum.activation_share_pct
              << " merge_share_pct=" << sum.merge_share_pct
              << " pack_layout_share_pct=" << sum.pack_layout_share_pct
              << "\n";
    return correctness[0] == 'p' ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    int iterations = 1;
    if (argc > 1) {
        iterations = std::max(1, std::atoi(argv[1]));
    }
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
        std::cout << "candidate=scalar_reference_int8_lut correctness_status=reference_fail status=" << status << "\n";
        return 1;
    }

    std::cout << "subset=candidate_I_model4_split_first_branch shape_class=full_shape_model4_branch_entry iterations="
              << iterations << "\n";
    int failures = 0;
    failures += run_candidate(fixture,
                              "scalar_reference_int8_lut",
                              Y26_ACTIVATION_MODE_INT8_LUT,
                              false,
                              iterations,
                              expected_split1,
                              expected_branch0,
                              expected_branch0_act,
                              model4_cv1_i32);
    if (y26_vmadot_4x4x8_ime_available_buildtime()) {
        (void)y26_k1x_ime_probe_once();
        failures += run_candidate(fixture,
                                  "stage16A_IME_A2_rvv_f32_lut",
                                  Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT,
                                  true,
                                  iterations,
                                  expected_split1,
                                  expected_branch0,
                                  expected_branch0_act,
                                  model4_cv1_i32);
    } else {
        std::cout << "candidate=stage16A_IME_A2_rvv_f32_lut correctness_status=not_built\n";
    }
    return failures == 0 ? 0 : 1;
}
