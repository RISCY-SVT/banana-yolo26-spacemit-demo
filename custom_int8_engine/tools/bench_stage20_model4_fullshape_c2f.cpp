#define Y26_STAGE16_NO_TEST_MAIN 1
#include "../tests/test_stage16_model4_c2f_runner.cpp"

#include "y26_k1x_threaded_conv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kFullH = 80;
constexpr int kFullW = 80;
constexpr int kModel4Cv1C = 64;
constexpr int kSplitC = 32;

struct Protocol {
    int warmup = 10;
    int runs = 100;
    int repeats = 5;
};

struct Timing {
    double conv_us = 0.0;
    double activation_requant_us = 0.0;
    double split_us = 0.0;
    double add_us = 0.0;
    double concat_us = 0.0;
    double post_concat_qdq_us = 0.0;
    double pack_layout_us = 0.0;
    double thread_overhead_us = 0.0;
    double correction_us = 0.0;
    double total_us = 0.0;
    double branch0_conv_us = 0.0;
    double branch1_conv_us = 0.0;
    double model4_cv2_conv_us = 0.0;
};

struct Summary {
    double mean_total_us = 0.0;
    double stddev_total_us = 0.0;
    double min_total_us = 0.0;
    double max_total_us = 0.0;
    double cv_total_pct = 0.0;
    double mean_conv_us = 0.0;
    double mean_activation_requant_us = 0.0;
    double mean_merge_us = 0.0;
    double mean_thread_overhead_us = 0.0;
    double mean_branch0_conv_us = 0.0;
    double mean_branch1_conv_us = 0.0;
    double mean_model4_cv2_conv_us = 0.0;
    double mean_correction_us = 0.0;
    std::size_t mismatches = 0;
    long long checksum = 0;
    int status = Y26_CONV_STATUS_SUCCESS;
    int affinity_ok = 1;
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

std::size_t output_count_for_kernel(const Y26Stage7ConvNodeConfig& cfg) {
    return static_cast<std::size_t>(output_h_for_kernel(cfg.params, cfg.kernel_h)) *
           static_cast<std::size_t>(output_w_for_kernel(cfg.params, cfg.kernel_w)) *
           static_cast<std::size_t>(cfg.params.output_c);
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
    const int output_h = output_h_for_kernel(cfg.params, cfg.kernel_h);
    const int output_w = output_w_for_kernel(cfg.params, cfg.kernel_w);
    return y26_conv2d_apply_u8_as_s8_correction_nhwc(raw_i32,
                                                     cfg.bias_i32,
                                                     y26_prepacked_conv_weights_sums(weights),
                                                     corrected_i32,
                                                     output_h * output_w,
                                                     cfg.params.output_c,
                                                     cfg.activation_zero_point_u8);
}

int run_conv_scalar(const Y26Stage7ConvNodeConfig& cfg,
                    const Y26PrepackedConvWeights* weights,
                    const std::int8_t* input_s8,
                    std::vector<std::int32_t>& raw,
                    std::vector<std::int32_t>& output,
                    double& conv_us,
                    double& correction_us) {
    const auto begin = Clock::now();
    int status = scalar_raw_dot(cfg, input_s8, raw.data());
    const auto correction_begin = Clock::now();
    if (status == Y26_CONV_STATUS_SUCCESS) {
        status = apply_correction(cfg, weights, raw.data(), output.data());
    }
    const auto end = Clock::now();
    conv_us = elapsed_us(begin, end);
    correction_us = elapsed_us(correction_begin, end);
    return status;
}

int run_conv_ime(const Y26Stage7ConvNodeConfig& cfg,
                 const Y26PrepackedConvWeights* weights,
                 Y26ConvWorkspace* workspace,
                 const std::int8_t* input_s8,
                 std::vector<std::int32_t>& raw,
                 std::vector<std::int32_t>& output,
                 double& conv_us,
                 double& correction_us) {
    const auto begin = Clock::now();
    int status = y26_conv2d_i8s8s32_nhwc_ime_prepacked_v1(input_s8,
                                                          weights,
                                                          raw.data(),
                                                          cfg.input_storage_zero_point_s8,
                                                          workspace,
                                                          Y26_CONV_LOOP_ORDER_M_MAJOR);
    const auto correction_begin = Clock::now();
    if (status == Y26_CONV_STATUS_SUCCESS) {
        status = apply_correction(cfg, weights, raw.data(), output.data());
    }
    const auto end = Clock::now();
    conv_us = elapsed_us(begin, end);
    correction_us = elapsed_us(correction_begin, end);
    return status;
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

int apply_activation(int activation_mode,
                     const Y26ActivationRequantParams& params,
                     const std::int8_t* lut_s8,
                     const std::int32_t* input_i32,
                     std::int8_t* output_s8) {
    if (activation_mode == Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT) {
        return y26_activation_requant_silu_int8_lut_rvv_f32(&params, input_i32, lut_s8, output_s8);
    }
    return y26_activation_requant_silu_int8_lut(&params, input_i32, lut_s8, output_s8);
}

float silu_f32(float value) {
    return value / (1.0f + std::exp(-value));
}

std::uint8_t accumulator_to_conv_code(const Y26Stage7ConvNodeConfig& cfg, std::int32_t acc, int channel) {
    const float acc_scale = cfg.input_scale * cfg.weight_scales[channel];
    const float conv_float = static_cast<float>(acc) * acc_scale;
    return y26_quantize_u8_nearest_even_f32(conv_float, cfg.output_scale, cfg.output_zero_point_u8);
}

float accumulator_to_silu_float(const Y26Stage7ConvNodeConfig& cfg, std::int32_t acc, int channel) {
    const std::uint8_t code = accumulator_to_conv_code(cfg, acc, channel);
    const float x = (static_cast<int>(code) - cfg.output_zero_point_u8) * cfg.output_scale;
    return silu_f32(x);
}

float dequant_signed_storage(std::int8_t value, float scale, int zero_point_u8) {
    const int code = static_cast<int>(value) + 128;
    return static_cast<float>(code - zero_point_u8) * scale;
}

std::int8_t quantize_s8(float value, float scale, int zero_point_u8) {
    const std::uint8_t code = y26_quantize_u8_nearest_even_f32(value, scale, zero_point_u8);
    return static_cast<std::int8_t>(static_cast<int>(code) - 128);
}

Y26Stage7ConvNodeConfig fullshape_model4_cv1_producer(
    const y26_stage15_model4_branch_fixture::Model4BranchFixture& fixture) {
    Y26Stage7ConvNodeConfig cfg =
        config_from_fixture(*fixture.stage14_fixture, Y26_ACTIVATION_MODE_INT8_LUT).model4_cv1;
    cfg.params.input_h = kFullH;
    cfg.params.input_w = kFullW;
    return cfg;
}

Y26Stage7ConvNodeConfig fullshape_branch0(
    const y26_stage15_model4_branch_fixture::Model4BranchFixture& fixture) {
    Y26Stage7ConvNodeConfig cfg = stage15_branch0_config_from_fixture(fixture);
    cfg.params.input_h = kFullH;
    cfg.params.input_w = kFullW;
    return cfg;
}

Y26Stage7ConvNodeConfig fullshape_branch1(
    const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture) {
    Y26Stage7ConvNodeConfig cfg = branch1_config_from_fixture(fixture);
    cfg.params.input_h = kFullH;
    cfg.params.input_w = kFullW;
    return cfg;
}

Y26Stage7ConvNodeConfig fullshape_model4_cv2(
    const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture) {
    Y26Stage7ConvNodeConfig cfg = model4_cv2_config_from_fixture(fixture);
    cfg.params.input_h = kFullH;
    cfg.params.input_w = kFullW;
    return cfg;
}

void fill_model4_cv1_i32(const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture,
                         std::vector<std::int32_t>& values) {
    const std::int32_t* pattern = fixture.stage15_fixture->stage14_fixture->expected_model4_cv1_i32_nhwc;
    const std::size_t pattern_count = fixture.stage15_fixture->stage14_fixture->expected_model4_cv1_count;
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = pattern[i % pattern_count];
    }
}

long long checksum_i32(const std::vector<std::int32_t>& values) {
    long long sum = 0;
    for (std::int32_t value : values) {
        sum += value;
    }
    return sum;
}

std::size_t mismatches_i32(const std::vector<std::int32_t>& actual, const std::vector<std::int32_t>& expected) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        mismatches += actual[i] != expected[i] ? 1U : 0U;
    }
    return mismatches;
}

struct Context {
    explicit Context(const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture_in)
        : fixture(fixture_in),
          producer(fullshape_model4_cv1_producer(*fixture_in.stage15_fixture)),
          branch0(fullshape_branch0(*fixture_in.stage15_fixture)),
          branch1(fullshape_branch1(fixture_in)),
          model4_cv2(fullshape_model4_cv2(fixture_in)),
          model4_cv1_i32(static_cast<std::size_t>(kFullH) * kFullW * kModel4Cv1C),
          model4_cv1_act_s8(model4_cv1_i32.size()),
          model4_cv1_concat_s8(model4_cv1_i32.size()),
          split1_s8(static_cast<std::size_t>(kFullH) * kFullW * kSplitC),
          branch0_raw(output_count_for_kernel(branch0)),
          branch0_i32(output_count_for_kernel(branch0)),
          branch0_act_s8(output_count_for_kernel(branch0)),
          branch1_raw(output_count_for_kernel(branch1)),
          branch1_i32(output_count_for_kernel(branch1)),
          branch1_act_f32(output_count_for_kernel(branch1)),
          concat_s8(static_cast<std::size_t>(kFullH) * kFullW * kSplitC * 3),
          model4_cv2_raw(output_count_for_kernel(model4_cv2)),
          output_i32(output_count_for_kernel(model4_cv2)) {}

    const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture;
    Y26Stage7ConvNodeConfig producer;
    Y26Stage7ConvNodeConfig branch0;
    Y26Stage7ConvNodeConfig branch1;
    Y26Stage7ConvNodeConfig model4_cv2;
    Y26PrepackedConvWeights* branch0_weights = nullptr;
    Y26PrepackedConvWeights* branch1_weights = nullptr;
    Y26PrepackedConvWeights* model4_cv2_weights = nullptr;
    Y26ConvWorkspace* branch0_workspace = nullptr;
    Y26ConvWorkspace* branch1_workspace = nullptr;
    Y26ConvWorkspace* model4_cv2_workspace = nullptr;
    Y26ThreadedConvWorkspace* branch0_threaded = nullptr;
    std::vector<std::int32_t> model4_cv1_i32;
    std::vector<std::int8_t> model4_cv1_act_s8;
    std::vector<std::int8_t> model4_cv1_concat_s8;
    std::vector<std::int8_t> split1_s8;
    std::vector<std::int32_t> branch0_raw;
    std::vector<std::int32_t> branch0_i32;
    std::vector<std::int8_t> branch0_act_s8;
    std::vector<std::int32_t> branch1_raw;
    std::vector<std::int32_t> branch1_i32;
    std::vector<float> branch1_act_f32;
    std::vector<std::int8_t> concat_s8;
    std::vector<std::int32_t> model4_cv2_raw;
    std::vector<std::int32_t> output_i32;
    std::int8_t split_lut[256] {};
    std::int8_t split0_concat_lut[256] {};
    std::int8_t branch0_lut[256] {};
};

bool prepare(Context& ctx, int thread_count) {
    fill_model4_cv1_i32(ctx.fixture, ctx.model4_cv1_i32);
    ctx.branch0_weights = y26_prepacked_conv_weights_create_mmt4d_s8(
        ctx.branch0.weights_ohwi_s8, &ctx.branch0.params, ctx.branch0.kernel_h, ctx.branch0.kernel_w,
        ctx.branch0.node_name, ctx.branch0.weight_scales);
    ctx.branch1_weights = y26_prepacked_conv_weights_create_mmt4d_s8(
        ctx.branch1.weights_ohwi_s8, &ctx.branch1.params, ctx.branch1.kernel_h, ctx.branch1.kernel_w,
        ctx.branch1.node_name, ctx.branch1.weight_scales);
    ctx.model4_cv2_weights = y26_prepacked_conv_weights_create_mmt4d_s8(
        ctx.model4_cv2.weights_ohwi_s8, &ctx.model4_cv2.params, ctx.model4_cv2.kernel_h, ctx.model4_cv2.kernel_w,
        ctx.model4_cv2.node_name, ctx.model4_cv2.weight_scales);
    ctx.branch0_workspace = y26_conv_workspace_create(&ctx.branch0.params, ctx.branch0.kernel_h, ctx.branch0.kernel_w);
    ctx.branch1_workspace = y26_conv_workspace_create(&ctx.branch1.params, ctx.branch1.kernel_h, ctx.branch1.kernel_w);
    ctx.model4_cv2_workspace =
        y26_conv_workspace_create(&ctx.model4_cv2.params, ctx.model4_cv2.kernel_h, ctx.model4_cv2.kernel_w);
    if (thread_count > 0) {
        ctx.branch0_threaded = y26_threaded_conv_create_spatial_rows(&ctx.branch0, thread_count);
    }
    y26_build_silu_u8_to_s8_lut(ctx.producer.output_scale,
                                ctx.producer.output_zero_point_u8,
                                ctx.fixture.stage15_fixture->split1_output_scale,
                                ctx.fixture.stage15_fixture->split1_output_zero_point_u8,
                                ctx.split_lut);
    y26_build_silu_u8_to_s8_lut(ctx.producer.output_scale,
                                ctx.producer.output_zero_point_u8,
                                ctx.fixture.concat_output_scale,
                                ctx.fixture.concat_output_zero_point_u8,
                                ctx.split0_concat_lut);
    y26_build_silu_u8_to_s8_lut(ctx.branch0.output_scale,
                                ctx.branch0.output_zero_point_u8,
                                ctx.fixture.stage15_fixture->branch0_act_output_scale,
                                ctx.fixture.stage15_fixture->branch0_act_output_zero_point_u8,
                                ctx.branch0_lut);
    return ctx.branch0_weights != nullptr && ctx.branch1_weights != nullptr && ctx.model4_cv2_weights != nullptr &&
           ctx.branch0_workspace != nullptr && ctx.branch1_workspace != nullptr && ctx.model4_cv2_workspace != nullptr &&
           (thread_count == 0 || ctx.branch0_threaded != nullptr);
}

void destroy(Context& ctx) {
    y26_threaded_conv_destroy(ctx.branch0_threaded);
    y26_conv_workspace_destroy(ctx.model4_cv2_workspace);
    y26_conv_workspace_destroy(ctx.branch1_workspace);
    y26_conv_workspace_destroy(ctx.branch0_workspace);
    y26_prepacked_conv_weights_destroy(ctx.model4_cv2_weights);
    y26_prepacked_conv_weights_destroy(ctx.branch1_weights);
    y26_prepacked_conv_weights_destroy(ctx.branch0_weights);
}

void build_branch1_activation_float(Context& ctx) {
    const int channels = ctx.branch1.params.output_c;
    for (std::size_t i = 0; i < ctx.branch1_i32.size(); ++i) {
        const int c = static_cast<int>(i % static_cast<std::size_t>(channels));
        ctx.branch1_act_f32[i] = accumulator_to_silu_float(ctx.branch1, ctx.branch1_i32[i], c);
    }
}

void build_concat_qdq(Context& ctx, bool use_split0_concat_lut) {
    for (int m = 0; m < kFullH * kFullW; ++m) {
        std::int8_t* dst = ctx.concat_s8.data() + static_cast<std::size_t>(m) * kSplitC * 3;
        for (int c = 0; c < kSplitC; ++c) {
            if (use_split0_concat_lut) {
                dst[c] = ctx.model4_cv1_concat_s8[static_cast<std::size_t>(m) * kModel4Cv1C + c];
            } else {
                const float value =
                    accumulator_to_silu_float(ctx.producer, ctx.model4_cv1_i32[m * kModel4Cv1C + c], c);
                dst[c] = quantize_s8(value, ctx.fixture.concat_output_scale, ctx.fixture.concat_output_zero_point_u8);
            }
        }
        for (int c = 0; c < kSplitC; ++c) {
            const float split1_value = dequant_signed_storage(ctx.split1_s8[m * kSplitC + c],
                                                             ctx.fixture.stage15_fixture->split1_output_scale,
                                                             ctx.fixture.stage15_fixture->split1_output_zero_point_u8);
            dst[kSplitC + c] =
                quantize_s8(split1_value, ctx.fixture.concat_output_scale, ctx.fixture.concat_output_zero_point_u8);
            const float add_value = split1_value + ctx.branch1_act_f32[m * kSplitC + c];
            dst[kSplitC * 2 + c] =
                quantize_s8(add_value, ctx.fixture.concat_output_scale, ctx.fixture.concat_output_zero_point_u8);
        }
    }
}

int run_once(Context& ctx,
             int activation_mode,
             bool use_ime,
             bool use_threaded_branch0,
             bool use_split0_concat_lut,
             Timing& timing) {
    timing = Timing {};
    const auto total_begin = Clock::now();

    const auto act0_begin = Clock::now();
    Y26ActivationRequantParams split_params = activation_params(ctx.producer,
                                                                ctx.model4_cv1_i32.size(),
                                                                ctx.fixture.stage15_fixture->split1_output_scale,
                                                                ctx.fixture.stage15_fixture->split1_output_zero_point_u8);
    int status = apply_activation(
        activation_mode, split_params, ctx.split_lut, ctx.model4_cv1_i32.data(), ctx.model4_cv1_act_s8.data());
    const auto act0_end = Clock::now();
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    if (use_split0_concat_lut) {
        Y26ActivationRequantParams concat_params = activation_params(
            ctx.producer, ctx.model4_cv1_i32.size(), ctx.fixture.concat_output_scale, ctx.fixture.concat_output_zero_point_u8);
        status = apply_activation(activation_mode,
                                  concat_params,
                                  ctx.split0_concat_lut,
                                  ctx.model4_cv1_i32.data(),
                                  ctx.model4_cv1_concat_s8.data());
        if (status != Y26_CONV_STATUS_SUCCESS) {
            return status;
        }
    }

    const auto split_begin = Clock::now();
    for (int m = 0; m < kFullH * kFullW; ++m) {
        std::memcpy(ctx.split1_s8.data() + static_cast<std::size_t>(m) * kSplitC,
                    ctx.model4_cv1_act_s8.data() + static_cast<std::size_t>(m) * kModel4Cv1C + kSplitC,
                    kSplitC);
    }
    const auto split_end = Clock::now();

    double conv_us = 0.0;
    double correction_us = 0.0;
    if (use_threaded_branch0) {
        Y26ThreadedConvTimingUs threaded {};
        status =
            y26_threaded_conv_run_ime_cluster0(ctx.branch0_threaded, ctx.split1_s8.data(), ctx.branch0_i32.data(), &threaded);
        conv_us = threaded.total_us;
        correction_us = threaded.correction_us;
        timing.thread_overhead_us += std::max(0.0, threaded.total_us - threaded.worker_max_us);
    } else if (use_ime) {
        status = run_conv_ime(ctx.branch0,
                              ctx.branch0_weights,
                              ctx.branch0_workspace,
                              ctx.split1_s8.data(),
                              ctx.branch0_raw,
                              ctx.branch0_i32,
                              conv_us,
                              correction_us);
    } else {
        status = run_conv_scalar(ctx.branch0,
                                 ctx.branch0_weights,
                                 ctx.split1_s8.data(),
                                 ctx.branch0_raw,
                                 ctx.branch0_i32,
                                 conv_us,
                                 correction_us);
    }
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    timing.conv_us += conv_us;
    timing.branch0_conv_us += conv_us;
    timing.correction_us += correction_us;

    const auto act1_begin = Clock::now();
    Y26ActivationRequantParams branch0_params =
        activation_params(ctx.branch0,
                          ctx.branch0_i32.size(),
                          ctx.fixture.stage15_fixture->branch0_act_output_scale,
                          ctx.fixture.stage15_fixture->branch0_act_output_zero_point_u8);
    status = apply_activation(
        activation_mode, branch0_params, ctx.branch0_lut, ctx.branch0_i32.data(), ctx.branch0_act_s8.data());
    const auto act1_end = Clock::now();
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }

    double branch1_conv_us = 0.0;
    double branch1_correction_us = 0.0;
    status = use_ime ? run_conv_ime(ctx.branch1,
                                    ctx.branch1_weights,
                                    ctx.branch1_workspace,
                                    ctx.branch0_act_s8.data(),
                                    ctx.branch1_raw,
                                    ctx.branch1_i32,
                                    branch1_conv_us,
                                    branch1_correction_us)
                     : run_conv_scalar(ctx.branch1,
                                       ctx.branch1_weights,
                                       ctx.branch0_act_s8.data(),
                                       ctx.branch1_raw,
                                       ctx.branch1_i32,
                                       branch1_conv_us,
                                       branch1_correction_us);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    timing.conv_us += branch1_conv_us;
    timing.branch1_conv_us += branch1_conv_us;
    timing.correction_us += branch1_correction_us;

    const auto branch1_act_begin = Clock::now();
    build_branch1_activation_float(ctx);
    const auto branch1_act_end = Clock::now();

    const auto merge_begin = Clock::now();
    build_concat_qdq(ctx, use_split0_concat_lut);
    const auto merge_end = Clock::now();

    double model4_cv2_conv_us = 0.0;
    double model4_cv2_correction_us = 0.0;
    status = use_ime ? run_conv_ime(ctx.model4_cv2,
                                    ctx.model4_cv2_weights,
                                    ctx.model4_cv2_workspace,
                                    ctx.concat_s8.data(),
                                    ctx.model4_cv2_raw,
                                    ctx.output_i32,
                                    model4_cv2_conv_us,
                                    model4_cv2_correction_us)
                     : run_conv_scalar(ctx.model4_cv2,
                                       ctx.model4_cv2_weights,
                                       ctx.concat_s8.data(),
                                       ctx.model4_cv2_raw,
                                       ctx.output_i32,
                                       model4_cv2_conv_us,
                                       model4_cv2_correction_us);
    const auto total_end = Clock::now();
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    timing.conv_us += model4_cv2_conv_us;
    timing.model4_cv2_conv_us += model4_cv2_conv_us;
    timing.correction_us += model4_cv2_correction_us;
    timing.activation_requant_us += elapsed_us(act0_begin, act0_end);
    timing.activation_requant_us += elapsed_us(act1_begin, act1_end);
    timing.activation_requant_us += elapsed_us(branch1_act_begin, branch1_act_end);
    timing.split_us += elapsed_us(split_begin, split_end);
    timing.add_us += elapsed_us(merge_begin, merge_end);
    timing.concat_us += elapsed_us(merge_begin, merge_end);
    timing.post_concat_qdq_us += elapsed_us(merge_begin, merge_end);
    timing.total_us = elapsed_us(total_begin, total_end);
    return Y26_CONV_STATUS_SUCCESS;
}

Summary summarize_repeats(const std::vector<Timing>& repeats, std::size_t mismatches, long long checksum, int status) {
    Summary out {};
    out.mismatches = mismatches;
    out.checksum = checksum;
    out.status = status;
    if (repeats.empty()) {
        return out;
    }
    std::vector<double> totals;
    totals.reserve(repeats.size());
    for (const Timing& timing : repeats) {
        totals.push_back(timing.total_us);
        out.mean_conv_us += timing.conv_us;
        out.mean_activation_requant_us += timing.activation_requant_us;
        out.mean_merge_us += timing.post_concat_qdq_us;
        out.mean_thread_overhead_us += timing.thread_overhead_us;
        out.mean_branch0_conv_us += timing.branch0_conv_us;
        out.mean_branch1_conv_us += timing.branch1_conv_us;
        out.mean_model4_cv2_conv_us += timing.model4_cv2_conv_us;
        out.mean_correction_us += timing.correction_us;
    }
    const double denom = static_cast<double>(repeats.size());
    out.mean_total_us = std::accumulate(totals.begin(), totals.end(), 0.0) / denom;
    out.mean_conv_us /= denom;
    out.mean_activation_requant_us /= denom;
    out.mean_merge_us /= denom;
    out.mean_thread_overhead_us /= denom;
    out.mean_branch0_conv_us /= denom;
    out.mean_branch1_conv_us /= denom;
    out.mean_model4_cv2_conv_us /= denom;
    out.mean_correction_us /= denom;
    out.min_total_us = *std::min_element(totals.begin(), totals.end());
    out.max_total_us = *std::max_element(totals.begin(), totals.end());
    double variance = 0.0;
    for (double value : totals) {
        const double diff = value - out.mean_total_us;
        variance += diff * diff;
    }
    out.stddev_total_us = std::sqrt(variance / denom);
    out.cv_total_pct = out.mean_total_us > 0.0 ? 100.0 * out.stddev_total_us / out.mean_total_us : 0.0;
    return out;
}

Summary run_candidate(const char* candidate,
                      int thread_count,
                      int activation_mode,
                      bool use_ime,
                      bool use_threaded_branch0,
                      bool use_split0_concat_lut,
                      const Protocol& protocol,
                      const std::vector<std::int32_t>& expected) {
    Context ctx(y26_stage16_model4_c2f_fixture::kSyntheticSeededFixture);
    if (!prepare(ctx, use_threaded_branch0 ? thread_count : 0)) {
        Summary failed {};
        failed.status = Y26_CONV_STATUS_INVALID_ARGUMENT;
        return failed;
    }
    if (use_ime || use_threaded_branch0) {
        (void)y26_k1x_ime_probe_once();
    }
    Timing ignored {};
    for (int i = 0; i < protocol.warmup; ++i) {
        const int status = run_once(ctx, activation_mode, use_ime, use_threaded_branch0, use_split0_concat_lut, ignored);
        if (status != Y26_CONV_STATUS_SUCCESS) {
            destroy(ctx);
            Summary failed {};
            failed.status = status;
            return failed;
        }
    }
    std::vector<Timing> repeats;
    repeats.reserve(protocol.repeats);
    std::size_t mismatches = 0;
    int status = Y26_CONV_STATUS_SUCCESS;
    for (int repeat = 0; repeat < protocol.repeats; ++repeat) {
        Timing sum {};
        for (int run = 0; run < protocol.runs; ++run) {
            Timing timing {};
            status = run_once(ctx, activation_mode, use_ime, use_threaded_branch0, use_split0_concat_lut, timing);
            if (status != Y26_CONV_STATUS_SUCCESS) {
                break;
            }
            sum.conv_us += timing.conv_us;
            sum.activation_requant_us += timing.activation_requant_us;
            sum.split_us += timing.split_us;
            sum.add_us += timing.add_us;
            sum.concat_us += timing.concat_us;
            sum.post_concat_qdq_us += timing.post_concat_qdq_us;
            sum.pack_layout_us += timing.pack_layout_us;
            sum.thread_overhead_us += timing.thread_overhead_us;
            sum.correction_us += timing.correction_us;
            sum.total_us += timing.total_us;
            sum.branch0_conv_us += timing.branch0_conv_us;
            sum.branch1_conv_us += timing.branch1_conv_us;
            sum.model4_cv2_conv_us += timing.model4_cv2_conv_us;
        }
        const double denom = static_cast<double>(protocol.runs);
        sum.conv_us /= denom;
        sum.activation_requant_us /= denom;
        sum.split_us /= denom;
        sum.add_us /= denom;
        sum.concat_us /= denom;
        sum.post_concat_qdq_us /= denom;
        sum.pack_layout_us /= denom;
        sum.thread_overhead_us /= denom;
        sum.correction_us /= denom;
        sum.total_us /= denom;
        sum.branch0_conv_us /= denom;
        sum.branch1_conv_us /= denom;
        sum.model4_cv2_conv_us /= denom;
        repeats.push_back(sum);
        mismatches += mismatches_i32(ctx.output_i32, expected);
        if (status != Y26_CONV_STATUS_SUCCESS) {
            break;
        }
    }
    Summary summary = summarize_repeats(repeats, mismatches, checksum_i32(ctx.output_i32), status);
    summary.affinity_ok = !use_threaded_branch0 || ctx.branch0_threaded == nullptr
                              ? 1
                              : y26_threaded_conv_worker_affinity_ok(ctx.branch0_threaded);
    std::cout << "candidate=" << candidate
              << " shape_class=representative_full_shape_model4_c2f_synthetic"
              << " thread_count=" << thread_count
              << " warmup=" << protocol.warmup
              << " runs=" << protocol.runs
              << " repeats=" << protocol.repeats
              << " status=" << summary.status
              << " mismatches=" << summary.mismatches
              << " checksum=" << summary.checksum
              << " affinity_ok=" << summary.affinity_ok
              << " mean_total_us=" << summary.mean_total_us
              << " stddev_total_us=" << summary.stddev_total_us
              << " min_total_us=" << summary.min_total_us
              << " max_total_us=" << summary.max_total_us
              << " cv_total_pct=" << summary.cv_total_pct
              << " mean_conv_us=" << summary.mean_conv_us
              << " mean_activation_requant_us=" << summary.mean_activation_requant_us
              << " mean_merge_us=" << summary.mean_merge_us
              << " mean_thread_overhead_us=" << summary.mean_thread_overhead_us
              << " mean_branch0_conv_us=" << summary.mean_branch0_conv_us
              << " mean_branch1_conv_us=" << summary.mean_branch1_conv_us
              << " mean_model4_cv2_conv_us=" << summary.mean_model4_cv2_conv_us
              << " mean_correction_us=" << summary.mean_correction_us
              << " conv_share_pct="
              << (summary.mean_total_us > 0.0 ? 100.0 * summary.mean_conv_us / summary.mean_total_us : 0.0)
              << " activation_share_pct="
              << (summary.mean_total_us > 0.0 ? 100.0 * summary.mean_activation_requant_us / summary.mean_total_us : 0.0)
              << " merge_share_pct="
              << (summary.mean_total_us > 0.0 ? 100.0 * summary.mean_merge_us / summary.mean_total_us : 0.0)
              << "\n";
    destroy(ctx);
    return summary;
}

std::vector<std::int32_t> make_scalar_reference() {
    Context ctx(y26_stage16_model4_c2f_fixture::kSyntheticSeededFixture);
    if (!prepare(ctx, 0)) {
        return {};
    }
    Timing timing {};
    const int status = run_once(ctx, Y26_ACTIVATION_MODE_INT8_LUT, false, false, false, timing);
    std::vector<std::int32_t> expected = status == Y26_CONV_STATUS_SUCCESS ? ctx.output_i32 : std::vector<std::int32_t> {};
    std::cout << "reference=scalar_int8_lut status=" << status
              << " checksum=" << checksum_i32(expected)
              << " total_us=" << timing.total_us
              << " conv_us=" << timing.conv_us
              << " activation_requant_us=" << timing.activation_requant_us
              << " merge_us=" << timing.post_concat_qdq_us
              << "\n";
    destroy(ctx);
    return expected;
}

}  // namespace

int main(int argc, char** argv) {
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
    std::cout << "subset=candidate_K_model4_c2f_representative_fullshape_synthetic"
              << " h=" << kFullH
              << " w=" << kFullW
              << " model4_cv1_c=" << kModel4Cv1C
              << " note=selected-subset-microbench-not-model-fps"
              << "\n";
    std::vector<std::int32_t> expected = make_scalar_reference();
    if (expected.empty()) {
        return 1;
    }
    if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
        std::cout << "candidate=B0_ime_single_thread_rvv status=not_built\n";
        return 0;
    }
    int failures = 0;
    Summary b0 = run_candidate("B0_ime_single_thread_rvv",
                               0,
                               Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT,
                               true,
                               false,
                               false,
                               protocol,
                               expected);
    failures += b0.status == Y26_CONV_STATUS_SUCCESS && b0.mismatches == 0 ? 0 : 1;
    for (int threads = 1; threads <= 4; ++threads) {
        const char* label = threads == 1   ? "B1_threaded_branch0_1t"
                            : threads == 2 ? "B1_threaded_branch0_2t"
                            : threads == 3 ? "B1_threaded_branch0_3t"
                                           : "B1_threaded_branch0_4t";
        Summary threaded = run_candidate(label,
                                         threads,
                                         Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT,
                                         true,
                                         true,
                                         false,
                                         protocol,
                                         expected);
        failures += threaded.status == Y26_CONV_STATUS_SUCCESS && threaded.mismatches == 0 &&
                            threaded.affinity_ok == 1
                        ? 0
                        : 1;
    }
    Summary repaired = run_candidate("C2_split0_concat_lut_4t",
                                     4,
                                     Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT,
                                     true,
                                     true,
                                     true,
                                     protocol,
                                     expected);
    failures += repaired.status == Y26_CONV_STATUS_SUCCESS && repaired.mismatches == 0 && repaired.affinity_ok == 1 ? 0 : 1;
    return failures == 0 ? 0 : 1;
}
