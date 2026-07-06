#include "y26_k1x_model4_c2f_runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kStage16Alignment = 64;

bool kernel_supported(int kernel_h, int kernel_w) {
    return (kernel_h == 1 && kernel_w == 1) || (kernel_h == 3 && kernel_w == 3);
}

bool conv_params_valid(const Y26Conv2DParams& params) {
    return params.input_h > 0 && params.input_w > 0 && params.input_c > 0 && params.output_c > 0 &&
           params.stride_h > 0 && params.stride_w > 0 && params.pad_h >= 0 && params.pad_w >= 0;
}

int output_h_for_kernel(const Y26Conv2DParams& params, int kernel_h) {
    return kernel_h == 1 ? y26_conv1x1_output_h(&params) : y26_conv3x3_output_h(&params);
}

int output_w_for_kernel(const Y26Conv2DParams& params, int kernel_w) {
    return kernel_w == 1 ? y26_conv1x1_output_w(&params) : y26_conv3x3_output_w(&params);
}

std::size_t output_count_for_kernel(const Y26Conv2DParams& params, int kernel_h, int kernel_w) {
    const int output_h = output_h_for_kernel(params, kernel_h);
    const int output_w = output_w_for_kernel(params, kernel_w);
    if (output_h <= 0 || output_w <= 0) {
        return 0;
    }
    return static_cast<std::size_t>(output_h) * static_cast<std::size_t>(output_w) *
           static_cast<std::size_t>(params.output_c);
}

std::size_t expected_weight_count(const Y26Stage7ConvNodeConfig& cfg) {
    return static_cast<std::size_t>(cfg.params.output_c) * static_cast<std::size_t>(cfg.kernel_h) *
           static_cast<std::size_t>(cfg.kernel_w) * static_cast<std::size_t>(cfg.params.input_c);
}

bool conv_config_valid(const Y26Stage7ConvNodeConfig& cfg) {
    if (!conv_params_valid(cfg.params) || !kernel_supported(cfg.kernel_h, cfg.kernel_w) ||
        cfg.activation_zero_point_u8 < 0 || cfg.activation_zero_point_u8 > 255 ||
        cfg.input_storage_zero_point_s8 < static_cast<int>(std::numeric_limits<std::int8_t>::min()) ||
        cfg.input_storage_zero_point_s8 > static_cast<int>(std::numeric_limits<std::int8_t>::max()) ||
        cfg.output_zero_point_u8 < 0 || cfg.output_zero_point_u8 > 255 || cfg.input_scale <= 0.0f ||
        cfg.output_scale <= 0.0f || cfg.weight_scales == nullptr || cfg.weights_ohwi_s8 == nullptr ||
        cfg.bias_i32 == nullptr) {
        return false;
    }
    if (cfg.weight_scale_count < static_cast<std::size_t>(cfg.params.output_c) ||
        cfg.weight_count < expected_weight_count(cfg) ||
        cfg.bias_count < static_cast<std::size_t>(cfg.params.output_c)) {
        return false;
    }
    return output_count_for_kernel(cfg.params, cfg.kernel_h, cfg.kernel_w) > 0;
}

bool activation_mode_valid(int mode) {
    return mode == Y26_ACTIVATION_MODE_SCALAR_FLOAT_REFERENCE ||
           mode == Y26_ACTIVATION_MODE_FIXED_REQUANT_ONLY ||
           mode == Y26_ACTIVATION_MODE_INT8_LUT ||
           mode == Y26_ACTIVATION_MODE_STAGE9_SCALAR_UNROLLED_LUT ||
           mode == Y26_ACTIVATION_MODE_STAGE9_FIXED_REQUANT_LUT ||
           mode == Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT ||
           mode == Y26_ACTIVATION_MODE_STAGE9_FUSED_CURRENT_LAYOUT;
}

bool merge_mode_valid(int mode) {
    return mode == Y26_STAGE16_MERGE_MODE_A0_MATERIALIZED_FLOAT ||
           mode == Y26_STAGE16_MERGE_MODE_A2_FUSED_QDQ_NHWC ||
           mode == Y26_STAGE16_MERGE_MODE_C2_SPLIT0_CONCAT_LUT;
}

bool config_valid(const Y26Stage16Model4C2fConfig* cfg) {
    if (cfg == nullptr || y26_stage15_model4_branch_output_count(&cfg->stage15) == 0 ||
        !conv_config_valid(cfg->branch1) || !conv_config_valid(cfg->model4_cv2) ||
        cfg->concat_output_scale <= 0.0f || cfg->concat_output_zero_point_u8 < 0 ||
        cfg->concat_output_zero_point_u8 > 255 || !activation_mode_valid(cfg->activation_mode) ||
        !merge_mode_valid(cfg->merge_mode)) {
        return false;
    }
    const Y26Stage7ConvNodeConfig& branch0 = cfg->stage15.branch0;
    const int branch0_h = output_h_for_kernel(branch0.params, branch0.kernel_h);
    const int branch0_w = output_w_for_kernel(branch0.params, branch0.kernel_w);
    const Y26Stage7ConvNodeConfig& model4_cv1 = cfg->stage15.stage14.model4_cv1;
    const int model4_cv1_h = output_h_for_kernel(model4_cv1.params, model4_cv1.kernel_h);
    const int model4_cv1_w = output_w_for_kernel(model4_cv1.params, model4_cv1.kernel_w);
    const int split_channels = model4_cv1.params.output_c / 2;
    if (model4_cv1.params.output_c % 2 != 0 || split_channels <= 0) {
        return false;
    }
    return cfg->branch1.params.input_h == branch0_h && cfg->branch1.params.input_w == branch0_w &&
           cfg->branch1.params.input_c == branch0.params.output_c &&
           cfg->branch1.activation_zero_point_u8 == cfg->stage15.branch0_act_output_zero_point_u8 &&
           cfg->branch1.input_storage_zero_point_s8 == cfg->stage15.branch0_act_output_zero_point_u8 - 128 &&
           cfg->branch1.input_scale == cfg->stage15.branch0_act_output_scale &&
           cfg->branch1.params.output_c == split_channels && cfg->model4_cv2.kernel_h == 1 &&
           cfg->model4_cv2.kernel_w == 1 && cfg->model4_cv2.params.input_h == model4_cv1_h &&
           cfg->model4_cv2.params.input_w == model4_cv1_w &&
           cfg->model4_cv2.params.input_c == split_channels * 3 &&
           cfg->model4_cv2.activation_zero_point_u8 == cfg->concat_output_zero_point_u8 &&
           cfg->model4_cv2.input_storage_zero_point_s8 == cfg->concat_output_zero_point_u8 - 128 &&
           cfg->model4_cv2.input_scale == cfg->concat_output_scale;
}

void* allocate_aligned(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    return ::operator new(bytes, std::align_val_t(kStage16Alignment));
}

void free_aligned(void* ptr) {
    ::operator delete(ptr, std::align_val_t(kStage16Alignment));
}

std::int32_t* allocate_i32(std::size_t count) {
    return static_cast<std::int32_t*>(allocate_aligned(count * sizeof(std::int32_t)));
}

std::int8_t* allocate_i8(std::size_t count) {
    return static_cast<std::int8_t*>(allocate_aligned(count));
}

float* allocate_f32(std::size_t count) {
    return static_cast<float*>(allocate_aligned(count * sizeof(float)));
}

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000.0;
}

void timing_reset(Y26Stage16TimingUs* timing) {
    if (timing != nullptr) {
        std::memset(timing, 0, sizeof(*timing));
    }
}

float silu_f32(float value) {
    return value / (1.0f + std::exp(-value));
}

std::int8_t signed_storage_from_u8(std::uint8_t value) {
    return static_cast<std::int8_t>(static_cast<int>(value) - 128);
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

std::int8_t quantize_concat_s8(float value, float scale, int zero_point_u8) {
    const std::uint8_t code = y26_quantize_u8_nearest_even_f32(value, scale, zero_point_u8);
    return signed_storage_from_u8(code);
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

int apply_activation_lut_mode(int activation_mode,
                              const Y26ActivationRequantParams& params,
                              const std::int8_t* lut_s8,
                              const std::int32_t* producer_i32,
                              std::int8_t* consumer_input_s8) {
    switch (activation_mode) {
        case Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT:
            return y26_activation_requant_silu_int8_lut_rvv_f32(&params, producer_i32, lut_s8, consumer_input_s8);
        case Y26_ACTIVATION_MODE_STAGE9_SCALAR_UNROLLED_LUT:
        case Y26_ACTIVATION_MODE_STAGE9_FUSED_CURRENT_LAYOUT:
            return y26_activation_requant_silu_int8_lut_scalar_unrolled(
                &params, producer_i32, lut_s8, consumer_input_s8);
        case Y26_ACTIVATION_MODE_INT8_LUT:
        default:
            return y26_activation_requant_silu_int8_lut(&params, producer_i32, lut_s8, consumer_input_s8);
    }
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

int run_conv_scalar(const Y26Stage7ConvNodeConfig& cfg,
                    const Y26PrepackedConvWeights* weights,
                    const std::int8_t* input_s8,
                    std::int32_t* raw_i32,
                    std::int32_t* output_i32,
                    double* conv_us,
                    double* correction_us) {
    const auto begin = Clock::now();
    int status = scalar_raw_dot(cfg, input_s8, raw_i32);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    const auto correction_begin = Clock::now();
    status = apply_correction(cfg, weights, raw_i32, output_i32);
    const auto end = Clock::now();
    if (conv_us != nullptr) {
        *conv_us = elapsed_us(begin, end);
    }
    if (correction_us != nullptr) {
        *correction_us = elapsed_us(correction_begin, end);
    }
    return status;
}

int run_conv_ime(const Y26Stage7ConvNodeConfig& cfg,
                 const Y26PrepackedConvWeights* weights,
                 Y26ConvWorkspace* workspace,
                 const std::int8_t* input_s8,
                 std::int32_t* raw_i32,
                 std::int32_t* output_i32,
                 double* conv_us,
                 double* correction_us) {
    const auto begin = Clock::now();
    int status = y26_conv2d_i8s8s32_nhwc_ime_prepacked_v1(input_s8,
                                                          weights,
                                                          raw_i32,
                                                          cfg.input_storage_zero_point_s8,
                                                          workspace,
                                                          Y26_CONV_LOOP_ORDER_M_MAJOR);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    const auto correction_begin = Clock::now();
    status = apply_correction(cfg, weights, raw_i32, output_i32);
    const auto end = Clock::now();
    if (conv_us != nullptr) {
        *conv_us = elapsed_us(begin, end);
    }
    if (correction_us != nullptr) {
        *correction_us = elapsed_us(correction_begin, end);
    }
    return status;
}

void accumulate_stage15_timing(Y26Stage16TimingUs& dst, const Y26Stage15TimingUs& src) {
    dst.stage15_timing_us = src;
    dst.conv_us += src.conv_us;
    dst.activation_requant_us += src.activation_requant_us;
    dst.split_us += src.split_us;
    dst.merge_us += src.merge_us;
    dst.post_qdq_us += src.post_qdq_us;
    dst.pack_layout_us += src.pack_layout_us;
    dst.correction_us += src.correction_us;
    dst.thread_overhead_us += src.thread_overhead_us;
}

void finalize_timing(Y26Stage16TimingUs& timing) {
    if (timing.total_us <= 0.0) {
        return;
    }
    timing.activation_share_pct = 100.0 * timing.activation_requant_us / timing.total_us;
    timing.conv_share_pct = 100.0 * timing.conv_us / timing.total_us;
    timing.merge_share_pct = 100.0 * timing.merge_us / timing.total_us;
    timing.pack_layout_share_pct = 100.0 * timing.pack_layout_us / timing.total_us;
}

int build_concat_qdq_nhwc(const Y26Stage16Model4C2fConfig& cfg,
                          const Y26Stage16Model4C2fWorkspace& ws,
                          Y26Stage16TimingUs* timing) {
    const Y26Stage7ConvNodeConfig& model4_cv1 = cfg.stage15.stage14.model4_cv1;
    const int h = output_h_for_kernel(model4_cv1.params, model4_cv1.kernel_h);
    const int w = output_w_for_kernel(model4_cv1.params, model4_cv1.kernel_w);
    const int split_c = model4_cv1.params.output_c / 2;
    const int spatial = h * w;
    const std::int32_t* model4_cv1_i32 = y26_stage15_model4_branch_model4_cv1_i32(&ws.stage15_ws);
    const std::int8_t* split1_s8 = y26_stage15_model4_branch_split1_input_s8(&ws.stage15_ws);
    if (model4_cv1_i32 == nullptr || split1_s8 == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    const bool use_split0_concat_lut = cfg.merge_mode == Y26_STAGE16_MERGE_MODE_C2_SPLIT0_CONCAT_LUT;
    if (use_split0_concat_lut && ws.split0_concat_s8 == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }

    const auto begin = Clock::now();
    for (int m = 0; m < spatial; ++m) {
        std::int8_t* dst = ws.concat_s8 + m * split_c * 3;
        for (int c = 0; c < split_c; ++c) {
            if (use_split0_concat_lut) {
                dst[c] = ws.split0_concat_s8[m * split_c * 2 + c];
            } else {
                const float value = accumulator_to_silu_float(model4_cv1, model4_cv1_i32[m * split_c * 2 + c], c);
                dst[c] = quantize_concat_s8(value, cfg.concat_output_scale, cfg.concat_output_zero_point_u8);
            }
        }
        for (int c = 0; c < split_c; ++c) {
            const float split1_value =
                dequant_signed_storage(split1_s8[m * split_c + c],
                                       cfg.stage15.split1_output_scale,
                                       cfg.stage15.split1_output_zero_point_u8);
            dst[split_c + c] = quantize_concat_s8(split1_value,
                                                  cfg.concat_output_scale,
                                                  cfg.concat_output_zero_point_u8);
        }
        for (int c = 0; c < split_c; ++c) {
            const float split1_value =
                dequant_signed_storage(split1_s8[m * split_c + c],
                                       cfg.stage15.split1_output_scale,
                                       cfg.stage15.split1_output_zero_point_u8);
            const float add_value = split1_value + ws.branch1_act_f32[m * split_c + c];
            dst[split_c * 2 + c] =
                quantize_concat_s8(add_value, cfg.concat_output_scale, cfg.concat_output_zero_point_u8);
        }
    }
    const auto end = Clock::now();
    if (timing != nullptr) {
        const double qdq_us = elapsed_us(begin, end);
        timing->add_us += qdq_us;
        timing->concat_us += qdq_us;
        timing->post_qdq_us += qdq_us;
        timing->merge_us += qdq_us;
    }
    return Y26_CONV_STATUS_SUCCESS;
}

int build_split0_concat_lut_activation(const Y26Stage16Model4C2fConfig& cfg,
                                       Y26Stage16Model4C2fWorkspace& ws,
                                       Y26Stage16TimingUs* timing) {
    if (cfg.merge_mode != Y26_STAGE16_MERGE_MODE_C2_SPLIT0_CONCAT_LUT) {
        return Y26_CONV_STATUS_SUCCESS;
    }
    const std::int32_t* model4_cv1_i32 = y26_stage15_model4_branch_model4_cv1_i32(&ws.stage15_ws);
    if (model4_cv1_i32 == nullptr || ws.split0_concat_s8 == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    const auto begin = Clock::now();
    const Y26Stage7ConvNodeConfig& model4_cv1 = cfg.stage15.stage14.model4_cv1;
    const Y26ActivationRequantParams concat_params = activation_params(model4_cv1,
                                                                       ws.stage15_ws.model4_cv1_output_count,
                                                                       cfg.concat_output_scale,
                                                                       cfg.concat_output_zero_point_u8);
    const int status = apply_activation_lut_mode(cfg.activation_mode,
                                                 concat_params,
                                                 ws.model4_cv1_to_concat_lut_s8,
                                                 model4_cv1_i32,
                                                 ws.split0_concat_s8);
    const auto end = Clock::now();
    if (timing != nullptr) {
        timing->activation_requant_us += elapsed_us(begin, end);
    }
    return status;
}

void build_branch1_activation_float(const Y26Stage16Model4C2fConfig& cfg,
                                    const Y26Stage16Model4C2fWorkspace& ws) {
    const int channels = cfg.branch1.params.output_c;
    for (std::size_t i = 0; i < ws.branch1_output_count; ++i) {
        const int c = static_cast<int>(i % static_cast<std::size_t>(channels));
        ws.branch1_act_f32[i] = accumulator_to_silu_float(cfg.branch1, ws.branch1_i32[i], c);
    }
}

int run_after_stage15(const Y26Stage16Model4C2fConfig& cfg,
                      Y26Stage16Model4C2fWorkspace& ws,
                      std::int32_t* output_i32_nhwc,
                      Y26Stage16TimingUs* timing,
                      bool use_ime) {
    double branch1_conv_us = 0.0;
    double branch1_correction_us = 0.0;
    int status = use_ime ? run_conv_ime(cfg.branch1,
                                        ws.branch1_weights,
                                        ws.branch1_workspace,
                                        y26_stage15_model4_branch_branch0_act_s8(&ws.stage15_ws),
                                        ws.branch1_raw_i32,
                                        ws.branch1_i32,
                                        &branch1_conv_us,
                                        &branch1_correction_us)
                         : run_conv_scalar(cfg.branch1,
                                           ws.branch1_weights,
                                           y26_stage15_model4_branch_branch0_act_s8(&ws.stage15_ws),
                                           ws.branch1_raw_i32,
                                           ws.branch1_i32,
                                           &branch1_conv_us,
                                           &branch1_correction_us);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }

    const auto branch1_act_begin = Clock::now();
    build_branch1_activation_float(cfg, ws);
    const auto branch1_act_end = Clock::now();

    status = build_split0_concat_lut_activation(cfg, ws, timing);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }

    status = build_concat_qdq_nhwc(cfg, ws, timing);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }

    double model4_cv2_conv_us = 0.0;
    double model4_cv2_correction_us = 0.0;
    status = use_ime ? run_conv_ime(cfg.model4_cv2,
                                    ws.model4_cv2_weights,
                                    ws.model4_cv2_workspace,
                                    ws.concat_s8,
                                    ws.model4_cv2_raw_i32,
                                    output_i32_nhwc,
                                    &model4_cv2_conv_us,
                                    &model4_cv2_correction_us)
                     : run_conv_scalar(cfg.model4_cv2,
                                       ws.model4_cv2_weights,
                                       ws.concat_s8,
                                       ws.model4_cv2_raw_i32,
                                       output_i32_nhwc,
                                       &model4_cv2_conv_us,
                                       &model4_cv2_correction_us);

    if (timing != nullptr) {
        const double branch1_activation_us = elapsed_us(branch1_act_begin, branch1_act_end);
        timing->conv_us += branch1_conv_us + model4_cv2_conv_us;
        timing->branch1_conv_us += branch1_conv_us;
        timing->model4_cv2_conv_us += model4_cv2_conv_us;
        timing->correction_us += branch1_correction_us + model4_cv2_correction_us;
        timing->branch1_correction_us += branch1_correction_us;
        timing->model4_cv2_correction_us += model4_cv2_correction_us;
        timing->activation_requant_us += branch1_activation_us;
        timing->branch1_activation_us += branch1_activation_us;
    }
    return status;
}

}  // namespace

extern "C" int y26_stage16_model4_c2f_prepare(const Y26Stage16Model4C2fConfig* cfg,
                                               Y26Stage16Model4C2fWorkspace* ws) {
    if (!config_valid(cfg) || ws == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    std::memset(ws, 0, sizeof(*ws));
    int status = y26_stage15_model4_branch_prepare(&cfg->stage15, &ws->stage15_ws);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    ws->branch1_weights = y26_prepacked_conv_weights_create_mmt4d_s8(cfg->branch1.weights_ohwi_s8,
                                                                     &cfg->branch1.params,
                                                                     cfg->branch1.kernel_h,
                                                                     cfg->branch1.kernel_w,
                                                                     cfg->branch1.node_name,
                                                                     cfg->branch1.weight_scales);
    ws->model4_cv2_weights = y26_prepacked_conv_weights_create_mmt4d_s8(cfg->model4_cv2.weights_ohwi_s8,
                                                                        &cfg->model4_cv2.params,
                                                                        cfg->model4_cv2.kernel_h,
                                                                        cfg->model4_cv2.kernel_w,
                                                                        cfg->model4_cv2.node_name,
                                                                        cfg->model4_cv2.weight_scales);
    ws->branch1_workspace =
        y26_conv_workspace_create(&cfg->branch1.params, cfg->branch1.kernel_h, cfg->branch1.kernel_w);
    ws->model4_cv2_workspace =
        y26_conv_workspace_create(&cfg->model4_cv2.params, cfg->model4_cv2.kernel_h, cfg->model4_cv2.kernel_w);
    ws->stage15_output_count = y26_stage15_model4_branch_output_count(&cfg->stage15);
    ws->branch1_output_count = output_count_for_kernel(cfg->branch1.params, cfg->branch1.kernel_h, cfg->branch1.kernel_w);
    const int spatial = output_h_for_kernel(cfg->branch1.params, cfg->branch1.kernel_h) *
                        output_w_for_kernel(cfg->branch1.params, cfg->branch1.kernel_w);
    ws->concat_count = static_cast<std::size_t>(spatial) * static_cast<std::size_t>(cfg->model4_cv2.params.input_c);
    ws->model4_cv2_output_count = y26_stage16_model4_c2f_output_count(cfg);
    ws->stage15_output_i32 = allocate_i32(ws->stage15_output_count);
    ws->branch1_raw_i32 = allocate_i32(ws->branch1_output_count);
    ws->branch1_i32 = allocate_i32(ws->branch1_output_count);
    ws->branch1_act_f32 = allocate_f32(ws->branch1_output_count);
    ws->split0_concat_s8 = allocate_i8(ws->stage15_ws.model4_cv1_output_count);
    ws->concat_s8 = allocate_i8(ws->concat_count);
    ws->model4_cv2_raw_i32 = allocate_i32(ws->model4_cv2_output_count);

    if (ws->branch1_weights == nullptr || ws->model4_cv2_weights == nullptr || ws->branch1_workspace == nullptr ||
        ws->model4_cv2_workspace == nullptr || ws->stage15_output_i32 == nullptr || ws->branch1_raw_i32 == nullptr ||
        ws->branch1_i32 == nullptr || ws->branch1_act_f32 == nullptr || ws->split0_concat_s8 == nullptr ||
        ws->concat_s8 == nullptr || ws->model4_cv2_raw_i32 == nullptr) {
        y26_stage16_model4_c2f_release(ws);
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    const Y26Stage7ConvNodeConfig& model4_cv1 = cfg->stage15.stage14.model4_cv1;
    if (y26_build_silu_u8_to_s8_lut(model4_cv1.output_scale,
                                    model4_cv1.output_zero_point_u8,
                                    cfg->concat_output_scale,
                                    cfg->concat_output_zero_point_u8,
                                    ws->model4_cv1_to_concat_lut_s8) != Y26_CONV_STATUS_SUCCESS) {
        y26_stage16_model4_c2f_release(ws);
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    ws->prepacked_bytes = y26_prepacked_conv_weights_total_bytes(ws->branch1_weights) +
                          y26_prepacked_conv_weights_total_bytes(ws->model4_cv2_weights);
    ws->workspace_bytes = y26_conv_workspace_bytes(ws->branch1_workspace) +
                          y26_conv_workspace_bytes(ws->model4_cv2_workspace);
    ws->prepared = 1;
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" int y26_stage16_model4_c2f_prepare_threaded_branch0(const Y26Stage16Model4C2fConfig* cfg,
                                                                Y26Stage16Model4C2fWorkspace* ws,
                                                                int thread_count) {
    if (!config_valid(cfg) || ws == nullptr || ws->prepared != 1) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    return y26_stage15_model4_branch_prepare_threaded_conv(&cfg->stage15, &ws->stage15_ws, thread_count);
}

extern "C" void y26_stage16_model4_c2f_release(Y26Stage16Model4C2fWorkspace* ws) {
    if (ws == nullptr) {
        return;
    }
    y26_stage15_model4_branch_release(&ws->stage15_ws);
    y26_prepacked_conv_weights_destroy(ws->branch1_weights);
    y26_prepacked_conv_weights_destroy(ws->model4_cv2_weights);
    y26_conv_workspace_destroy(ws->branch1_workspace);
    y26_conv_workspace_destroy(ws->model4_cv2_workspace);
    free_aligned(ws->stage15_output_i32);
    free_aligned(ws->branch1_raw_i32);
    free_aligned(ws->branch1_i32);
    free_aligned(ws->branch1_act_f32);
    free_aligned(ws->split0_concat_s8);
    free_aligned(ws->concat_s8);
    free_aligned(ws->model4_cv2_raw_i32);
    std::memset(ws, 0, sizeof(*ws));
}

extern "C" std::size_t y26_stage16_model4_c2f_output_count(const Y26Stage16Model4C2fConfig* cfg) {
    if (cfg == nullptr || !conv_params_valid(cfg->model4_cv2.params) ||
        !kernel_supported(cfg->model4_cv2.kernel_h, cfg->model4_cv2.kernel_w)) {
        return 0;
    }
    return output_count_for_kernel(cfg->model4_cv2.params, cfg->model4_cv2.kernel_h, cfg->model4_cv2.kernel_w);
}

extern "C" int y26_stage16_model4_c2f_run_scalar(const Y26Stage16Model4C2fConfig* cfg,
                                                  Y26Stage16Model4C2fWorkspace* ws,
                                                  const std::int8_t* input_nhwc_s8,
                                                  std::int32_t* output_i32_nhwc,
                                                  Y26Stage16TimingUs* timing) {
    if (!config_valid(cfg) || ws == nullptr || ws->prepared != 1 || input_nhwc_s8 == nullptr ||
        output_i32_nhwc == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    timing_reset(timing);
    const auto begin = Clock::now();
    Y26Stage15TimingUs stage15_timing {};
    int status = y26_stage15_model4_branch_run_scalar(
        &cfg->stage15, &ws->stage15_ws, input_nhwc_s8, ws->stage15_output_i32, &stage15_timing);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    if (timing != nullptr) {
        accumulate_stage15_timing(*timing, stage15_timing);
    }
    status = run_after_stage15(*cfg, *ws, output_i32_nhwc, timing, false);
    const auto end = Clock::now();
    if (timing != nullptr) {
        timing->total_us = elapsed_us(begin, end);
        finalize_timing(*timing);
    }
    return status;
}

extern "C" int y26_stage16_model4_c2f_run_ime_cluster0_hotpath(const Y26Stage16Model4C2fConfig* cfg,
                                                                Y26Stage16Model4C2fWorkspace* ws,
                                                                const std::int8_t* input_nhwc_s8,
                                                                std::int32_t* output_i32_nhwc,
                                                                Y26Stage16TimingUs* timing) {
    if (!config_valid(cfg) || ws == nullptr || ws->prepared != 1 || input_nhwc_s8 == nullptr ||
        output_i32_nhwc == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    timing_reset(timing);
    const auto begin = Clock::now();
    Y26Stage15TimingUs stage15_timing {};
    int status = y26_stage15_model4_branch_run_ime_cluster0_hotpath(
        &cfg->stage15, &ws->stage15_ws, input_nhwc_s8, ws->stage15_output_i32, &stage15_timing);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    if (timing != nullptr) {
        accumulate_stage15_timing(*timing, stage15_timing);
    }
    status = run_after_stage15(*cfg, *ws, output_i32_nhwc, timing, true);
    const auto end = Clock::now();
    if (timing != nullptr) {
        timing->total_us = elapsed_us(begin, end);
        finalize_timing(*timing);
    }
    return status;
}

extern "C" int y26_stage16_model4_c2f_run_ime_threaded_branch0_cluster0_hotpath(
    const Y26Stage16Model4C2fConfig* cfg,
    Y26Stage16Model4C2fWorkspace* ws,
    const std::int8_t* input_nhwc_s8,
    std::int32_t* output_i32_nhwc,
    int thread_activation,
    Y26Stage16TimingUs* timing) {
    if (!config_valid(cfg) || ws == nullptr || ws->prepared != 1 || input_nhwc_s8 == nullptr ||
        output_i32_nhwc == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    timing_reset(timing);
    const auto begin = Clock::now();
    Y26Stage15TimingUs stage15_timing {};
    int status = y26_stage15_model4_branch_run_ime_threaded_conv_cluster0_hotpath(
        &cfg->stage15, &ws->stage15_ws, input_nhwc_s8, ws->stage15_output_i32, thread_activation, &stage15_timing);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    if (timing != nullptr) {
        accumulate_stage15_timing(*timing, stage15_timing);
    }
    status = run_after_stage15(*cfg, *ws, output_i32_nhwc, timing, true);
    const auto end = Clock::now();
    if (timing != nullptr) {
        timing->total_us = elapsed_us(begin, end);
        finalize_timing(*timing);
    }
    return status;
}

extern "C" int y26_stage16_model4_c2f_threaded_worker_affinity_ok(const Y26Stage16Model4C2fWorkspace* ws) {
    return ws != nullptr ? y26_stage15_model4_branch_threaded_worker_affinity_ok(&ws->stage15_ws) : 0;
}

extern "C" int y26_stage16_model4_c2f_threaded_thread_count(const Y26Stage16Model4C2fWorkspace* ws) {
    return ws != nullptr ? y26_stage15_model4_branch_threaded_thread_count(&ws->stage15_ws) : 0;
}

extern "C" const std::int8_t* y26_stage16_model4_c2f_concat_s8(const Y26Stage16Model4C2fWorkspace* ws) {
    return ws != nullptr ? ws->concat_s8 : nullptr;
}

extern "C" const std::int32_t* y26_stage16_model4_c2f_branch1_i32(
    const Y26Stage16Model4C2fWorkspace* ws) {
    return ws != nullptr ? ws->branch1_i32 : nullptr;
}
