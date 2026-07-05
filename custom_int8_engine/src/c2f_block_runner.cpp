#include "y26_k1x_c2f_block_runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kStage12Alignment = 64;

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
    return mode == Y26_STAGE12_MERGE_MODE_A0_MATERIALIZED_FLOAT ||
           mode == Y26_STAGE13_MERGE_MODE_A1_FUSED_ADD_CONCAT ||
           mode == Y26_STAGE13_MERGE_MODE_A2_FUSED_QDQ_NHWC;
}

bool config_valid(const Y26Stage12C2fBlockConfig* cfg) {
    if (cfg == nullptr || y26_stage11_branch_block_output_count(&cfg->stage11) == 0 ||
        !conv_config_valid(cfg->model2_cv2) || cfg->split1_output_scale <= 0.0f ||
        cfg->split1_output_zero_point_u8 < 0 || cfg->split1_output_zero_point_u8 > 255 ||
        cfg->concat_output_scale <= 0.0f || cfg->concat_output_zero_point_u8 < 0 ||
        cfg->concat_output_zero_point_u8 > 255 || !activation_mode_valid(cfg->activation_mode) ||
        !merge_mode_valid(cfg->merge_mode)) {
        return false;
    }
    const Y26Stage10BackboneExpansionConfig& stage10 = cfg->stage11.stage10;
    const Y26Stage7ConvNodeConfig& producer = stage10.stage9.conv2;
    const int producer_h = output_h_for_kernel(producer.params, producer.kernel_h);
    const int producer_w = output_w_for_kernel(producer.params, producer.kernel_w);
    const int split0_channels = stage10.split_output1_channel_offset;
    const int split1_channels = stage10.split_output1_channels;
    const int concat_channels = split0_channels + split1_channels + split1_channels;
    if (split0_channels <= 0 || split1_channels <= 0 ||
        split0_channels + split1_channels > producer.params.output_c) {
        return false;
    }
    return cfg->model2_cv2.kernel_h == 1 && cfg->model2_cv2.kernel_w == 1 &&
           cfg->model2_cv2.params.input_h == producer_h && cfg->model2_cv2.params.input_w == producer_w &&
           cfg->model2_cv2.params.input_c == concat_channels &&
           cfg->model2_cv2.activation_zero_point_u8 == cfg->concat_output_zero_point_u8 &&
           cfg->model2_cv2.input_storage_zero_point_s8 == cfg->concat_output_zero_point_u8 - 128 &&
           cfg->model2_cv2.input_scale == cfg->concat_output_scale;
}

void* allocate_aligned(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    return ::operator new(bytes, std::align_val_t(kStage12Alignment));
}

void free_aligned(void* ptr) {
    ::operator delete(ptr, std::align_val_t(kStage12Alignment));
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

void timing_reset(Y26Stage12TimingUs* timing) {
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

float qdq_float(float value, float scale, int zero_point_u8) {
    const std::uint8_t code = y26_quantize_u8_nearest_even_f32(value, scale, zero_point_u8);
    return (static_cast<int>(code) - zero_point_u8) * scale;
}

[[maybe_unused]] std::int8_t quantize_concat_s8(float value, float scale, int zero_point_u8) {
    const std::uint8_t code = y26_quantize_u8_nearest_even_f32(value, scale, zero_point_u8);
    return signed_storage_from_u8(code);
}

void quantize_concat_segment(const float* src,
                             std::int8_t* dst,
                             std::size_t count,
                             float scale,
                             int zero_point_u8) {
#if defined(__riscv_vector)
    alignas(64) std::int32_t code_tmp[256] {};
    std::size_t offset = 0;
    while (offset < count) {
        const std::size_t vl = __riscv_vsetvl_e32m4(count - offset);
        vfloat32m4_t value = __riscv_vle32_v_f32m4(src + offset, vl);
        value = __riscv_vfdiv_vf_f32m4(value, scale, vl);
        vint32m4_t code = __riscv_vfcvt_x_f_v_i32m4_rm(value, __RISCV_FRM_RNE, vl);
        code = __riscv_vadd_vx_i32m4(code, zero_point_u8, vl);
        code = __riscv_vmax_vx_i32m4(code, 0, vl);
        code = __riscv_vmin_vx_i32m4(code, 255, vl);
        __riscv_vse32_v_i32m4(code_tmp, code, vl);
        for (std::size_t i = 0; i < vl; ++i) {
            dst[offset + i] = signed_storage_from_u8(static_cast<std::uint8_t>(code_tmp[i]));
        }
        offset += vl;
    }
#else
    for (std::size_t i = 0; i < count; ++i) {
        dst[i] = quantize_concat_s8(src[i], scale, zero_point_u8);
    }
#endif
}

void quantize_concat_add_segment(const float* lhs,
                                 const float* rhs,
                                 std::int8_t* dst,
                                 std::size_t count,
                                 float scale,
                                 int zero_point_u8) {
#if defined(__riscv_vector)
    alignas(64) std::int32_t code_tmp[256] {};
    std::size_t offset = 0;
    while (offset < count) {
        const std::size_t vl = __riscv_vsetvl_e32m4(count - offset);
        vfloat32m4_t value = __riscv_vle32_v_f32m4(lhs + offset, vl);
        vfloat32m4_t rhs_value = __riscv_vle32_v_f32m4(rhs + offset, vl);
        value = __riscv_vfadd_vv_f32m4(value, rhs_value, vl);
        value = __riscv_vfdiv_vf_f32m4(value, scale, vl);
        vint32m4_t code = __riscv_vfcvt_x_f_v_i32m4_rm(value, __RISCV_FRM_RNE, vl);
        code = __riscv_vadd_vx_i32m4(code, zero_point_u8, vl);
        code = __riscv_vmax_vx_i32m4(code, 0, vl);
        code = __riscv_vmin_vx_i32m4(code, 255, vl);
        __riscv_vse32_v_i32m4(code_tmp, code, vl);
        for (std::size_t i = 0; i < vl; ++i) {
            dst[offset + i] = signed_storage_from_u8(static_cast<std::uint8_t>(code_tmp[i]));
        }
        offset += vl;
    }
#else
    for (std::size_t i = 0; i < count; ++i) {
        dst[i] = quantize_concat_s8(lhs[i] + rhs[i], scale, zero_point_u8);
    }
#endif
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

void materialize_split_floats(const Y26Stage12C2fBlockConfig& cfg, Y26Stage12C2fBlockWorkspace& ws) {
    const Y26Stage10BackboneExpansionConfig& stage10 = cfg.stage11.stage10;
    const Y26Stage7ConvNodeConfig& producer = stage10.stage9.conv2;
    const int channels = producer.params.output_c;
    const int split0_channels = stage10.split_output1_channel_offset;
    const int split1_channels = stage10.split_output1_channels;
    const std::size_t pixels =
        y26_stage7_backbone_subset_conv2_output_count(&stage10.stage9) / static_cast<std::size_t>(channels);
    const std::int32_t* src = ws.stage11_ws.stage10_ws.conv2_i32;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        for (int c = 0; c < channels; ++c) {
            const float value = accumulator_to_silu_float(producer, src[pixel * channels + c], c);
            if (c < split0_channels) {
                ws.split0_f32[pixel * split0_channels + c] = value;
            } else if (c < split0_channels + split1_channels) {
                const int dst_c = c - split0_channels;
                ws.split1_f32[pixel * split1_channels + dst_c] =
                    qdq_float(value, cfg.split1_output_scale, cfg.split1_output_zero_point_u8);
            }
        }
    }
}

void materialize_split_floats_reuse_split1_qdq(const Y26Stage12C2fBlockConfig& cfg,
                                               Y26Stage12C2fBlockWorkspace& ws) {
    const Y26Stage10BackboneExpansionConfig& stage10 = cfg.stage11.stage10;
    const Y26Stage7ConvNodeConfig& producer = stage10.stage9.conv2;
    const int channels = producer.params.output_c;
    const int split0_channels = stage10.split_output1_channel_offset;
    const int split1_channels = stage10.split_output1_channels;
    const std::size_t pixels =
        y26_stage7_backbone_subset_conv2_output_count(&stage10.stage9) / static_cast<std::size_t>(channels);
    const std::int32_t* src = ws.stage11_ws.stage10_ws.conv2_i32;
    const std::int8_t* split1_s8 = ws.stage11_ws.stage10_ws.split_output1_s8;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        for (int c = 0; c < split0_channels; ++c) {
            ws.split0_f32[pixel * split0_channels + c] =
                accumulator_to_silu_float(producer, src[pixel * channels + c], c);
        }
        for (int c = 0; c < split1_channels; ++c) {
            const int code = static_cast<int>(split1_s8[pixel * split1_channels + c]) + 128;
            ws.split1_f32[pixel * split1_channels + c] =
                (code - cfg.split1_output_zero_point_u8) * cfg.split1_output_scale;
        }
    }
}

void materialize_branch1_activation_float(const Y26Stage12C2fBlockConfig& cfg, Y26Stage12C2fBlockWorkspace& ws) {
    const Y26Stage7ConvNodeConfig& branch1 = cfg.stage11.branch1;
    const int channels = branch1.params.output_c;
    const std::size_t pixels = ws.add_count / static_cast<std::size_t>(channels);
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        for (int c = 0; c < channels; ++c) {
            const std::size_t index = pixel * channels + c;
            ws.branch1_act_f32[index] = accumulator_to_silu_float(branch1, ws.branch1_i32[index], c);
        }
    }
}

void materialize_add(const Y26Stage12C2fBlockWorkspace& ws) {
    for (std::size_t i = 0; i < ws.add_count; ++i) {
        ws.add_f32[i] = ws.split1_f32[i] + ws.branch1_act_f32[i];
    }
}

void materialize_concat(const Y26Stage12C2fBlockWorkspace& ws, int split0_channels, int split1_channels) {
    const std::size_t pixels = ws.add_count / static_cast<std::size_t>(split1_channels);
    const int concat_channels = split0_channels + split1_channels + split1_channels;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        float* dst = ws.concat_f32 + pixel * concat_channels;
        std::memcpy(dst,
                    ws.split0_f32 + pixel * split0_channels,
                    static_cast<std::size_t>(split0_channels) * sizeof(float));
        std::memcpy(dst + split0_channels,
                    ws.split1_f32 + pixel * split1_channels,
                    static_cast<std::size_t>(split1_channels) * sizeof(float));
        std::memcpy(dst + split0_channels + split1_channels,
                    ws.add_f32 + pixel * split1_channels,
                    static_cast<std::size_t>(split1_channels) * sizeof(float));
    }
}

void materialize_concat_fused_add(const Y26Stage12C2fBlockWorkspace& ws,
                                  int split0_channels,
                                  int split1_channels) {
    const std::size_t pixels = ws.add_count / static_cast<std::size_t>(split1_channels);
    const int concat_channels = split0_channels + split1_channels + split1_channels;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        float* dst = ws.concat_f32 + pixel * concat_channels;
        const float* split0 = ws.split0_f32 + pixel * split0_channels;
        const float* split1 = ws.split1_f32 + pixel * split1_channels;
        const float* branch1 = ws.branch1_act_f32 + pixel * split1_channels;
        std::memcpy(dst, split0, static_cast<std::size_t>(split0_channels) * sizeof(float));
        std::memcpy(dst + split0_channels, split1, static_cast<std::size_t>(split1_channels) * sizeof(float));
        for (int c = 0; c < split1_channels; ++c) {
            dst[split0_channels + split1_channels + c] = split1[c] + branch1[c];
        }
    }
}

void quantize_concat(const Y26Stage12C2fBlockConfig& cfg, Y26Stage12C2fBlockWorkspace& ws) {
    quantize_concat_segment(
        ws.concat_f32, ws.concat_s8, ws.concat_count, cfg.concat_output_scale, cfg.concat_output_zero_point_u8);
}

void quantize_concat_fused(const Y26Stage12C2fBlockConfig& cfg,
                           Y26Stage12C2fBlockWorkspace& ws,
                           int split0_channels,
                           int split1_channels) {
    const std::size_t pixels = ws.add_count / static_cast<std::size_t>(split1_channels);
    const int concat_channels = split0_channels + split1_channels + split1_channels;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        std::int8_t* dst = ws.concat_s8 + pixel * concat_channels;
        const float* split0 = ws.split0_f32 + pixel * split0_channels;
        const float* split1 = ws.split1_f32 + pixel * split1_channels;
        const float* branch1 = ws.branch1_act_f32 + pixel * split1_channels;
        quantize_concat_segment(split0,
                                dst,
                                static_cast<std::size_t>(split0_channels),
                                cfg.concat_output_scale,
                                cfg.concat_output_zero_point_u8);
        quantize_concat_segment(split1,
                                dst + split0_channels,
                                static_cast<std::size_t>(split1_channels),
                                cfg.concat_output_scale,
                                cfg.concat_output_zero_point_u8);
        quantize_concat_add_segment(split1,
                                    branch1,
                                    dst + split0_channels + split1_channels,
                                    static_cast<std::size_t>(split1_channels),
                                    cfg.concat_output_scale,
                                    cfg.concat_output_zero_point_u8);
    }
}

int run_model2_cv2_scalar(const Y26Stage12C2fBlockConfig& cfg,
                          Y26Stage12C2fBlockWorkspace& ws,
                          std::int32_t* output_i32_nhwc,
                          Y26Stage12TimingUs* timing) {
    const auto begin = Clock::now();
    int status = scalar_raw_dot(cfg.model2_cv2, ws.concat_s8, ws.model2_cv2_raw_i32);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    const auto correction_begin = Clock::now();
    status = apply_correction(cfg.model2_cv2, ws.model2_cv2_weights, ws.model2_cv2_raw_i32, output_i32_nhwc);
    const auto end = Clock::now();
    if (timing != nullptr) {
        timing->model2_cv2_conv_us = elapsed_us(begin, end);
        timing->correction_us = elapsed_us(correction_begin, end);
    }
    return status;
}

int run_model2_cv2_ime(const Y26Stage12C2fBlockConfig& cfg,
                       Y26Stage12C2fBlockWorkspace& ws,
                       std::int32_t* output_i32_nhwc,
                       Y26Stage12TimingUs* timing) {
    const auto begin = Clock::now();
    int status = y26_conv2d_i8s8s32_nhwc_ime_prepacked_v1(ws.concat_s8,
                                                          ws.model2_cv2_weights,
                                                          ws.model2_cv2_raw_i32,
                                                          cfg.model2_cv2.input_storage_zero_point_s8,
                                                          ws.model2_cv2_workspace,
                                                          Y26_CONV_LOOP_ORDER_M_MAJOR);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    const auto correction_begin = Clock::now();
    status = apply_correction(cfg.model2_cv2, ws.model2_cv2_weights, ws.model2_cv2_raw_i32, output_i32_nhwc);
    const auto end = Clock::now();
    if (timing != nullptr) {
        timing->model2_cv2_conv_us = elapsed_us(begin, end);
        timing->correction_us = elapsed_us(correction_begin, end);
    }
    return status;
}

void accumulate_stage11_timing(Y26Stage12TimingUs& dst, const Y26Stage11TimingUs& src) {
    dst.stage11_timing_us = src;
    dst.conv_us = src.conv0_ime_us + src.conv1_ime_us + src.conv2_ime_us + src.branch_cv1_conv_us +
                  src.branch_cv2_conv_us;
    dst.activation_requant_us = src.act0_requant_lut_us + src.act1_requant_lut_us +
                                src.act2_requant_lut_us + src.branch_cv1_activation_us;
    dst.split_us = src.split_us;
    dst.pack_layout_us = src.layout_or_pack_us;
    dst.correction_us = src.branch_cv2_correction_us;
}

void finalize_timing(Y26Stage12TimingUs& timing) {
    timing.conv_us += timing.model2_cv2_conv_us;
    timing.split_us += timing.split_copy_us;
    timing.add_us = timing.add_compute_us;
    timing.concat_us = timing.concat_materialize_us;
    timing.pack_layout_us += timing.pack_for_model2_cv2_us + timing.layout_copy_us;
    timing.merge_total_us = timing.split_copy_us + timing.add_compute_us + timing.concat_materialize_us +
                            timing.post_concat_qdq_us + timing.layout_copy_us;
    const double add_concat = timing.add_compute_us + timing.concat_materialize_us + timing.post_concat_qdq_us;
    if (timing.total_us > 0.0) {
        timing.activation_share_pct = 100.0 * timing.activation_requant_us / timing.total_us;
        timing.conv_share_pct = 100.0 * timing.conv_us / timing.total_us;
        timing.add_concat_share_pct = 100.0 * add_concat / timing.total_us;
        timing.pack_layout_share_pct = 100.0 * timing.pack_layout_us / timing.total_us;
        timing.merge_share_pct = 100.0 * timing.merge_total_us / timing.total_us;
    }
}

int run_after_stage11(const Y26Stage12C2fBlockConfig& cfg,
                      Y26Stage12C2fBlockWorkspace& ws,
                      std::int32_t* output_i32_nhwc,
                      Y26Stage12TimingUs* timing,
                      bool use_ime) {
    const int split0_channels = cfg.stage11.stage10.split_output1_channel_offset;
    const int split1_channels = cfg.stage11.stage10.split_output1_channels;

    const auto split_begin = Clock::now();
    if (cfg.merge_mode == Y26_STAGE12_MERGE_MODE_A0_MATERIALIZED_FLOAT) {
        materialize_split_floats(cfg, ws);
    } else {
        materialize_split_floats_reuse_split1_qdq(cfg, ws);
    }
    const auto split_end = Clock::now();
    materialize_branch1_activation_float(cfg, ws);
    const auto branch_act_end = Clock::now();
    auto add_end = branch_act_end;
    auto concat_end = branch_act_end;
    auto qdq_end = branch_act_end;
    if (cfg.merge_mode == Y26_STAGE12_MERGE_MODE_A0_MATERIALIZED_FLOAT) {
        materialize_add(ws);
        add_end = Clock::now();
        materialize_concat(ws, split0_channels, split1_channels);
        concat_end = Clock::now();
        quantize_concat(cfg, ws);
        qdq_end = Clock::now();
    } else if (cfg.merge_mode == Y26_STAGE13_MERGE_MODE_A1_FUSED_ADD_CONCAT) {
        materialize_concat_fused_add(ws, split0_channels, split1_channels);
        concat_end = Clock::now();
        quantize_concat(cfg, ws);
        qdq_end = Clock::now();
    } else {
        quantize_concat_fused(cfg, ws, split0_channels, split1_channels);
        qdq_end = Clock::now();
    }

    int status = use_ime ? run_model2_cv2_ime(cfg, ws, output_i32_nhwc, timing)
                         : run_model2_cv2_scalar(cfg, ws, output_i32_nhwc, timing);
    if (timing != nullptr) {
        timing->merge_mode = cfg.merge_mode;
        timing->split_copy_us += elapsed_us(split_begin, split_end);
        timing->activation_requant_us += elapsed_us(split_end, branch_act_end);
        timing->add_compute_us = cfg.merge_mode == Y26_STAGE12_MERGE_MODE_A0_MATERIALIZED_FLOAT
                                     ? elapsed_us(branch_act_end, add_end)
                                     : 0.0;
        timing->concat_materialize_us =
            cfg.merge_mode == Y26_STAGE13_MERGE_MODE_A2_FUSED_QDQ_NHWC ? 0.0 : elapsed_us(add_end, concat_end);
        timing->post_concat_qdq_us =
            cfg.merge_mode == Y26_STAGE13_MERGE_MODE_A2_FUSED_QDQ_NHWC ? elapsed_us(branch_act_end, qdq_end)
                                                                       : elapsed_us(concat_end, qdq_end);
    }
    return status;
}

}  // namespace

extern "C" int y26_stage12_c2f_block_prepare(const Y26Stage12C2fBlockConfig* cfg,
                                              Y26Stage12C2fBlockWorkspace* ws) {
    if (!config_valid(cfg) || ws == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    std::memset(ws, 0, sizeof(*ws));
    int status = y26_stage11_branch_block_prepare(&cfg->stage11, &ws->stage11_ws);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    ws->model2_cv2_weights = y26_prepacked_conv_weights_create_mmt4d_s8(cfg->model2_cv2.weights_ohwi_s8,
                                                                         &cfg->model2_cv2.params,
                                                                         cfg->model2_cv2.kernel_h,
                                                                         cfg->model2_cv2.kernel_w,
                                                                         cfg->model2_cv2.node_name,
                                                                         cfg->model2_cv2.weight_scales);
    ws->model2_cv2_workspace =
        y26_conv_workspace_create(&cfg->model2_cv2.params, cfg->model2_cv2.kernel_h, cfg->model2_cv2.kernel_w);

    const Y26Stage10BackboneExpansionConfig& stage10 = cfg->stage11.stage10;
    const Y26Stage7ConvNodeConfig& producer = stage10.stage9.conv2;
    const int producer_channels = producer.params.output_c;
    const int split0_channels = stage10.split_output1_channel_offset;
    const int split1_channels = stage10.split_output1_channels;
    const std::size_t pixels =
        y26_stage7_backbone_subset_conv2_output_count(&stage10.stage9) / static_cast<std::size_t>(producer_channels);

    ws->split0_count = pixels * static_cast<std::size_t>(split0_channels);
    ws->split1_count = pixels * static_cast<std::size_t>(split1_channels);
    ws->add_count = ws->split1_count;
    ws->concat_count = pixels * static_cast<std::size_t>(cfg->model2_cv2.params.input_c);
    ws->model2_cv2_output_count = output_count_for_kernel(cfg->model2_cv2.params,
                                                          cfg->model2_cv2.kernel_h,
                                                          cfg->model2_cv2.kernel_w);
    ws->branch1_i32 = allocate_i32(y26_stage11_branch_block_output_count(&cfg->stage11));
    ws->split0_f32 = allocate_f32(ws->split0_count);
    ws->split1_f32 = allocate_f32(ws->split1_count);
    ws->branch1_act_f32 = allocate_f32(ws->add_count);
    ws->add_f32 = allocate_f32(ws->add_count);
    ws->concat_f32 = allocate_f32(ws->concat_count);
    ws->concat_s8 = allocate_i8(ws->concat_count);
    ws->model2_cv2_raw_i32 = allocate_i32(ws->model2_cv2_output_count);

    if (ws->model2_cv2_weights == nullptr || ws->model2_cv2_workspace == nullptr || ws->branch1_i32 == nullptr ||
        ws->split0_f32 == nullptr || ws->split1_f32 == nullptr || ws->branch1_act_f32 == nullptr ||
        ws->add_f32 == nullptr || ws->concat_f32 == nullptr || ws->concat_s8 == nullptr ||
        ws->model2_cv2_raw_i32 == nullptr) {
        y26_stage12_c2f_block_release(ws);
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }

    ws->prepacked_bytes = y26_prepacked_conv_weights_total_bytes(ws->model2_cv2_weights) +
                          ws->stage11_ws.prepacked_bytes;
    ws->workspace_bytes = y26_conv_workspace_bytes(ws->model2_cv2_workspace) + ws->stage11_ws.workspace_bytes +
                          ws->split0_count * sizeof(float) + ws->split1_count * sizeof(float) +
                          ws->add_count * sizeof(float) * 3 + ws->concat_count + ws->model2_cv2_output_count *
                          sizeof(std::int32_t);
    ws->prepared = 1;
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" void y26_stage12_c2f_block_release(Y26Stage12C2fBlockWorkspace* ws) {
    if (ws == nullptr) {
        return;
    }
    y26_stage11_branch_block_release(&ws->stage11_ws);
    y26_prepacked_conv_weights_destroy(ws->model2_cv2_weights);
    y26_conv_workspace_destroy(ws->model2_cv2_workspace);
    free_aligned(ws->branch1_i32);
    free_aligned(ws->split0_f32);
    free_aligned(ws->split1_f32);
    free_aligned(ws->branch1_act_f32);
    free_aligned(ws->add_f32);
    free_aligned(ws->concat_f32);
    free_aligned(ws->concat_s8);
    free_aligned(ws->model2_cv2_raw_i32);
    std::memset(ws, 0, sizeof(*ws));
}

extern "C" std::size_t y26_stage12_c2f_block_output_count(const Y26Stage12C2fBlockConfig* cfg) {
    return config_valid(cfg) ? output_count_for_kernel(cfg->model2_cv2.params,
                                                       cfg->model2_cv2.kernel_h,
                                                       cfg->model2_cv2.kernel_w)
                             : 0;
}

extern "C" int y26_stage12_c2f_block_run_scalar(const Y26Stage12C2fBlockConfig* cfg,
                                                 Y26Stage12C2fBlockWorkspace* ws,
                                                 const std::int8_t* input_nhwc_s8,
                                                 std::int32_t* output_i32_nhwc,
                                                 Y26Stage12TimingUs* timing) {
    if (!config_valid(cfg) || ws == nullptr || ws->prepared != 1 || input_nhwc_s8 == nullptr ||
        output_i32_nhwc == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    timing_reset(timing);
    const auto begin = Clock::now();
    Y26Stage11TimingUs stage11_timing {};
    int status = y26_stage11_branch_block_run_scalar(&cfg->stage11,
                                                     &ws->stage11_ws,
                                                     input_nhwc_s8,
                                                     ws->branch1_i32,
                                                     &stage11_timing);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    if (timing != nullptr) {
        accumulate_stage11_timing(*timing, stage11_timing);
    }
    status = run_after_stage11(*cfg, *ws, output_i32_nhwc, timing, false);
    const auto end = Clock::now();
    if (timing != nullptr) {
        timing->total_us = elapsed_us(begin, end);
        finalize_timing(*timing);
    }
    return status;
}

extern "C" int y26_stage12_c2f_block_run_ime_cluster0_hotpath(const Y26Stage12C2fBlockConfig* cfg,
                                                               Y26Stage12C2fBlockWorkspace* ws,
                                                               const std::int8_t* input_nhwc_s8,
                                                               std::int32_t* output_i32_nhwc,
                                                               Y26Stage12TimingUs* timing) {
    if (!config_valid(cfg) || ws == nullptr || ws->prepared != 1 || input_nhwc_s8 == nullptr ||
        output_i32_nhwc == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    timing_reset(timing);
    const auto begin = Clock::now();
    Y26Stage11TimingUs stage11_timing {};
    int status = y26_stage11_branch_block_run_ime_cluster0_hotpath(&cfg->stage11,
                                                                   &ws->stage11_ws,
                                                                   input_nhwc_s8,
                                                                   ws->branch1_i32,
                                                                   &stage11_timing);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    if (timing != nullptr) {
        accumulate_stage11_timing(*timing, stage11_timing);
    }
    status = run_after_stage11(*cfg, *ws, output_i32_nhwc, timing, true);
    const auto end = Clock::now();
    if (timing != nullptr) {
        timing->total_us = elapsed_us(begin, end);
        finalize_timing(*timing);
    }
    return status;
}

extern "C" const std::int8_t* y26_stage12_c2f_block_concat_s8(const Y26Stage12C2fBlockWorkspace* ws) {
    return ws != nullptr ? ws->concat_s8 : nullptr;
}
