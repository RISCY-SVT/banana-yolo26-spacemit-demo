#include "y26_k1x_activation.h"

#include "y26_k1x_conv_kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000.0;
}

bool activation_params_valid(const Y26ActivationRequantParams* params) {
    return params != nullptr && params->element_count > 0 && params->channels > 0 && params->input_scale > 0.0f &&
           params->weight_scales != nullptr && params->conv_output_scale > 0.0f &&
           params->conv_output_zero_point_u8 >= 0 && params->conv_output_zero_point_u8 <= 255 &&
           params->act_output_scale > 0.0f && params->act_output_zero_point_u8 >= 0 &&
           params->act_output_zero_point_u8 <= 255;
}

bool conv_output_quantize_params_valid(const Y26ConvOutputQuantizeParams* params) {
    return params != nullptr && params->element_count > 0 && params->channels > 0 &&
           params->input_scale > 0.0f && params->weight_scales != nullptr &&
           params->output_scale > 0.0f && params->output_zero_point_u8 >= 0 &&
           params->output_zero_point_u8 <= 255 &&
           params->element_count % static_cast<std::size_t>(params->channels) == 0 &&
           params->channels <= 256;
}

float silu_f32(float value) {
    return value / (1.0f + std::exp(-value));
}

std::uint8_t clamp_u8(long value) {
    return static_cast<std::uint8_t>(std::max<long>(0, std::min<long>(255, value)));
}

long round_nearest_even_independent(double value) {
    if (!std::isfinite(value)) {
        return value < 0.0 ? std::numeric_limits<long>::min() : std::numeric_limits<long>::max();
    }
    const double floored = std::floor(value);
    const double fraction = value - floored;
    long base = static_cast<long>(floored);
    if (fraction > 0.5) {
        return base + 1;
    }
    if (fraction < 0.5) {
        return base;
    }
    return (base & 1L) == 0 ? base : base + 1;
}

std::int8_t signed_storage_from_u8(std::uint8_t value) {
    return static_cast<std::int8_t>(static_cast<int>(value) - 128);
}

std::int64_t round_shift_right_nearest_even(std::int64_t value, int shift) {
    if (shift <= 0) {
        const int left_shift = -shift;
        if (left_shift >= 63) {
            return value >= 0 ? std::numeric_limits<std::int64_t>::max() : std::numeric_limits<std::int64_t>::min();
        }
        if (value > (std::numeric_limits<std::int64_t>::max() >> left_shift)) {
            return std::numeric_limits<std::int64_t>::max();
        }
        if (value < (std::numeric_limits<std::int64_t>::min() >> left_shift)) {
            return std::numeric_limits<std::int64_t>::min();
        }
        return value << left_shift;
    }

    const bool negative = value < 0;
    const std::uint64_t abs_value =
        negative ? static_cast<std::uint64_t>(-(value + 1)) + 1U : static_cast<std::uint64_t>(value);
    if (shift >= 63) {
        return 0;
    }
    const std::uint64_t divisor = std::uint64_t{1} << shift;
    std::uint64_t quotient = abs_value >> shift;
    const std::uint64_t remainder = abs_value & (divisor - 1U);
    const std::uint64_t half = divisor >> 1U;
    if (remainder > half || (remainder == half && (quotient & 1) != 0)) {
        ++quotient;
    }
    if (negative) {
        if (quotient > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1U) {
            return std::numeric_limits<std::int64_t>::min();
        }
        return -static_cast<std::int64_t>(quotient);
    }
    return quotient > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
               ? std::numeric_limits<std::int64_t>::max()
               : static_cast<std::int64_t>(quotient);
}

std::uint8_t requantize_accumulator_to_conv_code_float(const Y26ActivationRequantParams& params,
                                                       std::int32_t accumulator,
                                                       int channel) {
    const float acc_scale = params.input_scale * params.weight_scales[channel];
    const float conv_float = static_cast<float>(accumulator) * acc_scale;
    return y26_quantize_u8_nearest_even_f32(
        conv_float, params.conv_output_scale, params.conv_output_zero_point_u8);
}

std::int8_t silu_lut_value_reference(const std::int8_t* lut_256_s8, std::uint8_t conv_code) {
    return lut_256_s8[static_cast<unsigned>(conv_code)];
}

std::uint8_t requantize_accumulator_to_conv_code_float_scale(std::int32_t accumulator,
                                                             float acc_scale,
                                                             float conv_output_scale,
                                                             int conv_output_zero_point_u8) {
    const float conv_float = static_cast<float>(accumulator) * acc_scale;
    const double scaled = static_cast<double>(conv_float) / static_cast<double>(conv_output_scale);
    const long rounded = round_nearest_even_independent(scaled) + static_cast<long>(conv_output_zero_point_u8);
    return clamp_u8(rounded);
}

bool stage9_channel_loop_supported(const Y26ActivationRequantParams& params) {
    constexpr int kMaxStage9Channels = 256;
    return params.channels > 0 && params.channels <= kMaxStage9Channels &&
           params.element_count % static_cast<std::size_t>(params.channels) == 0;
}

void build_acc_scales(const Y26ActivationRequantParams& params, float* acc_scales) {
    for (int channel = 0; channel < params.channels; ++channel) {
        acc_scales[channel] = params.input_scale * params.weight_scales[channel];
    }
}

void build_output_acc_scales(const Y26ConvOutputQuantizeParams& params, float* acc_scales) {
    for (int channel = 0; channel < params.channels; ++channel) {
        acc_scales[channel] = params.input_scale * params.weight_scales[channel];
    }
}

void conv_output_quantize_scalar_unrolled_impl(const Y26ConvOutputQuantizeParams& params,
                                               const std::int32_t* __restrict producer_i32,
                                               std::uint8_t* __restrict output_u8) {
    float acc_scales[256] {};
    build_output_acc_scales(params, acc_scales);
    const std::size_t pixels = params.element_count / static_cast<std::size_t>(params.channels);
    const float output_scale = params.output_scale;
    const int output_zp = params.output_zero_point_u8;
    const std::int32_t* __restrict src = producer_i32;
    std::uint8_t* __restrict dst = output_u8;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        int channel = 0;
        for (; channel + 8 <= params.channels; channel += 8) {
            dst[channel + 0] =
                requantize_accumulator_to_conv_code_float_scale(src[channel + 0], acc_scales[channel + 0], output_scale, output_zp);
            dst[channel + 1] =
                requantize_accumulator_to_conv_code_float_scale(src[channel + 1], acc_scales[channel + 1], output_scale, output_zp);
            dst[channel + 2] =
                requantize_accumulator_to_conv_code_float_scale(src[channel + 2], acc_scales[channel + 2], output_scale, output_zp);
            dst[channel + 3] =
                requantize_accumulator_to_conv_code_float_scale(src[channel + 3], acc_scales[channel + 3], output_scale, output_zp);
            dst[channel + 4] =
                requantize_accumulator_to_conv_code_float_scale(src[channel + 4], acc_scales[channel + 4], output_scale, output_zp);
            dst[channel + 5] =
                requantize_accumulator_to_conv_code_float_scale(src[channel + 5], acc_scales[channel + 5], output_scale, output_zp);
            dst[channel + 6] =
                requantize_accumulator_to_conv_code_float_scale(src[channel + 6], acc_scales[channel + 6], output_scale, output_zp);
            dst[channel + 7] =
                requantize_accumulator_to_conv_code_float_scale(src[channel + 7], acc_scales[channel + 7], output_scale, output_zp);
        }
        for (; channel < params.channels; ++channel) {
            dst[channel] =
                requantize_accumulator_to_conv_code_float_scale(src[channel], acc_scales[channel], output_scale, output_zp);
        }
        src += params.channels;
        dst += params.channels;
    }
}

void requant_codes_scalar_unrolled(const Y26ActivationRequantParams& params,
                                   const std::int32_t* __restrict producer_i32,
                                   std::uint8_t* __restrict conv_code_u8) {
    float acc_scales[256] {};
    build_acc_scales(params, acc_scales);
    const std::size_t pixels = params.element_count / static_cast<std::size_t>(params.channels);
    const float conv_output_scale = params.conv_output_scale;
    const int conv_zp = params.conv_output_zero_point_u8;
    const std::int32_t* __restrict src = producer_i32;
    std::uint8_t* __restrict dst = conv_code_u8;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        int channel = 0;
        for (; channel + 8 <= params.channels; channel += 8) {
            dst[channel + 0] =
                requantize_accumulator_to_conv_code_float_scale(src[channel + 0], acc_scales[channel + 0], conv_output_scale, conv_zp);
            dst[channel + 1] =
                requantize_accumulator_to_conv_code_float_scale(src[channel + 1], acc_scales[channel + 1], conv_output_scale, conv_zp);
            dst[channel + 2] =
                requantize_accumulator_to_conv_code_float_scale(src[channel + 2], acc_scales[channel + 2], conv_output_scale, conv_zp);
            dst[channel + 3] =
                requantize_accumulator_to_conv_code_float_scale(src[channel + 3], acc_scales[channel + 3], conv_output_scale, conv_zp);
            dst[channel + 4] =
                requantize_accumulator_to_conv_code_float_scale(src[channel + 4], acc_scales[channel + 4], conv_output_scale, conv_zp);
            dst[channel + 5] =
                requantize_accumulator_to_conv_code_float_scale(src[channel + 5], acc_scales[channel + 5], conv_output_scale, conv_zp);
            dst[channel + 6] =
                requantize_accumulator_to_conv_code_float_scale(src[channel + 6], acc_scales[channel + 6], conv_output_scale, conv_zp);
            dst[channel + 7] =
                requantize_accumulator_to_conv_code_float_scale(src[channel + 7], acc_scales[channel + 7], conv_output_scale, conv_zp);
        }
        for (; channel < params.channels; ++channel) {
            dst[channel] =
                requantize_accumulator_to_conv_code_float_scale(src[channel], acc_scales[channel], conv_output_scale, conv_zp);
        }
        src += params.channels;
        dst += params.channels;
    }
}

void requant_codes_fixed_unrolled(const Y26ActivationRequantParams& params,
                                  const Y26FixedRequantParams* per_channel_params,
                                  const std::int32_t* __restrict producer_i32,
                                  std::uint8_t* __restrict conv_code_u8) {
    const std::size_t pixels = params.element_count / static_cast<std::size_t>(params.channels);
    const std::int32_t* __restrict src = producer_i32;
    std::uint8_t* __restrict dst = conv_code_u8;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        int channel = 0;
        for (; channel + 8 <= params.channels; channel += 8) {
            dst[channel + 0] = y26_requant_s32_to_u8_fixed_nearest_even(src[channel + 0], per_channel_params + channel + 0);
            dst[channel + 1] = y26_requant_s32_to_u8_fixed_nearest_even(src[channel + 1], per_channel_params + channel + 1);
            dst[channel + 2] = y26_requant_s32_to_u8_fixed_nearest_even(src[channel + 2], per_channel_params + channel + 2);
            dst[channel + 3] = y26_requant_s32_to_u8_fixed_nearest_even(src[channel + 3], per_channel_params + channel + 3);
            dst[channel + 4] = y26_requant_s32_to_u8_fixed_nearest_even(src[channel + 4], per_channel_params + channel + 4);
            dst[channel + 5] = y26_requant_s32_to_u8_fixed_nearest_even(src[channel + 5], per_channel_params + channel + 5);
            dst[channel + 6] = y26_requant_s32_to_u8_fixed_nearest_even(src[channel + 6], per_channel_params + channel + 6);
            dst[channel + 7] = y26_requant_s32_to_u8_fixed_nearest_even(src[channel + 7], per_channel_params + channel + 7);
        }
        for (; channel < params.channels; ++channel) {
            dst[channel] = y26_requant_s32_to_u8_fixed_nearest_even(src[channel], per_channel_params + channel);
        }
        src += params.channels;
        dst += params.channels;
    }
}

void lut_lookup_unrolled(const std::uint8_t* __restrict conv_code_u8,
                         const std::int8_t* __restrict lut_256_s8,
                         std::size_t element_count,
                         std::int8_t* __restrict consumer_input_s8) {
    std::size_t index = 0;
    for (; index + 8 <= element_count; index += 8) {
        consumer_input_s8[index + 0] = lut_256_s8[conv_code_u8[index + 0]];
        consumer_input_s8[index + 1] = lut_256_s8[conv_code_u8[index + 1]];
        consumer_input_s8[index + 2] = lut_256_s8[conv_code_u8[index + 2]];
        consumer_input_s8[index + 3] = lut_256_s8[conv_code_u8[index + 3]];
        consumer_input_s8[index + 4] = lut_256_s8[conv_code_u8[index + 4]];
        consumer_input_s8[index + 5] = lut_256_s8[conv_code_u8[index + 5]];
        consumer_input_s8[index + 6] = lut_256_s8[conv_code_u8[index + 6]];
        consumer_input_s8[index + 7] = lut_256_s8[conv_code_u8[index + 7]];
    }
    for (; index < element_count; ++index) {
        consumer_input_s8[index] = lut_256_s8[conv_code_u8[index]];
    }
}

void requant_lut_scalar_unrolled_fused(const Y26ActivationRequantParams& params,
                                       const std::int32_t* __restrict producer_i32,
                                       const std::int8_t* __restrict lut_256_s8,
                                       std::int8_t* __restrict consumer_input_s8) {
    float acc_scales[256] {};
    build_acc_scales(params, acc_scales);
    const std::size_t pixels = params.element_count / static_cast<std::size_t>(params.channels);
    const float conv_output_scale = params.conv_output_scale;
    const int conv_zp = params.conv_output_zero_point_u8;
    const std::int32_t* __restrict src = producer_i32;
    std::int8_t* __restrict dst = consumer_input_s8;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        int channel = 0;
        for (; channel + 8 <= params.channels; channel += 8) {
            dst[channel + 0] = lut_256_s8[requantize_accumulator_to_conv_code_float_scale(
                src[channel + 0], acc_scales[channel + 0], conv_output_scale, conv_zp)];
            dst[channel + 1] = lut_256_s8[requantize_accumulator_to_conv_code_float_scale(
                src[channel + 1], acc_scales[channel + 1], conv_output_scale, conv_zp)];
            dst[channel + 2] = lut_256_s8[requantize_accumulator_to_conv_code_float_scale(
                src[channel + 2], acc_scales[channel + 2], conv_output_scale, conv_zp)];
            dst[channel + 3] = lut_256_s8[requantize_accumulator_to_conv_code_float_scale(
                src[channel + 3], acc_scales[channel + 3], conv_output_scale, conv_zp)];
            dst[channel + 4] = lut_256_s8[requantize_accumulator_to_conv_code_float_scale(
                src[channel + 4], acc_scales[channel + 4], conv_output_scale, conv_zp)];
            dst[channel + 5] = lut_256_s8[requantize_accumulator_to_conv_code_float_scale(
                src[channel + 5], acc_scales[channel + 5], conv_output_scale, conv_zp)];
            dst[channel + 6] = lut_256_s8[requantize_accumulator_to_conv_code_float_scale(
                src[channel + 6], acc_scales[channel + 6], conv_output_scale, conv_zp)];
            dst[channel + 7] = lut_256_s8[requantize_accumulator_to_conv_code_float_scale(
                src[channel + 7], acc_scales[channel + 7], conv_output_scale, conv_zp)];
        }
        for (; channel < params.channels; ++channel) {
            dst[channel] = lut_256_s8[requantize_accumulator_to_conv_code_float_scale(
                src[channel], acc_scales[channel], conv_output_scale, conv_zp)];
        }
        src += params.channels;
        dst += params.channels;
    }
}

void requant_lut_fixed_unrolled_fused(const Y26ActivationRequantParams& params,
                                      const Y26FixedRequantParams* per_channel_params,
                                      const std::int32_t* __restrict producer_i32,
                                      const std::int8_t* __restrict lut_256_s8,
                                      std::int8_t* __restrict consumer_input_s8) {
    const std::size_t pixels = params.element_count / static_cast<std::size_t>(params.channels);
    const std::int32_t* __restrict src = producer_i32;
    std::int8_t* __restrict dst = consumer_input_s8;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        int channel = 0;
        for (; channel + 8 <= params.channels; channel += 8) {
            dst[channel + 0] =
                lut_256_s8[y26_requant_s32_to_u8_fixed_nearest_even(src[channel + 0], per_channel_params + channel + 0)];
            dst[channel + 1] =
                lut_256_s8[y26_requant_s32_to_u8_fixed_nearest_even(src[channel + 1], per_channel_params + channel + 1)];
            dst[channel + 2] =
                lut_256_s8[y26_requant_s32_to_u8_fixed_nearest_even(src[channel + 2], per_channel_params + channel + 2)];
            dst[channel + 3] =
                lut_256_s8[y26_requant_s32_to_u8_fixed_nearest_even(src[channel + 3], per_channel_params + channel + 3)];
            dst[channel + 4] =
                lut_256_s8[y26_requant_s32_to_u8_fixed_nearest_even(src[channel + 4], per_channel_params + channel + 4)];
            dst[channel + 5] =
                lut_256_s8[y26_requant_s32_to_u8_fixed_nearest_even(src[channel + 5], per_channel_params + channel + 5)];
            dst[channel + 6] =
                lut_256_s8[y26_requant_s32_to_u8_fixed_nearest_even(src[channel + 6], per_channel_params + channel + 6)];
            dst[channel + 7] =
                lut_256_s8[y26_requant_s32_to_u8_fixed_nearest_even(src[channel + 7], per_channel_params + channel + 7)];
        }
        for (; channel < params.channels; ++channel) {
            dst[channel] =
                lut_256_s8[y26_requant_s32_to_u8_fixed_nearest_even(src[channel], per_channel_params + channel)];
        }
        src += params.channels;
        dst += params.channels;
    }
}

#if defined(__riscv_vector)
int requant_lut_rvv_f32_impl(const Y26ActivationRequantParams& params,
                             const std::int32_t* producer_i32,
                             const std::int8_t* lut_256_s8,
                             std::int8_t* consumer_input_s8) {
    if (!stage9_channel_loop_supported(params)) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    float acc_scales[256] {};
    alignas(64) std::int32_t code_tmp[256] {};
    build_acc_scales(params, acc_scales);
    const std::size_t pixels = params.element_count / static_cast<std::size_t>(params.channels);
    const std::int32_t* src = producer_i32;
    std::int8_t* dst = consumer_input_s8;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        int channel = 0;
        while (channel < params.channels) {
            const std::size_t vl = __riscv_vsetvl_e32m4(static_cast<std::size_t>(params.channels - channel));
            vint32m4_t vacc = __riscv_vle32_v_i32m4(src + channel, vl);
            vfloat32m4_t vf = __riscv_vfcvt_f_x_v_f32m4(vacc, vl);
            vfloat32m4_t vscale = __riscv_vle32_v_f32m4(acc_scales + channel, vl);
            vf = __riscv_vfmul_vv_f32m4(vf, vscale, vl);
            vf = __riscv_vfdiv_vf_f32m4(vf, params.conv_output_scale, vl);
            vint32m4_t vcode = __riscv_vfcvt_x_f_v_i32m4_rm(vf, __RISCV_FRM_RNE, vl);
            vcode = __riscv_vadd_vx_i32m4(vcode, params.conv_output_zero_point_u8, vl);
            vcode = __riscv_vmax_vx_i32m4(vcode, 0, vl);
            vcode = __riscv_vmin_vx_i32m4(vcode, 255, vl);
            __riscv_vse32_v_i32m4(code_tmp + channel, vcode, vl);
            channel += static_cast<int>(vl);
        }
        for (int c = 0; c < params.channels; ++c) {
            dst[c] = lut_256_s8[static_cast<std::uint8_t>(code_tmp[c])];
        }
        src += params.channels;
        dst += params.channels;
    }
    return Y26_CONV_STATUS_SUCCESS;
}

int conv_output_quantize_rvv_f32_impl(const Y26ConvOutputQuantizeParams& params,
                                      const std::int32_t* producer_i32,
                                      std::uint8_t* output_u8) {
    float acc_scales[256] {};
    alignas(64) std::int32_t code_tmp[256] {};
    build_output_acc_scales(params, acc_scales);
    const std::size_t pixels = params.element_count / static_cast<std::size_t>(params.channels);
    const std::int32_t* src = producer_i32;
    std::uint8_t* dst = output_u8;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        int channel = 0;
        while (channel < params.channels) {
            const std::size_t vl = __riscv_vsetvl_e32m4(static_cast<std::size_t>(params.channels - channel));
            vint32m4_t vacc = __riscv_vle32_v_i32m4(src + channel, vl);
            vfloat32m4_t vf = __riscv_vfcvt_f_x_v_f32m4(vacc, vl);
            vfloat32m4_t vscale = __riscv_vle32_v_f32m4(acc_scales + channel, vl);
            vf = __riscv_vfmul_vv_f32m4(vf, vscale, vl);
            vf = __riscv_vfdiv_vf_f32m4(vf, params.output_scale, vl);
            vint32m4_t vcode = __riscv_vfcvt_x_f_v_i32m4_rm(vf, __RISCV_FRM_RNE, vl);
            vcode = __riscv_vadd_vx_i32m4(vcode, params.output_zero_point_u8, vl);
            vcode = __riscv_vmax_vx_i32m4(vcode, 0, vl);
            vcode = __riscv_vmin_vx_i32m4(vcode, 255, vl);
            __riscv_vse32_v_i32m4(code_tmp + channel, vcode, vl);
            channel += static_cast<int>(vl);
        }
        for (int c = 0; c < params.channels; ++c) {
            dst[c] = static_cast<std::uint8_t>(code_tmp[c]);
        }
        src += params.channels;
        dst += params.channels;
    }
    return Y26_CONV_STATUS_SUCCESS;
}

int conv_output_quantize_rvv_f32_direct_store_impl(const Y26ConvOutputQuantizeParams& params,
                                                   const std::int32_t* producer_i32,
                                                   std::uint8_t* output_u8) {
    float acc_scales[256] {};
    build_output_acc_scales(params, acc_scales);
    const std::size_t pixels = params.element_count / static_cast<std::size_t>(params.channels);
    const std::int32_t* src = producer_i32;
    std::uint8_t* dst = output_u8;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        int channel = 0;
        while (channel < params.channels) {
            const std::size_t vl = __riscv_vsetvl_e32m4(static_cast<std::size_t>(params.channels - channel));
            vint32m4_t vacc = __riscv_vle32_v_i32m4(src + channel, vl);
            vfloat32m4_t vf = __riscv_vfcvt_f_x_v_f32m4(vacc, vl);
            vfloat32m4_t vscale = __riscv_vle32_v_f32m4(acc_scales + channel, vl);
            vf = __riscv_vfmul_vv_f32m4(vf, vscale, vl);
            vf = __riscv_vfdiv_vf_f32m4(vf, params.output_scale, vl);
            vint32m4_t vcode = __riscv_vfcvt_x_f_v_i32m4_rm(vf, __RISCV_FRM_RNE, vl);
            vcode = __riscv_vadd_vx_i32m4(vcode, params.output_zero_point_u8, vl);
            vcode = __riscv_vmax_vx_i32m4(vcode, 0, vl);
            vcode = __riscv_vmin_vx_i32m4(vcode, 255, vl);
            vuint32m4_t vu32 = __riscv_vreinterpret_v_i32m4_u32m4(vcode);
            vuint16m2_t vu16 = __riscv_vnclipu_wx_u16m2(vu32, 0, __RISCV_VXRM_RNU, vl);
            vuint8m1_t vu8 = __riscv_vnclipu_wx_u8m1(vu16, 0, __RISCV_VXRM_RNU, vl);
            __riscv_vse8_v_u8m1(dst + channel, vu8, vl);
            channel += static_cast<int>(vl);
        }
        src += params.channels;
        dst += params.channels;
    }
    return Y26_CONV_STATUS_SUCCESS;
}
#endif

}  // namespace

extern "C" std::uint8_t y26_quantize_u8_nearest_even_f32(float value, float scale, int zero_point_u8) {
    if (scale <= 0.0f || zero_point_u8 < 0 || zero_point_u8 > 255) {
        return 0;
    }
    const double scaled = static_cast<double>(value) / static_cast<double>(scale);
    const long rounded = round_nearest_even_independent(scaled) + static_cast<long>(zero_point_u8);
    return clamp_u8(rounded);
}

extern "C" int y26_fixed_requant_params_from_multiplier(double multiplier,
                                                         int output_zero_point_u8,
                                                         Y26FixedRequantParams* params) {
    if (params == nullptr || multiplier < 0.0 || !std::isfinite(multiplier) || output_zero_point_u8 < 0 ||
        output_zero_point_u8 > 255) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    if (multiplier == 0.0) {
        params->multiplier_q31 = 0;
        params->exponent = 0;
        params->output_zero_point_u8 = output_zero_point_u8;
        return Y26_CONV_STATUS_SUCCESS;
    }

    int exponent = 0;
    const double fraction = std::frexp(multiplier, &exponent);
    std::int64_t q31 = static_cast<std::int64_t>(std::llround(fraction * 2147483648.0));
    if (q31 == 2147483648LL) {
        q31 >>= 1;
        ++exponent;
    }
    params->multiplier_q31 = q31;
    params->exponent = exponent;
    params->output_zero_point_u8 = output_zero_point_u8;
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" std::uint8_t y26_requant_s32_to_u8_fixed_nearest_even(std::int32_t value,
                                                                  const Y26FixedRequantParams* params) {
    if (params == nullptr || params->output_zero_point_u8 < 0 || params->output_zero_point_u8 > 255) {
        return 0;
    }
    const std::int64_t product = static_cast<std::int64_t>(value) * params->multiplier_q31;
    const int shift = 31 - params->exponent;
    const std::int64_t scaled = round_shift_right_nearest_even(product, shift);
    const std::int64_t shifted = scaled + static_cast<std::int64_t>(params->output_zero_point_u8);
    return clamp_u8(static_cast<long>(std::max<std::int64_t>(0, std::min<std::int64_t>(255, shifted))));
}

extern "C" int y26_build_silu_u8_to_s8_lut(float conv_output_scale,
                                            int conv_output_zero_point_u8,
                                            float act_output_scale,
                                            int act_output_zero_point_u8,
                                            std::int8_t* lut_256_s8) {
    if (lut_256_s8 == nullptr || conv_output_scale <= 0.0f || conv_output_zero_point_u8 < 0 ||
        conv_output_zero_point_u8 > 255 || act_output_scale <= 0.0f || act_output_zero_point_u8 < 0 ||
        act_output_zero_point_u8 > 255) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    for (int q = 0; q < 256; ++q) {
        const float conv_dq = static_cast<float>(q - conv_output_zero_point_u8) * conv_output_scale;
        const float activated = silu_f32(conv_dq);
        const std::uint8_t act_q =
            y26_quantize_u8_nearest_even_f32(activated, act_output_scale, act_output_zero_point_u8);
        lut_256_s8[q] = signed_storage_from_u8(act_q);
    }
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" int y26_build_fixed_requant_params_per_channel(const Y26ActivationRequantParams* params,
                                                           Y26FixedRequantParams* per_channel_params) {
    if (!activation_params_valid(params) || per_channel_params == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    for (int channel = 0; channel < params->channels; ++channel) {
        const double multiplier = static_cast<double>(params->input_scale) *
                                  static_cast<double>(params->weight_scales[channel]) /
                                  static_cast<double>(params->conv_output_scale);
        const int status = y26_fixed_requant_params_from_multiplier(
            multiplier, params->conv_output_zero_point_u8, per_channel_params + channel);
        if (status != Y26_CONV_STATUS_SUCCESS) {
            return status;
        }
    }
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" int y26_activation_requant_silu_scalar_float(const Y26ActivationRequantParams* params,
                                                         const std::int32_t* producer_i32,
                                                         std::int8_t* consumer_input_s8) {
    if (!activation_params_valid(params) || producer_i32 == nullptr || consumer_input_s8 == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t index = 0; index < params->element_count; ++index) {
        const int channel = static_cast<int>(index % static_cast<std::size_t>(params->channels));
        const std::uint8_t conv_q =
            requantize_accumulator_to_conv_code_float(*params, producer_i32[index], channel);
        const float conv_dq =
            static_cast<float>(static_cast<int>(conv_q) - params->conv_output_zero_point_u8) *
            params->conv_output_scale;
        const std::uint8_t act_q = y26_quantize_u8_nearest_even_f32(
            silu_f32(conv_dq), params->act_output_scale, params->act_output_zero_point_u8);
        consumer_input_s8[index] = signed_storage_from_u8(act_q);
    }
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" int y26_activation_requant_silu_int8_lut(const Y26ActivationRequantParams* params,
                                                     const std::int32_t* producer_i32,
                                                     const std::int8_t* lut_256_s8,
                                                     std::int8_t* consumer_input_s8) {
    if (!activation_params_valid(params) || producer_i32 == nullptr || lut_256_s8 == nullptr ||
        consumer_input_s8 == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t index = 0; index < params->element_count; ++index) {
        const int channel = static_cast<int>(index % static_cast<std::size_t>(params->channels));
        const std::uint8_t conv_q =
            requantize_accumulator_to_conv_code_float(*params, producer_i32[index], channel);
        consumer_input_s8[index] = silu_lut_value_reference(lut_256_s8, conv_q);
    }
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" int y26_activation_requant_silu_int8_lut_scalar_unrolled(const Y26ActivationRequantParams* params,
                                                                     const std::int32_t* producer_i32,
                                                                     const std::int8_t* lut_256_s8,
                                                                     std::int8_t* consumer_input_s8) {
    if (!activation_params_valid(params) || producer_i32 == nullptr || lut_256_s8 == nullptr ||
        consumer_input_s8 == nullptr || !stage9_channel_loop_supported(*params)) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    requant_lut_scalar_unrolled_fused(*params, producer_i32, lut_256_s8, consumer_input_s8);
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" int y26_activation_requant_silu_int8_lut_fixed_requant(const Y26ActivationRequantParams* params,
                                                                   const Y26FixedRequantParams* per_channel_params,
                                                                   const std::int32_t* producer_i32,
                                                                   const std::int8_t* lut_256_s8,
                                                                   std::int8_t* consumer_input_s8) {
    if (!activation_params_valid(params) || per_channel_params == nullptr || producer_i32 == nullptr ||
        lut_256_s8 == nullptr || consumer_input_s8 == nullptr || !stage9_channel_loop_supported(*params)) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    requant_lut_fixed_unrolled_fused(*params, per_channel_params, producer_i32, lut_256_s8, consumer_input_s8);
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" int y26_activation_requant_silu_int8_lut_rvv_f32(const Y26ActivationRequantParams* params,
                                                             const std::int32_t* producer_i32,
                                                             const std::int8_t* lut_256_s8,
                                                             std::int8_t* consumer_input_s8) {
    if (!activation_params_valid(params) || producer_i32 == nullptr || lut_256_s8 == nullptr ||
        consumer_input_s8 == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
#if defined(__riscv_vector)
    return requant_lut_rvv_f32_impl(*params, producer_i32, lut_256_s8, consumer_input_s8);
#else
    return Y26_CONV_STATUS_NOT_BUILT_WITH_IME;
#endif
}

extern "C" int y26_activation_requant_silu_int8_lut_scalar_unrolled_profile(
    const Y26ActivationRequantParams* params,
    const std::int32_t* producer_i32,
    const std::int8_t* lut_256_s8,
    std::uint8_t* conv_code_u8,
    std::int8_t* consumer_input_s8,
    Y26Stage9ActivationTimingUs* timing) {
    if (!activation_params_valid(params) || producer_i32 == nullptr || lut_256_s8 == nullptr ||
        conv_code_u8 == nullptr || consumer_input_s8 == nullptr || timing == nullptr ||
        !stage9_channel_loop_supported(*params)) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    *timing = Y26Stage9ActivationTimingUs {};
    const auto total_begin = Clock::now();
    auto begin = Clock::now();
    requant_codes_scalar_unrolled(*params, producer_i32, conv_code_u8);
    auto end = Clock::now();
    timing->requant_arithmetic_us = elapsed_us(begin, end);

    begin = Clock::now();
    lut_lookup_unrolled(conv_code_u8, lut_256_s8, params->element_count, consumer_input_s8);
    end = Clock::now();
    timing->lut_lookup_us = elapsed_us(begin, end);
    timing->store_write_us = timing->lut_lookup_us;
    timing->packa_handoff_us = 0.0;
    timing->total_us = elapsed_us(total_begin, Clock::now());
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" int y26_activation_requant_silu_int8_lut_fixed_requant_profile(
    const Y26ActivationRequantParams* params,
    const Y26FixedRequantParams* per_channel_params,
    const std::int32_t* producer_i32,
    const std::int8_t* lut_256_s8,
    std::uint8_t* conv_code_u8,
    std::int8_t* consumer_input_s8,
    Y26Stage9ActivationTimingUs* timing) {
    if (!activation_params_valid(params) || per_channel_params == nullptr || producer_i32 == nullptr ||
        lut_256_s8 == nullptr || conv_code_u8 == nullptr || consumer_input_s8 == nullptr || timing == nullptr ||
        !stage9_channel_loop_supported(*params)) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    *timing = Y26Stage9ActivationTimingUs {};
    const auto total_begin = Clock::now();
    auto begin = Clock::now();
    requant_codes_fixed_unrolled(*params, per_channel_params, producer_i32, conv_code_u8);
    auto end = Clock::now();
    timing->requant_arithmetic_us = elapsed_us(begin, end);

    begin = Clock::now();
    lut_lookup_unrolled(conv_code_u8, lut_256_s8, params->element_count, consumer_input_s8);
    end = Clock::now();
    timing->lut_lookup_us = elapsed_us(begin, end);
    timing->store_write_us = timing->lut_lookup_us;
    timing->packa_handoff_us = 0.0;
    timing->total_us = elapsed_us(total_begin, Clock::now());
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" int y26_activation_requant_silu_fixed_requant_only(const Y26ActivationRequantParams* params,
                                                               const Y26FixedRequantParams* per_channel_params,
                                                               const std::int32_t* producer_i32,
                                                               std::int8_t* consumer_input_s8) {
    if (!activation_params_valid(params) || per_channel_params == nullptr || producer_i32 == nullptr ||
        consumer_input_s8 == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t index = 0; index < params->element_count; ++index) {
        const int channel = static_cast<int>(index % static_cast<std::size_t>(params->channels));
        const std::uint8_t conv_q =
            y26_requant_s32_to_u8_fixed_nearest_even(producer_i32[index], per_channel_params + channel);
        const float conv_dq =
            static_cast<float>(static_cast<int>(conv_q) - params->conv_output_zero_point_u8) *
            params->conv_output_scale;
        const std::uint8_t act_q = y26_quantize_u8_nearest_even_f32(
            silu_f32(conv_dq), params->act_output_scale, params->act_output_zero_point_u8);
        consumer_input_s8[index] = signed_storage_from_u8(act_q);
    }
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" int y26_activation_packa_1x1_mmt4d_4x8_from_nhwc(const std::int8_t* input_nhwc_s8,
                                                             int input_h,
                                                             int input_w,
                                                             int input_c,
                                                             std::int8_t* packed_tiles,
                                                             std::size_t packed_tile_bytes) {
    if (input_nhwc_s8 == nullptr || packed_tiles == nullptr || input_h <= 0 || input_w <= 0 || input_c <= 0) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    const int output_m = input_h * input_w;
    const int k_padded = ((input_c + 7) / 8) * 8;
    const std::size_t expected = static_cast<std::size_t>((output_m + 3) / 4) *
                                 static_cast<std::size_t>(k_padded / 8) * 32U;
    if (packed_tile_bytes < expected) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    std::memset(packed_tiles, 0, expected);
    for (int m0 = 0; m0 < output_m; m0 += 4) {
        const std::size_t panel = static_cast<std::size_t>(m0 / 4) * static_cast<std::size_t>(k_padded / 8) * 32U;
        for (int m = 0; m < 4; ++m) {
            const int flat_m = m0 + m;
            if (flat_m >= output_m) {
                continue;
            }
            const std::int8_t* src = input_nhwc_s8 + static_cast<std::size_t>(flat_m) * input_c;
            for (int c = 0; c < input_c; ++c) {
                const int k_tile = c / 8;
                const int k_lane = c % 8;
                packed_tiles[panel + static_cast<std::size_t>(k_tile) * 32U + m * 8 + k_lane] = src[c];
            }
        }
    }
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" int y26_activation_unpacka_1x1_mmt4d_4x8_to_nhwc(const std::int8_t* packed_tiles,
                                                             int input_h,
                                                             int input_w,
                                                             int input_c,
                                                             std::int8_t* output_nhwc_s8) {
    if (packed_tiles == nullptr || output_nhwc_s8 == nullptr || input_h <= 0 || input_w <= 0 || input_c <= 0) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    const int output_m = input_h * input_w;
    const int k_padded = ((input_c + 7) / 8) * 8;
    for (int m0 = 0; m0 < output_m; m0 += 4) {
        const std::size_t panel = static_cast<std::size_t>(m0 / 4) * static_cast<std::size_t>(k_padded / 8) * 32U;
        for (int m = 0; m < 4; ++m) {
            const int flat_m = m0 + m;
            if (flat_m >= output_m) {
                continue;
            }
            std::int8_t* dst = output_nhwc_s8 + static_cast<std::size_t>(flat_m) * input_c;
            for (int c = 0; c < input_c; ++c) {
                const int k_tile = c / 8;
                const int k_lane = c % 8;
                dst[c] = packed_tiles[panel + static_cast<std::size_t>(k_tile) * 32U + m * 8 + k_lane];
            }
        }
    }
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" int y26_activation_requant_silu_profile_scalar_float(const Y26ActivationRequantParams* params,
                                                                 const std::int32_t* producer_i32,
                                                                 std::uint8_t* conv_code_u8,
                                                                 float* conv_dq_f32,
                                                                 float* silu_f32_out,
                                                                 std::uint8_t* act_code_u8,
                                                                 std::int8_t* consumer_input_s8,
                                                                 Y26ActivationSubbucketTimingUs* timing) {
    if (!activation_params_valid(params) || producer_i32 == nullptr || conv_code_u8 == nullptr ||
        conv_dq_f32 == nullptr || silu_f32_out == nullptr || act_code_u8 == nullptr ||
        consumer_input_s8 == nullptr || timing == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    *timing = Y26ActivationSubbucketTimingUs {};
    const auto total_begin = Clock::now();

    auto begin = Clock::now();
    for (std::size_t index = 0; index < params->element_count; ++index) {
        const int channel = static_cast<int>(index % static_cast<std::size_t>(params->channels));
        conv_code_u8[index] = requantize_accumulator_to_conv_code_float(*params, producer_i32[index], channel);
    }
    auto end = Clock::now();
    timing->corr_i32_to_conv_out_quant_code_us = elapsed_us(begin, end);

    begin = Clock::now();
    for (std::size_t index = 0; index < params->element_count; ++index) {
        conv_dq_f32[index] =
            static_cast<float>(static_cast<int>(conv_code_u8[index]) - params->conv_output_zero_point_u8) *
            params->conv_output_scale;
    }
    end = Clock::now();
    timing->conv_out_code_to_float_dequant_us = elapsed_us(begin, end);

    begin = Clock::now();
    for (std::size_t index = 0; index < params->element_count; ++index) {
        silu_f32_out[index] = silu_f32(conv_dq_f32[index]);
    }
    end = Clock::now();
    timing->float_silu_sigmoid_mul_us = elapsed_us(begin, end);

    begin = Clock::now();
    for (std::size_t index = 0; index < params->element_count; ++index) {
        act_code_u8[index] = y26_quantize_u8_nearest_even_f32(
            silu_f32_out[index], params->act_output_scale, params->act_output_zero_point_u8);
    }
    end = Clock::now();
    timing->act_quant_float_to_uint8_us = elapsed_us(begin, end);

    begin = Clock::now();
    for (std::size_t index = 0; index < params->element_count; ++index) {
        consumer_input_s8[index] = signed_storage_from_u8(act_code_u8[index]);
    }
    end = Clock::now();
    timing->signed_storage_shift_us = elapsed_us(begin, end);
    timing->layout_or_pack_handoff_us = 0.0;
    timing->combined_current_fallback_us = elapsed_us(total_begin, Clock::now());
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" int y26_conv_output_quantize_i32_to_u8_scalar_unrolled(const Y26ConvOutputQuantizeParams* params,
                                                                   const std::int32_t* producer_i32,
                                                                   std::uint8_t* output_u8) {
    if (!conv_output_quantize_params_valid(params) || producer_i32 == nullptr || output_u8 == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    conv_output_quantize_scalar_unrolled_impl(*params, producer_i32, output_u8);
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" int y26_conv_output_quantize_i32_to_u8_rvv_f32(const Y26ConvOutputQuantizeParams* params,
                                                           const std::int32_t* producer_i32,
                                                           std::uint8_t* output_u8) {
    if (!conv_output_quantize_params_valid(params) || producer_i32 == nullptr || output_u8 == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
#if defined(__riscv_vector)
    return conv_output_quantize_rvv_f32_impl(*params, producer_i32, output_u8);
#else
    conv_output_quantize_scalar_unrolled_impl(*params, producer_i32, output_u8);
    return Y26_CONV_STATUS_SUCCESS;
#endif
}

extern "C" int y26_conv_output_quantize_i32_to_u8_rvv_f32_direct_store(
    const Y26ConvOutputQuantizeParams* params,
    const std::int32_t* producer_i32,
    std::uint8_t* output_u8) {
    if (!conv_output_quantize_params_valid(params) || producer_i32 == nullptr || output_u8 == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
#if defined(__riscv_vector)
    return conv_output_quantize_rvv_f32_direct_store_impl(*params, producer_i32, output_u8);
#else
    conv_output_quantize_scalar_unrolled_impl(*params, producer_i32, output_u8);
    return Y26_CONV_STATUS_SUCCESS;
#endif
}
