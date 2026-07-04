#include "y26_k1x_activation.h"

#include "y26_k1x_conv_kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

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

float silu_f32(float value) {
    return value / (1.0f + std::exp(-value));
}

std::uint8_t clamp_u8(long value) {
    return static_cast<std::uint8_t>(std::max<long>(0, std::min<long>(255, value)));
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

}  // namespace

extern "C" std::uint8_t y26_quantize_u8_nearest_even_f32(float value, float scale, int zero_point_u8) {
    if (scale <= 0.0f || zero_point_u8 < 0 || zero_point_u8 > 255) {
        return 0;
    }
    const double scaled = static_cast<double>(value) / static_cast<double>(scale);
    const long rounded = static_cast<long>(std::nearbyint(scaled)) + static_cast<long>(zero_point_u8);
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
