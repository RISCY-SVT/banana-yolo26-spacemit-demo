#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

enum Y26ActivationMode {
    Y26_ACTIVATION_MODE_SCALAR_FLOAT_REFERENCE = 0,
    Y26_ACTIVATION_MODE_FIXED_REQUANT_ONLY = 1,
    Y26_ACTIVATION_MODE_INT8_LUT = 2,
    Y26_ACTIVATION_MODE_FUSED_LUT_PACK = 3,
    Y26_ACTIVATION_MODE_STAGE9_SCALAR_UNROLLED_LUT = 4,
    Y26_ACTIVATION_MODE_STAGE9_FIXED_REQUANT_LUT = 5,
    Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT = 6,
    Y26_ACTIVATION_MODE_STAGE9_FUSED_CURRENT_LAYOUT = 7,
};

struct Y26ActivationRequantParams {
    std::size_t element_count;
    int channels;
    float input_scale;
    const float* weight_scales;
    float conv_output_scale;
    int conv_output_zero_point_u8;
    float act_output_scale;
    int act_output_zero_point_u8;
};

struct Y26FixedRequantParams {
    std::int64_t multiplier_q31;
    int exponent;
    int output_zero_point_u8;
};

struct Y26ActivationSubbucketTimingUs {
    double corr_i32_to_conv_out_quant_code_us;
    double conv_out_code_to_float_dequant_us;
    double float_silu_sigmoid_mul_us;
    double act_quant_float_to_uint8_us;
    double signed_storage_shift_us;
    double layout_or_pack_handoff_us;
    double combined_current_fallback_us;
};

struct Y26Stage9ActivationTimingUs {
    double requant_arithmetic_us;
    double lut_lookup_us;
    double store_write_us;
    double packa_handoff_us;
    double total_us;
};

struct Y26ConvOutputQuantizeParams {
    std::size_t element_count;
    int channels;
    float input_scale;
    const float* weight_scales;
    float output_scale;
    int output_zero_point_u8;
};

std::uint8_t y26_quantize_u8_nearest_even_f32(float value, float scale, int zero_point_u8);

int y26_fixed_requant_params_from_multiplier(double multiplier,
                                             int output_zero_point_u8,
                                             Y26FixedRequantParams* params);

std::uint8_t y26_requant_s32_to_u8_fixed_nearest_even(std::int32_t value,
                                                      const Y26FixedRequantParams* params);

int y26_build_silu_u8_to_s8_lut(float conv_output_scale,
                                int conv_output_zero_point_u8,
                                float act_output_scale,
                                int act_output_zero_point_u8,
                                std::int8_t* lut_256_s8);

int y26_build_fixed_requant_params_per_channel(const Y26ActivationRequantParams* params,
                                               Y26FixedRequantParams* per_channel_params);

int y26_activation_requant_silu_scalar_float(const Y26ActivationRequantParams* params,
                                             const std::int32_t* producer_i32,
                                             std::int8_t* consumer_input_s8);

int y26_activation_requant_silu_int8_lut(const Y26ActivationRequantParams* params,
                                         const std::int32_t* producer_i32,
                                         const std::int8_t* lut_256_s8,
                                         std::int8_t* consumer_input_s8);

int y26_activation_requant_silu_int8_lut_scalar_unrolled(const Y26ActivationRequantParams* params,
                                                         const std::int32_t* producer_i32,
                                                         const std::int8_t* lut_256_s8,
                                                         std::int8_t* consumer_input_s8);

int y26_activation_requant_silu_int8_lut_fixed_requant(const Y26ActivationRequantParams* params,
                                                       const Y26FixedRequantParams* per_channel_params,
                                                       const std::int32_t* producer_i32,
                                                       const std::int8_t* lut_256_s8,
                                                       std::int8_t* consumer_input_s8);

int y26_activation_requant_silu_int8_lut_rvv_f32(const Y26ActivationRequantParams* params,
                                                 const std::int32_t* producer_i32,
                                                 const std::int8_t* lut_256_s8,
                                                 std::int8_t* consumer_input_s8);

int y26_activation_requant_silu_int8_lut_scalar_unrolled_profile(
    const Y26ActivationRequantParams* params,
    const std::int32_t* producer_i32,
    const std::int8_t* lut_256_s8,
    std::uint8_t* conv_code_u8,
    std::int8_t* consumer_input_s8,
    Y26Stage9ActivationTimingUs* timing);

int y26_activation_requant_silu_int8_lut_fixed_requant_profile(
    const Y26ActivationRequantParams* params,
    const Y26FixedRequantParams* per_channel_params,
    const std::int32_t* producer_i32,
    const std::int8_t* lut_256_s8,
    std::uint8_t* conv_code_u8,
    std::int8_t* consumer_input_s8,
    Y26Stage9ActivationTimingUs* timing);

int y26_activation_packa_1x1_mmt4d_4x8_from_nhwc(const std::int8_t* input_nhwc_s8,
                                                 int input_h,
                                                 int input_w,
                                                 int input_c,
                                                 std::int8_t* packed_tiles,
                                                 std::size_t packed_tile_bytes);

int y26_activation_unpacka_1x1_mmt4d_4x8_to_nhwc(const std::int8_t* packed_tiles,
                                                 int input_h,
                                                 int input_w,
                                                 int input_c,
                                                 std::int8_t* output_nhwc_s8);

int y26_activation_requant_silu_fixed_requant_only(const Y26ActivationRequantParams* params,
                                                   const Y26FixedRequantParams* per_channel_params,
                                                   const std::int32_t* producer_i32,
                                                   std::int8_t* consumer_input_s8);

int y26_activation_requant_silu_profile_scalar_float(const Y26ActivationRequantParams* params,
                                                     const std::int32_t* producer_i32,
                                                     std::uint8_t* conv_code_u8,
                                                     float* conv_dq_f32,
                                                     float* silu_f32,
                                                     std::uint8_t* act_code_u8,
                                                     std::int8_t* consumer_input_s8,
                                                     Y26ActivationSubbucketTimingUs* timing);

int y26_conv_output_quantize_i32_to_u8_scalar_unrolled(const Y26ConvOutputQuantizeParams* params,
                                                       const std::int32_t* producer_i32,
                                                       std::uint8_t* output_u8);

int y26_conv_output_quantize_i32_to_u8_rvv_f32(const Y26ConvOutputQuantizeParams* params,
                                               const std::int32_t* producer_i32,
                                               std::uint8_t* output_u8);

int y26_conv_output_quantize_i32_to_u8_rvv_f32_direct_store(const Y26ConvOutputQuantizeParams* params,
                                                            const std::int32_t* producer_i32,
                                                            std::uint8_t* output_u8);

}
