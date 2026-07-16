#pragma once

#include <cstdint>

namespace y26::stage51 {

struct VectorFixedPointState {
    std::uint32_t saved_vcsr = 0;
    bool active = false;
};

struct VectorFixedPointResult {
    bool saturated = false;
    bool restored = false;
};

// Establishes RNE with a clear saturation flag on the current worker.
bool begin_q62_vector_rne(VectorFixedPointState* state) noexcept;

// Requires an active vector-RNE state. M63 is the positive Q62 multiplier
// shifted left by one, so vsmul.e64 implements the exact V1 Q62 quotient.
void q62_vsmul_m63_i64x4(const std::int64_t* values,
                         const std::int64_t* multipliers_m63,
                         std::int64_t* rounded) noexcept;

// Stage52 E2c2 sidecar: exact vector multiply, output-zero-point add,
// clamp, signed-storage conversion, narrow, and contiguous C4 store.
void q62_vsmul_m63_i64x4_to_s8(const std::int64_t* values,
                               const std::int64_t* multipliers_m63,
                               std::int64_t output_zero_point,
                               std::int8_t* output_s8) noexcept;

// Stage54 E2c3 sidecar: one exact C8 requant/store, optionally followed by an
// indexed byte LUT without a scalar lane loop or temporary output array.
void q62_vsmul_m63_i64x8_to_s8(const std::int64_t* values,
                               const std::int64_t* multipliers_m63,
                               std::int64_t output_zero_point,
                               std::int8_t* output_s8) noexcept;

void q62_vsmul_m63_i64x8_lut_to_s8(const std::int64_t* values,
                                   const std::int64_t* multipliers_m63,
                                   std::int64_t output_zero_point,
                                   const std::int8_t* lut_s8,
                                   std::int8_t* output_s8) noexcept;

// Stage55 E2c4 sidecar: widen two C4 accumulator groups, add C8 corrected
// bias, requantize, and store without constructing an intermediate i64 array.
void q62_e2c4_i32x4x2_bias_to_s8(const std::int32_t* values_low,
                                 const std::int32_t* values_high,
                                 const std::int64_t* corrected_bias,
                                 const std::int64_t* multipliers_m63,
                                 std::int64_t output_zero_point,
                                 std::int8_t* output_s8) noexcept;

void q62_e2c4_i32x4x2_bias_lut_to_s8(const std::int32_t* values_low,
                                     const std::int32_t* values_high,
                                     const std::int64_t* corrected_bias,
                                     const std::int64_t* multipliers_m63,
                                     std::int64_t output_zero_point,
                                     const std::int8_t* lut_s8,
                                     std::int8_t* output_s8) noexcept;

// Stage57 E2c5 sidecar: two independent C4 chains avoid the E2c4 C8
// vslideup dependency while retaining the exact Q62/RNE contract.
void q62_e2c5_i32x4x2_bias_to_s8(const std::int32_t* values_low,
                                 const std::int32_t* values_high,
                                 const std::int64_t* corrected_bias,
                                 const std::int64_t* multipliers_m63,
                                 std::int64_t output_zero_point,
                                 std::int8_t* output_s8) noexcept;

void q62_e2c5_i32x4x2_bias_lut_to_s8(const std::int32_t* values_low,
                                     const std::int32_t* values_high,
                                     const std::int64_t* corrected_bias,
                                     const std::int64_t* multipliers_m63,
                                     std::int64_t output_zero_point,
                                     const std::int8_t* lut_s8,
                                     std::int8_t* output_s8) noexcept;

// Attention MatMul C8 epilogue. The row correction is common to all eight
// columns; per-column right sums remain exact signed int64 inputs.
void q62_attention_i32x4x2_to_s8(const std::int32_t* values_low,
                                 const std::int32_t* values_high,
                                 const std::int64_t* right_sums,
                                 std::int64_t left_correction,
                                 std::int64_t common_correction,
                                 std::int64_t multiplier_m63,
                                 std::int64_t output_zero_point,
                                 std::int8_t* output_s8) noexcept;

VectorFixedPointResult end_q62_vector_rne(VectorFixedPointState* state) noexcept;

// Test/diagnostic wrapper that brackets one vector operation and restores vcsr.
bool q62_vsmul_m63_i64x4_guarded(const std::int64_t* values,
                                 const std::int64_t* multipliers_m63,
                                 std::int64_t* rounded,
                                 VectorFixedPointResult* result) noexcept;

}  // namespace y26::stage51
