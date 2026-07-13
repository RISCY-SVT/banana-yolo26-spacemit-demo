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

VectorFixedPointResult end_q62_vector_rne(VectorFixedPointState* state) noexcept;

// Test/diagnostic wrapper that brackets one vector operation and restores vcsr.
bool q62_vsmul_m63_i64x4_guarded(const std::int64_t* values,
                                 const std::int64_t* multipliers_m63,
                                 std::int64_t* rounded,
                                 VectorFixedPointResult* result) noexcept;

}  // namespace y26::stage51
