#include "y26_k1x_stage51_q62.h"

#include <algorithm>
#include <cstdint>

namespace y26::stage51 {
namespace {

#if !defined(__riscv)
__extension__ using Signed128 = __int128;
__extension__ using Unsigned128 = unsigned __int128;

std::int64_t round_shift_right_even(Signed128 product, unsigned shift) noexcept {
    const bool negative = product < 0;
    const Unsigned128 bits = static_cast<Unsigned128>(product);
    const Unsigned128 magnitude = negative ? (~bits) + 1U : bits;
    Unsigned128 quotient = magnitude >> shift;
    const Unsigned128 remainder = magnitude & ((static_cast<Unsigned128>(1) << shift) - 1U);
    const Unsigned128 half = static_cast<Unsigned128>(1) << (shift - 1U);
    if (remainder > half || (remainder == half && (quotient & 1U) != 0)) ++quotient;
    const auto rounded = static_cast<std::int64_t>(quotient);
    return negative ? -rounded : rounded;
}
#endif

}  // namespace

bool begin_q62_vector_rne(VectorFixedPointState* state) noexcept {
    if (state == nullptr || state->active) return false;
#if defined(__riscv)
    std::uint32_t saved = 0;
    asm volatile(
        "csrr %[saved], vcsr\n\t"
        "csrwi vcsr, 2\n\t"
        : [saved] "=r"(saved)
        :
        : "memory");
    state->saved_vcsr = saved & 7U;
#else
    state->saved_vcsr = 0;
#endif
    state->active = true;
    return true;
}

void q62_vsmul_m63_i64x4(const std::int64_t* values,
                         const std::int64_t* multipliers_m63,
                         std::int64_t* rounded) noexcept {
#if defined(__riscv)
    asm volatile(
        "vsetivli zero, 4, e64, m1, ta, ma\n\t"
        "vle64.v v0, (%[values])\n\t"
        "vle64.v v1, (%[multipliers])\n\t"
        "vsmul.vv v2, v0, v1\n\t"
        "vse64.v v2, (%[rounded])\n\t"
        :
        : [values] "r"(values), [multipliers] "r"(multipliers_m63), [rounded] "r"(rounded)
        : "v0", "v1", "v2", "memory");
#else
    for (int lane = 0; lane < 4; ++lane) {
        // M63 is exactly 2 * the package's Q62 multiplier.
        rounded[lane] = round_shift_right_even(
            static_cast<Signed128>(values[lane]) * static_cast<Signed128>(multipliers_m63[lane]), 63U);
    }
#endif
}

void q62_vsmul_m63_i64x4_to_s8(const std::int64_t* values,
                               const std::int64_t* multipliers_m63,
                               std::int64_t output_zero_point,
                               std::int8_t* output_s8) noexcept {
#if defined(__riscv)
    const std::int64_t maximum = 255;
    const std::int64_t signed_offset = -128;
    asm volatile(
        "vsetivli zero, 4, e64, m1, ta, ma\n\t"
        "vle64.v v0, (%[values])\n\t"
        "vle64.v v1, (%[multipliers])\n\t"
        "vsmul.vv v2, v0, v1\n\t"
        "vadd.vx v2, v2, %[zero_point]\n\t"
        "vmax.vx v2, v2, zero\n\t"
        "vmin.vx v2, v2, %[maximum]\n\t"
        "vadd.vx v2, v2, %[signed_offset]\n\t"
        "vsetivli zero, 4, e32, mf2, ta, ma\n\t"
        "vnclip.wi v4, v2, 0\n\t"
        "vsetivli zero, 4, e16, mf4, ta, ma\n\t"
        "vnclip.wi v6, v4, 0\n\t"
        "vsetivli zero, 4, e8, mf8, ta, ma\n\t"
        "vnclip.wi v8, v6, 0\n\t"
        "vse8.v v8, (%[output])\n\t"
        :
        : [values] "r"(values), [multipliers] "r"(multipliers_m63),
          [zero_point] "r"(output_zero_point), [maximum] "r"(maximum),
          [signed_offset] "r"(signed_offset), [output] "r"(output_s8)
        : "v0", "v1", "v2", "v4", "v6", "v8", "memory");
#else
    std::int64_t rounded[4] {};
    q62_vsmul_m63_i64x4(values, multipliers_m63, rounded);
    for (int lane = 0; lane < 4; ++lane) {
        const std::int64_t code = std::clamp(rounded[lane] + output_zero_point,
                                             std::int64_t {0}, std::int64_t {255});
        output_s8[lane] = static_cast<std::int8_t>(code - 128);
    }
#endif
}

void q62_vsmul_m63_i64x8_to_s8(const std::int64_t* values,
                               const std::int64_t* multipliers_m63,
                               std::int64_t output_zero_point,
                               std::int8_t* output_s8) noexcept {
#if defined(__riscv)
    const std::int64_t maximum = 255;
    const std::int64_t signed_mask = 128;
    asm volatile(
        "vsetivli zero, 8, e64, m2, ta, ma\n\t"
        "vle64.v v0, (%[values])\n\t"
        "vle64.v v2, (%[multipliers])\n\t"
        "vsmul.vv v4, v0, v2\n\t"
        "vadd.vx v4, v4, %[zero_point]\n\t"
        "vmax.vx v4, v4, zero\n\t"
        "vmin.vx v4, v4, %[maximum]\n\t"
        "vsetivli zero, 8, e32, m1, ta, ma\n\t"
        "vnclipu.wi v6, v4, 0\n\t"
        "vsetivli zero, 8, e16, mf2, ta, ma\n\t"
        "vnclipu.wi v8, v6, 0\n\t"
        "vsetivli zero, 8, e8, mf4, ta, ma\n\t"
        "vnclipu.wi v10, v8, 0\n\t"
        "vxor.vx v10, v10, %[signed_mask]\n\t"
        "vse8.v v10, (%[output])\n\t"
        :
        : [values] "r"(values), [multipliers] "r"(multipliers_m63),
          [zero_point] "r"(output_zero_point), [maximum] "r"(maximum),
          [signed_mask] "r"(signed_mask), [output] "r"(output_s8)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v8", "v10", "memory");
#else
    for (int lane = 0; lane < 8; ++lane) {
        const std::int64_t rounded = round_shift_right_even(
            static_cast<Signed128>(values[lane]) *
            static_cast<Signed128>(multipliers_m63[lane]), 63U);
        const std::int64_t code = std::clamp(
            rounded + output_zero_point, std::int64_t {0}, std::int64_t {255});
        output_s8[lane] = static_cast<std::int8_t>(code - 128);
    }
#endif
}

void q62_vsmul_m63_i64x8_lut_to_s8(const std::int64_t* values,
                                   const std::int64_t* multipliers_m63,
                                   std::int64_t output_zero_point,
                                   const std::int8_t* lut_s8,
                                   std::int8_t* output_s8) noexcept {
#if defined(__riscv)
    const std::int64_t maximum = 255;
    asm volatile(
        "vsetivli zero, 8, e64, m2, ta, ma\n\t"
        "vle64.v v0, (%[values])\n\t"
        "vle64.v v2, (%[multipliers])\n\t"
        "vsmul.vv v4, v0, v2\n\t"
        "vadd.vx v4, v4, %[zero_point]\n\t"
        "vmax.vx v4, v4, zero\n\t"
        "vmin.vx v4, v4, %[maximum]\n\t"
        "vsetivli zero, 8, e32, m1, ta, ma\n\t"
        "vnclipu.wi v6, v4, 0\n\t"
        "vsetivli zero, 8, e16, mf2, ta, ma\n\t"
        "vnclipu.wi v8, v6, 0\n\t"
        "vsetivli zero, 8, e8, mf4, ta, ma\n\t"
        "vnclipu.wi v10, v8, 0\n\t"
        "vluxei8.v v12, (%[lut]), v10\n\t"
        "vse8.v v12, (%[output])\n\t"
        :
        : [values] "r"(values), [multipliers] "r"(multipliers_m63),
          [zero_point] "r"(output_zero_point), [maximum] "r"(maximum),
          [lut] "r"(lut_s8), [output] "r"(output_s8)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v8", "v10", "v12", "memory");
#else
    std::int8_t quantized[8] {};
    q62_vsmul_m63_i64x8_to_s8(values, multipliers_m63, output_zero_point, quantized);
    for (int lane = 0; lane < 8; ++lane) {
        const std::uint8_t code = static_cast<std::uint8_t>(
            static_cast<int>(quantized[lane]) + 128);
        output_s8[lane] = lut_s8[code];
    }
#endif
}

void q62_e2c4_i32x4x2_bias_to_s8(const std::int32_t* values_low,
                                 const std::int32_t* values_high,
                                 const std::int64_t* corrected_bias,
                                 const std::int64_t* multipliers_m63,
                                 std::int64_t output_zero_point,
                                 std::int8_t* output_s8) noexcept {
#if defined(__riscv)
    const std::int64_t maximum = 255;
    const std::int64_t signed_mask = 128;
    asm volatile(
        "vsetivli zero, 4, e32, mf2, ta, ma\n\t"
        "vle32.v v0, (%[values_low])\n\t"
        "vle32.v v1, (%[values_high])\n\t"
        "vwadd.vx v2, v0, zero\n\t"
        "vwadd.vx v4, v1, zero\n\t"
        "vsetivli zero, 4, e64, m1, ta, ma\n\t"
        "vle64.v v6, (%[bias])\n\t"
        "addi t0, %[bias], 32\n\t"
        "vle64.v v7, (t0)\n\t"
        "vadd.vv v2, v2, v6\n\t"
        "vadd.vv v4, v4, v7\n\t"
        "vsetivli zero, 4, e64, m2, tu, ma\n\t"
        "vmv.v.v v8, v2\n\t"
        "vsetivli zero, 8, e64, m2, tu, ma\n\t"
        "vslideup.vi v8, v4, 4\n\t"
        "vle64.v v10, (%[multipliers])\n\t"
        "vsmul.vv v12, v8, v10\n\t"
        "vadd.vx v12, v12, %[zero_point]\n\t"
        "vmax.vx v12, v12, zero\n\t"
        "vmin.vx v12, v12, %[maximum]\n\t"
        "vsetivli zero, 8, e32, m1, ta, ma\n\t"
        "vnclipu.wi v14, v12, 0\n\t"
        "vsetivli zero, 8, e16, mf2, ta, ma\n\t"
        "vnclipu.wi v16, v14, 0\n\t"
        "vsetivli zero, 8, e8, mf4, ta, ma\n\t"
        "vnclipu.wi v18, v16, 0\n\t"
        "vxor.vx v18, v18, %[signed_mask]\n\t"
        "vse8.v v18, (%[output])\n\t"
        :
        : [values_low] "r"(values_low), [values_high] "r"(values_high),
          [bias] "r"(corrected_bias), [multipliers] "r"(multipliers_m63),
          [zero_point] "r"(output_zero_point), [maximum] "r"(maximum),
          [signed_mask] "r"(signed_mask), [output] "r"(output_s8)
        : "t0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v16", "v18", "memory");
#else
    std::int64_t corrected[8] {};
    for (int lane = 0; lane < 4; ++lane) {
        corrected[lane] = static_cast<std::int64_t>(values_low[lane]) + corrected_bias[lane];
        corrected[lane + 4] = static_cast<std::int64_t>(values_high[lane]) + corrected_bias[lane + 4];
    }
    q62_vsmul_m63_i64x8_to_s8(
        corrected, multipliers_m63, output_zero_point, output_s8);
#endif
}

void q62_e2c4_i32x4x2_bias_lut_to_s8(const std::int32_t* values_low,
                                     const std::int32_t* values_high,
                                     const std::int64_t* corrected_bias,
                                     const std::int64_t* multipliers_m63,
                                     std::int64_t output_zero_point,
                                     const std::int8_t* lut_s8,
                                     std::int8_t* output_s8) noexcept {
#if defined(__riscv)
    const std::int64_t maximum = 255;
    asm volatile(
        "vsetivli zero, 4, e32, mf2, ta, ma\n\t"
        "vle32.v v0, (%[values_low])\n\t"
        "vle32.v v1, (%[values_high])\n\t"
        "vwadd.vx v2, v0, zero\n\t"
        "vwadd.vx v4, v1, zero\n\t"
        "vsetivli zero, 4, e64, m1, ta, ma\n\t"
        "vle64.v v6, (%[bias])\n\t"
        "addi t0, %[bias], 32\n\t"
        "vle64.v v7, (t0)\n\t"
        "vadd.vv v2, v2, v6\n\t"
        "vadd.vv v4, v4, v7\n\t"
        "vsetivli zero, 4, e64, m2, tu, ma\n\t"
        "vmv.v.v v8, v2\n\t"
        "vsetivli zero, 8, e64, m2, tu, ma\n\t"
        "vslideup.vi v8, v4, 4\n\t"
        "vle64.v v10, (%[multipliers])\n\t"
        "vsmul.vv v12, v8, v10\n\t"
        "vadd.vx v12, v12, %[zero_point]\n\t"
        "vmax.vx v12, v12, zero\n\t"
        "vmin.vx v12, v12, %[maximum]\n\t"
        "vsetivli zero, 8, e32, m1, ta, ma\n\t"
        "vnclipu.wi v14, v12, 0\n\t"
        "vsetivli zero, 8, e16, mf2, ta, ma\n\t"
        "vnclipu.wi v16, v14, 0\n\t"
        "vsetivli zero, 8, e8, mf4, ta, ma\n\t"
        "vnclipu.wi v18, v16, 0\n\t"
        "vluxei8.v v20, (%[lut]), v18\n\t"
        "vse8.v v20, (%[output])\n\t"
        :
        : [values_low] "r"(values_low), [values_high] "r"(values_high),
          [bias] "r"(corrected_bias), [multipliers] "r"(multipliers_m63),
          [zero_point] "r"(output_zero_point), [maximum] "r"(maximum),
          [lut] "r"(lut_s8), [output] "r"(output_s8)
        : "t0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v16", "v18", "v20", "memory");
#else
    std::int64_t corrected[8] {};
    for (int lane = 0; lane < 4; ++lane) {
        corrected[lane] = static_cast<std::int64_t>(values_low[lane]) + corrected_bias[lane];
        corrected[lane + 4] = static_cast<std::int64_t>(values_high[lane]) + corrected_bias[lane + 4];
    }
    q62_vsmul_m63_i64x8_lut_to_s8(
        corrected, multipliers_m63, output_zero_point, lut_s8, output_s8);
#endif
}

VectorFixedPointResult end_q62_vector_rne(VectorFixedPointState* state) noexcept {
    VectorFixedPointResult result;
    if (state == nullptr || !state->active) return result;
#if defined(__riscv)
    std::uint32_t saturated = 0;
    std::uint32_t restored = 0;
    asm volatile(
        "csrr %[saturated], vxsat\n\t"
        "csrw vcsr, %[saved]\n\t"
        "csrr %[restored], vcsr\n\t"
        : [saturated] "=r"(saturated), [restored] "=r"(restored)
        : [saved] "r"(state->saved_vcsr)
        : "memory");
    result.saturated = (saturated & 1U) != 0;
    result.restored = (restored & 7U) == state->saved_vcsr;
#else
    result.restored = true;
#endif
    state->active = false;
    return result;
}

bool q62_vsmul_m63_i64x4_guarded(const std::int64_t* values,
                                 const std::int64_t* multipliers_m63,
                                 std::int64_t* rounded,
                                 VectorFixedPointResult* result) noexcept {
    if (values == nullptr || multipliers_m63 == nullptr || rounded == nullptr || result == nullptr) return false;
    VectorFixedPointState state;
    if (!begin_q62_vector_rne(&state)) return false;
    q62_vsmul_m63_i64x4(values, multipliers_m63, rounded);
    *result = end_q62_vector_rne(&state);
    return result->restored && !result->saturated;
}

}  // namespace y26::stage51
