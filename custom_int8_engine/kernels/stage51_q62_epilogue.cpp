#include "y26_k1x_stage51_q62.h"

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
