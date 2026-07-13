#include "y26_k1x_stage51_q62.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

__extension__ using Signed128 = __int128;
__extension__ using Unsigned128 = unsigned __int128;

std::int64_t reference(std::int64_t value, std::int64_t multiplier_m63) {
    const Signed128 product = static_cast<Signed128>(value) * multiplier_m63;
    const bool negative = product < 0;
    const Unsigned128 bits = static_cast<Unsigned128>(product);
    const Unsigned128 magnitude = negative ? (~bits) + 1U : bits;
    Unsigned128 quotient = magnitude >> 63U;
    const Unsigned128 remainder = magnitude & ((static_cast<Unsigned128>(1) << 63U) - 1U);
    const Unsigned128 half = static_cast<Unsigned128>(1) << 62U;
    if (remainder > half || (remainder == half && (quotient & 1U) != 0)) ++quotient;
    const auto rounded = static_cast<std::int64_t>(quotient);
    return negative ? -rounded : rounded;
}

#if defined(__riscv)
std::uint32_t read_vcsr() noexcept {
    std::uint32_t value = 0;
    asm volatile("csrr %0, vcsr" : "=r"(value));
    return value & 7U;
}

void write_vcsr(std::uint32_t value) noexcept {
    asm volatile("csrw vcsr, %0" : : "r"(value & 7U) : "memory");
}
#endif

}  // namespace

int main() {
    constexpr std::array<std::int64_t, 32> values {
        0, 1, -1, 2, -2, 3, -3, 4,
        -4, 5, -5, 7, -7, 15, -15, 31,
        -31, 127, -127, 255, -255, 32767, -32767, 1048575,
        -1048575, 36074272, -36074272, std::numeric_limits<std::int32_t>::max(),
        std::numeric_limits<std::int32_t>::min(), 9, -9, 11,
    };
    constexpr std::array<std::int64_t, 4> multipliers {
        std::int64_t{1} << 62U,
        (std::int64_t{1} << 62U) + 2,
        (std::int64_t{1} << 61U) + 104729,
        std::numeric_limits<std::int64_t>::max() - 2,
    };

    std::size_t cases = 0;
    std::size_t mismatches = 0;
    bool state_restored = true;
    bool saturation_clear = true;
    for (std::uint32_t ambient = 0; ambient < 8; ++ambient) {
#if defined(__riscv)
        write_vcsr(ambient);
#endif
        for (std::size_t base = 0; base < values.size(); base += 4) {
            std::array<std::int64_t, 4> input {};
            std::array<std::int64_t, 4> m63 {};
            std::array<std::int64_t, 4> actual {};
            for (std::size_t lane = 0; lane < 4; ++lane) {
                input[lane] = values[base + lane];
                m63[lane] = multipliers[(base / 4 + lane) % multipliers.size()];
            }
            y26::stage51::VectorFixedPointResult result;
            const bool status = y26::stage51::q62_vsmul_m63_i64x4_guarded(
                input.data(), m63.data(), actual.data(), &result);
            state_restored = state_restored && result.restored;
            saturation_clear = saturation_clear && !result.saturated;
            if (!status) ++mismatches;
            for (std::size_t lane = 0; lane < 4; ++lane) {
                ++cases;
                if (actual[lane] != reference(input[lane], m63[lane])) ++mismatches;
            }
#if defined(__riscv)
            if (read_vcsr() != ambient) {
                state_restored = false;
                ++mismatches;
            }
#endif
        }
    }
    std::cout << "contract=K1X_INT8_V1\nimplementation=explicit_vsmul_e64\n"
              << "cases=" << cases << "\nmismatches=" << mismatches
              << "\nvcsr_restored=" << (state_restored ? 1 : 0)
              << "\nvxsat_clear=" << (saturation_clear ? 1 : 0)
              << "\nrvv_execution="
#if defined(__riscv)
              << 1
#else
              << 0
#endif
              << '\n';
    return mismatches == 0 && state_restored && saturation_clear ? 0 : 3;
}
