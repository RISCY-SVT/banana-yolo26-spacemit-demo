#include "y26_k1x_int8_v1.h"
#include "y26_k1x_stage51_q62.h"

#include <array>
#include <cstdint>
#include <iostream>

int main() {
    constexpr std::array<std::int64_t, 4> multipliers {
        INT64_C(4611686018427387903), INT64_C(3458764513820540928),
        INT64_C(2305843009213693952), INT64_C(1152921504606846976),
    };
    const std::array<std::int64_t, 4> m63 {
        multipliers[0] * 2, multipliers[1] * 2,
        multipliers[2] * 2, multipliers[3] * 2,
    };
    const std::array<std::array<std::int64_t, 4>, 8> cases {{
        {{0, 1, -1, 2}},
        {{3, -3, 5, -5}},
        {{127, 128, 129, 130}},
        {{-127, -128, -129, -130}},
        {{1023, -1023, 2047, -2047}},
        {{INT32_MAX, INT32_MIN, INT32_MAX - 1, INT32_MIN + 1}},
        {{255, 511, 1024, 4096}},
        {{-255, -511, -1024, -4096}},
    }};
    for (const auto& values : cases) {
        std::array<std::int8_t, 4> actual {};
        y26::stage51::VectorFixedPointState state;
        if (!y26::stage51::begin_q62_vector_rne(&state)) return 1;
        y26::stage51::q62_vsmul_m63_i64x4_to_s8(
            values.data(), m63.data(), 117, actual.data());
        const auto result = y26::stage51::end_q62_vector_rne(&state);
        if (!result.restored || result.saturated) return 2;
        for (std::size_t lane = 0; lane < actual.size(); ++lane) {
            const y26::int8_v1::RequantAsset asset {multipliers[lane], 62, 117, 0, 255};
            std::uint8_t expected = 0;
            if (!y26::int8_v1::requantize_u8(values[lane], asset, &expected)) return 3;
            if (actual[lane] != y26::int8_v1::signed_storage(expected)) {
                std::cerr << "E2c2 mismatch at lane " << lane << '\n';
                return 4;
            }
        }
    }
    std::array<std::int8_t, 256> inverse_lut {};
    for (std::size_t code = 0; code < inverse_lut.size(); ++code) {
        inverse_lut[code] = y26::int8_v1::signed_storage(
            static_cast<std::uint8_t>(255U - code));
    }
    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
        std::array<std::int64_t, 8> values {};
        std::array<std::int64_t, 8> multipliers8 {};
        std::array<std::int64_t, 8> m63_8 {};
        for (std::size_t lane = 0; lane < values.size(); ++lane) {
            values[lane] = cases[(case_index + lane / 4U) % cases.size()][lane % 4U];
            multipliers8[lane] = multipliers[lane % 4U];
            m63_8[lane] = multipliers8[lane] * 2;
        }
        std::array<std::int8_t, 8> actual {};
        std::array<std::int8_t, 8> actual_lut {};
        y26::stage51::VectorFixedPointState state;
        if (!y26::stage51::begin_q62_vector_rne(&state)) return 5;
        y26::stage51::q62_vsmul_m63_i64x8_to_s8(
            values.data(), m63_8.data(), 117, actual.data());
        y26::stage51::q62_vsmul_m63_i64x8_lut_to_s8(
            values.data(), m63_8.data(), 117, inverse_lut.data(), actual_lut.data());
        const auto result = y26::stage51::end_q62_vector_rne(&state);
        if (!result.restored || result.saturated) return 6;
        for (std::size_t lane = 0; lane < actual.size(); ++lane) {
            const y26::int8_v1::RequantAsset asset {multipliers8[lane], 62, 117, 0, 255};
            std::uint8_t expected = 0;
            if (!y26::int8_v1::requantize_u8(values[lane], asset, &expected)) return 7;
            if (actual[lane] != y26::int8_v1::signed_storage(expected) ||
                actual_lut[lane] != inverse_lut[expected]) {
                std::cerr << "E2c3 mismatch at lane " << lane << '\n';
                return 8;
            }
        }
    }
    return 0;
}
