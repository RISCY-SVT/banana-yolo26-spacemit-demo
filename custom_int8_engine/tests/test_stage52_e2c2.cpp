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
    return 0;
}
