#include "y26_k1x_vmadot.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace {

struct Case {
    std::string_view name;
    std::array<std::int8_t, 32> a {};
    std::array<std::int8_t, 32> b {};
    std::array<std::int32_t, 16> init {};
    bool accumulate = false;
};

std::uint32_t lcg_next(std::uint32_t state) {
    return state * 1664525U + 1013904223U;
}

std::int8_t sample_from_seed(std::uint32_t& state) {
    state = lcg_next(state);
    return static_cast<std::int8_t>((state >> 24U) - 128U);
}

Case make_random_case(std::uint32_t seed) {
    Case tc {};
    tc.name = "random";
    for (std::size_t i = 0; i < tc.a.size(); ++i) {
        tc.a[i] = sample_from_seed(seed);
        tc.b[i] = sample_from_seed(seed);
    }
    return tc;
}

void reference(const Case& tc, std::array<std::int32_t, 16>& out) {
    out = tc.init;
    for (std::size_t m = 0; m < 4; ++m) {
        for (std::size_t n = 0; n < 4; ++n) {
            std::int32_t acc = tc.accumulate ? tc.init[m * 4 + n] : 0;
            for (std::size_t k = 0; k < 8; ++k) {
                acc += static_cast<std::int32_t>(tc.a[m * 8 + k]) * static_cast<std::int32_t>(tc.b[n * 8 + k]);
            }
            out[m * 4 + n] = acc;
        }
    }
}

std::array<Case, 8> cases() {
    std::array<Case, 8> v {};
    v[0].name = "all_zeros";

    v[1].name = "all_ones";
    v[1].a.fill(1);
    v[1].b.fill(1);

    v[2].name = "ramp";
    for (std::size_t i = 0; i < 32; ++i) {
        v[2].a[i] = static_cast<std::int8_t>(static_cast<int>(i) - 16);
        v[2].b[i] = static_cast<std::int8_t>(15 - static_cast<int>(i));
    }

    v[3].name = "alternating_edges";
    for (std::size_t i = 0; i < 32; ++i) {
        v[3].a[i] = (i % 2U) == 0U ? static_cast<std::int8_t>(-128) : static_cast<std::int8_t>(127);
        v[3].b[i] = (i % 2U) == 0U ? static_cast<std::int8_t>(127) : static_cast<std::int8_t>(-128);
    }

    v[4] = make_random_case(0U);
    v[5] = make_random_case(1U);
    v[6] = make_random_case(12345U);

    v[7] = v[6];
    v[7].name = "accumulate_true";
    v[7].accumulate = true;
    for (std::size_t i = 0; i < 16; ++i) {
        v[7].init[i] = static_cast<std::int32_t>(static_cast<int>(i) * 19 - 143);
    }

    return v;
}

}  // namespace

int main() {
    for (const auto& tc : cases()) {
        std::array<std::int32_t, 16> expected {};
        std::array<std::int32_t, 16> actual = tc.init;
        reference(tc, expected);
        y26_vmadot_4x4x8_scalar_s8s8s32(tc.a.data(), tc.b.data(), actual.data(), tc.accumulate);
        assert(actual == expected);
    }

    return 0;
}
