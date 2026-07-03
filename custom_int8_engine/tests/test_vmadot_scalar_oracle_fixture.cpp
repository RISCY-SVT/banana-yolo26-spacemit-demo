#include "y26_k1x_engine.h"

#include <array>
#include <cassert>

int main() {
    std::array<std::int8_t, 32> a{};
    std::array<std::int8_t, 32> b{};
    for (std::ptrdiff_t i = 0; i < 32; ++i) {
        a[static_cast<std::size_t>(i)] = static_cast<std::int8_t>((i % 9) - 4);
        b[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(3 - (i % 7));
    }

    std::array<std::int32_t, 16> c{};
    y26_k1x::vmadot_scalar_4x4x8(a.data(), b.data(), c.data(), 4, false);

    for (std::ptrdiff_t m = 0; m < 4; ++m) {
        for (std::ptrdiff_t n = 0; n < 4; ++n) {
            std::int32_t expected = 0;
            for (std::ptrdiff_t k = 0; k < 8; ++k) {
                expected += static_cast<std::int32_t>(a[static_cast<std::size_t>(m * 8 + k)]) *
                            static_cast<std::int32_t>(b[static_cast<std::size_t>(n * 8 + k)]);
            }
            assert(c[static_cast<std::size_t>(m * 4 + n)] == expected);
        }
    }

    y26_k1x::vmadot_scalar_4x4x8(a.data(), b.data(), c.data(), 4, true);
    for (std::ptrdiff_t i = 0; i < 16; ++i) {
        assert((c[static_cast<std::size_t>(i)] % 2) == 0);
    }
    return 0;
}
