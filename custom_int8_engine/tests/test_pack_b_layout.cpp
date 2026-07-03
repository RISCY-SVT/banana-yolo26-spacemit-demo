#include "y26_k1x_engine.h"

#include <array>
#include <cassert>

int main() {
    std::array<std::int8_t, 4 * 9> src{};
    for (std::ptrdiff_t n = 0; n < 4; ++n) {
        for (std::ptrdiff_t k = 0; k < 9; ++k) {
            src[static_cast<std::size_t>(n * 9 + k)] = static_cast<std::int8_t>(50 + n * 9 + k);
        }
    }

    std::array<std::int8_t, 32> packed{};
    y26_k1x::pack_b_transposed_4x8(src.data(), 9, packed);

    for (std::ptrdiff_t n = 0; n < 4; ++n) {
        for (std::ptrdiff_t k = 0; k < 8; ++k) {
            assert(packed[static_cast<std::size_t>(n * 8 + k)] == src[static_cast<std::size_t>(n * 9 + k)]);
        }
    }
    return 0;
}
