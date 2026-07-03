#include "y26_k1x_engine.h"

#include <array>
#include <cassert>

int main() {
    std::array<std::int8_t, 4 * 10> src{};
    for (std::size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<std::int8_t>(i);
    }

    std::array<std::int8_t, 32> packed{};
    y26_k1x::pack_a_row_major_4x8(src.data(), 10, packed);

    for (std::ptrdiff_t m = 0; m < 4; ++m) {
        for (std::ptrdiff_t k = 0; k < 8; ++k) {
            assert(packed[static_cast<std::size_t>(m * 8 + k)] == src[static_cast<std::size_t>(m * 10 + k)]);
        }
    }
    return 0;
}
