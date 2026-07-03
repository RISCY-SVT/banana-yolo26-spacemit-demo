#include "y26_k1x_conv_kernels.h"
#include "y26_k1x_engine.h"

#include <array>

int main() {
    std::array<std::int8_t, 4 * 10> src{};
    for (std::size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<std::int8_t>(i);
    }

    std::array<std::int8_t, 32> packed{};
    y26_k1x::pack_a_row_major_4x8(src.data(), 10, packed);

    for (std::ptrdiff_t m = 0; m < 4; ++m) {
        for (std::ptrdiff_t k = 0; k < 8; ++k) {
            if (packed[static_cast<std::size_t>(m * 8 + k)] != src[static_cast<std::size_t>(m * 10 + k)]) {
                return 1;
            }
        }
    }

    std::array<std::int8_t, 3 * 5> tail_src {};
    for (std::size_t i = 0; i < tail_src.size(); ++i) {
        tail_src[i] = static_cast<std::int8_t>(20 + static_cast<int>(i));
    }
    std::array<std::int8_t, 32> tail_packed {};
    y26_pack_a_mmt4d_4x8_s8(tail_src.data(), 3, 5, 5, 1, 2, tail_packed.data());
    for (int m = 0; m < 4; ++m) {
        for (int k = 0; k < 8; ++k) {
            const int src_m = 1 + m;
            const int src_k = 2 + k;
            const auto expected =
                (src_m < 3 && src_k < 5) ? tail_src[static_cast<std::size_t>(src_m * 5 + src_k)] : 0;
            if (tail_packed[static_cast<std::size_t>(m * 8 + k)] != expected) {
                return 1;
            }
        }
    }
    return 0;
}
