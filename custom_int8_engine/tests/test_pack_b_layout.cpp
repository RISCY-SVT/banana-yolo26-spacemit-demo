#include "y26_k1x_conv_kernels.h"
#include "y26_k1x_engine.h"

#include <array>

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
            if (packed[static_cast<std::size_t>(n * 8 + k)] != src[static_cast<std::size_t>(n * 9 + k)]) {
                return 1;
            }
        }
    }

    std::array<std::int8_t, 3 * 5> tail_src {};
    for (std::size_t i = 0; i < tail_src.size(); ++i) {
        tail_src[i] = static_cast<std::int8_t>(80 + static_cast<int>(i));
    }
    std::array<std::int8_t, 32> tail_packed {};
    y26_pack_b_mmt4d_4x8_s8(tail_src.data(), 3, 5, 5, 1, 3, tail_packed.data());
    for (int n = 0; n < 4; ++n) {
        for (int k = 0; k < 8; ++k) {
            const int src_n = 1 + n;
            const int src_k = 3 + k;
            const auto expected =
                (src_n < 3 && src_k < 5) ? tail_src[static_cast<std::size_t>(src_n * 5 + src_k)] : 0;
            if (tail_packed[static_cast<std::size_t>(n * 8 + k)] != expected) {
                return 1;
            }
        }
    }
    return 0;
}
