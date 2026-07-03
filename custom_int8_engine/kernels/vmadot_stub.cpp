#include "y26_k1x_engine.h"

namespace y26_k1x {

void pack_a_row_major_4x8(const std::int8_t* src, std::ptrdiff_t row_stride, std::span<std::int8_t, 32> dst) {
    for (std::ptrdiff_t m = 0; m < 4; ++m) {
        for (std::ptrdiff_t k = 0; k < 8; ++k) {
            dst[static_cast<std::size_t>(m * 8 + k)] = src[m * row_stride + k];
        }
    }
}

void pack_b_transposed_4x8(const std::int8_t* src, std::ptrdiff_t row_stride, std::span<std::int8_t, 32> dst) {
    for (std::ptrdiff_t n = 0; n < 4; ++n) {
        for (std::ptrdiff_t k = 0; k < 8; ++k) {
            dst[static_cast<std::size_t>(n * 8 + k)] = src[n * row_stride + k];
        }
    }
}

void vmadot_scalar_4x4x8(const std::int8_t* a_panel,
                         const std::int8_t* b_transposed_panel,
                         std::int32_t* c_tile,
                         std::ptrdiff_t c_stride,
                         bool accumulate) {
    for (std::ptrdiff_t m = 0; m < 4; ++m) {
        for (std::ptrdiff_t n = 0; n < 4; ++n) {
            std::int32_t sum = accumulate ? c_tile[m * c_stride + n] : 0;
            for (std::ptrdiff_t k = 0; k < 8; ++k) {
                const auto a = static_cast<std::int32_t>(a_panel[m * 8 + k]);
                const auto b = static_cast<std::int32_t>(b_transposed_panel[n * 8 + k]);
                sum += a * b;
            }
            c_tile[m * c_stride + n] = sum;
        }
    }
}

}  // namespace y26_k1x
