#include "y26_k1x_vmadot.h"

#include <cstddef>

extern "C" void y26_vmadot_4x4x8_scalar_s8s8s32(const std::int8_t* a_4x8_row_major,
                                                 const std::int8_t* b_4x8_transposed_nk,
                                                 std::int32_t* c_4x4_row_major,
                                                 bool accumulate) {
    if (a_4x8_row_major == nullptr || b_4x8_transposed_nk == nullptr || c_4x4_row_major == nullptr) {
        return;
    }

    for (std::size_t m = 0; m < 4; ++m) {
        for (std::size_t n = 0; n < 4; ++n) {
            std::int32_t acc = accumulate ? c_4x4_row_major[m * 4 + n] : 0;
            for (std::size_t k = 0; k < 8; ++k) {
                const auto a = static_cast<std::int32_t>(a_4x8_row_major[m * 8 + k]);
                const auto b = static_cast<std::int32_t>(b_4x8_transposed_nk[n * 8 + k]);
                acc += a * b;
            }
            c_4x4_row_major[m * 4 + n] = acc;
        }
    }
}
