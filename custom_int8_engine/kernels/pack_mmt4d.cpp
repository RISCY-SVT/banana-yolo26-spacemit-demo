#include "y26_k1x_conv_kernels.h"

void y26_pack_a_mmt4d_4x8_s8(const std::int8_t* matrix_mk,
                             int rows,
                             int cols,
                             int row_stride,
                             int row_offset,
                             int col_offset,
                             std::int8_t* dst_4x8) {
    if (matrix_mk == nullptr || dst_4x8 == nullptr || rows < 0 || cols < 0 || row_stride < cols) {
        return;
    }
    for (int m = 0; m < 4; ++m) {
        for (int k = 0; k < 8; ++k) {
            const int src_m = row_offset + m;
            const int src_k = col_offset + k;
            dst_4x8[m * 8 + k] =
                (src_m < rows && src_k < cols) ? matrix_mk[src_m * row_stride + src_k] : 0;
        }
    }
}

void y26_pack_b_mmt4d_4x8_s8(const std::int8_t* matrix_nk,
                             int rows_n,
                             int cols_k,
                             int row_stride,
                             int row_offset_n,
                             int col_offset_k,
                             std::int8_t* dst_4x8_transposed_nk) {
    if (matrix_nk == nullptr || dst_4x8_transposed_nk == nullptr || rows_n < 0 || cols_k < 0 ||
        row_stride < cols_k) {
        return;
    }
    for (int n = 0; n < 4; ++n) {
        for (int k = 0; k < 8; ++k) {
            const int src_n = row_offset_n + n;
            const int src_k = col_offset_k + k;
            dst_4x8_transposed_nk[n * 8 + k] =
                (src_n < rows_n && src_k < cols_k) ? matrix_nk[src_n * row_stride + src_k] : 0;
        }
    }
}
