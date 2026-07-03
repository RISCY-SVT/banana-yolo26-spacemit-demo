#pragma once

#include <cstdint>

extern "C" {

enum Y26ConvStatusCode {
    Y26_CONV_STATUS_SUCCESS = 0,
    Y26_CONV_STATUS_NOT_BUILT_WITH_IME = 1,
    Y26_CONV_STATUS_RUNTIME_SAFETY_FAILED = 2,
    Y26_CONV_STATUS_SIGILL_CAUGHT = 3,
    Y26_CONV_STATUS_INVALID_ARGUMENT = 4,
};

struct Y26Conv2DParams {
    int input_h;
    int input_w;
    int input_c;
    int output_c;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
};

int y26_conv1x1_output_h(const Y26Conv2DParams* params);
int y26_conv1x1_output_w(const Y26Conv2DParams* params);
int y26_conv3x3_output_h(const Y26Conv2DParams* params);
int y26_conv3x3_output_w(const Y26Conv2DParams* params);

void y26_pack_a_mmt4d_4x8_s8(const std::int8_t* matrix_mk,
                             int rows,
                             int cols,
                             int row_stride,
                             int row_offset,
                             int col_offset,
                             std::int8_t* dst_4x8);

void y26_pack_b_mmt4d_4x8_s8(const std::int8_t* matrix_nk,
                             int rows_n,
                             int cols_k,
                             int row_stride,
                             int row_offset_n,
                             int col_offset_k,
                             std::int8_t* dst_4x8_transposed_nk);

int y26_conv1x1_i8s8s32_nhwc_scalar(const std::int8_t* input_nhwc,
                                     const std::int8_t* weights_oc_ic,
                                     const std::int32_t* bias_oc,
                                     std::int32_t* output_nhwc,
                                     const Y26Conv2DParams* params);

int y26_conv1x1_i8s8s32_nhwc_ime(const std::int8_t* input_nhwc,
                                  const std::int8_t* weights_oc_ic,
                                  const std::int32_t* bias_oc,
                                  std::int32_t* output_nhwc,
                                  const Y26Conv2DParams* params);

int y26_conv3x3_i8s8s32_nhwc_scalar(const std::int8_t* input_nhwc,
                                     const std::int8_t* weights_oc_kh_kw_ic,
                                     const std::int32_t* bias_oc,
                                     std::int32_t* output_nhwc,
                                     const Y26Conv2DParams* params);

int y26_conv3x3_i8s8s32_nhwc_ime(const std::int8_t* input_nhwc,
                                  const std::int8_t* weights_oc_kh_kw_ic,
                                  const std::int32_t* bias_oc,
                                  std::int32_t* output_nhwc,
                                  const Y26Conv2DParams* params);

}
