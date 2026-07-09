#pragma once

#include <cstddef>
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

enum Y26ConvLoopOrder {
    Y26_CONV_LOOP_ORDER_M_MAJOR = 0,
    Y26_CONV_LOOP_ORDER_N_MAJOR = 1,
};

struct Y26PrepackedConvWeights;
struct Y26ConvWorkspace;

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

std::size_t y26_mmt4d_packed_b_bytes(int output_c, int kernel_k);

std::size_t y26_conv_mmt4d_a_workspace_bytes(const Y26Conv2DParams* params,
                                             int kernel_h,
                                             int kernel_w);

int y26_conv1x1_prepack_weights_mmt4d_s8(const std::int8_t* weights_oc_ic,
                                         const Y26Conv2DParams* params,
                                         std::int8_t* packed_b_mmt4d,
                                         std::size_t packed_b_bytes,
                                         std::int32_t* weight_sums_oc);

int y26_conv3x3_prepack_weights_mmt4d_s8(const std::int8_t* weights_oc_kh_kw_ic,
                                         const Y26Conv2DParams* params,
                                         std::int8_t* packed_b_mmt4d,
                                         std::size_t packed_b_bytes,
                                         std::int32_t* weight_sums_oc);

int y26_conv1x1_i8s8s32_nhwc_ime_prepacked(const std::int8_t* input_nhwc_s8,
                                            const std::int8_t* packed_b_mmt4d,
                                            std::int32_t* raw_output_nhwc,
                                            const Y26Conv2DParams* params,
                                            int input_storage_zero_point_s8,
                                            std::int8_t* workspace,
                                            std::size_t workspace_bytes);

int y26_conv3x3_i8s8s32_nhwc_ime_prepacked(const std::int8_t* input_nhwc_s8,
                                            const std::int8_t* packed_b_mmt4d,
                                            std::int32_t* raw_output_nhwc,
                                            const Y26Conv2DParams* params,
                                            int input_storage_zero_point_s8,
                                            std::int8_t* workspace,
                                            std::size_t workspace_bytes);

Y26PrepackedConvWeights* y26_prepacked_conv_weights_create_mmt4d_s8(
    const std::int8_t* weights_oc_kh_kw_ic,
    const Y26Conv2DParams* params,
    int kernel_h,
    int kernel_w,
    const char* source_tensor_name,
    const void* quant_scale_metadata);

void y26_prepacked_conv_weights_destroy(Y26PrepackedConvWeights* weights);

const std::int8_t* y26_prepacked_conv_weights_packed_b(const Y26PrepackedConvWeights* weights);
const std::int32_t* y26_prepacked_conv_weights_sums(const Y26PrepackedConvWeights* weights);
std::size_t y26_prepacked_conv_weights_packed_b_bytes(const Y26PrepackedConvWeights* weights);
std::size_t y26_prepacked_conv_weights_total_bytes(const Y26PrepackedConvWeights* weights);
const char* y26_prepacked_conv_weights_source_tensor_name(const Y26PrepackedConvWeights* weights);

Y26ConvWorkspace* y26_conv_workspace_create(const Y26Conv2DParams* params,
                                            int kernel_h,
                                            int kernel_w);

void y26_conv_workspace_destroy(Y26ConvWorkspace* workspace);

std::size_t y26_conv_workspace_bytes(const Y26ConvWorkspace* workspace);
std::size_t y26_conv_workspace_peak_bytes(const Y26ConvWorkspace* workspace);

int y26_conv2d_i8s8s32_nhwc_ime_prepacked_v1(const std::int8_t* input_nhwc_s8,
                                             const Y26PrepackedConvWeights* weights,
                                             std::int32_t* raw_output_nhwc,
                                             int input_storage_zero_point_s8,
                                             Y26ConvWorkspace* workspace,
                                             int loop_order);

void y26_conv_mmt4d_set_stage38_pack_timing_enabled(int enabled);
double y26_conv_mmt4d_last_im2col_pack_us();

int y26_conv2d_i8s8s32_nhwc_ime_prepacked_stage36_pipelined_v1(
    const std::int8_t* input_nhwc_s8,
    const Y26PrepackedConvWeights* weights,
    std::int32_t* raw_output_nhwc,
    int input_storage_zero_point_s8,
    Y26ConvWorkspace* workspace,
    int accumulator_groups,
    int loop_order);

int y26_conv2d_i8s8s32_nhwc_ime_prepacked_stage37_pipelined_v1(
    const std::int8_t* input_nhwc_s8,
    const Y26PrepackedConvWeights* weights,
    std::int32_t* raw_output_nhwc,
    int input_storage_zero_point_s8,
    Y26ConvWorkspace* workspace,
    int accumulator_groups,
    int loop_order);

int y26_conv2d_i8s8s32_nhwc_ime_prepacked_stage39_fastpack_v1(
    const std::int8_t* input_nhwc_s8,
    const Y26PrepackedConvWeights* weights,
    std::int32_t* raw_output_nhwc,
    int input_storage_zero_point_s8,
    Y26ConvWorkspace* workspace,
    int accumulator_groups,
    int loop_order);

int y26_conv2d_u8s8s32_nhwc_ime_prepacked_fused_correction_v1(
    const std::int8_t* input_nhwc_s8_storage,
    const Y26PrepackedConvWeights* weights,
    const std::int32_t* bias_oc,
    std::int32_t* corrected_output_nhwc,
    int activation_zero_point_u8,
    Y26ConvWorkspace* workspace,
    int loop_order);

int y26_conv2d_apply_u8_as_s8_correction_nhwc(const std::int32_t* raw_dot_nhwc,
                                               const std::int32_t* bias_oc,
                                               const std::int32_t* weight_sums_oc,
                                               std::int32_t* corrected_output_nhwc,
                                               int output_m,
                                               int output_c,
                                               int activation_zero_point_u8);

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
