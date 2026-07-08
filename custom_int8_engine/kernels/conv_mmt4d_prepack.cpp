#include "conv_kernel_common.h"
#include "y26_k1x_vmadot.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

namespace {

constexpr std::size_t kStage4Alignment = 64;

int align_up(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

bool kernel_shape_supported(int kernel_h, int kernel_w) {
    return (kernel_h == 1 && kernel_w == 1) || (kernel_h == 3 && kernel_w == 3);
}

bool storage_zero_point_valid(int value) {
    return value >= static_cast<int>(std::numeric_limits<std::int8_t>::min()) &&
           value <= static_cast<int>(std::numeric_limits<std::int8_t>::max());
}

bool loop_order_valid(int loop_order) {
    return loop_order == Y26_CONV_LOOP_ORDER_M_MAJOR || loop_order == Y26_CONV_LOOP_ORDER_N_MAJOR;
}

int output_h_for_kernel(const Y26Conv2DParams* params, int kernel_h) {
    if (kernel_h == 1) {
        return y26_conv1x1_output_h(params);
    }
    if (kernel_h == 3) {
        return y26_conv3x3_output_h(params);
    }
    return 0;
}

int output_w_for_kernel(const Y26Conv2DParams* params, int kernel_w) {
    if (kernel_w == 1) {
        return y26_conv1x1_output_w(params);
    }
    if (kernel_w == 3) {
        return y26_conv3x3_output_w(params);
    }
    return 0;
}

std::int8_t conv_weight_at(const std::int8_t* weights_oc_kh_kw_ic,
                           const Y26Conv2DParams& params,
                           int kernel_h,
                           int kernel_w,
                           int oc,
                           int flat_k) {
    if (oc >= params.output_c || flat_k >= kernel_h * kernel_w * params.input_c) {
        return 0;
    }
    const int kh = flat_k / (kernel_w * params.input_c);
    const int rem = flat_k % (kernel_w * params.input_c);
    const int kw = rem / params.input_c;
    const int ic = rem % params.input_c;
    return weights_oc_kh_kw_ic[((oc * kernel_h + kh) * kernel_w + kw) * params.input_c + ic];
}

void* allocate_aligned(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    return ::operator new(bytes, std::align_val_t(kStage4Alignment));
}

void free_aligned(void* ptr) {
    ::operator delete(ptr, std::align_val_t(kStage4Alignment));
}

int prepack_weights_mmt4d(const std::int8_t* weights_oc_kh_kw_ic,
                          const Y26Conv2DParams* params,
                          int kernel_h,
                          int kernel_w,
                          std::int8_t* packed_b_mmt4d,
                          std::size_t packed_b_bytes,
                          std::int32_t* weight_sums_oc) {
    if (weights_oc_kh_kw_ic == nullptr || packed_b_mmt4d == nullptr ||
        !y26_k1x::kernels::conv_params_valid(params) || !kernel_shape_supported(kernel_h, kernel_w)) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    const int kernel_k = kernel_h * kernel_w * params->input_c;
    const std::size_t expected_bytes = y26_mmt4d_packed_b_bytes(params->output_c, kernel_k);
    if (expected_bytes == 0 || packed_b_bytes < expected_bytes) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }

    if (weight_sums_oc != nullptr) {
        for (int oc = 0; oc < params->output_c; ++oc) {
            std::int32_t sum = 0;
            for (int k = 0; k < kernel_k; ++k) {
                sum += static_cast<std::int32_t>(
                    conv_weight_at(weights_oc_kh_kw_ic, *params, kernel_h, kernel_w, oc, k));
            }
            weight_sums_oc[oc] = sum;
        }
    }

    const int k_tiles = align_up(kernel_k, 8) / 8;
    for (int n0 = 0; n0 < params->output_c; n0 += 4) {
        const int n_tile = n0 / 4;
        for (int k0 = 0; k0 < kernel_k || k0 == 0; k0 += 8) {
            const int k_tile = k0 / 8;
            std::int8_t* dst = packed_b_mmt4d + (n_tile * k_tiles + k_tile) * 32;
            for (int n = 0; n < 4; ++n) {
                const int oc = n0 + n;
                for (int k = 0; k < 8; ++k) {
                    dst[n * 8 + k] = conv_weight_at(weights_oc_kh_kw_ic, *params, kernel_h, kernel_w, oc, k0 + k);
                }
            }
        }
    }
    return Y26_CONV_STATUS_SUCCESS;
}

void store_a_value(std::int8_t* a_tiles, int m, int flat_k, std::int8_t value) {
    const int k_tile = flat_k / 8;
    const int k_lane = flat_k % 8;
    a_tiles[k_tile * 32 + m * 8 + k_lane] = value;
}

std::uint8_t u8_code_from_s8_storage(std::int8_t value) {
    return static_cast<std::uint8_t>(static_cast<int>(value) + 128);
}

void store_a_u8_value(std::uint8_t* a_tiles, int m, int flat_k, std::uint8_t value) {
    const int k_tile = flat_k / 8;
    const int k_lane = flat_k % 8;
    a_tiles[k_tile * 32 + m * 8 + k_lane] = value;
}

void pack_conv1x1_a_panel(const std::int8_t* input_nhwc_s8,
                          const Y26Conv2DParams& params,
                          int output_w,
                          int output_m,
                          int m0,
                          std::int8_t input_storage_zero_point_s8,
                          std::int8_t* a_tiles) {
    for (int m = 0; m < 4; ++m) {
        const int flat_m = m0 + m;
        if (flat_m >= output_m) {
            continue;
        }
        const int oh = flat_m / output_w;
        const int ow = flat_m - oh * output_w;
        const int ih = oh * params.stride_h - params.pad_h;
        const int iw = ow * params.stride_w - params.pad_w;
        const bool inside = ih >= 0 && iw >= 0 && ih < params.input_h && iw < params.input_w;
        const std::int8_t* src =
            inside ? input_nhwc_s8 + (ih * params.input_w + iw) * params.input_c : nullptr;
        for (int ic = 0; ic < params.input_c; ++ic) {
            store_a_value(a_tiles, m, ic, inside ? src[ic] : input_storage_zero_point_s8);
        }
    }
}

void pack_conv3x3_a_panel(const std::int8_t* input_nhwc_s8,
                          const Y26Conv2DParams& params,
                          int output_w,
                          int output_m,
                          int m0,
                          std::int8_t input_storage_zero_point_s8,
                          std::int8_t* a_tiles) {
    for (int m = 0; m < 4; ++m) {
        const int flat_m = m0 + m;
        if (flat_m >= output_m) {
            continue;
        }
        const int oh = flat_m / output_w;
        const int ow = flat_m - oh * output_w;
        int flat_k = 0;
        for (int kh = 0; kh < 3; ++kh) {
            const int ih = oh * params.stride_h + kh - params.pad_h;
            const bool valid_h = ih >= 0 && ih < params.input_h;
            for (int kw = 0; kw < 3; ++kw) {
                const int iw = ow * params.stride_w + kw - params.pad_w;
                const bool inside = valid_h && iw >= 0 && iw < params.input_w;
                const std::int8_t* src =
                    inside ? input_nhwc_s8 + (ih * params.input_w + iw) * params.input_c : nullptr;
                for (int ic = 0; ic < params.input_c; ++ic, ++flat_k) {
                    store_a_value(a_tiles, m, flat_k, inside ? src[ic] : input_storage_zero_point_s8);
                }
            }
        }
    }
}

void pack_a_panel_4xk_tile_contiguous(const std::int8_t* input_nhwc_s8,
                                      const Y26Conv2DParams& params,
                                      int kernel_h,
                                      int kernel_w,
                                      int output_w,
                                      int output_m,
                                      int m0,
                                      int k_padded,
                                      std::int8_t input_storage_zero_point_s8,
                                      std::int8_t* a_tiles) {
    std::fill(a_tiles, a_tiles + 4 * k_padded, std::int8_t{0});
    if (kernel_h == 1 && kernel_w == 1) {
        pack_conv1x1_a_panel(
            input_nhwc_s8, params, output_w, output_m, m0, input_storage_zero_point_s8, a_tiles);
    } else {
        pack_conv3x3_a_panel(
            input_nhwc_s8, params, output_w, output_m, m0, input_storage_zero_point_s8, a_tiles);
    }
}

void pack_conv1x1_a_panel_u8_from_s8_storage(const std::int8_t* input_nhwc_s8,
                                             const Y26Conv2DParams& params,
                                             int output_w,
                                             int output_m,
                                             int m0,
                                             std::uint8_t activation_zero_point_u8,
                                             std::uint8_t* a_tiles_u8) {
    for (int m = 0; m < 4; ++m) {
        const int flat_m = m0 + m;
        if (flat_m >= output_m) {
            continue;
        }
        const int oh = flat_m / output_w;
        const int ow = flat_m - oh * output_w;
        const int ih = oh * params.stride_h - params.pad_h;
        const int iw = ow * params.stride_w - params.pad_w;
        const bool inside = ih >= 0 && iw >= 0 && ih < params.input_h && iw < params.input_w;
        const std::int8_t* src =
            inside ? input_nhwc_s8 + (ih * params.input_w + iw) * params.input_c : nullptr;
        for (int ic = 0; ic < params.input_c; ++ic) {
            store_a_u8_value(a_tiles_u8,
                             m,
                             ic,
                             inside ? u8_code_from_s8_storage(src[ic]) : activation_zero_point_u8);
        }
    }
}

void pack_conv3x3_a_panel_u8_from_s8_storage(const std::int8_t* input_nhwc_s8,
                                             const Y26Conv2DParams& params,
                                             int output_w,
                                             int output_m,
                                             int m0,
                                             std::uint8_t activation_zero_point_u8,
                                             std::uint8_t* a_tiles_u8) {
    for (int m = 0; m < 4; ++m) {
        const int flat_m = m0 + m;
        if (flat_m >= output_m) {
            continue;
        }
        const int oh = flat_m / output_w;
        const int ow = flat_m - oh * output_w;
        int flat_k = 0;
        for (int kh = 0; kh < 3; ++kh) {
            const int ih = oh * params.stride_h + kh - params.pad_h;
            const bool valid_h = ih >= 0 && ih < params.input_h;
            for (int kw = 0; kw < 3; ++kw) {
                const int iw = ow * params.stride_w + kw - params.pad_w;
                const bool inside = valid_h && iw >= 0 && iw < params.input_w;
                const std::int8_t* src =
                    inside ? input_nhwc_s8 + (ih * params.input_w + iw) * params.input_c : nullptr;
                for (int ic = 0; ic < params.input_c; ++ic, ++flat_k) {
                    store_a_u8_value(a_tiles_u8,
                                     m,
                                     flat_k,
                                     inside ? u8_code_from_s8_storage(src[ic]) : activation_zero_point_u8);
                }
            }
        }
    }
}

void pack_a_panel_4xk_u8_from_s8_storage(const std::int8_t* input_nhwc_s8,
                                         const Y26Conv2DParams& params,
                                         int kernel_h,
                                         int kernel_w,
                                         int output_w,
                                         int output_m,
                                         int m0,
                                         int k_padded,
                                         std::uint8_t activation_zero_point_u8,
                                         std::uint8_t* a_tiles_u8) {
    std::fill(a_tiles_u8, a_tiles_u8 + 4 * k_padded, std::uint8_t{0});
    if (kernel_h == 1 && kernel_w == 1) {
        pack_conv1x1_a_panel_u8_from_s8_storage(
            input_nhwc_s8, params, output_w, output_m, m0, activation_zero_point_u8, a_tiles_u8);
    } else {
        pack_conv3x3_a_panel_u8_from_s8_storage(
            input_nhwc_s8, params, output_w, output_m, m0, activation_zero_point_u8, a_tiles_u8);
    }
}

int run_c_tile(const std::int8_t* a_tiles,
               const std::int8_t* packed_b_mmt4d,
               std::int32_t* raw_output_nhwc,
               const Y26Conv2DParams& params,
               int output_m,
               int m0,
               int n0,
               int k_tiles,
               std::array<std::int32_t, 16>& c_tile) {
    std::fill(c_tile.begin(), c_tile.end(), 0);
    const int n_tile = n0 / 4;
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const std::int8_t* a_tile = a_tiles + k_tile * 32;
        const std::int8_t* b_tile = packed_b_mmt4d + (n_tile * k_tiles + k_tile) * 32;
        const int status = y26_k1x_vmadot_4x4x8_unsafe_cluster0_s8s8s32(
            a_tile, b_tile, c_tile.data(), true);
        if (status != Y26_VMADOT_STATUS_SUCCESS) {
            return y26_k1x::kernels::conv_status_from_vmadot_status(status);
        }
    }
    for (int m = 0; m < 4; ++m) {
        const int flat_m = m0 + m;
        if (flat_m >= output_m) {
            continue;
        }
        for (int n = 0; n < 4; ++n) {
            const int oc = n0 + n;
            if (oc < params.output_c) {
                raw_output_nhwc[flat_m * params.output_c + oc] = c_tile[m * 4 + n];
            }
        }
    }
    return Y26_CONV_STATUS_SUCCESS;
}

int run_c_tile_u8s8_fused_correction(const std::uint8_t* a_tiles_u8,
                                      const std::int8_t* packed_b_mmt4d,
                                      const std::int32_t* bias_oc,
                                      const std::int32_t* weight_sums_oc,
                                      std::int32_t* corrected_output_nhwc,
                                      const Y26Conv2DParams& params,
                                      int output_m,
                                      int m0,
                                      int n0,
                                      int k_tiles,
                                      int activation_zero_point_u8,
                                      std::array<std::int32_t, 16>& c_tile) {
    for (int m = 0; m < 4; ++m) {
        for (int n = 0; n < 4; ++n) {
            const int oc = n0 + n;
            c_tile[m * 4 + n] =
                oc < params.output_c
                    ? static_cast<std::int32_t>(
                          static_cast<std::int64_t>(bias_oc[oc]) -
                          static_cast<std::int64_t>(activation_zero_point_u8) *
                              static_cast<std::int64_t>(weight_sums_oc[oc]))
                    : 0;
        }
    }
    const int n_tile = n0 / 4;
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const std::uint8_t* a_tile = a_tiles_u8 + k_tile * 32;
        const std::int8_t* b_tile = packed_b_mmt4d + (n_tile * k_tiles + k_tile) * 32;
        const int status = y26_k1x_vmadot_4x4x8_unsafe_cluster0_u8s8s32(
            a_tile, b_tile, c_tile.data(), true);
        if (status != Y26_VMADOT_STATUS_SUCCESS) {
            return y26_k1x::kernels::conv_status_from_vmadot_status(status);
        }
    }
    for (int m = 0; m < 4; ++m) {
        const int flat_m = m0 + m;
        if (flat_m >= output_m) {
            continue;
        }
        for (int n = 0; n < 4; ++n) {
            const int oc = n0 + n;
            if (oc < params.output_c) {
                corrected_output_nhwc[flat_m * params.output_c + oc] = c_tile[m * 4 + n];
            }
        }
    }
    return Y26_CONV_STATUS_SUCCESS;
}

int conv_ime_prepacked_core(const std::int8_t* input_nhwc_s8,
                            const std::int8_t* packed_b_mmt4d,
                            std::int32_t* raw_output_nhwc,
                            const Y26Conv2DParams* params,
                            int kernel_h,
                            int kernel_w,
                            int input_storage_zero_point_s8,
                            std::int8_t* a_workspace_tiles,
                            std::size_t workspace_bytes,
                            int loop_order) {
    if (input_nhwc_s8 == nullptr || packed_b_mmt4d == nullptr || raw_output_nhwc == nullptr ||
        a_workspace_tiles == nullptr || !y26_k1x::kernels::conv_params_valid(params) ||
        !kernel_shape_supported(kernel_h, kernel_w) || !storage_zero_point_valid(input_storage_zero_point_s8) ||
        !loop_order_valid(loop_order)) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
        return Y26_CONV_STATUS_NOT_BUILT_WITH_IME;
    }
    if (!y26_k1x_ime_hotpath_allowed_on_current_cpu()) {
        const auto snapshot = y26_k1x_ime_runtime_state_snapshot();
        return y26_k1x::kernels::conv_status_from_vmadot_status(snapshot.probe_status);
    }

    const int output_h = output_h_for_kernel(params, kernel_h);
    const int output_w = output_w_for_kernel(params, kernel_w);
    if (output_h <= 0 || output_w <= 0) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    const int kernel_k = kernel_h * kernel_w * params->input_c;
    const int k_padded = align_up(kernel_k, 8);
    const int k_tiles = k_padded / 8;
    if (workspace_bytes < static_cast<std::size_t>(4 * k_padded)) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }

    const int output_m = output_h * output_w;
    std::array<std::int32_t, 16> c_tile {};
    if (loop_order == Y26_CONV_LOOP_ORDER_M_MAJOR) {
        for (int m0 = 0; m0 < output_m; m0 += 4) {
            pack_a_panel_4xk_tile_contiguous(input_nhwc_s8,
                                             *params,
                                             kernel_h,
                                             kernel_w,
                                             output_w,
                                             output_m,
                                             m0,
                                             k_padded,
                                             static_cast<std::int8_t>(input_storage_zero_point_s8),
                                             a_workspace_tiles);
            for (int n0 = 0; n0 < params->output_c; n0 += 4) {
                const int status = run_c_tile(a_workspace_tiles,
                                              packed_b_mmt4d,
                                              raw_output_nhwc,
                                              *params,
                                              output_m,
                                              m0,
                                              n0,
                                              k_tiles,
                                              c_tile);
                if (status != Y26_CONV_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
        return Y26_CONV_STATUS_SUCCESS;
    }

    for (int n0 = 0; n0 < params->output_c; n0 += 4) {
        for (int m0 = 0; m0 < output_m; m0 += 4) {
            pack_a_panel_4xk_tile_contiguous(input_nhwc_s8,
                                             *params,
                                             kernel_h,
                                             kernel_w,
                                             output_w,
                                             output_m,
                                             m0,
                                             k_padded,
                                             static_cast<std::int8_t>(input_storage_zero_point_s8),
                                             a_workspace_tiles);
            const int status = run_c_tile(a_workspace_tiles,
                                          packed_b_mmt4d,
                                          raw_output_nhwc,
                                          *params,
                                          output_m,
                                          m0,
                                          n0,
                                          k_tiles,
                                          c_tile);
            if (status != Y26_CONV_STATUS_SUCCESS) {
                return status;
            }
        }
    }
    return Y26_CONV_STATUS_SUCCESS;
}

int conv_u8s8_fused_correction_core(const std::int8_t* input_nhwc_s8_storage,
                                    const std::int8_t* packed_b_mmt4d,
                                    const std::int32_t* bias_oc,
                                    const std::int32_t* weight_sums_oc,
                                    std::int32_t* corrected_output_nhwc,
                                    const Y26Conv2DParams* params,
                                    int kernel_h,
                                    int kernel_w,
                                    int activation_zero_point_u8,
                                    std::uint8_t* a_workspace_tiles_u8,
                                    std::size_t workspace_bytes,
                                    int loop_order) {
    if (input_nhwc_s8_storage == nullptr || packed_b_mmt4d == nullptr || bias_oc == nullptr ||
        weight_sums_oc == nullptr || corrected_output_nhwc == nullptr || a_workspace_tiles_u8 == nullptr ||
        !y26_k1x::kernels::conv_params_valid(params) || !kernel_shape_supported(kernel_h, kernel_w) ||
        activation_zero_point_u8 < 0 || activation_zero_point_u8 > 255 || !loop_order_valid(loop_order)) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
        return Y26_CONV_STATUS_NOT_BUILT_WITH_IME;
    }
    if (!y26_k1x_ime_hotpath_allowed_on_current_cpu()) {
        const auto snapshot = y26_k1x_ime_runtime_state_snapshot();
        return y26_k1x::kernels::conv_status_from_vmadot_status(snapshot.probe_status);
    }

    const int output_h = output_h_for_kernel(params, kernel_h);
    const int output_w = output_w_for_kernel(params, kernel_w);
    if (output_h <= 0 || output_w <= 0) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    const int kernel_k = kernel_h * kernel_w * params->input_c;
    const int k_padded = align_up(kernel_k, 8);
    const int k_tiles = k_padded / 8;
    if (workspace_bytes < static_cast<std::size_t>(4 * k_padded)) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }

    const int output_m = output_h * output_w;
    std::array<std::int32_t, 16> c_tile {};
    if (loop_order == Y26_CONV_LOOP_ORDER_M_MAJOR) {
        for (int m0 = 0; m0 < output_m; m0 += 4) {
            pack_a_panel_4xk_u8_from_s8_storage(input_nhwc_s8_storage,
                                                *params,
                                                kernel_h,
                                                kernel_w,
                                                output_w,
                                                output_m,
                                                m0,
                                                k_padded,
                                                static_cast<std::uint8_t>(activation_zero_point_u8),
                                                a_workspace_tiles_u8);
            for (int n0 = 0; n0 < params->output_c; n0 += 4) {
                const int status = run_c_tile_u8s8_fused_correction(a_workspace_tiles_u8,
                                                                    packed_b_mmt4d,
                                                                    bias_oc,
                                                                    weight_sums_oc,
                                                                    corrected_output_nhwc,
                                                                    *params,
                                                                    output_m,
                                                                    m0,
                                                                    n0,
                                                                    k_tiles,
                                                                    activation_zero_point_u8,
                                                                    c_tile);
                if (status != Y26_CONV_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
        return Y26_CONV_STATUS_SUCCESS;
    }

    for (int n0 = 0; n0 < params->output_c; n0 += 4) {
        for (int m0 = 0; m0 < output_m; m0 += 4) {
            pack_a_panel_4xk_u8_from_s8_storage(input_nhwc_s8_storage,
                                                *params,
                                                kernel_h,
                                                kernel_w,
                                                output_w,
                                                output_m,
                                                m0,
                                                k_padded,
                                                static_cast<std::uint8_t>(activation_zero_point_u8),
                                                a_workspace_tiles_u8);
            const int status = run_c_tile_u8s8_fused_correction(a_workspace_tiles_u8,
                                                                packed_b_mmt4d,
                                                                bias_oc,
                                                                weight_sums_oc,
                                                                corrected_output_nhwc,
                                                                *params,
                                                                output_m,
                                                                m0,
                                                                n0,
                                                                k_tiles,
                                                                activation_zero_point_u8,
                                                                c_tile);
            if (status != Y26_CONV_STATUS_SUCCESS) {
                return status;
            }
        }
    }
    return Y26_CONV_STATUS_SUCCESS;
}

}  // namespace

struct Y26PrepackedConvWeights {
    Y26Conv2DParams params {};
    int kernel_h = 0;
    int kernel_w = 0;
    int kernel_k = 0;
    std::size_t packed_b_bytes = 0;
    std::size_t total_bytes = 0;
    const char* source_tensor_name = nullptr;
    const void* quant_scale_metadata = nullptr;
    std::int8_t* packed_b_mmt4d = nullptr;
    std::int32_t* weight_sums_oc = nullptr;
};

struct Y26ConvWorkspace {
    Y26Conv2DParams params {};
    int kernel_h = 0;
    int kernel_w = 0;
    int k_padded = 0;
    std::size_t bytes = 0;
    std::size_t peak_bytes = 0;
    std::int8_t* a_tiles = nullptr;
};

extern "C" std::size_t y26_mmt4d_packed_b_bytes(int output_c, int kernel_k) {
    if (output_c <= 0 || kernel_k <= 0) {
        return 0;
    }
    const int n_tiles = align_up(output_c, 4) / 4;
    const int k_tiles = align_up(kernel_k, 8) / 8;
    return static_cast<std::size_t>(n_tiles) * static_cast<std::size_t>(k_tiles) * 32U;
}

extern "C" std::size_t y26_conv_mmt4d_a_workspace_bytes(const Y26Conv2DParams* params,
                                                         int kernel_h,
                                                         int kernel_w) {
    if (!y26_k1x::kernels::conv_params_valid(params) || !kernel_shape_supported(kernel_h, kernel_w)) {
        return 0;
    }
    const int kernel_k = kernel_h * kernel_w * params->input_c;
    return static_cast<std::size_t>(4 * align_up(kernel_k, 8));
}

extern "C" int y26_conv1x1_prepack_weights_mmt4d_s8(const std::int8_t* weights_oc_ic,
                                                     const Y26Conv2DParams* params,
                                                     std::int8_t* packed_b_mmt4d,
                                                     std::size_t packed_b_bytes,
                                                     std::int32_t* weight_sums_oc) {
    return prepack_weights_mmt4d(weights_oc_ic, params, 1, 1, packed_b_mmt4d, packed_b_bytes, weight_sums_oc);
}

extern "C" int y26_conv3x3_prepack_weights_mmt4d_s8(const std::int8_t* weights_oc_kh_kw_ic,
                                                     const Y26Conv2DParams* params,
                                                     std::int8_t* packed_b_mmt4d,
                                                     std::size_t packed_b_bytes,
                                                     std::int32_t* weight_sums_oc) {
    return prepack_weights_mmt4d(
        weights_oc_kh_kw_ic, params, 3, 3, packed_b_mmt4d, packed_b_bytes, weight_sums_oc);
}

extern "C" int y26_conv1x1_i8s8s32_nhwc_ime_prepacked(const std::int8_t* input_nhwc_s8,
                                                       const std::int8_t* packed_b_mmt4d,
                                                       std::int32_t* raw_output_nhwc,
                                                       const Y26Conv2DParams* params,
                                                       int input_storage_zero_point_s8,
                                                       std::int8_t* workspace,
                                                       std::size_t workspace_bytes) {
    return conv_ime_prepacked_core(input_nhwc_s8,
                                   packed_b_mmt4d,
                                   raw_output_nhwc,
                                   params,
                                   1,
                                   1,
                                   input_storage_zero_point_s8,
                                   workspace,
                                   workspace_bytes,
                                   Y26_CONV_LOOP_ORDER_M_MAJOR);
}

extern "C" int y26_conv3x3_i8s8s32_nhwc_ime_prepacked(const std::int8_t* input_nhwc_s8,
                                                       const std::int8_t* packed_b_mmt4d,
                                                       std::int32_t* raw_output_nhwc,
                                                       const Y26Conv2DParams* params,
                                                       int input_storage_zero_point_s8,
                                                       std::int8_t* workspace,
                                                       std::size_t workspace_bytes) {
    return conv_ime_prepacked_core(input_nhwc_s8,
                                   packed_b_mmt4d,
                                   raw_output_nhwc,
                                   params,
                                   3,
                                   3,
                                   input_storage_zero_point_s8,
                                   workspace,
                                   workspace_bytes,
                                   Y26_CONV_LOOP_ORDER_M_MAJOR);
}

extern "C" Y26PrepackedConvWeights* y26_prepacked_conv_weights_create_mmt4d_s8(
    const std::int8_t* weights_oc_kh_kw_ic,
    const Y26Conv2DParams* params,
    int kernel_h,
    int kernel_w,
    const char* source_tensor_name,
    const void* quant_scale_metadata) {
    if (weights_oc_kh_kw_ic == nullptr || !y26_k1x::kernels::conv_params_valid(params) ||
        !kernel_shape_supported(kernel_h, kernel_w)) {
        return nullptr;
    }
    const int kernel_k = kernel_h * kernel_w * params->input_c;
    const std::size_t packed_b_bytes = y26_mmt4d_packed_b_bytes(params->output_c, kernel_k);
    if (packed_b_bytes == 0) {
        return nullptr;
    }
    Y26PrepackedConvWeights* result = nullptr;
    try {
        result = new Y26PrepackedConvWeights;
        result->params = *params;
        result->kernel_h = kernel_h;
        result->kernel_w = kernel_w;
        result->kernel_k = kernel_k;
        result->packed_b_bytes = packed_b_bytes;
        result->total_bytes = packed_b_bytes + static_cast<std::size_t>(params->output_c) * sizeof(std::int32_t);
        result->source_tensor_name = source_tensor_name;
        result->quant_scale_metadata = quant_scale_metadata;
        result->packed_b_mmt4d = static_cast<std::int8_t*>(allocate_aligned(packed_b_bytes));
        result->weight_sums_oc = static_cast<std::int32_t*>(
            allocate_aligned(static_cast<std::size_t>(params->output_c) * sizeof(std::int32_t)));
        std::memset(result->packed_b_mmt4d, 0, packed_b_bytes);
        std::memset(result->weight_sums_oc, 0, static_cast<std::size_t>(params->output_c) * sizeof(std::int32_t));
        const int status = prepack_weights_mmt4d(weights_oc_kh_kw_ic,
                                                 params,
                                                 kernel_h,
                                                 kernel_w,
                                                 result->packed_b_mmt4d,
                                                 result->packed_b_bytes,
                                                 result->weight_sums_oc);
        if (status != Y26_CONV_STATUS_SUCCESS) {
            y26_prepacked_conv_weights_destroy(result);
            return nullptr;
        }
        return result;
    } catch (const std::bad_alloc&) {
        y26_prepacked_conv_weights_destroy(result);
        return nullptr;
    }
}

extern "C" void y26_prepacked_conv_weights_destroy(Y26PrepackedConvWeights* weights) {
    if (weights == nullptr) {
        return;
    }
    free_aligned(weights->packed_b_mmt4d);
    free_aligned(weights->weight_sums_oc);
    delete weights;
}

extern "C" const std::int8_t* y26_prepacked_conv_weights_packed_b(const Y26PrepackedConvWeights* weights) {
    return weights != nullptr ? weights->packed_b_mmt4d : nullptr;
}

extern "C" const std::int32_t* y26_prepacked_conv_weights_sums(const Y26PrepackedConvWeights* weights) {
    return weights != nullptr ? weights->weight_sums_oc : nullptr;
}

extern "C" std::size_t y26_prepacked_conv_weights_packed_b_bytes(const Y26PrepackedConvWeights* weights) {
    return weights != nullptr ? weights->packed_b_bytes : 0;
}

extern "C" std::size_t y26_prepacked_conv_weights_total_bytes(const Y26PrepackedConvWeights* weights) {
    return weights != nullptr ? weights->total_bytes : 0;
}

extern "C" const char* y26_prepacked_conv_weights_source_tensor_name(const Y26PrepackedConvWeights* weights) {
    return weights != nullptr ? weights->source_tensor_name : nullptr;
}

extern "C" Y26ConvWorkspace* y26_conv_workspace_create(const Y26Conv2DParams* params,
                                                        int kernel_h,
                                                        int kernel_w) {
    if (!y26_k1x::kernels::conv_params_valid(params) || !kernel_shape_supported(kernel_h, kernel_w)) {
        return nullptr;
    }
    const std::size_t bytes = y26_conv_mmt4d_a_workspace_bytes(params, kernel_h, kernel_w);
    if (bytes == 0) {
        return nullptr;
    }
    Y26ConvWorkspace* workspace = nullptr;
    try {
        workspace = new Y26ConvWorkspace;
        workspace->params = *params;
        workspace->kernel_h = kernel_h;
        workspace->kernel_w = kernel_w;
        workspace->k_padded = align_up(kernel_h * kernel_w * params->input_c, 8);
        workspace->bytes = bytes;
        workspace->peak_bytes = bytes;
        workspace->a_tiles = static_cast<std::int8_t*>(allocate_aligned(bytes));
        std::memset(workspace->a_tiles, 0, bytes);
        return workspace;
    } catch (const std::bad_alloc&) {
        y26_conv_workspace_destroy(workspace);
        return nullptr;
    }
}

extern "C" void y26_conv_workspace_destroy(Y26ConvWorkspace* workspace) {
    if (workspace == nullptr) {
        return;
    }
    free_aligned(workspace->a_tiles);
    delete workspace;
}

extern "C" std::size_t y26_conv_workspace_bytes(const Y26ConvWorkspace* workspace) {
    return workspace != nullptr ? workspace->bytes : 0;
}

extern "C" std::size_t y26_conv_workspace_peak_bytes(const Y26ConvWorkspace* workspace) {
    return workspace != nullptr ? workspace->peak_bytes : 0;
}

extern "C" int y26_conv2d_i8s8s32_nhwc_ime_prepacked_v1(const std::int8_t* input_nhwc_s8,
                                                         const Y26PrepackedConvWeights* weights,
                                                         std::int32_t* raw_output_nhwc,
                                                         int input_storage_zero_point_s8,
                                                         Y26ConvWorkspace* workspace,
                                                         int loop_order) {
    if (weights == nullptr || workspace == nullptr || weights->packed_b_mmt4d == nullptr ||
        workspace->a_tiles == nullptr || weights->kernel_h != workspace->kernel_h ||
        weights->kernel_w != workspace->kernel_w || weights->params.input_h != workspace->params.input_h ||
        weights->params.input_w != workspace->params.input_w ||
        weights->params.input_c != workspace->params.input_c ||
        weights->params.output_c != workspace->params.output_c) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    return conv_ime_prepacked_core(input_nhwc_s8,
                                   weights->packed_b_mmt4d,
                                   raw_output_nhwc,
                                   &weights->params,
                                   weights->kernel_h,
                                   weights->kernel_w,
                                   input_storage_zero_point_s8,
                                   workspace->a_tiles,
                                   workspace->bytes,
                                   loop_order);
}

extern "C" int y26_conv2d_u8s8s32_nhwc_ime_prepacked_fused_correction_v1(
    const std::int8_t* input_nhwc_s8_storage,
    const Y26PrepackedConvWeights* weights,
    const std::int32_t* bias_oc,
    std::int32_t* corrected_output_nhwc,
    int activation_zero_point_u8,
    Y26ConvWorkspace* workspace,
    int loop_order) {
    if (weights == nullptr || workspace == nullptr || weights->packed_b_mmt4d == nullptr ||
        weights->weight_sums_oc == nullptr || workspace->a_tiles == nullptr ||
        weights->kernel_h != workspace->kernel_h || weights->kernel_w != workspace->kernel_w ||
        weights->params.input_h != workspace->params.input_h || weights->params.input_w != workspace->params.input_w ||
        weights->params.input_c != workspace->params.input_c ||
        weights->params.output_c != workspace->params.output_c) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    return conv_u8s8_fused_correction_core(input_nhwc_s8_storage,
                                           weights->packed_b_mmt4d,
                                           bias_oc,
                                           weights->weight_sums_oc,
                                           corrected_output_nhwc,
                                           &weights->params,
                                           weights->kernel_h,
                                           weights->kernel_w,
                                           activation_zero_point_u8,
                                           reinterpret_cast<std::uint8_t*>(workspace->a_tiles),
                                           workspace->bytes,
                                           loop_order);
}

extern "C" int y26_conv2d_apply_u8_as_s8_correction_nhwc(const std::int32_t* raw_dot_nhwc,
                                                          const std::int32_t* bias_oc,
                                                          const std::int32_t* weight_sums_oc,
                                                          std::int32_t* corrected_output_nhwc,
                                                          int output_m,
                                                          int output_c,
                                                          int activation_zero_point_u8) {
    if (raw_dot_nhwc == nullptr || bias_oc == nullptr || weight_sums_oc == nullptr ||
        corrected_output_nhwc == nullptr || output_m <= 0 || output_c <= 0 ||
        activation_zero_point_u8 < 0 || activation_zero_point_u8 > 255) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    const std::int64_t correction_offset = 128 - static_cast<std::int64_t>(activation_zero_point_u8);
    for (int m = 0; m < output_m; ++m) {
        for (int oc = 0; oc < output_c; ++oc) {
            const std::int64_t corrected =
                static_cast<std::int64_t>(raw_dot_nhwc[m * output_c + oc]) +
                correction_offset * static_cast<std::int64_t>(weight_sums_oc[oc]) +
                static_cast<std::int64_t>(bias_oc[oc]);
            corrected_output_nhwc[m * output_c + oc] = static_cast<std::int32_t>(corrected);
        }
    }
    return Y26_CONV_STATUS_SUCCESS;
}
