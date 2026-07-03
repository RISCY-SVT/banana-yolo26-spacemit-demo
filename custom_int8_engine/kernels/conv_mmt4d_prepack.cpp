#include "conv_kernel_common.h"
#include "y26_k1x_vmadot.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

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

void pack_a_panel_4xk(const std::int8_t* input_nhwc_s8,
                      const Y26Conv2DParams& params,
                      int kernel_h,
                      int kernel_w,
                      int output_w,
                      int output_m,
                      int m0,
                      std::int8_t input_storage_zero_point_s8,
                      std::int8_t* workspace) {
    const int kernel_k = kernel_h * kernel_w * params.input_c;
    const int k_padded = align_up(kernel_k, 8);
    for (int m = 0; m < 4; ++m) {
        const int flat_m = m0 + m;
        for (int k = 0; k < k_padded; ++k) {
            std::int8_t value = 0;
            if (flat_m < output_m && k < kernel_k) {
                const int oh = y26_k1x::kernels::flat_output_m_to_h(flat_m, output_w);
                const int ow = y26_k1x::kernels::flat_output_m_to_w(flat_m, output_w);
                const int kh = k / (kernel_w * params.input_c);
                const int rem = k % (kernel_w * params.input_c);
                const int kw = rem / params.input_c;
                const int ic = rem % params.input_c;
                const int ih = oh * params.stride_h + kh - params.pad_h;
                const int iw = ow * params.stride_w + kw - params.pad_w;
                if (ih < 0 || iw < 0 || ih >= params.input_h || iw >= params.input_w) {
                    value = input_storage_zero_point_s8;
                } else {
                    value = input_nhwc_s8[(ih * params.input_w + iw) * params.input_c + ic];
                }
            }
            workspace[m * k_padded + k] = value;
        }
    }
}

int conv_ime_prepacked(const std::int8_t* input_nhwc_s8,
                       const std::int8_t* packed_b_mmt4d,
                       std::int32_t* raw_output_nhwc,
                       const Y26Conv2DParams* params,
                       int kernel_h,
                       int kernel_w,
                       int input_storage_zero_point_s8,
                       std::int8_t* workspace,
                       std::size_t workspace_bytes) {
    if (input_nhwc_s8 == nullptr || packed_b_mmt4d == nullptr || raw_output_nhwc == nullptr ||
        workspace == nullptr || !y26_k1x::kernels::conv_params_valid(params) ||
        !kernel_shape_supported(kernel_h, kernel_w) || !storage_zero_point_valid(input_storage_zero_point_s8)) {
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

    for (int m0 = 0; m0 < output_m; m0 += 4) {
        pack_a_panel_4xk(input_nhwc_s8,
                         *params,
                         kernel_h,
                         kernel_w,
                         output_w,
                         output_m,
                         m0,
                         static_cast<std::int8_t>(input_storage_zero_point_s8),
                         workspace);
        for (int n0 = 0; n0 < params->output_c; n0 += 4) {
            std::fill(c_tile.begin(), c_tile.end(), 0);
            const int n_tile = n0 / 4;
            for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
                std::array<std::int8_t, 32> a_tile {};
                const int k0 = k_tile * 8;
                for (int m = 0; m < 4; ++m) {
                    for (int k = 0; k < 8; ++k) {
                        a_tile[m * 8 + k] = workspace[m * k_padded + k0 + k];
                    }
                }
                const std::int8_t* b_tile = packed_b_mmt4d + (n_tile * k_tiles + k_tile) * 32;
                const int status = y26_k1x_vmadot_4x4x8_unsafe_cluster0_s8s8s32(
                    a_tile.data(), b_tile, c_tile.data(), true);
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
                    if (oc < params->output_c) {
                        raw_output_nhwc[flat_m * params->output_c + oc] = c_tile[m * 4 + n];
                    }
                }
            }
        }
    }
    return Y26_CONV_STATUS_SUCCESS;
}

}  // namespace

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
    return conv_ime_prepacked(input_nhwc_s8,
                              packed_b_mmt4d,
                              raw_output_nhwc,
                              params,
                              1,
                              1,
                              input_storage_zero_point_s8,
                              workspace,
                              workspace_bytes);
}

extern "C" int y26_conv3x3_i8s8s32_nhwc_ime_prepacked(const std::int8_t* input_nhwc_s8,
                                                       const std::int8_t* packed_b_mmt4d,
                                                       std::int32_t* raw_output_nhwc,
                                                       const Y26Conv2DParams* params,
                                                       int input_storage_zero_point_s8,
                                                       std::int8_t* workspace,
                                                       std::size_t workspace_bytes) {
    return conv_ime_prepacked(input_nhwc_s8,
                              packed_b_mmt4d,
                              raw_output_nhwc,
                              params,
                              3,
                              3,
                              input_storage_zero_point_s8,
                              workspace,
                              workspace_bytes);
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
