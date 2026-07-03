#include "conv_kernel_common.h"
#include "y26_k1x_vmadot.h"

#include <array>

namespace {

void pack_conv1x1_a_tile(const std::int8_t* input_nhwc,
                         const Y26Conv2DParams& params,
                         int output_w,
                         int m_offset,
                         int k_offset,
                         std::array<std::int8_t, 32>& dst) {
    for (int m = 0; m < 4; ++m) {
        const int flat_m = m_offset + m;
        const int oh = y26_k1x::kernels::flat_output_m_to_h(flat_m, output_w);
        const int ow = y26_k1x::kernels::flat_output_m_to_w(flat_m, output_w);
        const int ih = oh * params.stride_h - params.pad_h;
        const int iw = ow * params.stride_w - params.pad_w;
        for (int k = 0; k < 8; ++k) {
            const int ic = k_offset + k;
            dst[m * 8 + k] = y26_k1x::kernels::input_nhwc_or_zero(input_nhwc, params, ih, iw, ic);
        }
    }
}

void pack_conv1x1_b_tile(const std::int8_t* weights_oc_ic,
                         const Y26Conv2DParams& params,
                         int n_offset,
                         int k_offset,
                         std::array<std::int8_t, 32>& dst) {
    for (int n = 0; n < 4; ++n) {
        const int oc = n_offset + n;
        for (int k = 0; k < 8; ++k) {
            const int ic = k_offset + k;
            dst[n * 8 + k] =
                (oc < params.output_c && ic < params.input_c) ? weights_oc_ic[oc * params.input_c + ic] : 0;
        }
    }
}

}  // namespace

extern "C" int y26_conv1x1_i8s8s32_nhwc_ime(const std::int8_t* input_nhwc,
                                             const std::int8_t* weights_oc_ic,
                                             const std::int32_t* bias_oc,
                                             std::int32_t* output_nhwc,
                                             const Y26Conv2DParams* params) {
    if (input_nhwc == nullptr || weights_oc_ic == nullptr || output_nhwc == nullptr ||
        !y26_k1x::kernels::conv_params_valid(params)) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
        return Y26_CONV_STATUS_NOT_BUILT_WITH_IME;
    }
    if (!y26_k1x_ime_hotpath_allowed_on_current_cpu()) {
        const auto snapshot = y26_k1x_ime_runtime_state_snapshot();
        return y26_k1x::kernels::conv_status_from_vmadot_status(snapshot.probe_status);
    }

    const int output_h = y26_conv1x1_output_h(params);
    const int output_w = y26_conv1x1_output_w(params);
    if (output_h <= 0 || output_w <= 0) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }

    const int output_m = output_h * output_w;
    std::array<std::int8_t, 32> a_tile {};
    std::array<std::int8_t, 32> b_tile {};
    std::array<std::int32_t, 16> c_tile {};

    for (int m0 = 0; m0 < output_m; m0 += 4) {
        for (int n0 = 0; n0 < params->output_c; n0 += 4) {
            for (int m = 0; m < 4; ++m) {
                for (int n = 0; n < 4; ++n) {
                    const int oc = n0 + n;
                    c_tile[m * 4 + n] = (oc < params->output_c && bias_oc != nullptr) ? bias_oc[oc] : 0;
                }
            }
            for (int k0 = 0; k0 < params->input_c; k0 += 8) {
                pack_conv1x1_a_tile(input_nhwc, *params, output_w, m0, k0, a_tile);
                pack_conv1x1_b_tile(weights_oc_ic, *params, n0, k0, b_tile);
                const int status = y26_k1x_vmadot_4x4x8_unsafe_cluster0_s8s8s32(
                    a_tile.data(), b_tile.data(), c_tile.data(), true);
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
                    if (oc >= params->output_c) {
                        continue;
                    }
                    output_nhwc[flat_m * params->output_c + oc] = c_tile[m * 4 + n];
                }
            }
        }
    }
    return Y26_CONV_STATUS_SUCCESS;
}
