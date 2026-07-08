#include "y26_k1x_vmadot123_direct_conv.h"

#include "conv_kernel_common.h"
#include "y26_k1x_vmadot.h"
#include "y26_k1x_vmadot123_probe.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kKernelH = 3;
constexpr int kKernelW = 3;
constexpr int kOutputMStep = 7;

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000.0;
}

int align_up(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

int output_dim(int input, int kernel, int stride, int pad) {
    return (input + 2 * pad - kernel) / stride + 1;
}

bool storage_zero_point_valid(int zp) {
    return zp >= -128 && zp <= 127;
}

int conv_status_from_vmadot_status_local(int status) {
    switch (status) {
    case Y26_VMADOT_STATUS_SUCCESS:
        return Y26_CONV_STATUS_SUCCESS;
    case Y26_VMADOT_STATUS_NOT_BUILT_WITH_IME:
        return Y26_CONV_STATUS_NOT_BUILT_WITH_IME;
    case Y26_VMADOT_STATUS_RUNTIME_SAFETY_FAILED:
        return Y26_CONV_STATUS_RUNTIME_SAFETY_FAILED;
    case Y26_VMADOT_STATUS_SIGILL_CAUGHT:
        return Y26_CONV_STATUS_SIGILL_CAUGHT;
    default:
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
}

bool supported_params(const Y26Conv2DParams* params) {
    return y26_k1x::kernels::conv_params_valid(params) && params->stride_h == 1 && params->stride_w == 1 &&
           params->pad_h == 1 && params->pad_w == 1 && params->input_c > 0 && params->output_c > 0;
}

std::int8_t input_or_pad(const std::int8_t* input,
                         const Y26Conv2DParams& params,
                         int ih,
                         int iw,
                         int ic,
                         std::int8_t pad_value) {
    if (ih < 0 || ih >= params.input_h || iw < 0 || iw >= params.input_w) {
        return pad_value;
    }
    return input[(static_cast<std::size_t>(ih) * params.input_w + static_cast<std::size_t>(iw)) *
                     static_cast<std::size_t>(params.input_c) +
                 static_cast<std::size_t>(ic)];
}

void pack_a_panel_8xk_tiles(const std::int8_t* input,
                            const Y26Conv2DParams& params,
                            int output_w,
                            int output_m,
                            int m0,
                            int k_padded,
                            std::int8_t pad_value,
                            std::int8_t* panel) {
    std::fill(panel, panel + static_cast<std::size_t>(8 * k_padded), std::int8_t{0});
    for (int m = 0; m < 8; ++m) {
        const int flat_m = m0 + m;
        if (flat_m >= output_m) {
            continue;
        }
        const int oh = flat_m / output_w;
        const int ow = flat_m - oh * output_w;
        int flat_k = 0;
        for (int kh = 0; kh < kKernelH; ++kh) {
            const int ih = oh + kh - params.pad_h;
            for (int kw = 0; kw < kKernelW; ++kw) {
                const int iw = ow + kw - params.pad_w;
                for (int ic = 0; ic < params.input_c; ++ic, ++flat_k) {
                    const int k_tile = flat_k / 8;
                    const int k_lane = flat_k - k_tile * 8;
                    panel[static_cast<std::size_t>(k_tile) * 64U + static_cast<std::size_t>(m) * 8U +
                          static_cast<std::size_t>(k_lane)] =
                        input_or_pad(input, params, ih, iw, ic, pad_value);
                }
            }
        }
    }
}

void store_c_rows(const std::array<std::int32_t, 16>& c_tile,
                  int c_row,
                  int flat_m,
                  int n0,
                  int output_m,
                  int output_c,
                  std::int32_t* raw_output) {
    if (flat_m >= output_m) {
        return;
    }
    for (int n = 0; n < 4; ++n) {
        const int oc = n0 + n;
        if (oc < output_c) {
            raw_output[static_cast<std::size_t>(flat_m) * output_c + static_cast<std::size_t>(oc)] =
                c_tile[static_cast<std::size_t>(c_row) * 4U + static_cast<std::size_t>(n)];
        }
    }
}

int run_one_c_group(const std::int8_t* a_panel,
                    const std::int8_t* packed_b,
                    int n_tile,
                    int k_tiles,
                    std::array<std::int32_t, 16>& c0,
                    std::array<std::int32_t, 16>& c1,
                    std::array<std::int32_t, 16>& c2,
                    std::array<std::int32_t, 16>& c3) {
    c0.fill(0);
    c1.fill(0);
    c2.fill(0);
    c3.fill(0);
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const std::int8_t* a_tile = a_panel + static_cast<std::size_t>(k_tile) * 64U;
        const std::int8_t* b_tile = packed_b + (static_cast<std::size_t>(n_tile) *
                                                    static_cast<std::size_t>(k_tiles) +
                                                static_cast<std::size_t>(k_tile)) *
                                                   32U;
        const bool accumulate = k_tile != 0;
        int status = y26_k1x_vmadot_4x4x8_unsafe_cluster0_s8s8s32(a_tile, b_tile, c0.data(), accumulate);
        if (status != Y26_VMADOT_STATUS_SUCCESS) {
            return conv_status_from_vmadot_status_local(status);
        }
        status = y26_k1x_vmadot123_unsafe_cluster0_s8s8s32(
            Y26_VMADOT123_VARIANT_1, a_tile, b_tile, c1.data(), accumulate);
        if (status != Y26_VMADOT_STATUS_SUCCESS) {
            return conv_status_from_vmadot_status_local(status);
        }
        status = y26_k1x_vmadot123_unsafe_cluster0_s8s8s32(
            Y26_VMADOT123_VARIANT_2, a_tile, b_tile, c2.data(), accumulate);
        if (status != Y26_VMADOT_STATUS_SUCCESS) {
            return conv_status_from_vmadot_status_local(status);
        }
        status = y26_k1x_vmadot123_unsafe_cluster0_s8s8s32(
            Y26_VMADOT123_VARIANT_3, a_tile, b_tile, c3.data(), accumulate);
        if (status != Y26_VMADOT_STATUS_SUCCESS) {
            return conv_status_from_vmadot_status_local(status);
        }
    }
    return Y26_CONV_STATUS_SUCCESS;
}

}  // namespace

struct Y26Vmadot123DirectConvWorkspace {
    Y26Conv2DParams params {};
    int output_h = 0;
    int output_w = 0;
    int output_m = 0;
    int kernel_k = 0;
    int k_padded = 0;
    int k_tiles = 0;
    std::vector<std::int8_t> a_panel;
    std::vector<std::int32_t> raw_output;
};

extern "C" Y26Vmadot123DirectConvWorkspace* y26_vmadot123_direct_conv3x3_workspace_create(
    const Y26Conv2DParams* params) {
    if (!supported_params(params)) {
        return nullptr;
    }
    auto* workspace = new (std::nothrow) Y26Vmadot123DirectConvWorkspace();
    if (workspace == nullptr) {
        return nullptr;
    }
    workspace->params = *params;
    workspace->output_h = output_dim(params->input_h, kKernelH, params->stride_h, params->pad_h);
    workspace->output_w = output_dim(params->input_w, kKernelW, params->stride_w, params->pad_w);
    if (workspace->output_h <= 0 || workspace->output_w <= 0) {
        delete workspace;
        return nullptr;
    }
    workspace->output_m = workspace->output_h * workspace->output_w;
    workspace->kernel_k = kKernelH * kKernelW * params->input_c;
    workspace->k_padded = align_up(workspace->kernel_k, 8);
    workspace->k_tiles = workspace->k_padded / 8;
    workspace->a_panel.resize(static_cast<std::size_t>(8 * workspace->k_padded));
    workspace->raw_output.resize(static_cast<std::size_t>(workspace->output_m) *
                                 static_cast<std::size_t>(params->output_c));
    return workspace;
}

extern "C" void y26_vmadot123_direct_conv3x3_workspace_destroy(
    Y26Vmadot123DirectConvWorkspace* workspace) {
    delete workspace;
}

extern "C" std::size_t y26_vmadot123_direct_conv3x3_workspace_bytes(
    const Y26Vmadot123DirectConvWorkspace* workspace) {
    if (workspace == nullptr) {
        return 0;
    }
    return workspace->a_panel.size() + workspace->raw_output.size() * sizeof(std::int32_t);
}

extern "C" int y26_vmadot123_direct_conv3x3_i8s8s32_nhwc_single_thread(
    const std::int8_t* input_nhwc_s8,
    const Y26PrepackedConvWeights* weights,
    const std::int32_t* bias_oc,
    std::int32_t* corrected_output_nhwc,
    const Y26Conv2DParams* params,
    int input_storage_zero_point_s8,
    int activation_zero_point_u8,
    Y26Vmadot123DirectConvWorkspace* workspace,
    Y26Vmadot123DirectConvTimingUs* timing) {
    if (timing != nullptr) {
        std::memset(timing, 0, sizeof(*timing));
        timing->output_m_step = kOutputMStep;
    }
    if (input_nhwc_s8 == nullptr || weights == nullptr || bias_oc == nullptr || corrected_output_nhwc == nullptr ||
        !supported_params(params) || workspace == nullptr || !storage_zero_point_valid(input_storage_zero_point_s8) ||
        activation_zero_point_u8 < 0 || activation_zero_point_u8 > 255) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    if (workspace->params.input_h != params->input_h || workspace->params.input_w != params->input_w ||
        workspace->params.input_c != params->input_c || workspace->params.output_c != params->output_c) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    if (!y26_vmadot_4x4x8_ime_available_buildtime() || !y26_vmadot123_probe_available_buildtime()) {
        return Y26_CONV_STATUS_NOT_BUILT_WITH_IME;
    }
    if (!y26_k1x_ime_hotpath_allowed_on_current_cpu()) {
        const auto snapshot = y26_k1x_ime_runtime_state_snapshot();
        return conv_status_from_vmadot_status_local(snapshot.probe_status);
    }
    const std::int8_t* packed_b = y26_prepacked_conv_weights_packed_b(weights);
    const std::int32_t* weight_sums = y26_prepacked_conv_weights_sums(weights);
    if (packed_b == nullptr || weight_sums == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }

    const auto total_begin = Clock::now();
    double panel_build_us = 0.0;
    double kernel_compute_us = 0.0;
    double writeback_us = 0.0;
    std::fill(workspace->raw_output.begin(), workspace->raw_output.end(), 0);
    std::array<std::int32_t, 16> c0 {};
    std::array<std::int32_t, 16> c1 {};
    std::array<std::int32_t, 16> c2 {};
    std::array<std::int32_t, 16> c3 {};
    const int n_tiles = align_up(params->output_c, 4) / 4;

    for (int m0 = 0; m0 < workspace->output_m; m0 += kOutputMStep) {
        const auto panel_begin = Clock::now();
        pack_a_panel_8xk_tiles(input_nhwc_s8,
                               *params,
                               workspace->output_w,
                               workspace->output_m,
                               m0,
                               workspace->k_padded,
                               static_cast<std::int8_t>(input_storage_zero_point_s8),
                               workspace->a_panel.data());
        const auto panel_end = Clock::now();
        panel_build_us += elapsed_us(panel_begin, panel_end);

        for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
            const int n0 = n_tile * 4;
            const auto compute_begin = Clock::now();
            const int status = run_one_c_group(workspace->a_panel.data(),
                                               packed_b,
                                               n_tile,
                                               workspace->k_tiles,
                                               c0,
                                               c1,
                                               c2,
                                               c3);
            const auto compute_end = Clock::now();
            kernel_compute_us += elapsed_us(compute_begin, compute_end);
            if (status != Y26_CONV_STATUS_SUCCESS) {
                return status;
            }

            const auto write_begin = Clock::now();
            store_c_rows(c0, 0, m0 + 0, n0, workspace->output_m, params->output_c, workspace->raw_output.data());
            store_c_rows(c0, 1, m0 + 1, n0, workspace->output_m, params->output_c, workspace->raw_output.data());
            store_c_rows(c0, 2, m0 + 2, n0, workspace->output_m, params->output_c, workspace->raw_output.data());
            store_c_rows(c0, 3, m0 + 3, n0, workspace->output_m, params->output_c, workspace->raw_output.data());
            store_c_rows(c1, 3, m0 + 4, n0, workspace->output_m, params->output_c, workspace->raw_output.data());
            store_c_rows(c2, 3, m0 + 5, n0, workspace->output_m, params->output_c, workspace->raw_output.data());
            store_c_rows(c3, 3, m0 + 6, n0, workspace->output_m, params->output_c, workspace->raw_output.data());
            const auto write_end = Clock::now();
            writeback_us += elapsed_us(write_begin, write_end);
        }
    }

    const auto correction_begin = Clock::now();
    int status = y26_conv2d_apply_u8_as_s8_correction_nhwc(workspace->raw_output.data(),
                                                           bias_oc,
                                                           weight_sums,
                                                           corrected_output_nhwc,
                                                           workspace->output_m,
                                                           params->output_c,
                                                           activation_zero_point_u8);
    const auto correction_end = Clock::now();
    const auto total_end = Clock::now();
    if (timing != nullptr) {
        timing->panel_build_us = panel_build_us;
        timing->kernel_compute_us = kernel_compute_us;
        timing->writeback_us = writeback_us;
        timing->correction_us = elapsed_us(correction_begin, correction_end);
        timing->total_us = elapsed_us(total_begin, total_end);
        timing->used_vmadot123 = 1;
    }
    return status;
}
