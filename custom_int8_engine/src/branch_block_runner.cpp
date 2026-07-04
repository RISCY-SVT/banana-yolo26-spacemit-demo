#include "y26_k1x_branch_block_runner.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kStage11Alignment = 64;

bool kernel_supported(int kernel_h, int kernel_w) {
    return (kernel_h == 1 && kernel_w == 1) || (kernel_h == 3 && kernel_w == 3);
}

bool conv_params_valid(const Y26Conv2DParams& params) {
    return params.input_h > 0 && params.input_w > 0 && params.input_c > 0 && params.output_c > 0 &&
           params.stride_h > 0 && params.stride_w > 0 && params.pad_h >= 0 && params.pad_w >= 0;
}

int output_h_for_kernel(const Y26Conv2DParams& params, int kernel_h) {
    return kernel_h == 1 ? y26_conv1x1_output_h(&params) : y26_conv3x3_output_h(&params);
}

int output_w_for_kernel(const Y26Conv2DParams& params, int kernel_w) {
    return kernel_w == 1 ? y26_conv1x1_output_w(&params) : y26_conv3x3_output_w(&params);
}

std::size_t output_count_for_kernel(const Y26Conv2DParams& params, int kernel_h, int kernel_w) {
    const int output_h = output_h_for_kernel(params, kernel_h);
    const int output_w = output_w_for_kernel(params, kernel_w);
    if (output_h <= 0 || output_w <= 0) {
        return 0;
    }
    return static_cast<std::size_t>(output_h) * static_cast<std::size_t>(output_w) *
           static_cast<std::size_t>(params.output_c);
}

std::size_t expected_weight_count(const Y26Stage7ConvNodeConfig& cfg) {
    return static_cast<std::size_t>(cfg.params.output_c) * static_cast<std::size_t>(cfg.kernel_h) *
           static_cast<std::size_t>(cfg.kernel_w) * static_cast<std::size_t>(cfg.params.input_c);
}

bool conv_config_valid(const Y26Stage7ConvNodeConfig& cfg) {
    if (!conv_params_valid(cfg.params) || !kernel_supported(cfg.kernel_h, cfg.kernel_w) ||
        cfg.activation_zero_point_u8 < 0 || cfg.activation_zero_point_u8 > 255 ||
        cfg.input_storage_zero_point_s8 < static_cast<int>(std::numeric_limits<std::int8_t>::min()) ||
        cfg.input_storage_zero_point_s8 > static_cast<int>(std::numeric_limits<std::int8_t>::max()) ||
        cfg.output_zero_point_u8 < 0 || cfg.output_zero_point_u8 > 255 || cfg.input_scale <= 0.0f ||
        cfg.output_scale <= 0.0f || cfg.weight_scales == nullptr || cfg.weights_ohwi_s8 == nullptr ||
        cfg.bias_i32 == nullptr) {
        return false;
    }
    if (cfg.weight_scale_count < static_cast<std::size_t>(cfg.params.output_c) ||
        cfg.weight_count < expected_weight_count(cfg) ||
        cfg.bias_count < static_cast<std::size_t>(cfg.params.output_c)) {
        return false;
    }
    return output_count_for_kernel(cfg.params, cfg.kernel_h, cfg.kernel_w) > 0;
}

bool activation_mode_valid(int mode) {
    return mode == Y26_ACTIVATION_MODE_SCALAR_FLOAT_REFERENCE ||
           mode == Y26_ACTIVATION_MODE_FIXED_REQUANT_ONLY ||
           mode == Y26_ACTIVATION_MODE_INT8_LUT ||
           mode == Y26_ACTIVATION_MODE_STAGE9_SCALAR_UNROLLED_LUT ||
           mode == Y26_ACTIVATION_MODE_STAGE9_FIXED_REQUANT_LUT ||
           mode == Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT ||
           mode == Y26_ACTIVATION_MODE_STAGE9_FUSED_CURRENT_LAYOUT;
}

bool config_valid(const Y26Stage11BranchBlockConfig* cfg) {
    if (cfg == nullptr || y26_stage10_backbone_expansion_output_count(&cfg->stage10) == 0 ||
        !conv_config_valid(cfg->branch1) || cfg->branch0_act_output_scale <= 0.0f ||
        cfg->branch0_act_output_zero_point_u8 < 0 || cfg->branch0_act_output_zero_point_u8 > 255 ||
        !activation_mode_valid(cfg->activation_mode)) {
        return false;
    }

    const Y26Stage7ConvNodeConfig& branch0 = cfg->stage10.branch0;
    const int branch0_output_h = output_h_for_kernel(branch0.params, branch0.kernel_h);
    const int branch0_output_w = output_w_for_kernel(branch0.params, branch0.kernel_w);
    return cfg->branch1.params.input_h == branch0_output_h &&
           cfg->branch1.params.input_w == branch0_output_w &&
           cfg->branch1.params.input_c == branch0.params.output_c &&
           cfg->branch1.activation_zero_point_u8 == cfg->branch0_act_output_zero_point_u8 &&
           cfg->branch1.input_storage_zero_point_s8 == cfg->branch0_act_output_zero_point_u8 - 128;
}

void* allocate_aligned(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    return ::operator new(bytes, std::align_val_t(kStage11Alignment));
}

void free_aligned(void* ptr) {
    ::operator delete(ptr, std::align_val_t(kStage11Alignment));
}

std::int32_t* allocate_i32(std::size_t count) {
    return static_cast<std::int32_t*>(allocate_aligned(count * sizeof(std::int32_t)));
}

std::int8_t* allocate_i8(std::size_t count) {
    return static_cast<std::int8_t*>(allocate_aligned(count));
}

Y26FixedRequantParams* allocate_fixed_requant(std::size_t count) {
    return static_cast<Y26FixedRequantParams*>(allocate_aligned(count * sizeof(Y26FixedRequantParams)));
}

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000.0;
}

void timing_reset(Y26Stage11TimingUs* timing) {
    if (timing != nullptr) {
        std::memset(timing, 0, sizeof(*timing));
    }
}

Y26ActivationRequantParams branch0_activation_params(const Y26Stage11BranchBlockConfig& cfg) {
    return Y26ActivationRequantParams{y26_stage10_backbone_expansion_output_count(&cfg.stage10),
                                      cfg.stage10.branch0.params.output_c,
                                      cfg.stage10.branch0.input_scale,
                                      cfg.stage10.branch0.weight_scales,
                                      cfg.stage10.branch0.output_scale,
                                      cfg.stage10.branch0.output_zero_point_u8,
                                      cfg.branch0_act_output_scale,
                                      cfg.branch0_act_output_zero_point_u8};
}

int apply_branch0_activation(const Y26Stage11BranchBlockConfig& cfg,
                             const Y26Stage11BranchBlockWorkspace& ws,
                             const std::int32_t* producer_i32,
                             std::int8_t* consumer_input_s8) {
    const Y26ActivationRequantParams params = branch0_activation_params(cfg);
    switch (cfg.activation_mode) {
        case Y26_ACTIVATION_MODE_INT8_LUT:
            return y26_activation_requant_silu_int8_lut(&params, producer_i32, ws.branch0_act_lut_s8, consumer_input_s8);
        case Y26_ACTIVATION_MODE_STAGE9_SCALAR_UNROLLED_LUT:
        case Y26_ACTIVATION_MODE_STAGE9_FUSED_CURRENT_LAYOUT:
            return y26_activation_requant_silu_int8_lut_scalar_unrolled(
                &params, producer_i32, ws.branch0_act_lut_s8, consumer_input_s8);
        case Y26_ACTIVATION_MODE_STAGE9_FIXED_REQUANT_LUT:
            return y26_activation_requant_silu_int8_lut_fixed_requant(
                &params, ws.branch0_fixed_requant, producer_i32, ws.branch0_act_lut_s8, consumer_input_s8);
        case Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT:
            return y26_activation_requant_silu_int8_lut_rvv_f32(
                &params, producer_i32, ws.branch0_act_lut_s8, consumer_input_s8);
        case Y26_ACTIVATION_MODE_FIXED_REQUANT_ONLY:
            return y26_activation_requant_silu_fixed_requant_only(
                &params, ws.branch0_fixed_requant, producer_i32, consumer_input_s8);
        default:
            return y26_activation_requant_silu_scalar_float(&params, producer_i32, consumer_input_s8);
    }
}

std::int8_t weight_at(const Y26Stage7ConvNodeConfig& cfg, int oc, int kh, int kw, int ic) {
    const int index = ((oc * cfg.kernel_h + kh) * cfg.kernel_w + kw) * cfg.params.input_c + ic;
    return cfg.weights_ohwi_s8[index];
}

int scalar_raw_dot(const Y26Stage7ConvNodeConfig& cfg,
                   const std::int8_t* input_nhwc_s8,
                   std::int32_t* raw_i32_nhwc) {
    const int output_h = output_h_for_kernel(cfg.params, cfg.kernel_h);
    const int output_w = output_w_for_kernel(cfg.params, cfg.kernel_w);
    const std::int8_t pad = static_cast<std::int8_t>(cfg.input_storage_zero_point_s8);
    for (int oh = 0; oh < output_h; ++oh) {
        for (int ow = 0; ow < output_w; ++ow) {
            for (int oc = 0; oc < cfg.params.output_c; ++oc) {
                std::int32_t acc = 0;
                for (int kh = 0; kh < cfg.kernel_h; ++kh) {
                    const int ih = oh * cfg.params.stride_h + kh - cfg.params.pad_h;
                    const bool valid_h = ih >= 0 && ih < cfg.params.input_h;
                    for (int kw = 0; kw < cfg.kernel_w; ++kw) {
                        const int iw = ow * cfg.params.stride_w + kw - cfg.params.pad_w;
                        const bool inside = valid_h && iw >= 0 && iw < cfg.params.input_w;
                        const std::int8_t* src =
                            inside ? input_nhwc_s8 + (ih * cfg.params.input_w + iw) * cfg.params.input_c : nullptr;
                        for (int ic = 0; ic < cfg.params.input_c; ++ic) {
                            const std::int8_t a = inside ? src[ic] : pad;
                            acc += static_cast<std::int32_t>(a) *
                                   static_cast<std::int32_t>(weight_at(cfg, oc, kh, kw, ic));
                        }
                    }
                }
                raw_i32_nhwc[(oh * output_w + ow) * cfg.params.output_c + oc] = acc;
            }
        }
    }
    return Y26_CONV_STATUS_SUCCESS;
}

int apply_correction(const Y26Stage7ConvNodeConfig& cfg,
                     const Y26PrepackedConvWeights* weights,
                     const std::int32_t* raw_i32,
                     std::int32_t* corrected_i32) {
    return y26_conv2d_apply_u8_as_s8_correction_nhwc(raw_i32,
                                                     cfg.bias_i32,
                                                     y26_prepacked_conv_weights_sums(weights),
                                                     corrected_i32,
                                                     output_h_for_kernel(cfg.params, cfg.kernel_h) *
                                                         output_w_for_kernel(cfg.params, cfg.kernel_w),
                                                     cfg.params.output_c,
                                                     cfg.activation_zero_point_u8);
}

int run_branch1_scalar(const Y26Stage11BranchBlockConfig& cfg,
                       Y26Stage11BranchBlockWorkspace& ws,
                       std::int32_t* output_i32_nhwc,
                       Y26Stage11TimingUs* timing) {
    const auto branch_begin = Clock::now();
    int status = scalar_raw_dot(cfg.branch1, ws.branch0_act_s8, ws.branch1_raw_i32);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    const auto correction_begin = Clock::now();
    status = apply_correction(cfg.branch1, ws.branch1_weights, ws.branch1_raw_i32, output_i32_nhwc);
    const auto end = Clock::now();
    if (timing != nullptr) {
        timing->branch_cv2_conv_us = elapsed_us(branch_begin, end);
        timing->branch_cv2_correction_us = elapsed_us(correction_begin, end);
    }
    return status;
}

int run_branch1_ime(const Y26Stage11BranchBlockConfig& cfg,
                    Y26Stage11BranchBlockWorkspace& ws,
                    std::int32_t* output_i32_nhwc,
                    Y26Stage11TimingUs* timing) {
    const auto branch_begin = Clock::now();
    int status = y26_conv2d_i8s8s32_nhwc_ime_prepacked_v1(ws.branch0_act_s8,
                                                          ws.branch1_weights,
                                                          ws.branch1_raw_i32,
                                                          cfg.branch1.input_storage_zero_point_s8,
                                                          ws.branch1_workspace,
                                                          Y26_CONV_LOOP_ORDER_M_MAJOR);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    const auto correction_begin = Clock::now();
    status = apply_correction(cfg.branch1, ws.branch1_weights, ws.branch1_raw_i32, output_i32_nhwc);
    const auto end = Clock::now();
    if (timing != nullptr) {
        timing->branch_cv2_conv_us = elapsed_us(branch_begin, end);
        timing->branch_cv2_correction_us = elapsed_us(correction_begin, end);
    }
    return status;
}

void copy_stage10_timing(Y26Stage11TimingUs& dst, const Y26Stage10TimingUs& src) {
    dst.stage10_timing_us = src;
    dst.conv0_ime_us = src.conv0_ime_us;
    dst.act0_requant_lut_us = src.act0_requant_lut_us;
    dst.conv1_ime_us = src.conv1_ime_us;
    dst.act1_requant_lut_us = src.act1_requant_lut_us;
    dst.conv2_ime_us = src.conv2_ime_us;
    dst.act2_requant_lut_us = src.act2_requant_lut_us;
    dst.split_us = src.split_us;
    dst.branch_cv1_conv_us = src.branch_conv_us;
    dst.layout_or_pack_us = src.pack_layout_us;
}

int run_after_stage10(const Y26Stage11BranchBlockConfig& cfg,
                      Y26Stage11BranchBlockWorkspace& ws,
                      std::int32_t* output_i32_nhwc,
                      Y26Stage11TimingUs* timing,
                      bool use_ime) {
    const auto act_begin = Clock::now();
    int status = apply_branch0_activation(cfg, ws, ws.branch0_i32, ws.branch0_act_s8);
    const auto act_end = Clock::now();
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    if (timing != nullptr) {
        timing->branch_cv1_activation_us = elapsed_us(act_begin, act_end);
    }
    return use_ime ? run_branch1_ime(cfg, ws, output_i32_nhwc, timing)
                   : run_branch1_scalar(cfg, ws, output_i32_nhwc, timing);
}

}  // namespace

extern "C" int y26_stage11_branch_block_prepare(const Y26Stage11BranchBlockConfig* cfg,
                                                 Y26Stage11BranchBlockWorkspace* ws) {
    if (!config_valid(cfg) || ws == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    std::memset(ws, 0, sizeof(*ws));
    int status = y26_stage10_backbone_expansion_prepare(&cfg->stage10, &ws->stage10_ws);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }

    ws->branch1_weights = y26_prepacked_conv_weights_create_mmt4d_s8(cfg->branch1.weights_ohwi_s8,
                                                                      &cfg->branch1.params,
                                                                      cfg->branch1.kernel_h,
                                                                      cfg->branch1.kernel_w,
                                                                      cfg->branch1.node_name,
                                                                      cfg->branch1.weight_scales);
    ws->branch1_workspace = y26_conv_workspace_create(&cfg->branch1.params, cfg->branch1.kernel_h, cfg->branch1.kernel_w);
    ws->branch0_output_count = y26_stage10_backbone_expansion_output_count(&cfg->stage10);
    ws->branch0_act_count = ws->branch0_output_count;
    ws->branch1_output_count = output_count_for_kernel(cfg->branch1.params, cfg->branch1.kernel_h, cfg->branch1.kernel_w);
    ws->branch0_i32 = allocate_i32(ws->branch0_output_count);
    ws->branch0_act_s8 = allocate_i8(ws->branch0_act_count);
    ws->branch1_raw_i32 = allocate_i32(ws->branch1_output_count);
    ws->branch0_fixed_requant = allocate_fixed_requant(static_cast<std::size_t>(cfg->stage10.branch0.params.output_c));

    const Y26ActivationRequantParams act_params = branch0_activation_params(*cfg);
    status = y26_build_silu_u8_to_s8_lut(cfg->stage10.branch0.output_scale,
                                         cfg->stage10.branch0.output_zero_point_u8,
                                         cfg->branch0_act_output_scale,
                                         cfg->branch0_act_output_zero_point_u8,
                                         ws->branch0_act_lut_s8);
    if (status == Y26_CONV_STATUS_SUCCESS) {
        status = y26_build_fixed_requant_params_per_channel(&act_params, ws->branch0_fixed_requant);
    }
    if (status != Y26_CONV_STATUS_SUCCESS || ws->branch1_weights == nullptr || ws->branch1_workspace == nullptr ||
        ws->branch0_i32 == nullptr || ws->branch0_act_s8 == nullptr || ws->branch1_raw_i32 == nullptr ||
        ws->branch0_fixed_requant == nullptr) {
        y26_stage11_branch_block_release(ws);
        return status == Y26_CONV_STATUS_SUCCESS ? Y26_CONV_STATUS_INVALID_ARGUMENT : status;
    }

    ws->prepacked_bytes = y26_prepacked_conv_weights_total_bytes(ws->branch1_weights) + ws->stage10_ws.prepacked_bytes;
    ws->workspace_bytes = y26_conv_workspace_bytes(ws->branch1_workspace) + ws->stage10_ws.workspace_bytes;
    ws->prepared = 1;
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" void y26_stage11_branch_block_release(Y26Stage11BranchBlockWorkspace* ws) {
    if (ws == nullptr) {
        return;
    }
    y26_stage10_backbone_expansion_release(&ws->stage10_ws);
    y26_prepacked_conv_weights_destroy(ws->branch1_weights);
    y26_conv_workspace_destroy(ws->branch1_workspace);
    free_aligned(ws->branch0_i32);
    free_aligned(ws->branch0_act_s8);
    free_aligned(ws->branch1_raw_i32);
    free_aligned(ws->branch0_fixed_requant);
    std::memset(ws, 0, sizeof(*ws));
}

extern "C" std::size_t y26_stage11_branch_block_output_count(const Y26Stage11BranchBlockConfig* cfg) {
    return config_valid(cfg) ? output_count_for_kernel(cfg->branch1.params, cfg->branch1.kernel_h, cfg->branch1.kernel_w)
                             : 0;
}

extern "C" int y26_stage11_branch_block_run_scalar(const Y26Stage11BranchBlockConfig* cfg,
                                                    Y26Stage11BranchBlockWorkspace* ws,
                                                    const std::int8_t* input_nhwc_s8,
                                                    std::int32_t* output_i32_nhwc,
                                                    Y26Stage11TimingUs* timing) {
    if (!config_valid(cfg) || ws == nullptr || ws->prepared == 0 || input_nhwc_s8 == nullptr ||
        output_i32_nhwc == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    timing_reset(timing);
    const auto begin = Clock::now();
    Y26Stage10TimingUs stage10_timing {};
    int status = y26_stage10_backbone_expansion_run_scalar(
        &cfg->stage10, &ws->stage10_ws, input_nhwc_s8, ws->branch0_i32, &stage10_timing);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    if (timing != nullptr) {
        copy_stage10_timing(*timing, stage10_timing);
    }
    status = run_after_stage10(*cfg, *ws, output_i32_nhwc, timing, false);
    if (timing != nullptr) {
        timing->total_us = elapsed_us(begin, Clock::now());
    }
    return status;
}

extern "C" int y26_stage11_branch_block_run_ime_cluster0_hotpath(const Y26Stage11BranchBlockConfig* cfg,
                                                                  Y26Stage11BranchBlockWorkspace* ws,
                                                                  const std::int8_t* input_nhwc_s8,
                                                                  std::int32_t* output_i32_nhwc,
                                                                  Y26Stage11TimingUs* timing) {
    if (!config_valid(cfg) || ws == nullptr || ws->prepared == 0 || input_nhwc_s8 == nullptr ||
        output_i32_nhwc == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    timing_reset(timing);
    const auto begin = Clock::now();
    Y26Stage10TimingUs stage10_timing {};
    int status = y26_stage10_backbone_expansion_run_ime_cluster0_hotpath(
        &cfg->stage10, &ws->stage10_ws, input_nhwc_s8, ws->branch0_i32, &stage10_timing);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    if (timing != nullptr) {
        copy_stage10_timing(*timing, stage10_timing);
    }
    status = run_after_stage10(*cfg, *ws, output_i32_nhwc, timing, true);
    if (timing != nullptr) {
        timing->total_us = elapsed_us(begin, Clock::now());
    }
    return status;
}

extern "C" const std::int8_t* y26_stage11_branch_block_branch0_activation_s8(
    const Y26Stage11BranchBlockWorkspace* ws) {
    return ws != nullptr ? ws->branch0_act_s8 : nullptr;
}
