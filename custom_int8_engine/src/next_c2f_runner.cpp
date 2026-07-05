#include "y26_k1x_next_c2f_runner.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kStage14Alignment = 64;

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

bool config_valid(const Y26Stage14NextC2fConfig* cfg) {
    if (cfg == nullptr || y26_stage12_c2f_block_output_count(&cfg->stage13) == 0 ||
        !conv_config_valid(cfg->model3) || !conv_config_valid(cfg->model4_cv1) ||
        cfg->model2_cv2_act_output_scale <= 0.0f || cfg->model2_cv2_act_output_zero_point_u8 < 0 ||
        cfg->model2_cv2_act_output_zero_point_u8 > 255 || cfg->model3_act_output_scale <= 0.0f ||
        cfg->model3_act_output_zero_point_u8 < 0 || cfg->model3_act_output_zero_point_u8 > 255 ||
        !activation_mode_valid(cfg->activation_mode)) {
        return false;
    }
    const Y26Stage7ConvNodeConfig& producer = cfg->stage13.model2_cv2;
    const int producer_h = output_h_for_kernel(producer.params, producer.kernel_h);
    const int producer_w = output_w_for_kernel(producer.params, producer.kernel_w);
    const int model3_h = output_h_for_kernel(cfg->model3.params, cfg->model3.kernel_h);
    const int model3_w = output_w_for_kernel(cfg->model3.params, cfg->model3.kernel_w);
    return cfg->model3.params.input_h == producer_h && cfg->model3.params.input_w == producer_w &&
           cfg->model3.params.input_c == producer.params.output_c &&
           cfg->model3.activation_zero_point_u8 == cfg->model2_cv2_act_output_zero_point_u8 &&
           cfg->model3.input_storage_zero_point_s8 == cfg->model2_cv2_act_output_zero_point_u8 - 128 &&
           cfg->model3.input_scale == cfg->model2_cv2_act_output_scale &&
           cfg->model4_cv1.params.input_h == model3_h && cfg->model4_cv1.params.input_w == model3_w &&
           cfg->model4_cv1.params.input_c == cfg->model3.params.output_c &&
           cfg->model4_cv1.activation_zero_point_u8 == cfg->model3_act_output_zero_point_u8 &&
           cfg->model4_cv1.input_storage_zero_point_s8 == cfg->model3_act_output_zero_point_u8 - 128 &&
           cfg->model4_cv1.input_scale == cfg->model3_act_output_scale;
}

void* allocate_aligned(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    return ::operator new(bytes, std::align_val_t(kStage14Alignment));
}

void free_aligned(void* ptr) {
    ::operator delete(ptr, std::align_val_t(kStage14Alignment));
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

void timing_reset(Y26Stage14TimingUs* timing) {
    if (timing != nullptr) {
        std::memset(timing, 0, sizeof(*timing));
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

Y26ActivationRequantParams activation_params(const Y26Stage7ConvNodeConfig& producer,
                                             std::size_t output_count,
                                             float act_output_scale,
                                             int act_output_zero_point_u8) {
    return Y26ActivationRequantParams{output_count,
                                      producer.params.output_c,
                                      producer.input_scale,
                                      producer.weight_scales,
                                      producer.output_scale,
                                      producer.output_zero_point_u8,
                                      act_output_scale,
                                      act_output_zero_point_u8};
}

int apply_activation_requant(const Y26Stage14NextC2fConfig& cfg,
                             const Y26Stage14NextC2fWorkspace& ws,
                             const Y26ActivationRequantParams& params,
                             const Y26FixedRequantParams* fixed_requant,
                             const std::int8_t* lut_s8,
                             const std::int32_t* producer_i32,
                             std::int8_t* consumer_input_s8) {
    switch (cfg.activation_mode) {
        case Y26_ACTIVATION_MODE_INT8_LUT:
            return y26_activation_requant_silu_int8_lut(&params, producer_i32, lut_s8, consumer_input_s8);
        case Y26_ACTIVATION_MODE_STAGE9_SCALAR_UNROLLED_LUT:
        case Y26_ACTIVATION_MODE_STAGE9_FUSED_CURRENT_LAYOUT:
            return y26_activation_requant_silu_int8_lut_scalar_unrolled(
                &params, producer_i32, lut_s8, consumer_input_s8);
        case Y26_ACTIVATION_MODE_STAGE9_FIXED_REQUANT_LUT:
            return y26_activation_requant_silu_int8_lut_fixed_requant(
                &params, fixed_requant, producer_i32, lut_s8, consumer_input_s8);
        case Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT:
            return y26_activation_requant_silu_int8_lut_rvv_f32(
                &params, producer_i32, lut_s8, consumer_input_s8);
        case Y26_ACTIVATION_MODE_FIXED_REQUANT_ONLY:
            (void)ws;
            return y26_activation_requant_silu_fixed_requant_only(
                &params, fixed_requant, producer_i32, consumer_input_s8);
        default:
            return y26_activation_requant_silu_scalar_float(&params, producer_i32, consumer_input_s8);
    }
}

int run_conv_scalar(const Y26Stage7ConvNodeConfig& cfg,
                    const Y26PrepackedConvWeights* weights,
                    const std::int8_t* input_s8,
                    std::int32_t* raw_i32,
                    std::int32_t* output_i32,
                    double* conv_us,
                    double* correction_us) {
    const auto begin = Clock::now();
    int status = scalar_raw_dot(cfg, input_s8, raw_i32);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    const auto correction_begin = Clock::now();
    status = apply_correction(cfg, weights, raw_i32, output_i32);
    const auto end = Clock::now();
    if (conv_us != nullptr) {
        *conv_us = elapsed_us(begin, end);
    }
    if (correction_us != nullptr) {
        *correction_us = elapsed_us(correction_begin, end);
    }
    return status;
}

int run_conv_ime(const Y26Stage7ConvNodeConfig& cfg,
                 const Y26PrepackedConvWeights* weights,
                 Y26ConvWorkspace* workspace,
                 const std::int8_t* input_s8,
                 std::int32_t* raw_i32,
                 std::int32_t* output_i32,
                 double* conv_us,
                 double* correction_us) {
    const auto begin = Clock::now();
    int status = y26_conv2d_i8s8s32_nhwc_ime_prepacked_v1(input_s8,
                                                          weights,
                                                          raw_i32,
                                                          cfg.input_storage_zero_point_s8,
                                                          workspace,
                                                          Y26_CONV_LOOP_ORDER_M_MAJOR);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    const auto correction_begin = Clock::now();
    status = apply_correction(cfg, weights, raw_i32, output_i32);
    const auto end = Clock::now();
    if (conv_us != nullptr) {
        *conv_us = elapsed_us(begin, end);
    }
    if (correction_us != nullptr) {
        *correction_us = elapsed_us(correction_begin, end);
    }
    return status;
}

void accumulate_stage13_timing(Y26Stage14TimingUs& dst, const Y26Stage12TimingUs& src) {
    dst.stage13_timing_us = src;
    dst.conv_us = src.conv_us;
    dst.activation_requant_us = src.activation_requant_us;
    dst.split_copy_us = src.split_copy_us;
    dst.merge_us = src.merge_total_us;
    dst.post_qdq_us = src.post_concat_qdq_us;
    dst.pack_layout_us = src.pack_layout_us;
    dst.add_us = src.add_compute_us;
    dst.concat_us = src.concat_materialize_us;
    dst.correction_us = src.correction_us;
}

void finalize_timing(Y26Stage14TimingUs& timing) {
    timing.conv_us += timing.model3_conv_us + timing.model4_cv1_conv_us;
    timing.correction_us += timing.model3_correction_us + timing.model4_cv1_correction_us;
    if (timing.total_us > 0.0) {
        timing.conv_share_pct = 100.0 * timing.conv_us / timing.total_us;
        timing.activation_share_pct = 100.0 * timing.activation_requant_us / timing.total_us;
        timing.merge_share_pct = 100.0 * timing.merge_us / timing.total_us;
        timing.pack_layout_share_pct = 100.0 * timing.pack_layout_us / timing.total_us;
        timing.split_branch_share_pct = 100.0 * (timing.split_copy_us + timing.add_us + timing.concat_us) /
                                        timing.total_us;
    }
}

int run_after_stage13(const Y26Stage14NextC2fConfig& cfg,
                      Y26Stage14NextC2fWorkspace& ws,
                      std::int32_t* output_i32_nhwc,
                      Y26Stage14TimingUs* timing,
                      bool use_ime) {
    const auto act0_begin = Clock::now();
    Y26ActivationRequantParams model2_cv2_act =
        activation_params(cfg.stage13.model2_cv2,
                          ws.model2_cv2_output_count,
                          cfg.model2_cv2_act_output_scale,
                          cfg.model2_cv2_act_output_zero_point_u8);
    int status = apply_activation_requant(cfg,
                                          ws,
                                          model2_cv2_act,
                                          ws.model2_cv2_fixed_requant,
                                          ws.model2_cv2_act_lut_s8,
                                          ws.model2_cv2_i32,
                                          ws.model3_input_s8);
    const auto act0_end = Clock::now();
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    if (timing != nullptr) {
        timing->activation_requant_us += elapsed_us(act0_begin, act0_end);
    }

    status = use_ime ? run_conv_ime(cfg.model3,
                                    ws.model3_weights,
                                    ws.model3_workspace,
                                    ws.model3_input_s8,
                                    ws.model3_raw_i32,
                                    ws.model3_i32,
                                    timing != nullptr ? &timing->model3_conv_us : nullptr,
                                    timing != nullptr ? &timing->model3_correction_us : nullptr)
                     : run_conv_scalar(cfg.model3,
                                       ws.model3_weights,
                                       ws.model3_input_s8,
                                       ws.model3_raw_i32,
                                       ws.model3_i32,
                                       timing != nullptr ? &timing->model3_conv_us : nullptr,
                                       timing != nullptr ? &timing->model3_correction_us : nullptr);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }

    const auto act1_begin = Clock::now();
    Y26ActivationRequantParams model3_act =
        activation_params(cfg.model3, ws.model3_output_count, cfg.model3_act_output_scale, cfg.model3_act_output_zero_point_u8);
    status = apply_activation_requant(cfg,
                                      ws,
                                      model3_act,
                                      ws.model3_fixed_requant,
                                      ws.model3_act_lut_s8,
                                      ws.model3_i32,
                                      ws.model4_cv1_input_s8);
    const auto act1_end = Clock::now();
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    if (timing != nullptr) {
        timing->activation_requant_us += elapsed_us(act1_begin, act1_end);
    }

    return use_ime ? run_conv_ime(cfg.model4_cv1,
                                  ws.model4_cv1_weights,
                                  ws.model4_cv1_workspace,
                                  ws.model4_cv1_input_s8,
                                  ws.model4_cv1_raw_i32,
                                  output_i32_nhwc,
                                  timing != nullptr ? &timing->model4_cv1_conv_us : nullptr,
                                  timing != nullptr ? &timing->model4_cv1_correction_us : nullptr)
                   : run_conv_scalar(cfg.model4_cv1,
                                     ws.model4_cv1_weights,
                                     ws.model4_cv1_input_s8,
                                     ws.model4_cv1_raw_i32,
                                     output_i32_nhwc,
                                     timing != nullptr ? &timing->model4_cv1_conv_us : nullptr,
                                     timing != nullptr ? &timing->model4_cv1_correction_us : nullptr);
}

}  // namespace

extern "C" int y26_stage14_next_c2f_prepare(const Y26Stage14NextC2fConfig* cfg,
                                             Y26Stage14NextC2fWorkspace* ws) {
    if (!config_valid(cfg) || ws == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    std::memset(ws, 0, sizeof(*ws));
    int status = y26_stage12_c2f_block_prepare(&cfg->stage13, &ws->stage13_ws);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }

    ws->model3_weights = y26_prepacked_conv_weights_create_mmt4d_s8(
        cfg->model3.weights_ohwi_s8, &cfg->model3.params, cfg->model3.kernel_h, cfg->model3.kernel_w,
        cfg->model3.node_name, cfg->model3.weight_scales);
    ws->model4_cv1_weights = y26_prepacked_conv_weights_create_mmt4d_s8(
        cfg->model4_cv1.weights_ohwi_s8, &cfg->model4_cv1.params, cfg->model4_cv1.kernel_h,
        cfg->model4_cv1.kernel_w, cfg->model4_cv1.node_name, cfg->model4_cv1.weight_scales);
    ws->model3_workspace = y26_conv_workspace_create(&cfg->model3.params, cfg->model3.kernel_h, cfg->model3.kernel_w);
    ws->model4_cv1_workspace =
        y26_conv_workspace_create(&cfg->model4_cv1.params, cfg->model4_cv1.kernel_h, cfg->model4_cv1.kernel_w);

    ws->model2_cv2_output_count = y26_stage12_c2f_block_output_count(&cfg->stage13);
    ws->model3_input_count = ws->model2_cv2_output_count;
    ws->model3_output_count = output_count_for_kernel(cfg->model3.params, cfg->model3.kernel_h, cfg->model3.kernel_w);
    ws->model4_cv1_input_count = ws->model3_output_count;
    ws->model4_cv1_output_count =
        output_count_for_kernel(cfg->model4_cv1.params, cfg->model4_cv1.kernel_h, cfg->model4_cv1.kernel_w);

    ws->model2_cv2_i32 = allocate_i32(ws->model2_cv2_output_count);
    ws->model3_input_s8 = allocate_i8(ws->model3_input_count);
    ws->model3_raw_i32 = allocate_i32(ws->model3_output_count);
    ws->model3_i32 = allocate_i32(ws->model3_output_count);
    ws->model4_cv1_input_s8 = allocate_i8(ws->model4_cv1_input_count);
    ws->model4_cv1_raw_i32 = allocate_i32(ws->model4_cv1_output_count);
    ws->model2_cv2_fixed_requant = allocate_fixed_requant(static_cast<std::size_t>(cfg->stage13.model2_cv2.params.output_c));
    ws->model3_fixed_requant = allocate_fixed_requant(static_cast<std::size_t>(cfg->model3.params.output_c));

    if (ws->model3_weights == nullptr || ws->model4_cv1_weights == nullptr || ws->model3_workspace == nullptr ||
        ws->model4_cv1_workspace == nullptr || ws->model2_cv2_i32 == nullptr || ws->model3_input_s8 == nullptr ||
        ws->model3_raw_i32 == nullptr || ws->model3_i32 == nullptr || ws->model4_cv1_input_s8 == nullptr ||
        ws->model4_cv1_raw_i32 == nullptr || ws->model2_cv2_fixed_requant == nullptr ||
        ws->model3_fixed_requant == nullptr) {
        y26_stage14_next_c2f_release(ws);
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }

    Y26ActivationRequantParams model2_cv2_act =
        activation_params(cfg->stage13.model2_cv2,
                          ws->model2_cv2_output_count,
                          cfg->model2_cv2_act_output_scale,
                          cfg->model2_cv2_act_output_zero_point_u8);
    Y26ActivationRequantParams model3_act =
        activation_params(cfg->model3, ws->model3_output_count, cfg->model3_act_output_scale, cfg->model3_act_output_zero_point_u8);
    status = y26_build_silu_u8_to_s8_lut(cfg->stage13.model2_cv2.output_scale,
                                         cfg->stage13.model2_cv2.output_zero_point_u8,
                                         cfg->model2_cv2_act_output_scale,
                                         cfg->model2_cv2_act_output_zero_point_u8,
                                         ws->model2_cv2_act_lut_s8);
    if (status == Y26_CONV_STATUS_SUCCESS) {
        status = y26_build_silu_u8_to_s8_lut(cfg->model3.output_scale,
                                             cfg->model3.output_zero_point_u8,
                                             cfg->model3_act_output_scale,
                                             cfg->model3_act_output_zero_point_u8,
                                             ws->model3_act_lut_s8);
    }
    if (status == Y26_CONV_STATUS_SUCCESS) {
        status = y26_build_fixed_requant_params_per_channel(&model2_cv2_act, ws->model2_cv2_fixed_requant);
    }
    if (status == Y26_CONV_STATUS_SUCCESS) {
        status = y26_build_fixed_requant_params_per_channel(&model3_act, ws->model3_fixed_requant);
    }
    if (status != Y26_CONV_STATUS_SUCCESS) {
        y26_stage14_next_c2f_release(ws);
        return status;
    }

    ws->prepacked_bytes = ws->stage13_ws.prepacked_bytes +
                          y26_prepacked_conv_weights_total_bytes(ws->model3_weights) +
                          y26_prepacked_conv_weights_total_bytes(ws->model4_cv1_weights);
    ws->workspace_bytes = ws->stage13_ws.workspace_bytes + y26_conv_workspace_bytes(ws->model3_workspace) +
                          y26_conv_workspace_bytes(ws->model4_cv1_workspace) +
                          (ws->model2_cv2_output_count + ws->model3_output_count + ws->model4_cv1_output_count) *
                              sizeof(std::int32_t) +
                          ws->model3_input_count + ws->model4_cv1_input_count +
                          (static_cast<std::size_t>(cfg->stage13.model2_cv2.params.output_c) +
                           static_cast<std::size_t>(cfg->model3.params.output_c)) *
                              sizeof(Y26FixedRequantParams);
    ws->prepared = 1;
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" void y26_stage14_next_c2f_release(Y26Stage14NextC2fWorkspace* ws) {
    if (ws == nullptr) {
        return;
    }
    y26_stage12_c2f_block_release(&ws->stage13_ws);
    y26_prepacked_conv_weights_destroy(ws->model3_weights);
    y26_prepacked_conv_weights_destroy(ws->model4_cv1_weights);
    y26_conv_workspace_destroy(ws->model3_workspace);
    y26_conv_workspace_destroy(ws->model4_cv1_workspace);
    free_aligned(ws->model2_cv2_i32);
    free_aligned(ws->model3_input_s8);
    free_aligned(ws->model3_raw_i32);
    free_aligned(ws->model3_i32);
    free_aligned(ws->model4_cv1_input_s8);
    free_aligned(ws->model4_cv1_raw_i32);
    free_aligned(ws->model2_cv2_fixed_requant);
    free_aligned(ws->model3_fixed_requant);
    std::memset(ws, 0, sizeof(*ws));
}

extern "C" std::size_t y26_stage14_next_c2f_output_count(const Y26Stage14NextC2fConfig* cfg) {
    return config_valid(cfg) ? output_count_for_kernel(cfg->model4_cv1.params,
                                                       cfg->model4_cv1.kernel_h,
                                                       cfg->model4_cv1.kernel_w)
                             : 0;
}

extern "C" int y26_stage14_next_c2f_run_scalar(const Y26Stage14NextC2fConfig* cfg,
                                                Y26Stage14NextC2fWorkspace* ws,
                                                const std::int8_t* input_nhwc_s8,
                                                std::int32_t* output_i32_nhwc,
                                                Y26Stage14TimingUs* timing) {
    if (!config_valid(cfg) || ws == nullptr || ws->prepared != 1 || input_nhwc_s8 == nullptr ||
        output_i32_nhwc == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    timing_reset(timing);
    const auto begin = Clock::now();
    Y26Stage12TimingUs stage13_timing {};
    int status = y26_stage12_c2f_block_run_scalar(
        &cfg->stage13, &ws->stage13_ws, input_nhwc_s8, ws->model2_cv2_i32, &stage13_timing);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    if (timing != nullptr) {
        accumulate_stage13_timing(*timing, stage13_timing);
    }
    status = run_after_stage13(*cfg, *ws, output_i32_nhwc, timing, false);
    const auto end = Clock::now();
    if (timing != nullptr) {
        timing->total_us = elapsed_us(begin, end);
        finalize_timing(*timing);
    }
    return status;
}

extern "C" int y26_stage14_next_c2f_run_ime_cluster0_hotpath(const Y26Stage14NextC2fConfig* cfg,
                                                              Y26Stage14NextC2fWorkspace* ws,
                                                              const std::int8_t* input_nhwc_s8,
                                                              std::int32_t* output_i32_nhwc,
                                                              Y26Stage14TimingUs* timing) {
    if (!config_valid(cfg) || ws == nullptr || ws->prepared != 1 || input_nhwc_s8 == nullptr ||
        output_i32_nhwc == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    timing_reset(timing);
    const auto begin = Clock::now();
    Y26Stage12TimingUs stage13_timing {};
    int status = y26_stage12_c2f_block_run_ime_cluster0_hotpath(
        &cfg->stage13, &ws->stage13_ws, input_nhwc_s8, ws->model2_cv2_i32, &stage13_timing);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    if (timing != nullptr) {
        accumulate_stage13_timing(*timing, stage13_timing);
    }
    status = run_after_stage13(*cfg, *ws, output_i32_nhwc, timing, true);
    const auto end = Clock::now();
    if (timing != nullptr) {
        timing->total_us = elapsed_us(begin, end);
        finalize_timing(*timing);
    }
    return status;
}

extern "C" const std::int8_t* y26_stage14_next_c2f_model3_input_s8(const Y26Stage14NextC2fWorkspace* ws) {
    return ws != nullptr ? ws->model3_input_s8 : nullptr;
}

extern "C" const std::int32_t* y26_stage14_next_c2f_model3_i32(const Y26Stage14NextC2fWorkspace* ws) {
    return ws != nullptr ? ws->model3_i32 : nullptr;
}

extern "C" const std::int8_t* y26_stage14_next_c2f_model4_cv1_input_s8(const Y26Stage14NextC2fWorkspace* ws) {
    return ws != nullptr ? ws->model4_cv1_input_s8 : nullptr;
}
