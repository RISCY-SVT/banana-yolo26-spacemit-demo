#include "y26_k1x_model4_branch_runner.h"

#include "y26_k1x_threaded_conv.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kStage15Alignment = 64;

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

bool config_valid(const Y26Stage15Model4BranchConfig* cfg) {
    if (cfg == nullptr || y26_stage14_next_c2f_output_count(&cfg->stage14) == 0 ||
        !conv_config_valid(cfg->branch0) || cfg->split1_output_scale <= 0.0f ||
        cfg->branch0_act_output_scale <= 0.0f || cfg->split1_output_zero_point_u8 < 0 ||
        cfg->split1_output_zero_point_u8 > 255 || cfg->branch0_act_output_zero_point_u8 < 0 ||
        cfg->branch0_act_output_zero_point_u8 > 255 || !activation_mode_valid(cfg->activation_mode)) {
        return false;
    }
    const Y26Stage7ConvNodeConfig& producer = cfg->stage14.model4_cv1;
    const int producer_h = output_h_for_kernel(producer.params, producer.kernel_h);
    const int producer_w = output_w_for_kernel(producer.params, producer.kernel_w);
    return producer.params.output_c == cfg->branch0.params.input_c * 2 &&
           cfg->branch0.params.input_h == producer_h && cfg->branch0.params.input_w == producer_w &&
           cfg->branch0.activation_zero_point_u8 == cfg->split1_output_zero_point_u8 &&
           cfg->branch0.input_storage_zero_point_s8 == cfg->split1_output_zero_point_u8 - 128 &&
           cfg->branch0.input_scale == cfg->split1_output_scale;
}

void* allocate_aligned(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    return ::operator new(bytes, std::align_val_t(kStage15Alignment));
}

void free_aligned(void* ptr) {
    ::operator delete(ptr, std::align_val_t(kStage15Alignment));
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

void timing_reset(Y26Stage15TimingUs* timing) {
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

int apply_activation_requant(int activation_mode,
                             const Y26ActivationRequantParams& params,
                             const Y26FixedRequantParams* fixed_requant,
                             const std::int8_t* lut_s8,
                             const std::int32_t* producer_i32,
                             std::int8_t* consumer_input_s8) {
    switch (activation_mode) {
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

int run_conv_threaded(const Y26Stage15Model4BranchWorkspace& ws,
                      const std::int8_t* input_s8,
                      std::int32_t* output_i32,
                      double* conv_us,
                      double* correction_us,
                      double* thread_overhead_us,
                      double* im2col_pack_us,
                      double* compute_us,
                      double* copy_us,
                      double* worker_other_us) {
    Y26ThreadedConvTimingUs threaded_timing {};
    const int status =
        y26_threaded_conv_run_ime_cluster0(ws.branch0_threaded_workspace, input_s8, output_i32, &threaded_timing);
    if (conv_us != nullptr) {
        *conv_us = threaded_timing.total_us;
    }
    if (correction_us != nullptr) {
        *correction_us = threaded_timing.correction_us;
    }
    if (thread_overhead_us != nullptr) {
        *thread_overhead_us = std::max(0.0, threaded_timing.total_us - threaded_timing.worker_max_us);
    }
    if (im2col_pack_us != nullptr) {
        *im2col_pack_us = threaded_timing.worker_im2col_pack_us;
    }
    if (compute_us != nullptr) {
        *compute_us = threaded_timing.worker_compute_us;
    }
    if (copy_us != nullptr) {
        *copy_us = threaded_timing.worker_copy_us;
    }
    if (worker_other_us != nullptr) {
        *worker_other_us = threaded_timing.worker_other_us;
    }
    return status;
}

int apply_activation_threaded(const Y26Stage15Model4BranchWorkspace& ws,
                              const Y26ActivationRequantParams& params,
                              const std::int8_t* lut_s8,
                              const std::int32_t* producer_i32,
                              std::int8_t* consumer_input_s8,
                              double* activation_us,
                              double* thread_overhead_us) {
    Y26ThreadedActivationTimingUs activation_timing {};
    const int status = y26_threaded_conv_run_activation_rvv_f32_rows(ws.branch0_threaded_workspace,
                                                                     &params,
                                                                     producer_i32,
                                                                     lut_s8,
                                                                     consumer_input_s8,
                                                                     &activation_timing);
    if (activation_us != nullptr) {
        *activation_us = activation_timing.total_us;
    }
    if (thread_overhead_us != nullptr) {
        *thread_overhead_us = std::max(0.0, activation_timing.total_us - activation_timing.worker_max_us);
    }
    return status;
}

void accumulate_stage14_timing(Y26Stage15TimingUs& dst, const Y26Stage14TimingUs& src) {
    dst.stage14_timing_us = src;
    dst.conv_us += src.conv_us;
    dst.activation_requant_us += src.activation_requant_us;
    dst.split_us += src.split_copy_us;
    dst.merge_us += src.merge_us;
    dst.post_qdq_us += src.post_qdq_us;
    dst.pack_layout_us += src.pack_layout_us;
    dst.correction_us += src.correction_us;
}

void finalize_timing(Y26Stage15TimingUs& timing) {
    if (timing.total_us <= 0.0) {
        return;
    }
    timing.conv_share_pct = 100.0 * timing.conv_us / timing.total_us;
    timing.activation_share_pct = 100.0 * timing.activation_requant_us / timing.total_us;
    timing.merge_share_pct = 100.0 * timing.merge_us / timing.total_us;
    timing.pack_layout_share_pct = 100.0 * timing.pack_layout_us / timing.total_us;
    timing.split_branch_share_pct = 100.0 * (timing.split_us + timing.add_us + timing.concat_us) / timing.total_us;
}

void split_second_half_nhwc(const std::int8_t* full_nhwc_s8,
                            int h,
                            int w,
                            int c,
                            std::int8_t* split1_s8) {
    const int split_c = c / 2;
    const int spatial = h * w;
    for (int m = 0; m < spatial; ++m) {
        std::memcpy(split1_s8 + m * split_c, full_nhwc_s8 + m * c + split_c, static_cast<std::size_t>(split_c));
    }
}

int run_after_stage14(const Y26Stage15Model4BranchConfig& cfg,
                      Y26Stage15Model4BranchWorkspace& ws,
                      std::int32_t* output_i32_nhwc,
                      Y26Stage15TimingUs* timing,
                      bool use_ime,
                      bool use_threaded_conv,
                      bool use_threaded_activation) {
    if ((use_threaded_conv || use_threaded_activation) && ws.branch0_threaded_workspace == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    const Y26Stage7ConvNodeConfig& producer = cfg.stage14.model4_cv1;
    Y26ActivationRequantParams split_params = activation_params(
        producer, ws.model4_cv1_output_count, cfg.split1_output_scale, cfg.split1_output_zero_point_u8);
    const auto act0_begin = Clock::now();
    double split_activation_thread_overhead_us = 0.0;
    int status = use_threaded_activation
                     ? apply_activation_threaded(ws,
                                                 split_params,
                                                 ws.model4_cv1_to_split1_lut_s8,
                                                 ws.model4_cv1_i32,
                                                 ws.model4_cv1_act_s8,
                                                 nullptr,
                                                 &split_activation_thread_overhead_us)
                     : apply_activation_requant(cfg.activation_mode,
                                                split_params,
                                                ws.model4_cv1_fixed_requant,
                                                ws.model4_cv1_to_split1_lut_s8,
                                                ws.model4_cv1_i32,
                                                ws.model4_cv1_act_s8);
    const auto act0_end = Clock::now();
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    const auto split_begin = Clock::now();
    split_second_half_nhwc(ws.model4_cv1_act_s8,
                           output_h_for_kernel(producer.params, producer.kernel_h),
                           output_w_for_kernel(producer.params, producer.kernel_w),
                           producer.params.output_c,
                           ws.split1_input_s8);
    const auto split_end = Clock::now();

    double branch0_conv_us = 0.0;
    double branch0_correction_us = 0.0;
    double branch0_im2col_pack_us = 0.0;
    double branch0_compute_us = 0.0;
    double branch0_copy_us = 0.0;
    double branch0_worker_other_us = 0.0;
    double branch0_thread_overhead_us = 0.0;
    if (use_threaded_conv) {
        status = run_conv_threaded(ws,
                                   ws.split1_input_s8,
                                   ws.branch0_i32,
                                   &branch0_conv_us,
                                   &branch0_correction_us,
                                   &branch0_thread_overhead_us,
                                   &branch0_im2col_pack_us,
                                   &branch0_compute_us,
                                   &branch0_copy_us,
                                   &branch0_worker_other_us);
    } else {
        status = use_ime ? run_conv_ime(cfg.branch0,
                                        ws.branch0_weights,
                                        ws.branch0_workspace,
                                        ws.split1_input_s8,
                                        ws.branch0_raw_i32,
                                        ws.branch0_i32,
                                        &branch0_conv_us,
                                        &branch0_correction_us)
                         : run_conv_scalar(cfg.branch0,
                                           ws.branch0_weights,
                                           ws.split1_input_s8,
                                           ws.branch0_raw_i32,
                                           ws.branch0_i32,
                                           &branch0_conv_us,
                                           &branch0_correction_us);
        branch0_compute_us = std::max(0.0, branch0_conv_us - branch0_correction_us);
        branch0_im2col_pack_us = use_ime ? y26_conv_mmt4d_last_im2col_pack_us() : 0.0;
    }
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }

    Y26ActivationRequantParams branch0_act_params = activation_params(
        cfg.branch0, ws.branch0_output_count, cfg.branch0_act_output_scale, cfg.branch0_act_output_zero_point_u8);
    const auto act1_begin = Clock::now();
    double branch_activation_thread_overhead_us = 0.0;
    status = use_threaded_activation
                 ? apply_activation_threaded(ws,
                                             branch0_act_params,
                                             ws.branch0_act_lut_s8,
                                             ws.branch0_i32,
                                             ws.branch0_act_s8,
                                             nullptr,
                                             &branch_activation_thread_overhead_us)
                 : apply_activation_requant(cfg.activation_mode,
                                            branch0_act_params,
                                            ws.branch0_fixed_requant,
                                            ws.branch0_act_lut_s8,
                                            ws.branch0_i32,
                                            ws.branch0_act_s8);
    const auto act1_end = Clock::now();
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    std::memcpy(output_i32_nhwc, ws.branch0_i32, ws.branch0_output_count * sizeof(std::int32_t));

    if (timing != nullptr) {
        const double split_activation_us = elapsed_us(act0_begin, act0_end);
        const double branch_activation_us = elapsed_us(act1_begin, act1_end);
        timing->activation_requant_us += split_activation_us + branch_activation_us;
        timing->split_us += elapsed_us(split_begin, split_end);
        timing->merge_us += elapsed_us(split_begin, split_end);
        timing->conv_us += branch0_conv_us;
        timing->branch0_conv_us += branch0_conv_us;
        timing->correction_us += branch0_correction_us;
        timing->branch0_correction_us += branch0_correction_us;
        timing->conv_im2col_pack_us += branch0_im2col_pack_us;
        timing->branch0_im2col_pack_us += branch0_im2col_pack_us;
        timing->conv_compute_us += branch0_compute_us;
        timing->conv_copy_us += branch0_copy_us;
        timing->conv_worker_other_us += branch0_worker_other_us;
        timing->branch0_compute_us += branch0_compute_us;
        timing->branch0_copy_us += branch0_copy_us;
        timing->branch0_worker_other_us += branch0_worker_other_us;
        timing->branch0_activation_us += branch_activation_us;
        timing->thread_overhead_us += split_activation_thread_overhead_us + branch0_thread_overhead_us +
                                      branch_activation_thread_overhead_us;
    }
    return Y26_CONV_STATUS_SUCCESS;
}

}  // namespace

extern "C" int y26_stage15_model4_branch_prepare(const Y26Stage15Model4BranchConfig* cfg,
                                                  Y26Stage15Model4BranchWorkspace* ws) {
    if (!config_valid(cfg) || ws == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    std::memset(ws, 0, sizeof(*ws));
    int status = y26_stage14_next_c2f_prepare(&cfg->stage14, &ws->stage14_ws);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    ws->branch0_weights = y26_prepacked_conv_weights_create_mmt4d_s8(cfg->branch0.weights_ohwi_s8,
                                                                     &cfg->branch0.params,
                                                                     cfg->branch0.kernel_h,
                                                                     cfg->branch0.kernel_w,
                                                                     cfg->branch0.node_name,
                                                                     cfg->branch0.weight_scales);
    ws->branch0_workspace =
        y26_conv_workspace_create(&cfg->branch0.params, cfg->branch0.kernel_h, cfg->branch0.kernel_w);
    ws->model4_cv1_output_count = y26_stage14_next_c2f_output_count(&cfg->stage14);
    const int producer_h = output_h_for_kernel(cfg->stage14.model4_cv1.params, cfg->stage14.model4_cv1.kernel_h);
    const int producer_w = output_w_for_kernel(cfg->stage14.model4_cv1.params, cfg->stage14.model4_cv1.kernel_w);
    ws->split1_count = static_cast<std::size_t>(producer_h) * static_cast<std::size_t>(producer_w) *
                       static_cast<std::size_t>(cfg->branch0.params.input_c);
    ws->branch0_output_count = y26_stage15_model4_branch_output_count(cfg);

    ws->model4_cv1_i32 = allocate_i32(ws->model4_cv1_output_count);
    ws->model4_cv1_act_s8 = allocate_i8(ws->model4_cv1_output_count);
    ws->split1_input_s8 = allocate_i8(ws->split1_count);
    ws->branch0_raw_i32 = allocate_i32(ws->branch0_output_count);
    ws->branch0_i32 = allocate_i32(ws->branch0_output_count);
    ws->branch0_act_s8 = allocate_i8(ws->branch0_output_count);
    ws->model4_cv1_fixed_requant = allocate_fixed_requant(cfg->stage14.model4_cv1.params.output_c);
    ws->branch0_fixed_requant = allocate_fixed_requant(cfg->branch0.params.output_c);

    if (ws->branch0_weights == nullptr || ws->branch0_workspace == nullptr || ws->model4_cv1_i32 == nullptr ||
        ws->model4_cv1_act_s8 == nullptr || ws->split1_input_s8 == nullptr || ws->branch0_raw_i32 == nullptr ||
        ws->branch0_i32 == nullptr || ws->branch0_act_s8 == nullptr || ws->model4_cv1_fixed_requant == nullptr ||
        ws->branch0_fixed_requant == nullptr) {
        y26_stage15_model4_branch_release(ws);
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }

    Y26ActivationRequantParams split_params = activation_params(cfg->stage14.model4_cv1,
                                                                ws->model4_cv1_output_count,
                                                                cfg->split1_output_scale,
                                                                cfg->split1_output_zero_point_u8);
    Y26ActivationRequantParams branch0_act_params = activation_params(cfg->branch0,
                                                                      ws->branch0_output_count,
                                                                      cfg->branch0_act_output_scale,
                                                                      cfg->branch0_act_output_zero_point_u8);
    if (y26_build_silu_u8_to_s8_lut(cfg->stage14.model4_cv1.output_scale,
                                    cfg->stage14.model4_cv1.output_zero_point_u8,
                                    cfg->split1_output_scale,
                                    cfg->split1_output_zero_point_u8,
                                    ws->model4_cv1_to_split1_lut_s8) != Y26_CONV_STATUS_SUCCESS ||
        y26_build_silu_u8_to_s8_lut(cfg->branch0.output_scale,
                                    cfg->branch0.output_zero_point_u8,
                                    cfg->branch0_act_output_scale,
                                    cfg->branch0_act_output_zero_point_u8,
                                    ws->branch0_act_lut_s8) != Y26_CONV_STATUS_SUCCESS ||
        y26_build_fixed_requant_params_per_channel(&split_params, ws->model4_cv1_fixed_requant) !=
            Y26_CONV_STATUS_SUCCESS ||
        y26_build_fixed_requant_params_per_channel(&branch0_act_params, ws->branch0_fixed_requant) !=
            Y26_CONV_STATUS_SUCCESS) {
        y26_stage15_model4_branch_release(ws);
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }

    ws->prepacked_bytes = y26_prepacked_conv_weights_total_bytes(ws->branch0_weights);
    ws->workspace_bytes = y26_conv_workspace_bytes(ws->branch0_workspace);
    ws->prepared = 1;
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" int y26_stage15_model4_branch_prepare_threaded_conv(const Y26Stage15Model4BranchConfig* cfg,
                                                                Y26Stage15Model4BranchWorkspace* ws,
                                                                int thread_count) {
    if (!config_valid(cfg) || ws == nullptr || ws->prepared != 1) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    y26_threaded_conv_destroy(ws->branch0_threaded_workspace);
    ws->branch0_threaded_workspace = y26_threaded_conv_create_spatial_rows(&cfg->branch0, thread_count);
    if (ws->branch0_threaded_workspace == nullptr) {
        ws->branch0_thread_count = 0;
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    ws->branch0_thread_count = thread_count;
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" void y26_stage15_model4_branch_release(Y26Stage15Model4BranchWorkspace* ws) {
    if (ws == nullptr) {
        return;
    }
    y26_stage14_next_c2f_release(&ws->stage14_ws);
    y26_threaded_conv_destroy(ws->branch0_threaded_workspace);
    y26_prepacked_conv_weights_destroy(ws->branch0_weights);
    y26_conv_workspace_destroy(ws->branch0_workspace);
    free_aligned(ws->model4_cv1_i32);
    free_aligned(ws->model4_cv1_act_s8);
    free_aligned(ws->split1_input_s8);
    free_aligned(ws->branch0_raw_i32);
    free_aligned(ws->branch0_i32);
    free_aligned(ws->branch0_act_s8);
    free_aligned(ws->model4_cv1_fixed_requant);
    free_aligned(ws->branch0_fixed_requant);
    std::memset(ws, 0, sizeof(*ws));
}

extern "C" std::size_t y26_stage15_model4_branch_output_count(const Y26Stage15Model4BranchConfig* cfg) {
    if (cfg == nullptr || !conv_params_valid(cfg->branch0.params) ||
        !kernel_supported(cfg->branch0.kernel_h, cfg->branch0.kernel_w)) {
        return 0;
    }
    return output_count_for_kernel(cfg->branch0.params, cfg->branch0.kernel_h, cfg->branch0.kernel_w);
}

extern "C" int y26_stage15_model4_branch_run_scalar(const Y26Stage15Model4BranchConfig* cfg,
                                                     Y26Stage15Model4BranchWorkspace* ws,
                                                     const std::int8_t* input_nhwc_s8,
                                                     std::int32_t* output_i32_nhwc,
                                                     Y26Stage15TimingUs* timing) {
    if (!config_valid(cfg) || ws == nullptr || ws->prepared != 1 || input_nhwc_s8 == nullptr ||
        output_i32_nhwc == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    timing_reset(timing);
    const auto begin = Clock::now();
    Y26Stage14TimingUs stage14_timing {};
    int status = y26_stage14_next_c2f_run_scalar(
        &cfg->stage14, &ws->stage14_ws, input_nhwc_s8, ws->model4_cv1_i32, &stage14_timing);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    if (timing != nullptr) {
        accumulate_stage14_timing(*timing, stage14_timing);
    }
    status = run_after_stage14(*cfg, *ws, output_i32_nhwc, timing, false, false, false);
    const auto end = Clock::now();
    if (timing != nullptr) {
        timing->total_us = elapsed_us(begin, end);
        finalize_timing(*timing);
    }
    return status;
}

extern "C" int y26_stage15_model4_branch_run_ime_cluster0_hotpath(const Y26Stage15Model4BranchConfig* cfg,
                                                                   Y26Stage15Model4BranchWorkspace* ws,
                                                                   const std::int8_t* input_nhwc_s8,
                                                                   std::int32_t* output_i32_nhwc,
                                                                   Y26Stage15TimingUs* timing) {
    if (!config_valid(cfg) || ws == nullptr || ws->prepared != 1 || input_nhwc_s8 == nullptr ||
        output_i32_nhwc == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    timing_reset(timing);
    const auto begin = Clock::now();
    Y26Stage14TimingUs stage14_timing {};
    int status = y26_stage14_next_c2f_run_ime_cluster0_hotpath(
        &cfg->stage14, &ws->stage14_ws, input_nhwc_s8, ws->model4_cv1_i32, &stage14_timing);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    if (timing != nullptr) {
        accumulate_stage14_timing(*timing, stage14_timing);
    }
    status = run_after_stage14(*cfg, *ws, output_i32_nhwc, timing, true, false, false);
    const auto end = Clock::now();
    if (timing != nullptr) {
        timing->total_us = elapsed_us(begin, end);
        finalize_timing(*timing);
    }
    return status;
}

extern "C" int y26_stage15_model4_branch_run_ime_threaded_conv_cluster0_hotpath(
    const Y26Stage15Model4BranchConfig* cfg,
    Y26Stage15Model4BranchWorkspace* ws,
    const std::int8_t* input_nhwc_s8,
    std::int32_t* output_i32_nhwc,
    int thread_activation,
    Y26Stage15TimingUs* timing) {
    if (!config_valid(cfg) || ws == nullptr || ws->prepared != 1 || ws->branch0_threaded_workspace == nullptr ||
        input_nhwc_s8 == nullptr || output_i32_nhwc == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    timing_reset(timing);
    const auto begin = Clock::now();
    Y26Stage14TimingUs stage14_timing {};
    int status = y26_stage14_next_c2f_run_ime_cluster0_hotpath(
        &cfg->stage14, &ws->stage14_ws, input_nhwc_s8, ws->model4_cv1_i32, &stage14_timing);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    if (timing != nullptr) {
        accumulate_stage14_timing(*timing, stage14_timing);
    }
    status = run_after_stage14(*cfg, *ws, output_i32_nhwc, timing, true, true, thread_activation != 0);
    const auto end = Clock::now();
    if (timing != nullptr) {
        timing->total_us = elapsed_us(begin, end);
        finalize_timing(*timing);
    }
    return status;
}

extern "C" int y26_stage15_model4_branch_threaded_worker_affinity_ok(
    const Y26Stage15Model4BranchWorkspace* ws) {
    return ws != nullptr ? y26_threaded_conv_worker_affinity_ok(ws->branch0_threaded_workspace) : 0;
}

extern "C" int y26_stage15_model4_branch_threaded_thread_count(const Y26Stage15Model4BranchWorkspace* ws) {
    return ws != nullptr ? y26_threaded_conv_thread_count(ws->branch0_threaded_workspace) : 0;
}

extern "C" const std::int8_t* y26_stage15_model4_branch_split1_input_s8(
    const Y26Stage15Model4BranchWorkspace* ws) {
    return ws != nullptr ? ws->split1_input_s8 : nullptr;
}

extern "C" const std::int8_t* y26_stage15_model4_branch_branch0_act_s8(
    const Y26Stage15Model4BranchWorkspace* ws) {
    return ws != nullptr ? ws->branch0_act_s8 : nullptr;
}

extern "C" const std::int32_t* y26_stage15_model4_branch_model4_cv1_i32(
    const Y26Stage15Model4BranchWorkspace* ws) {
    return ws != nullptr ? ws->model4_cv1_i32 : nullptr;
}
