#include "y26_k1x_backbone_subset_runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kStage7Alignment = 64;

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
    const int oh = output_h_for_kernel(params, kernel_h);
    const int ow = output_w_for_kernel(params, kernel_w);
    if (oh <= 0 || ow <= 0) {
        return 0;
    }
    return static_cast<std::size_t>(oh) * static_cast<std::size_t>(ow) *
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

bool handoff_valid(const Y26Stage7ConvNodeConfig& producer,
                   const Y26Stage7ConvNodeConfig& consumer,
                   float act_scale,
                   int act_zero_point) {
    const int producer_oh = output_h_for_kernel(producer.params, producer.kernel_h);
    const int producer_ow = output_w_for_kernel(producer.params, producer.kernel_w);
    return act_scale > 0.0f && act_zero_point >= 0 && act_zero_point <= 255 &&
           consumer.params.input_h == producer_oh && consumer.params.input_w == producer_ow &&
           consumer.params.input_c == producer.params.output_c &&
           consumer.activation_zero_point_u8 == act_zero_point &&
           consumer.input_storage_zero_point_s8 == act_zero_point - 128;
}

bool activation_mode_valid(int mode) {
    return mode == Y26_ACTIVATION_MODE_SCALAR_FLOAT_REFERENCE ||
           mode == Y26_ACTIVATION_MODE_FIXED_REQUANT_ONLY ||
           mode == Y26_ACTIVATION_MODE_INT8_LUT ||
           mode == Y26_ACTIVATION_MODE_FUSED_LUT_PACK;
}

int normalized_activation_mode(const Y26Stage7BackboneSubsetConfig& cfg) {
    return activation_mode_valid(cfg.activation_mode) ? cfg.activation_mode
                                                      : Y26_ACTIVATION_MODE_SCALAR_FLOAT_REFERENCE;
}

bool config_valid(const Y26Stage7BackboneSubsetConfig* cfg) {
    return cfg != nullptr && conv_config_valid(cfg->conv0) && conv_config_valid(cfg->conv1) &&
           conv_config_valid(cfg->conv2) &&
           handoff_valid(cfg->conv0, cfg->conv1, cfg->act0_output_scale, cfg->act0_output_zero_point_u8) &&
           handoff_valid(cfg->conv1, cfg->conv2, cfg->act1_output_scale, cfg->act1_output_zero_point_u8) &&
           activation_mode_valid(normalized_activation_mode(*cfg));
}

std::int8_t weight_at(const Y26Stage7ConvNodeConfig& cfg, int oc, int kh, int kw, int ic) {
    const int index = ((oc * cfg.kernel_h + kh) * cfg.kernel_w + kw) * cfg.params.input_c + ic;
    return cfg.weights_ohwi_s8[index];
}

void* allocate_aligned(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    return ::operator new(bytes, std::align_val_t(kStage7Alignment));
}

void free_aligned(void* ptr) {
    ::operator delete(ptr, std::align_val_t(kStage7Alignment));
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

void timing_reset(Y26Stage7TimingUs* timing) {
    if (timing != nullptr) {
        std::memset(timing, 0, sizeof(*timing));
    }
}

int scalar_raw_dot(const Y26Stage7ConvNodeConfig& cfg,
                   const std::int8_t* input_nhwc_s8,
                   std::int32_t* raw_i32_nhwc) {
    const int oh_count = output_h_for_kernel(cfg.params, cfg.kernel_h);
    const int ow_count = output_w_for_kernel(cfg.params, cfg.kernel_w);
    const int input_h = cfg.params.input_h;
    const int input_w = cfg.params.input_w;
    const int input_c = cfg.params.input_c;
    const std::int8_t pad = static_cast<std::int8_t>(cfg.input_storage_zero_point_s8);
    for (int oh = 0; oh < oh_count; ++oh) {
        for (int ow = 0; ow < ow_count; ++ow) {
            for (int oc = 0; oc < cfg.params.output_c; ++oc) {
                std::int32_t acc = 0;
                for (int kh = 0; kh < cfg.kernel_h; ++kh) {
                    const int ih = oh * cfg.params.stride_h + kh - cfg.params.pad_h;
                    const bool valid_h = ih >= 0 && ih < input_h;
                    for (int kw = 0; kw < cfg.kernel_w; ++kw) {
                        const int iw = ow * cfg.params.stride_w + kw - cfg.params.pad_w;
                        const bool inside = valid_h && iw >= 0 && iw < input_w;
                        const std::int8_t* src =
                            inside ? input_nhwc_s8 + (ih * input_w + iw) * input_c : nullptr;
                        for (int ic = 0; ic < input_c; ++ic) {
                            const std::int8_t a = inside ? src[ic] : pad;
                            acc += static_cast<std::int32_t>(a) *
                                   static_cast<std::int32_t>(weight_at(cfg, oc, kh, kw, ic));
                        }
                    }
                }
                raw_i32_nhwc[(oh * ow_count + ow) * cfg.params.output_c + oc] = acc;
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

Y26ActivationRequantParams activation_params_for(const Y26Stage7ConvNodeConfig& producer,
                                                 float act_output_scale,
                                                 int act_output_zero_point_u8) {
    return Y26ActivationRequantParams{
        output_count_for_kernel(producer.params, producer.kernel_h, producer.kernel_w),
        producer.params.output_c,
        producer.input_scale,
        producer.weight_scales,
        producer.output_scale,
        producer.output_zero_point_u8,
        act_output_scale,
        act_output_zero_point_u8,
    };
}

int apply_activation_requant(const Y26Stage7ConvNodeConfig& producer,
                             float act_output_scale,
                             int act_output_zero_point_u8,
                             const std::int32_t* producer_i32,
                             std::int8_t* consumer_input_s8,
                             int activation_mode,
                             const std::int8_t* lut_256_s8,
                             const Y26FixedRequantParams* fixed_requant_params,
                             Y26ActivationSubbucketTimingUs* subbucket_timing) {
    if (subbucket_timing != nullptr) {
        *subbucket_timing = Y26ActivationSubbucketTimingUs {};
    }
    const Y26ActivationRequantParams params =
        activation_params_for(producer, act_output_scale, act_output_zero_point_u8);
    if (activation_mode == Y26_ACTIVATION_MODE_INT8_LUT ||
        activation_mode == Y26_ACTIVATION_MODE_FUSED_LUT_PACK) {
        return y26_activation_requant_silu_int8_lut(&params, producer_i32, lut_256_s8, consumer_input_s8);
    }
    if (activation_mode == Y26_ACTIVATION_MODE_FIXED_REQUANT_ONLY) {
        return y26_activation_requant_silu_fixed_requant_only(
            &params, fixed_requant_params, producer_i32, consumer_input_s8);
    }
    return y26_activation_requant_silu_scalar_float(&params, producer_i32, consumer_input_s8);
}

int validate_run_args(const Y26Stage7BackboneSubsetConfig* cfg,
                      const Y26Stage7BackboneSubsetWorkspace* ws,
                      const std::int8_t* input_nhwc_s8,
                      const std::int32_t* output_i32_nhwc) {
    if (!config_valid(cfg) || ws == nullptr || ws->prepared == 0 || input_nhwc_s8 == nullptr ||
        output_i32_nhwc == nullptr || ws->conv0_weights == nullptr || ws->conv1_weights == nullptr ||
        ws->conv2_weights == nullptr || ws->conv0_workspace == nullptr || ws->conv1_workspace == nullptr ||
        ws->conv2_workspace == nullptr || ws->conv0_raw_i32 == nullptr || ws->conv0_i32 == nullptr ||
        ws->conv1_input_s8 == nullptr || ws->conv1_raw_i32 == nullptr || ws->conv1_i32 == nullptr ||
        ws->conv2_input_s8 == nullptr || ws->conv2_raw_i32 == nullptr ||
        ws->conv0_fixed_requant == nullptr || ws->conv1_fixed_requant == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    return Y26_CONV_STATUS_SUCCESS;
}

Y26PrepackedConvWeights* create_weights(const Y26Stage7ConvNodeConfig& cfg) {
    return y26_prepacked_conv_weights_create_mmt4d_s8(
        cfg.weights_ohwi_s8, &cfg.params, cfg.kernel_h, cfg.kernel_w, cfg.node_name, cfg.weight_scales);
}

Y26ConvWorkspace* create_workspace(const Y26Stage7ConvNodeConfig& cfg) {
    return y26_conv_workspace_create(&cfg.params, cfg.kernel_h, cfg.kernel_w);
}

}  // namespace

extern "C" void y26_stage7_backbone_subset_release(Y26Stage7BackboneSubsetWorkspace* ws) {
    if (ws == nullptr) {
        return;
    }
    y26_prepacked_conv_weights_destroy(ws->conv0_weights);
    y26_prepacked_conv_weights_destroy(ws->conv1_weights);
    y26_prepacked_conv_weights_destroy(ws->conv2_weights);
    y26_conv_workspace_destroy(ws->conv0_workspace);
    y26_conv_workspace_destroy(ws->conv1_workspace);
    y26_conv_workspace_destroy(ws->conv2_workspace);
    free_aligned(ws->conv0_raw_i32);
    free_aligned(ws->conv0_i32);
    free_aligned(ws->conv1_input_s8);
    free_aligned(ws->conv1_raw_i32);
    free_aligned(ws->conv1_i32);
    free_aligned(ws->conv2_input_s8);
    free_aligned(ws->conv2_raw_i32);
    free_aligned(ws->conv0_fixed_requant);
    free_aligned(ws->conv1_fixed_requant);
    std::memset(ws, 0, sizeof(*ws));
}

extern "C" int y26_stage7_backbone_subset_prepare(const Y26Stage7BackboneSubsetConfig* cfg,
                                                   Y26Stage7BackboneSubsetWorkspace* ws) {
    if (!config_valid(cfg) || ws == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    y26_stage7_backbone_subset_release(ws);
    const std::size_t conv0_count = output_count_for_kernel(cfg->conv0.params, cfg->conv0.kernel_h, cfg->conv0.kernel_w);
    const std::size_t conv1_count = output_count_for_kernel(cfg->conv1.params, cfg->conv1.kernel_h, cfg->conv1.kernel_w);
    const std::size_t conv2_count = output_count_for_kernel(cfg->conv2.params, cfg->conv2.kernel_h, cfg->conv2.kernel_w);
    try {
        ws->conv0_weights = create_weights(cfg->conv0);
        ws->conv1_weights = create_weights(cfg->conv1);
        ws->conv2_weights = create_weights(cfg->conv2);
        ws->conv0_workspace = create_workspace(cfg->conv0);
        ws->conv1_workspace = create_workspace(cfg->conv1);
        ws->conv2_workspace = create_workspace(cfg->conv2);
        ws->conv0_raw_i32 = allocate_i32(conv0_count);
        ws->conv0_i32 = allocate_i32(conv0_count);
        ws->conv1_input_s8 = allocate_i8(conv0_count);
        ws->conv1_raw_i32 = allocate_i32(conv1_count);
        ws->conv1_i32 = allocate_i32(conv1_count);
        ws->conv2_input_s8 = allocate_i8(conv1_count);
        ws->conv2_raw_i32 = allocate_i32(conv2_count);
        ws->conv0_fixed_requant = allocate_fixed_requant(static_cast<std::size_t>(cfg->conv0.params.output_c));
        ws->conv1_fixed_requant = allocate_fixed_requant(static_cast<std::size_t>(cfg->conv1.params.output_c));
        if (ws->conv0_weights == nullptr || ws->conv1_weights == nullptr || ws->conv2_weights == nullptr ||
            ws->conv0_workspace == nullptr || ws->conv1_workspace == nullptr || ws->conv2_workspace == nullptr ||
            ws->conv0_raw_i32 == nullptr || ws->conv0_i32 == nullptr || ws->conv1_input_s8 == nullptr ||
            ws->conv1_raw_i32 == nullptr || ws->conv1_i32 == nullptr || ws->conv2_input_s8 == nullptr ||
            ws->conv2_raw_i32 == nullptr || ws->conv0_fixed_requant == nullptr ||
            ws->conv1_fixed_requant == nullptr) {
            y26_stage7_backbone_subset_release(ws);
            return Y26_CONV_STATUS_INVALID_ARGUMENT;
        }
        const Y26ActivationRequantParams act0_params =
            activation_params_for(cfg->conv0, cfg->act0_output_scale, cfg->act0_output_zero_point_u8);
        const Y26ActivationRequantParams act1_params =
            activation_params_for(cfg->conv1, cfg->act1_output_scale, cfg->act1_output_zero_point_u8);
        if (y26_build_silu_u8_to_s8_lut(cfg->conv0.output_scale,
                                        cfg->conv0.output_zero_point_u8,
                                        cfg->act0_output_scale,
                                        cfg->act0_output_zero_point_u8,
                                        ws->act0_lut_s8) != Y26_CONV_STATUS_SUCCESS ||
            y26_build_silu_u8_to_s8_lut(cfg->conv1.output_scale,
                                        cfg->conv1.output_zero_point_u8,
                                        cfg->act1_output_scale,
                                        cfg->act1_output_zero_point_u8,
                                        ws->act1_lut_s8) != Y26_CONV_STATUS_SUCCESS ||
            y26_build_fixed_requant_params_per_channel(&act0_params, ws->conv0_fixed_requant) !=
                Y26_CONV_STATUS_SUCCESS ||
            y26_build_fixed_requant_params_per_channel(&act1_params, ws->conv1_fixed_requant) !=
                Y26_CONV_STATUS_SUCCESS) {
            y26_stage7_backbone_subset_release(ws);
            return Y26_CONV_STATUS_INVALID_ARGUMENT;
        }
        ws->conv0_output_count = conv0_count;
        ws->conv1_input_count = conv0_count;
        ws->conv1_output_count = conv1_count;
        ws->conv2_input_count = conv1_count;
        ws->conv2_output_count = conv2_count;
        ws->prepacked_bytes = y26_prepacked_conv_weights_total_bytes(ws->conv0_weights) +
                              y26_prepacked_conv_weights_total_bytes(ws->conv1_weights) +
                              y26_prepacked_conv_weights_total_bytes(ws->conv2_weights);
        ws->workspace_bytes = y26_conv_workspace_bytes(ws->conv0_workspace) +
                              y26_conv_workspace_bytes(ws->conv1_workspace) +
                              y26_conv_workspace_bytes(ws->conv2_workspace) +
                              conv0_count * sizeof(std::int32_t) * 2 + conv0_count +
                              conv1_count * sizeof(std::int32_t) * 2 + conv1_count +
                              conv2_count * sizeof(std::int32_t) +
                              static_cast<std::size_t>(cfg->conv0.params.output_c) * sizeof(Y26FixedRequantParams) +
                              static_cast<std::size_t>(cfg->conv1.params.output_c) * sizeof(Y26FixedRequantParams) +
                              sizeof(ws->act0_lut_s8) + sizeof(ws->act1_lut_s8);
        std::memset(ws->conv0_raw_i32, 0, conv0_count * sizeof(std::int32_t));
        std::memset(ws->conv0_i32, 0, conv0_count * sizeof(std::int32_t));
        std::memset(ws->conv1_input_s8, 0, conv0_count);
        std::memset(ws->conv1_raw_i32, 0, conv1_count * sizeof(std::int32_t));
        std::memset(ws->conv1_i32, 0, conv1_count * sizeof(std::int32_t));
        std::memset(ws->conv2_input_s8, 0, conv1_count);
        std::memset(ws->conv2_raw_i32, 0, conv2_count * sizeof(std::int32_t));
        ws->prepared = 1;
        return Y26_CONV_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
        y26_stage7_backbone_subset_release(ws);
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
}

extern "C" std::size_t y26_stage7_backbone_subset_conv0_output_count(const Y26Stage7BackboneSubsetConfig* cfg) {
    return config_valid(cfg) ? output_count_for_kernel(cfg->conv0.params, cfg->conv0.kernel_h, cfg->conv0.kernel_w) : 0;
}

extern "C" std::size_t y26_stage7_backbone_subset_conv1_output_count(const Y26Stage7BackboneSubsetConfig* cfg) {
    return config_valid(cfg) ? output_count_for_kernel(cfg->conv1.params, cfg->conv1.kernel_h, cfg->conv1.kernel_w) : 0;
}

extern "C" std::size_t y26_stage7_backbone_subset_conv2_output_count(const Y26Stage7BackboneSubsetConfig* cfg) {
    return config_valid(cfg) ? output_count_for_kernel(cfg->conv2.params, cfg->conv2.kernel_h, cfg->conv2.kernel_w) : 0;
}

extern "C" int y26_stage7_backbone_subset_run_scalar(const Y26Stage7BackboneSubsetConfig* cfg,
                                                      Y26Stage7BackboneSubsetWorkspace* ws,
                                                      const std::int8_t* input_nhwc_s8,
                                                      std::int32_t* output_i32_nhwc,
                                                      Y26Stage7TimingUs* timing) {
    timing_reset(timing);
    const auto total_begin = Clock::now();
    int status = validate_run_args(cfg, ws, input_nhwc_s8, output_i32_nhwc);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    const int activation_mode = normalized_activation_mode(*cfg);

    auto begin = Clock::now();
    status = scalar_raw_dot(cfg->conv0, input_nhwc_s8, ws->conv0_raw_i32);
    if (status == Y26_CONV_STATUS_SUCCESS) {
        status = apply_correction(cfg->conv0, ws->conv0_weights, ws->conv0_raw_i32, ws->conv0_i32);
    }
    auto end = Clock::now();
    if (timing != nullptr) {
        timing->conv0_us = elapsed_us(begin, end);
    }
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }

    begin = Clock::now();
    status = apply_activation_requant(cfg->conv0,
                                      cfg->act0_output_scale,
                                      cfg->act0_output_zero_point_u8,
                                      ws->conv0_i32,
                                      ws->conv1_input_s8,
                                      activation_mode,
                                      ws->act0_lut_s8,
                                      ws->conv0_fixed_requant,
                                      timing != nullptr ? &timing->act0_subbucket_us : nullptr);
    end = Clock::now();
    if (timing != nullptr) {
        timing->act0_requant_us = elapsed_us(begin, end);
    }
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }

    begin = Clock::now();
    status = scalar_raw_dot(cfg->conv1, ws->conv1_input_s8, ws->conv1_raw_i32);
    if (status == Y26_CONV_STATUS_SUCCESS) {
        status = apply_correction(cfg->conv1, ws->conv1_weights, ws->conv1_raw_i32, ws->conv1_i32);
    }
    end = Clock::now();
    if (timing != nullptr) {
        timing->conv1_us = elapsed_us(begin, end);
    }
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }

    begin = Clock::now();
    status = apply_activation_requant(cfg->conv1,
                                      cfg->act1_output_scale,
                                      cfg->act1_output_zero_point_u8,
                                      ws->conv1_i32,
                                      ws->conv2_input_s8,
                                      activation_mode,
                                      ws->act1_lut_s8,
                                      ws->conv1_fixed_requant,
                                      timing != nullptr ? &timing->act1_subbucket_us : nullptr);
    end = Clock::now();
    if (timing != nullptr) {
        timing->act1_requant_us = elapsed_us(begin, end);
    }
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }

    begin = Clock::now();
    status = scalar_raw_dot(cfg->conv2, ws->conv2_input_s8, ws->conv2_raw_i32);
    if (status == Y26_CONV_STATUS_SUCCESS) {
        status = apply_correction(cfg->conv2, ws->conv2_weights, ws->conv2_raw_i32, output_i32_nhwc);
    }
    end = Clock::now();
    if (timing != nullptr) {
        timing->conv2_us = elapsed_us(begin, end);
        timing->total_us = elapsed_us(total_begin, Clock::now());
    }
    return status;
}

extern "C" int y26_stage7_backbone_subset_run_ime_cluster0_hotpath(const Y26Stage7BackboneSubsetConfig* cfg,
                                                                    Y26Stage7BackboneSubsetWorkspace* ws,
                                                                    const std::int8_t* input_nhwc_s8,
                                                                    std::int32_t* output_i32_nhwc,
                                                                    Y26Stage7TimingUs* timing) {
    timing_reset(timing);
    const auto total_begin = Clock::now();
    int status = validate_run_args(cfg, ws, input_nhwc_s8, output_i32_nhwc);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    const int activation_mode = normalized_activation_mode(*cfg);

    auto begin = Clock::now();
    status = y26_conv2d_i8s8s32_nhwc_ime_prepacked_v1(input_nhwc_s8,
                                                      ws->conv0_weights,
                                                      ws->conv0_raw_i32,
                                                      cfg->conv0.input_storage_zero_point_s8,
                                                      ws->conv0_workspace,
                                                      Y26_CONV_LOOP_ORDER_M_MAJOR);
    if (status == Y26_CONV_STATUS_SUCCESS) {
        status = apply_correction(cfg->conv0, ws->conv0_weights, ws->conv0_raw_i32, ws->conv0_i32);
    }
    auto end = Clock::now();
    if (timing != nullptr) {
        timing->conv0_us = elapsed_us(begin, end);
    }
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }

    begin = Clock::now();
    status = apply_activation_requant(cfg->conv0,
                                      cfg->act0_output_scale,
                                      cfg->act0_output_zero_point_u8,
                                      ws->conv0_i32,
                                      ws->conv1_input_s8,
                                      activation_mode,
                                      ws->act0_lut_s8,
                                      ws->conv0_fixed_requant,
                                      timing != nullptr ? &timing->act0_subbucket_us : nullptr);
    end = Clock::now();
    if (timing != nullptr) {
        timing->act0_requant_us = elapsed_us(begin, end);
    }
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }

    begin = Clock::now();
    status = y26_conv2d_i8s8s32_nhwc_ime_prepacked_v1(ws->conv1_input_s8,
                                                      ws->conv1_weights,
                                                      ws->conv1_raw_i32,
                                                      cfg->conv1.input_storage_zero_point_s8,
                                                      ws->conv1_workspace,
                                                      Y26_CONV_LOOP_ORDER_M_MAJOR);
    if (status == Y26_CONV_STATUS_SUCCESS) {
        status = apply_correction(cfg->conv1, ws->conv1_weights, ws->conv1_raw_i32, ws->conv1_i32);
    }
    end = Clock::now();
    if (timing != nullptr) {
        timing->conv1_us = elapsed_us(begin, end);
    }
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }

    begin = Clock::now();
    status = apply_activation_requant(cfg->conv1,
                                      cfg->act1_output_scale,
                                      cfg->act1_output_zero_point_u8,
                                      ws->conv1_i32,
                                      ws->conv2_input_s8,
                                      activation_mode,
                                      ws->act1_lut_s8,
                                      ws->conv1_fixed_requant,
                                      timing != nullptr ? &timing->act1_subbucket_us : nullptr);
    end = Clock::now();
    if (timing != nullptr) {
        timing->act1_requant_us = elapsed_us(begin, end);
    }
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }

    begin = Clock::now();
    status = y26_conv2d_i8s8s32_nhwc_ime_prepacked_v1(ws->conv2_input_s8,
                                                      ws->conv2_weights,
                                                      ws->conv2_raw_i32,
                                                      cfg->conv2.input_storage_zero_point_s8,
                                                      ws->conv2_workspace,
                                                      Y26_CONV_LOOP_ORDER_M_MAJOR);
    if (status == Y26_CONV_STATUS_SUCCESS) {
        status = apply_correction(cfg->conv2, ws->conv2_weights, ws->conv2_raw_i32, output_i32_nhwc);
    }
    end = Clock::now();
    if (timing != nullptr) {
        timing->conv2_us = elapsed_us(begin, end);
        timing->total_us = elapsed_us(total_begin, Clock::now());
    }
    return status;
}

extern "C" const std::int32_t* y26_stage7_backbone_subset_conv0_i32(const Y26Stage7BackboneSubsetWorkspace* ws) {
    return ws != nullptr ? ws->conv0_i32 : nullptr;
}

extern "C" const std::int8_t* y26_stage7_backbone_subset_conv1_input_s8(const Y26Stage7BackboneSubsetWorkspace* ws) {
    return ws != nullptr ? ws->conv1_input_s8 : nullptr;
}

extern "C" const std::int32_t* y26_stage7_backbone_subset_conv1_i32(const Y26Stage7BackboneSubsetWorkspace* ws) {
    return ws != nullptr ? ws->conv1_i32 : nullptr;
}

extern "C" const std::int8_t* y26_stage7_backbone_subset_conv2_input_s8(const Y26Stage7BackboneSubsetWorkspace* ws) {
    return ws != nullptr ? ws->conv2_input_s8 : nullptr;
}
