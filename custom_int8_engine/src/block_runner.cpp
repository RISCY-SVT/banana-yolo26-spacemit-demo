#include "y26_k1x_block_runner.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

namespace {

constexpr std::size_t kStage5Alignment = 64;

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

std::size_t expected_weight_count(const Y26Stage5Block0Config& cfg) {
    return static_cast<std::size_t>(cfg.conv0_params.output_c) * static_cast<std::size_t>(cfg.kernel_h) *
           static_cast<std::size_t>(cfg.kernel_w) * static_cast<std::size_t>(cfg.conv0_params.input_c);
}

bool config_valid(const Y26Stage5Block0Config* cfg) {
    if (cfg == nullptr || !conv_params_valid(cfg->conv0_params) || !kernel_supported(cfg->kernel_h, cfg->kernel_w) ||
        cfg->activation_zero_point_u8 < 0 || cfg->activation_zero_point_u8 > 255 ||
        cfg->input_storage_zero_point_s8 < static_cast<int>(std::numeric_limits<std::int8_t>::min()) ||
        cfg->input_storage_zero_point_s8 > static_cast<int>(std::numeric_limits<std::int8_t>::max()) ||
        cfg->weights_ohwi_s8 == nullptr || cfg->bias_i32 == nullptr) {
        return false;
    }
    if (cfg->weight_count < expected_weight_count(*cfg) ||
        cfg->bias_count < static_cast<std::size_t>(cfg->conv0_params.output_c)) {
        return false;
    }
    const int oh = output_h_for_kernel(cfg->conv0_params, cfg->kernel_h);
    const int ow = output_w_for_kernel(cfg->conv0_params, cfg->kernel_w);
    return oh > 0 && ow > 0;
}

std::int8_t weight_at(const Y26Stage5Block0Config& cfg, int oc, int kh, int kw, int ic) {
    const int index = ((oc * cfg.kernel_h + kh) * cfg.kernel_w + kw) * cfg.conv0_params.input_c + ic;
    return cfg.weights_ohwi_s8[index];
}

void* allocate_aligned(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    return ::operator new(bytes, std::align_val_t(kStage5Alignment));
}

void free_aligned(void* ptr) {
    ::operator delete(ptr, std::align_val_t(kStage5Alignment));
}

int apply_stage5_correction(const Y26Stage5Block0Config& cfg,
                            Y26Stage5Block0Workspace& ws,
                            std::int32_t* output_i32_nhwc) {
    return y26_conv2d_apply_u8_as_s8_correction_nhwc(ws.raw_i32,
                                                     cfg.bias_i32,
                                                     y26_prepacked_conv_weights_sums(ws.conv0_weights),
                                                     output_i32_nhwc,
                                                     output_h_for_kernel(cfg.conv0_params, cfg.kernel_h) *
                                                         output_w_for_kernel(cfg.conv0_params, cfg.kernel_w),
                                                     cfg.conv0_params.output_c,
                                                     cfg.activation_zero_point_u8);
}

int scalar_raw_dot(const Y26Stage5Block0Config& cfg,
                   const std::int8_t* input_nhwc_s8,
                   std::int32_t* raw_i32_nhwc) {
    const int oh_count = output_h_for_kernel(cfg.conv0_params, cfg.kernel_h);
    const int ow_count = output_w_for_kernel(cfg.conv0_params, cfg.kernel_w);
    const int input_h = cfg.conv0_params.input_h;
    const int input_w = cfg.conv0_params.input_w;
    const int input_c = cfg.conv0_params.input_c;
    const std::int8_t pad = static_cast<std::int8_t>(cfg.input_storage_zero_point_s8);
    for (int oh = 0; oh < oh_count; ++oh) {
        for (int ow = 0; ow < ow_count; ++ow) {
            for (int oc = 0; oc < cfg.conv0_params.output_c; ++oc) {
                std::int32_t acc = 0;
                for (int kh = 0; kh < cfg.kernel_h; ++kh) {
                    const int ih = oh * cfg.conv0_params.stride_h + kh - cfg.conv0_params.pad_h;
                    const bool valid_h = ih >= 0 && ih < input_h;
                    for (int kw = 0; kw < cfg.kernel_w; ++kw) {
                        const int iw = ow * cfg.conv0_params.stride_w + kw - cfg.conv0_params.pad_w;
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
                raw_i32_nhwc[(oh * ow_count + ow) * cfg.conv0_params.output_c + oc] = acc;
            }
        }
    }
    return Y26_CONV_STATUS_SUCCESS;
}

}  // namespace

extern "C" void y26_stage5_block0_release(Y26Stage5Block0Workspace* ws) {
    if (ws == nullptr) {
        return;
    }
    y26_prepacked_conv_weights_destroy(ws->conv0_weights);
    y26_conv_workspace_destroy(ws->conv0_workspace);
    free_aligned(ws->raw_i32);
    std::memset(ws, 0, sizeof(*ws));
}

extern "C" int y26_stage5_block0_prepare(const Y26Stage5Block0Config* cfg,
                                          Y26Stage5Block0Workspace* ws) {
    if (!config_valid(cfg) || ws == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    y26_stage5_block0_release(ws);
    const int oh = output_h_for_kernel(cfg->conv0_params, cfg->kernel_h);
    const int ow = output_w_for_kernel(cfg->conv0_params, cfg->kernel_w);
    const std::size_t output_count =
        static_cast<std::size_t>(oh) * static_cast<std::size_t>(ow) *
        static_cast<std::size_t>(cfg->conv0_params.output_c);
    const std::size_t raw_bytes = output_count * sizeof(std::int32_t);
    try {
        ws->conv0_weights = y26_prepacked_conv_weights_create_mmt4d_s8(
            cfg->weights_ohwi_s8, &cfg->conv0_params, cfg->kernel_h, cfg->kernel_w, cfg->node_name, nullptr);
        ws->conv0_workspace = y26_conv_workspace_create(&cfg->conv0_params, cfg->kernel_h, cfg->kernel_w);
        ws->raw_i32 = static_cast<std::int32_t*>(allocate_aligned(raw_bytes));
        if (ws->conv0_weights == nullptr || ws->conv0_workspace == nullptr || ws->raw_i32 == nullptr) {
            y26_stage5_block0_release(ws);
            return Y26_CONV_STATUS_INVALID_ARGUMENT;
        }
        std::memset(ws->raw_i32, 0, raw_bytes);
        ws->raw_i32_count = output_count;
        ws->raw_i32_bytes = raw_bytes;
        ws->prepacked_bytes = y26_prepacked_conv_weights_total_bytes(ws->conv0_weights);
        ws->workspace_bytes = y26_conv_workspace_bytes(ws->conv0_workspace);
        ws->prepared = 1;
        return Y26_CONV_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
        y26_stage5_block0_release(ws);
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
}

extern "C" int y26_stage5_block0_output_h(const Y26Stage5Block0Config* cfg) {
    return config_valid(cfg) ? output_h_for_kernel(cfg->conv0_params, cfg->kernel_h) : 0;
}

extern "C" int y26_stage5_block0_output_w(const Y26Stage5Block0Config* cfg) {
    return config_valid(cfg) ? output_w_for_kernel(cfg->conv0_params, cfg->kernel_w) : 0;
}

extern "C" std::size_t y26_stage5_block0_output_count(const Y26Stage5Block0Config* cfg) {
    if (!config_valid(cfg)) {
        return 0;
    }
    return static_cast<std::size_t>(y26_stage5_block0_output_h(cfg)) *
           static_cast<std::size_t>(y26_stage5_block0_output_w(cfg)) *
           static_cast<std::size_t>(cfg->conv0_params.output_c);
}

extern "C" int y26_stage5_block0_run_scalar(const Y26Stage5Block0Config* cfg,
                                             Y26Stage5Block0Workspace* ws,
                                             const std::int8_t* input_nhwc_s8,
                                             std::int32_t* output_i32_nhwc) {
    if (!config_valid(cfg) || ws == nullptr || ws->prepared == 0 || input_nhwc_s8 == nullptr ||
        output_i32_nhwc == nullptr || ws->raw_i32 == nullptr || ws->conv0_weights == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    int status = scalar_raw_dot(*cfg, input_nhwc_s8, ws->raw_i32);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    return apply_stage5_correction(*cfg, *ws, output_i32_nhwc);
}

extern "C" int y26_stage5_block0_run_ime_cluster0_hotpath(const Y26Stage5Block0Config* cfg,
                                                           Y26Stage5Block0Workspace* ws,
                                                           const std::int8_t* input_nhwc_s8,
                                                           std::int32_t* output_i32_nhwc) {
    if (!config_valid(cfg) || ws == nullptr || ws->prepared == 0 || input_nhwc_s8 == nullptr ||
        output_i32_nhwc == nullptr || ws->raw_i32 == nullptr || ws->conv0_weights == nullptr ||
        ws->conv0_workspace == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    const int status = y26_conv2d_i8s8s32_nhwc_ime_prepacked_v1(input_nhwc_s8,
                                                                 ws->conv0_weights,
                                                                 ws->raw_i32,
                                                                 cfg->input_storage_zero_point_s8,
                                                                 ws->conv0_workspace,
                                                                 Y26_CONV_LOOP_ORDER_M_MAJOR);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    return apply_stage5_correction(*cfg, *ws, output_i32_nhwc);
}

extern "C" const std::int32_t* y26_stage5_block0_raw_scratch(const Y26Stage5Block0Workspace* ws) {
    return ws != nullptr ? ws->raw_i32 : nullptr;
}
