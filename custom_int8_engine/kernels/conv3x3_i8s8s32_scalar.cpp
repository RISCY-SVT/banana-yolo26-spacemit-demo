#include "conv_kernel_common.h"

extern "C" int y26_conv3x3_output_h(const Y26Conv2DParams* params) {
    if (!y26_k1x::kernels::conv_params_valid(params)) {
        return 0;
    }
    return y26_k1x::kernels::conv_output_dim(params->input_h, 3, params->stride_h, params->pad_h);
}

extern "C" int y26_conv3x3_output_w(const Y26Conv2DParams* params) {
    if (!y26_k1x::kernels::conv_params_valid(params)) {
        return 0;
    }
    return y26_k1x::kernels::conv_output_dim(params->input_w, 3, params->stride_w, params->pad_w);
}

extern "C" int y26_conv3x3_i8s8s32_nhwc_scalar(const std::int8_t* input_nhwc,
                                                const std::int8_t* weights_oc_kh_kw_ic,
                                                const std::int32_t* bias_oc,
                                                std::int32_t* output_nhwc,
                                                const Y26Conv2DParams* params) {
    if (input_nhwc == nullptr || weights_oc_kh_kw_ic == nullptr || output_nhwc == nullptr ||
        !y26_k1x::kernels::conv_params_valid(params)) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }

    const int output_h = y26_conv3x3_output_h(params);
    const int output_w = y26_conv3x3_output_w(params);
    if (output_h <= 0 || output_w <= 0) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }

    for (int oh = 0; oh < output_h; ++oh) {
        for (int ow = 0; ow < output_w; ++ow) {
            for (int oc = 0; oc < params->output_c; ++oc) {
                std::int32_t acc = bias_oc != nullptr ? bias_oc[oc] : 0;
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        const int ih = oh * params->stride_h + kh - params->pad_h;
                        const int iw = ow * params->stride_w + kw - params->pad_w;
                        for (int ic = 0; ic < params->input_c; ++ic) {
                            const auto a = static_cast<std::int32_t>(
                                y26_k1x::kernels::input_nhwc_or_zero(input_nhwc, *params, ih, iw, ic));
                            const int w_index = ((oc * 3 + kh) * 3 + kw) * params->input_c + ic;
                            const auto w = static_cast<std::int32_t>(weights_oc_kh_kw_ic[w_index]);
                            acc += a * w;
                        }
                    }
                }
                output_nhwc[(oh * output_w + ow) * params->output_c + oc] = acc;
            }
        }
    }
    return Y26_CONV_STATUS_SUCCESS;
}
