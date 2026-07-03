#pragma once

#include "y26_k1x_conv_kernels.h"

#include <cstdint>

namespace y26_k1x::kernels {

inline bool conv_params_valid(const Y26Conv2DParams* params) {
    return params != nullptr && params->input_h > 0 && params->input_w > 0 && params->input_c > 0 &&
           params->output_c > 0 && params->stride_h > 0 && params->stride_w > 0 && params->pad_h >= 0 &&
           params->pad_w >= 0;
}

inline int conv_output_dim(int input, int kernel, int stride, int pad) {
    return (input + 2 * pad - kernel) / stride + 1;
}

inline int flat_output_m_to_h(int m, int output_w) {
    return m / output_w;
}

inline int flat_output_m_to_w(int m, int output_w) {
    return m % output_w;
}

inline std::int8_t input_nhwc_or_zero(const std::int8_t* input_nhwc,
                                      const Y26Conv2DParams& params,
                                      int input_h,
                                      int input_w,
                                      int input_c) {
    if (input_h < 0 || input_h >= params.input_h || input_w < 0 || input_w >= params.input_w ||
        input_c < 0 || input_c >= params.input_c) {
        return 0;
    }
    const auto index = ((input_h * params.input_w + input_w) * params.input_c) + input_c;
    return input_nhwc[index];
}

inline int conv_status_from_vmadot_status(int status) {
    switch (status) {
    case 0:
        return Y26_CONV_STATUS_SUCCESS;
    case 1:
        return Y26_CONV_STATUS_NOT_BUILT_WITH_IME;
    case 2:
        return Y26_CONV_STATUS_RUNTIME_SAFETY_FAILED;
    case 3:
        return Y26_CONV_STATUS_SIGILL_CAUGHT;
    default:
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
}

}  // namespace y26_k1x::kernels
