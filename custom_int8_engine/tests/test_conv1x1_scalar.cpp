#include "test_conv_fixtures.h"

#include <cstdio>

namespace {

bool run_case(const Y26Conv2DParams& params) {
    const int output_h = y26_conv1x1_output_h(&params);
    const int output_w = y26_conv1x1_output_w(&params);
    if (output_h <= 0 || output_w <= 0) {
        return false;
    }

    auto input = y26_make_i8_vector(
        static_cast<std::size_t>(params.input_h * params.input_w * params.input_c), 11U);
    auto weights = y26_make_i8_vector(static_cast<std::size_t>(params.output_c * params.input_c), 17U);
    auto bias = y26_make_bias(params.output_c);
    std::vector<std::int32_t> output(static_cast<std::size_t>(output_h * output_w * params.output_c));
    std::vector<std::int32_t> expected(output.size());

    const int status = y26_conv1x1_i8s8s32_nhwc_scalar(
        input.data(), weights.data(), bias.data(), output.data(), &params);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return false;
    }

    for (int oh = 0; oh < output_h; ++oh) {
        for (int ow = 0; ow < output_w; ++ow) {
            const int ih = oh * params.stride_h - params.pad_h;
            const int iw = ow * params.stride_w - params.pad_w;
            for (int oc = 0; oc < params.output_c; ++oc) {
                std::int32_t acc = bias[static_cast<std::size_t>(oc)];
                for (int ic = 0; ic < params.input_c; ++ic) {
                    if (ih >= 0 && ih < params.input_h && iw >= 0 && iw < params.input_w) {
                        const auto a = static_cast<std::int32_t>(
                            input[static_cast<std::size_t>((ih * params.input_w + iw) * params.input_c + ic)]);
                        const auto w = static_cast<std::int32_t>(
                            weights[static_cast<std::size_t>(oc * params.input_c + ic)]);
                        acc += a * w;
                    }
                }
                expected[static_cast<std::size_t>((oh * output_w + ow) * params.output_c + oc)] = acc;
            }
        }
    }

    if (y26_count_mismatches(output, expected) != 0) {
        return false;
    }
    std::printf("conv1x1_scalar shape=%dx%dx%d->%d output=%dx%d checksum=%lld\n",
                params.input_h,
                params.input_w,
                params.input_c,
                params.output_c,
                output_h,
                output_w,
                static_cast<long long>(y26_checksum(output)));
    return true;
}

}  // namespace

int main() {
    return run_case(Y26Conv2DParams{3, 4, 5, 6, 1, 1, 0, 0}) &&
                   run_case(Y26Conv2DParams{5, 5, 9, 7, 2, 2, 0, 0})
               ? 0
               : 1;
}
