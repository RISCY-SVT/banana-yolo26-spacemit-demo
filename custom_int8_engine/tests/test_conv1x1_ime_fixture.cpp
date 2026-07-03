#include "test_conv_fixtures.h"
#include "y26_k1x_vmadot.h"

#include <cstdio>

namespace {

int run_case(const Y26Conv2DParams& params, std::uint32_t seed) {
    const int output_h = y26_conv1x1_output_h(&params);
    const int output_w = y26_conv1x1_output_w(&params);
    auto input =
        y26_make_i8_vector(static_cast<std::size_t>(params.input_h * params.input_w * params.input_c), seed);
    auto weights = y26_make_i8_vector(static_cast<std::size_t>(params.output_c * params.input_c), seed + 1U);
    auto bias = y26_make_bias(params.output_c);
    std::vector<std::int32_t> scalar(static_cast<std::size_t>(output_h * output_w * params.output_c));
    std::vector<std::int32_t> ime(scalar.size());

    if (y26_conv1x1_i8s8s32_nhwc_scalar(
            input.data(), weights.data(), bias.data(), scalar.data(), &params) != Y26_CONV_STATUS_SUCCESS) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    const int status =
        y26_conv1x1_i8s8s32_nhwc_ime(input.data(), weights.data(), bias.data(), ime.data(), &params);
    const int mismatches = status == Y26_CONV_STATUS_SUCCESS ? y26_count_mismatches(scalar, ime) : -1;
    std::printf("conv1x1_ime shape=%dx%dx%d->%d output=%dx%d status=%d mismatches=%d checksum_scalar=%lld checksum_ime=%lld\n",
                params.input_h,
                params.input_w,
                params.input_c,
                params.output_c,
                output_h,
                output_w,
                status,
                mismatches,
                static_cast<long long>(y26_checksum(scalar)),
                static_cast<long long>(status == Y26_CONV_STATUS_SUCCESS ? y26_checksum(ime) : 0));
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    return mismatches;
}

}  // namespace

int main() {
    if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
        std::printf("conv1x1_ime skipped: not built with IME\n");
        return 0;
    }
    if (!y26_k1x_ime_hotpath_allowed_on_current_cpu()) {
        std::printf("conv1x1_ime skipped: IME unavailable or not on cluster0\n");
        return 1;
    }

    if (run_case(Y26Conv2DParams{3, 4, 5, 6, 1, 1, 0, 0}, 101U) != 0) {
        return 1;
    }
    if (run_case(Y26Conv2DParams{4, 4, 8, 8, 1, 1, 0, 0}, 102U) != 0) {
        return 1;
    }
    return run_case(Y26Conv2DParams{5, 5, 9, 7, 2, 2, 0, 0}, 103U) == 0 ? 0 : 1;
}
