#include "y26_k1x_activation.h"
#include "y26_k1x_conv_kernels.h"

#include <cstdint>
#include <iostream>

namespace {

int expect_u8(std::uint8_t actual, int expected, const char* label) {
    if (static_cast<int>(actual) != expected) {
        std::cerr << label << " actual=" << static_cast<int>(actual) << " expected=" << expected << "\n";
        return 1;
    }
    return 0;
}

int test_multiplier(double multiplier, int zero_point) {
    Y26FixedRequantParams params {};
    const int status = y26_fixed_requant_params_from_multiplier(multiplier, zero_point, &params);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "params failed multiplier=" << multiplier << " status=" << status << "\n";
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    int failures = 0;

    failures += test_multiplier(0.5, 128);
    Y26FixedRequantParams half {};
    (void)y26_fixed_requant_params_from_multiplier(0.5, 128, &half);
    failures += expect_u8(y26_requant_s32_to_u8_fixed_nearest_even(1, &half), 128, "half ties to even +0");
    failures += expect_u8(y26_requant_s32_to_u8_fixed_nearest_even(2, &half), 129, "half 2 -> 1");
    failures += expect_u8(y26_requant_s32_to_u8_fixed_nearest_even(3, &half), 130, "half 3 -> 2");
    failures += expect_u8(y26_requant_s32_to_u8_fixed_nearest_even(-1, &half), 128, "half -1 ties to even 0");
    failures += expect_u8(y26_requant_s32_to_u8_fixed_nearest_even(-2, &half), 127, "half -2 -> -1");
    failures += expect_u8(y26_requant_s32_to_u8_fixed_nearest_even(-3, &half), 126, "half -3 -> -2");

    Y26FixedRequantParams quarter {};
    (void)y26_fixed_requant_params_from_multiplier(0.25, 10, &quarter);
    failures += expect_u8(y26_requant_s32_to_u8_fixed_nearest_even(2, &quarter), 10, "quarter tie even");
    failures += expect_u8(y26_requant_s32_to_u8_fixed_nearest_even(6, &quarter), 12, "quarter 1.5 -> even 2");
    failures += expect_u8(y26_requant_s32_to_u8_fixed_nearest_even(1000000, &quarter), 255, "u8 high clamp");
    failures += expect_u8(y26_requant_s32_to_u8_fixed_nearest_even(-1000000, &quarter), 0, "u8 low clamp");

    failures += test_multiplier(0.0, 0);
    if (y26_fixed_requant_params_from_multiplier(-0.25, 0, &quarter) == Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "negative multiplier unexpectedly accepted\n";
        ++failures;
    }

    return failures == 0 ? 0 : 1;
}
