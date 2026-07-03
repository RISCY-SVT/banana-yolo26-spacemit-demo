#include "y26_k1x_engine.h"

namespace {

int expect_equal(int got, int expected) {
    return got == expected ? 0 : 1;
}

}  // namespace

int main() {
    int failures = 0;
    y26_k1x::RequantParams params{};
    params.effective_scale = 0.5F;
    params.output_zero_point = 0;
    failures += expect_equal(y26_k1x::requantize_s32_to_s8(3, params), 2);
    failures += expect_equal(y26_k1x::requantize_s32_to_s8(-3, params), -2);

    params.output_zero_point = 10;
    failures += expect_equal(y26_k1x::requantize_s32_to_s8(4, params), 12);

    params.effective_scale = 2.0F;
    params.output_zero_point = 0;
    failures += expect_equal(y26_k1x::requantize_s32_to_s8(100, params), 127);
    failures += expect_equal(y26_k1x::requantize_s32_to_s8(-100, params), -128);
    return failures == 0 ? 0 : 1;
}
