#include "y26_k1x_engine.h"

#include <cassert>

int main() {
    y26_k1x::RequantParams params{};
    params.effective_scale = 0.5F;
    params.output_zero_point = 0;
    assert(y26_k1x::requantize_s32_to_s8(3, params) == 2);
    assert(y26_k1x::requantize_s32_to_s8(-3, params) == -2);

    params.output_zero_point = 10;
    assert(y26_k1x::requantize_s32_to_s8(4, params) == 12);

    params.effective_scale = 2.0F;
    params.output_zero_point = 0;
    assert(y26_k1x::requantize_s32_to_s8(100, params) == 127);
    assert(y26_k1x::requantize_s32_to_s8(-100, params) == -128);
    return 0;
}
