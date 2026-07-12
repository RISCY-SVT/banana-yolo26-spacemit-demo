#include "y26_k1x_int8_v1.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

int test_rounding() {
    struct Case {
        std::int64_t value;
        std::int64_t multiplier;
        int shift;
        std::int64_t expected;
    };
    constexpr std::array<Case, 12> cases{{
        {1, 1, 1, 0}, {3, 1, 1, 2}, {-1, 1, 1, 0}, {-3, 1, 1, -2},
        {2, 1, 2, 0}, {3, 1, 2, 1}, {4, 1, 2, 1}, {5, 1, 2, 1}, {6, 1, 2, 2},
        {7, 1, 2, 2}, {-6, 1, 2, -2}, {17, 5, 3, 11},
    }};
    for (const Case& test : cases) {
        std::int64_t actual = 0;
        if (!y26::int8_v1::round_product_right_even(test.value, test.multiplier, test.shift, &actual) ||
            actual != test.expected) {
            return 1;
        }
    }
    std::int64_t ignored = 0;
    if (y26::int8_v1::round_product_right_even(1, 1, -1, &ignored) ||
        y26::int8_v1::round_product_right_even(1, 1, 127, &ignored) ||
        y26::int8_v1::round_product_right_even(1, -1, 1, &ignored) ||
        y26::int8_v1::round_product_right_even(1, 1, 1, nullptr)) {
        return 2;
    }
    if (y26::int8_v1::round_product_right_even(
            std::numeric_limits<std::int64_t>::max(), std::numeric_limits<std::int64_t>::max(), 0, &ignored)) {
        return 3;
    }
    return 0;
}

int test_requant_saturation() {
    y26::int8_v1::RequantAsset asset{1, 0, 128, 0, 255};
    std::uint8_t value = 0;
    if (!y26::int8_v1::requantize_u8(-1000, asset, &value) || value != 0) return 1;
    if (!y26::int8_v1::requantize_u8(1000, asset, &value) || value != 255) return 2;
    asset.multiplier = 1;
    asset.right_shift = 1;
    asset.output_zero_point = 127;
    if (!y26::int8_v1::requantize_u8(-3, asset, &value) || value != 125) return 3;
    asset.right_shift = 127;
    if (y26::int8_v1::valid_requant_asset(asset) || y26::int8_v1::requantize_u8(0, asset, &value)) return 4;
    return 0;
}

int test_accumulator_bound() {
    const auto model5 = y26::int8_v1::accumulator_safety_bound(3U * 3U * 128U, 9, 128, 1000000);
    if (!model5.valid || !model5.int32_safe || model5.absolute_bound == 0) return 1;
    const auto unsafe = y26::int8_v1::accumulator_safety_bound(
        std::numeric_limits<std::size_t>::max(), 0, 255, std::numeric_limits<std::uint64_t>::max());
    if (unsafe.valid) return 2;
    return 0;
}

int test_layout_offset() {
    std::size_t offset = 0;
    if (!y26::int8_v1::nchwc8_offset(0, 0, 0, 0, 1, 16, 3, 5, &offset) || offset != 0) return 1;
    if (!y26::int8_v1::nchwc8_offset(0, 7, 0, 0, 1, 16, 3, 5, &offset) || offset != 7) return 2;
    if (!y26::int8_v1::nchwc8_offset(0, 8, 0, 0, 1, 16, 3, 5, &offset) || offset != 120) return 3;
    if (!y26::int8_v1::nchwc8_offset(0, 15, 2, 4, 1, 16, 3, 5, &offset) || offset != 239) return 4;
    if (y26::int8_v1::nchwc8_offset(0, 0, 0, 0, 1, 15, 3, 5, &offset) ||
        y26::int8_v1::nchwc8_offset(1, 0, 0, 0, 1, 16, 3, 5, &offset)) return 5;
    return 0;
}

}  // namespace

int main() {
    const int rounding = test_rounding();
    const int saturation = test_requant_saturation();
    const int accumulator = test_accumulator_bound();
    const int layout = test_layout_offset();
    std::cout << "rounding=" << rounding << '\n';
    std::cout << "saturation=" << saturation << '\n';
    std::cout << "accumulator=" << accumulator << '\n';
    std::cout << "layout=" << layout << '\n';
    return rounding == 0 && saturation == 0 && accumulator == 0 && layout == 0 ? 0 : 1;
}
