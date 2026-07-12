#include "y26_k1x_int8_v1.h"

#include <algorithm>
#include <limits>

namespace y26::int8_v1 {
namespace {

__extension__ using SignedInt128 = __int128;
__extension__ using UnsignedInt128 = unsigned __int128;

bool checked_mul(std::size_t lhs, std::size_t rhs, std::size_t* result) noexcept {
    if (result == nullptr || (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

bool checked_add(std::size_t lhs, std::size_t rhs, std::size_t* result) noexcept {
    if (result == nullptr || rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

UnsignedInt128 magnitude(SignedInt128 value) noexcept {
    const UnsignedInt128 bits = static_cast<UnsignedInt128>(value);
    return value < 0 ? (~bits) + 1U : bits;
}

bool restore_sign(UnsignedInt128 value, bool negative, std::int64_t* result) noexcept {
    if (result == nullptr) {
        return false;
    }
    const UnsignedInt128 positive_limit = static_cast<UnsignedInt128>(std::numeric_limits<std::int64_t>::max());
    const UnsignedInt128 negative_limit = positive_limit + 1U;
    if ((!negative && value > positive_limit) || (negative && value > negative_limit)) {
        return false;
    }
    if (!negative) {
        *result = static_cast<std::int64_t>(value);
    } else if (value == negative_limit) {
        *result = std::numeric_limits<std::int64_t>::min();
    } else {
        *result = -static_cast<std::int64_t>(value);
    }
    return true;
}

}  // namespace

bool valid_requant_asset(const RequantAsset& asset) noexcept {
    return asset.multiplier >= 0 && asset.right_shift >= 0 && asset.right_shift <= 126 &&
           asset.output_zero_point >= 0 && asset.output_zero_point <= 255 &&
           asset.clamp_min >= 0 && asset.clamp_min <= asset.clamp_max && asset.clamp_max <= 255;
}

bool round_product_right_even(std::int64_t value,
                              std::int64_t multiplier,
                              std::int32_t right_shift,
                              std::int64_t* rounded) noexcept {
    if (rounded == nullptr || multiplier < 0 || right_shift < 0 || right_shift > 126) {
        return false;
    }
    const SignedInt128 product = static_cast<SignedInt128>(value) * static_cast<SignedInt128>(multiplier);
    const bool negative = product < 0;
    const UnsignedInt128 absolute = magnitude(product);
    UnsignedInt128 quotient = absolute;
    if (right_shift != 0) {
        quotient = absolute >> right_shift;
        const UnsignedInt128 mask = (static_cast<UnsignedInt128>(1) << right_shift) - 1U;
        const UnsignedInt128 remainder = absolute & mask;
        const UnsignedInt128 half = static_cast<UnsignedInt128>(1) << (right_shift - 1);
        if (remainder > half || (remainder == half && (quotient & 1U) != 0)) {
            ++quotient;
        }
    }
    return restore_sign(quotient, negative, rounded);
}

bool requantize_u8(std::int64_t accumulator,
                   const RequantAsset& asset,
                   std::uint8_t* output_code) noexcept {
    if (output_code == nullptr || !valid_requant_asset(asset)) {
        return false;
    }
    std::int64_t rounded = 0;
    if (!round_product_right_even(accumulator, asset.multiplier, asset.right_shift, &rounded)) {
        *output_code = accumulator < 0 ? static_cast<std::uint8_t>(asset.clamp_min)
                                       : static_cast<std::uint8_t>(asset.clamp_max);
        return true;
    }
    const SignedInt128 shifted = static_cast<SignedInt128>(rounded) + asset.output_zero_point;
    const SignedInt128 clamped = std::clamp<SignedInt128>(shifted, asset.clamp_min, asset.clamp_max);
    *output_code = static_cast<std::uint8_t>(clamped);
    return true;
}

AccumulatorSafety accumulator_safety_bound(std::size_t k,
                                           std::uint8_t activation_zero_point,
                                           std::uint8_t maximum_weight_magnitude,
                                           std::uint64_t maximum_bias_magnitude) noexcept {
    AccumulatorSafety result;
    const std::uint64_t maximum_activation_magnitude = std::max<std::uint64_t>(
        activation_zero_point, 255U - activation_zero_point);
    const UnsignedInt128 bound = static_cast<UnsignedInt128>(k) * maximum_activation_magnitude *
                                     maximum_weight_magnitude +
                                 maximum_bias_magnitude;
    if (bound > std::numeric_limits<std::uint64_t>::max()) {
        return result;
    }
    result.absolute_bound = static_cast<std::uint64_t>(bound);
    result.int32_safe = result.absolute_bound <= static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
    result.valid = true;
    return result;
}

std::int8_t signed_storage(std::uint8_t semantic_code_value) noexcept {
    return static_cast<std::int8_t>(static_cast<int>(semantic_code_value) - 128);
}

std::uint8_t semantic_code(std::int8_t physical_storage) noexcept {
    return static_cast<std::uint8_t>(static_cast<int>(physical_storage) + 128);
}

bool nchwc8_offset(std::size_t n,
                   std::size_t channel,
                   std::size_t y,
                   std::size_t x,
                   std::size_t batches,
                   std::size_t channels,
                   std::size_t height,
                   std::size_t width,
                   std::size_t* offset) noexcept {
    if (offset == nullptr || batches == 0 || channels == 0 || channels % 8 != 0 || height == 0 || width == 0 ||
        n >= batches || channel >= channels || y >= height || x >= width) {
        return false;
    }
    const std::size_t blocks = channels / 8;
    const std::size_t block = channel / 8;
    const std::size_t inner = channel % 8;
    std::size_t value = 0;
    if (!checked_mul(n, blocks, &value) || !checked_add(value, block, &value) ||
        !checked_mul(value, height, &value) || !checked_add(value, y, &value) ||
        !checked_mul(value, width, &value) || !checked_add(value, x, &value) ||
        !checked_mul(value, 8, &value) || !checked_add(value, inner, &value)) {
        return false;
    }
    *offset = value;
    return true;
}

}  // namespace y26::int8_v1
