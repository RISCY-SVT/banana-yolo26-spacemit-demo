#pragma once

#include <cstddef>
#include <cstdint>

namespace y26::int8_v1 {

inline constexpr char kContractId[] = "K1X_INT8_V1";
inline constexpr char kGeneralProfile[] = "K1X_INT8_V1_GENERAL";
inline constexpr char kSymmetricProfile[] = "K1X_INT8_V1_SYMMETRIC";
inline constexpr char kNchwc8LayoutId[] = "NCHWc8_SPATIAL_INNER_V1";

struct RequantAsset {
    std::int64_t multiplier = 0;
    std::int32_t right_shift = 0;
    std::int32_t output_zero_point = 0;
    std::int32_t clamp_min = 0;
    std::int32_t clamp_max = 255;
};

struct AccumulatorSafety {
    std::uint64_t absolute_bound = 0;
    bool int32_safe = false;
    bool valid = false;
};

bool valid_requant_asset(const RequantAsset& asset) noexcept;

bool round_product_right_even(std::int64_t value,
                              std::int64_t multiplier,
                              std::int32_t right_shift,
                              std::int64_t* rounded) noexcept;

bool requantize_u8(std::int64_t accumulator,
                   const RequantAsset& asset,
                   std::uint8_t* output_code) noexcept;

AccumulatorSafety accumulator_safety_bound(std::size_t k,
                                           std::uint8_t activation_zero_point,
                                           std::uint8_t maximum_weight_magnitude,
                                           std::uint64_t maximum_bias_magnitude) noexcept;

std::int8_t signed_storage(std::uint8_t semantic_code) noexcept;
std::uint8_t semantic_code(std::int8_t physical_storage) noexcept;

bool nchwc8_offset(std::size_t n,
                   std::size_t channel,
                   std::size_t y,
                   std::size_t x,
                   std::size_t batches,
                   std::size_t channels,
                   std::size_t height,
                   std::size_t width,
                   std::size_t* offset) noexcept;

}  // namespace y26::int8_v1
