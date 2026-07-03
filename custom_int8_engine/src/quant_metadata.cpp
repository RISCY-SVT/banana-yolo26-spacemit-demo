#include "y26_k1x_engine.h"

#include <algorithm>
#include <cmath>

namespace y26_k1x {

std::int8_t requantize_s32_to_s8(std::int32_t accumulator, const RequantParams& params) {
    const float scaled = static_cast<float>(accumulator) * params.effective_scale;
    const float rounded = scaled >= 0.0F ? std::floor(scaled + 0.5F) : std::ceil(scaled - 0.5F);
    const auto shifted = static_cast<std::int32_t>(rounded) + params.output_zero_point;
    const auto clamped = std::clamp(shifted, params.qmin, params.qmax);
    return static_cast<std::int8_t>(clamped);
}

}  // namespace y26_k1x
