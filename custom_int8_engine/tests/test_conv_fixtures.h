#pragma once

#include "y26_k1x_conv_kernels.h"

#include <cstddef>
#include <cstdint>
#include <vector>

inline std::uint32_t y26_test_lcg_next(std::uint32_t state) {
    return state * 1664525U + 1013904223U;
}

inline std::int8_t y26_test_i8(std::uint32_t& state) {
    state = y26_test_lcg_next(state);
    const int value = static_cast<int>((state >> 24U) & 0xFFU) - 128;
    return static_cast<std::int8_t>(value);
}

inline std::vector<std::int8_t> y26_make_i8_vector(std::size_t count, std::uint32_t seed) {
    std::vector<std::int8_t> values(count);
    for (auto& value : values) {
        value = y26_test_i8(seed);
    }
    if (count >= 8) {
        values[0] = static_cast<std::int8_t>(-128);
        values[1] = static_cast<std::int8_t>(-127);
        values[2] = static_cast<std::int8_t>(-1);
        values[3] = static_cast<std::int8_t>(0);
        values[4] = static_cast<std::int8_t>(1);
        values[5] = static_cast<std::int8_t>(2);
        values[6] = static_cast<std::int8_t>(126);
        values[7] = static_cast<std::int8_t>(127);
    }
    return values;
}

inline std::vector<std::int32_t> y26_make_bias(int output_c) {
    std::vector<std::int32_t> bias(static_cast<std::size_t>(output_c));
    for (int oc = 0; oc < output_c; ++oc) {
        bias[static_cast<std::size_t>(oc)] = oc * 17 - 31;
    }
    return bias;
}

inline int y26_count_mismatches(const std::vector<std::int32_t>& lhs, const std::vector<std::int32_t>& rhs) {
    if (lhs.size() != rhs.size()) {
        return -1;
    }
    int mismatches = 0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) {
            ++mismatches;
        }
    }
    return mismatches;
}

inline std::int64_t y26_checksum(const std::vector<std::int32_t>& values) {
    std::int64_t sum = 0;
    for (const auto value : values) {
        sum += value;
    }
    return sum;
}
