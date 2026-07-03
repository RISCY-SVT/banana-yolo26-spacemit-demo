#include "y26_k1x_vmadot.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>

namespace {

std::int64_t checksum(const std::array<std::int32_t, 16>& values) {
    std::int64_t total = 0;
    for (const auto value : values) {
        total += value;
    }
    return total;
}

}  // namespace

int main() {
    const std::array<std::int8_t, 32> a_ramp {
        -16, -15, -14, -13, -12, -11, -10, -9,
        -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
    };
    const std::array<std::int8_t, 32> b_transposed {
        15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0,
        -1, -2, -3, -4, -5, -6, -7, -8,
        -9, -10, -11, -12, -13, -14, -15, -16,
    };
    const std::array<std::int32_t, 16> expected_ramp {
        -1192, -392, 408, 1208,
        -456, -168, 120, 408,
        280, 56, -168, -392,
        1016, 280, -456, -1192,
    };

    std::array<std::int32_t, 16> actual {};
    y26_vmadot_4x4x8_scalar_s8s8s32(a_ramp.data(), b_transposed.data(), actual.data(), false);
    if (actual != expected_ramp || checksum(actual) != -640) {
        std::fprintf(stderr, "ramp fixture mismatch\n");
        return 1;
    }

    std::array<std::int8_t, 32> a_edges {};
    std::array<std::int8_t, 32> b_edges {};
    for (std::size_t i = 0; i < 32; ++i) {
        a_edges[i] = (i % 2U) == 0U ? static_cast<std::int8_t>(-128) : static_cast<std::int8_t>(127);
        b_edges[i] = (i % 2U) == 0U ? static_cast<std::int8_t>(127) : static_cast<std::int8_t>(-128);
    }

    std::array<std::int32_t, 16> edge_actual {};
    y26_vmadot_4x4x8_scalar_s8s8s32(a_edges.data(), b_edges.data(), edge_actual.data(), false);
    for (const auto value : edge_actual) {
        if (value != -130048) {
            std::fprintf(stderr, "edge fixture mismatch\n");
            return 1;
        }
    }
    if (checksum(edge_actual) != -2080768) {
        std::fprintf(stderr, "edge checksum mismatch\n");
        return 1;
    }

    std::array<std::int32_t, 16> init {};
    for (std::size_t i = 0; i < init.size(); ++i) {
        init[i] = static_cast<std::int32_t>(static_cast<int>(i) * 7 - 23);
    }
    std::array<std::int32_t, 16> accum = init;
    y26_vmadot_4x4x8_scalar_s8s8s32(a_ramp.data(), b_transposed.data(), accum.data(), true);
    for (std::size_t i = 0; i < accum.size(); ++i) {
        if (accum[i] != expected_ramp[i] + init[i]) {
            std::fprintf(stderr, "accumulate fixture mismatch\n");
            return 1;
        }
    }

    if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
        const int status = y26_vmadot_4x4x8_ime_s8s8s32(a_ramp.data(), b_transposed.data(), actual.data(), false);
        if (status != Y26_VMADOT_STATUS_NOT_BUILT_WITH_IME) {
            std::fprintf(stderr, "unexpected host IME status %d\n", status);
            return 1;
        }
    }
    return 0;
}
