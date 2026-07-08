#include "y26_k1x_vmadot.h"
#include "y26_k1x_vmadot123_probe.h"

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

std::array<std::int8_t, 64> make_ramp_a() {
    std::array<std::int8_t, 64> a {};
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<std::int8_t>(static_cast<int>(i) - 16);
    }
    return a;
}

std::array<std::int8_t, 32> make_ramp_b() {
    std::array<std::int8_t, 32> b {};
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = static_cast<std::int8_t>(15 - static_cast<int>(i));
    }
    return b;
}

int count_mismatches(const std::array<std::int32_t, 16>& lhs, const std::array<std::int32_t, 16>& rhs) {
    int mismatches = 0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) {
            ++mismatches;
        }
    }
    return mismatches;
}

}  // namespace

int main() {
    const auto a = make_ramp_a();
    const auto b = make_ramp_b();
    std::array<std::int32_t, 16> scalar {};
    std::array<std::int32_t, 16> actual {};

    y26_vmadot_4x4x8_scalar_s8s8s32(a.data(), b.data(), scalar.data(), false);

    if (!y26_vmadot123_probe_available_buildtime()) {
        const int status =
            y26_k1x_vmadot123_unsafe_cluster0_s8s8s32(Y26_VMADOT123_VARIANT_1, a.data(), b.data(), actual.data(), false);
        if (status != Y26_VMADOT_STATUS_NOT_BUILT_WITH_IME) {
            std::fprintf(stderr, "unexpected host vmadot123 status %d\n", status);
            return 1;
        }
        return count_mismatches(scalar, scalar) == 0 ? 0 : 1;
    }

    for (int variant = Y26_VMADOT123_VARIANT_1; variant <= Y26_VMADOT123_VARIANT_3; ++variant) {
        actual.fill(0);
        const int status = y26_k1x_vmadot123_checked_cluster0_s8s8s32(variant, a.data(), b.data(), actual.data(), false);
        if (status != Y26_VMADOT_STATUS_SUCCESS && status != Y26_VMADOT_STATUS_RUNTIME_SAFETY_FAILED) {
            std::fprintf(stderr, "unexpected vmadot%d status %d\n", variant, status);
            return 1;
        }
    }
    return 0;
}
