#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "y26_k1x_vmadot.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string_view>

#if defined(__linux__)
#include <sched.h>
#endif

namespace {

struct Case {
    std::string_view name;
    std::array<std::int8_t, 32> a {};
    std::array<std::int8_t, 32> b {};
    std::array<std::int32_t, 16> init {};
    bool accumulate = false;
};

std::uint32_t lcg_next(std::uint32_t state) {
    return state * 1664525U + 1013904223U;
}

std::int8_t sample_from_seed(std::uint32_t& state) {
    state = lcg_next(state);
    return static_cast<std::int8_t>((state >> 24U) - 128U);
}

Case random_case(std::string_view name, std::uint32_t seed) {
    Case tc {};
    tc.name = name;
    for (std::size_t i = 0; i < 32; ++i) {
        tc.a[i] = sample_from_seed(seed);
        tc.b[i] = sample_from_seed(seed);
    }
    return tc;
}

std::array<Case, 8> make_cases() {
    std::array<Case, 8> cases {};
    cases[0].name = "all_zeros";

    cases[1].name = "all_ones";
    cases[1].a.fill(1);
    cases[1].b.fill(1);

    cases[2].name = "ramp";
    for (std::size_t i = 0; i < 32; ++i) {
        cases[2].a[i] = static_cast<std::int8_t>(static_cast<int>(i) - 16);
        cases[2].b[i] = static_cast<std::int8_t>(15 - static_cast<int>(i));
    }

    cases[3].name = "alternating_edges";
    for (std::size_t i = 0; i < 32; ++i) {
        cases[3].a[i] = (i % 2U) == 0U ? static_cast<std::int8_t>(-128) : static_cast<std::int8_t>(127);
        cases[3].b[i] = (i % 2U) == 0U ? static_cast<std::int8_t>(127) : static_cast<std::int8_t>(-128);
    }

    cases[4] = random_case("random_seed_0", 0U);
    cases[5] = random_case("random_seed_1", 1U);
    cases[6] = random_case("random_seed_12345", 12345U);
    cases[7] = cases[6];
    cases[7].name = "accumulate_true";
    cases[7].accumulate = true;
    for (std::size_t i = 0; i < cases[7].init.size(); ++i) {
        cases[7].init[i] = static_cast<std::int32_t>(static_cast<int>(i) * 19 - 143);
    }
    return cases;
}

std::int64_t checksum(const std::array<std::int32_t, 16>& values) {
    std::int64_t total = 0;
    for (const auto value : values) {
        total += value;
    }
    return total;
}

int mismatch_count(const std::array<std::int32_t, 16>& lhs, const std::array<std::int32_t, 16>& rhs) {
    int mismatches = 0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) {
            ++mismatches;
        }
    }
    return mismatches;
}

int current_cpu() {
#if defined(__linux__)
    return sched_getcpu();
#else
    return -1;
#endif
}

}  // namespace

int main() {
    std::printf("STAGE1_BOARD_PROBE_BEGIN\n");
    std::printf("ime_buildtime_available=%d\n", y26_vmadot_4x4x8_ime_available_buildtime() ? 1 : 0);
    std::printf("cpu_before=%d\n", current_cpu());
    std::printf("case\tstatus\tmismatches\tchecksum_scalar\tchecksum_ime\n");

    if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
        std::printf("STAGE1_BOARD_PROBE_END\n");
        return 0;
    }

    int total_mismatches = 0;
    int nonzero_status = 0;
    for (const auto& tc : make_cases()) {
        std::array<std::int32_t, 16> scalar = tc.init;
        std::array<std::int32_t, 16> ime = tc.init;
        y26_vmadot_4x4x8_scalar_s8s8s32(tc.a.data(), tc.b.data(), scalar.data(), tc.accumulate);
        const int status = y26_vmadot_4x4x8_ime_s8s8s32(tc.a.data(), tc.b.data(), ime.data(), tc.accumulate);
        const int mismatches = mismatch_count(scalar, ime);
        total_mismatches += mismatches;
        if (status != Y26_VMADOT_STATUS_SUCCESS) {
            ++nonzero_status;
        }
        std::printf("%.*s\t%d\t%d\t%lld\t%lld\n",
                    static_cast<int>(tc.name.size()),
                    tc.name.data(),
                    status,
                    mismatches,
                    static_cast<long long>(checksum(scalar)),
                    static_cast<long long>(checksum(ime)));
    }

    std::printf("total_mismatches=%d\n", total_mismatches);
    std::printf("nonzero_status=%d\n", nonzero_status);
    std::printf("cpu_after=%d\n", current_cpu());
    std::printf("STAGE1_BOARD_PROBE_END\n");
    return (total_mismatches == 0 && nonzero_status == 0) ? 0 : 1;
}
