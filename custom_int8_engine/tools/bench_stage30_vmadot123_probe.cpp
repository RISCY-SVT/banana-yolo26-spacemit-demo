#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "y26_k1x_vmadot.h"
#include "y26_k1x_vmadot123_probe.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <setjmp.h>
#include <signal.h>
#include <string_view>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace {

sigjmp_buf g_jump_env;
volatile sig_atomic_t g_sigill_seen = 0;

struct Case {
    std::string_view name;
    std::array<std::int8_t, 64> a {};
    std::array<std::int8_t, 32> b {};
    std::array<std::int32_t, 16> init {};
    bool accumulate = false;
};

void on_sigill(int, siginfo_t*, void*) {
    g_sigill_seen = 1;
    siglongjmp(g_jump_env, 1);
}

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
    for (std::size_t i = 0; i < tc.a.size(); ++i) {
        tc.a[i] = sample_from_seed(seed);
    }
    for (std::size_t i = 0; i < tc.b.size(); ++i) {
        tc.b[i] = sample_from_seed(seed);
    }
    return tc;
}

std::vector<Case> make_cases() {
    std::vector<Case> cases;
    cases.resize(7);
    cases[0].name = "all_zero";

    cases[1].name = "single_one";
    cases[1].a[0] = 1;
    cases[1].b[0] = 1;

    cases[2].name = "ramp";
    for (std::size_t i = 0; i < cases[2].a.size(); ++i) {
        cases[2].a[i] = static_cast<std::int8_t>(static_cast<int>(i) - 16);
    }
    for (std::size_t i = 0; i < cases[2].b.size(); ++i) {
        cases[2].b[i] = static_cast<std::int8_t>(15 - static_cast<int>(i));
    }

    cases[3].name = "alternating_signs";
    for (std::size_t i = 0; i < cases[3].a.size(); ++i) {
        cases[3].a[i] = (i % 2U) == 0U ? static_cast<std::int8_t>(-17) : static_cast<std::int8_t>(23);
    }
    for (std::size_t i = 0; i < cases[3].b.size(); ++i) {
        cases[3].b[i] = (i % 2U) == 0U ? static_cast<std::int8_t>(19) : static_cast<std::int8_t>(-29);
    }

    cases[4].name = "edge_values";
    static constexpr std::array<std::int8_t, 6> edge {-128, -127, -1, 0, 1, 127};
    for (std::size_t i = 0; i < cases[4].a.size(); ++i) {
        cases[4].a[i] = edge[i % edge.size()];
    }
    for (std::size_t i = 0; i < cases[4].b.size(); ++i) {
        cases[4].b[i] = edge[(i * 5U + 1U) % edge.size()];
    }

    cases[5] = random_case("random_seed_37", 37U);
    cases[6] = random_case("accumulate_random_seed_99", 99U);
    cases[6].accumulate = true;
    for (std::size_t i = 0; i < cases[6].init.size(); ++i) {
        cases[6].init[i] = static_cast<std::int32_t>(static_cast<int>(i) * 17 - 130);
    }
    return cases;
}

void scalar_gemm(const std::array<std::int8_t, 64>& a,
                 const std::array<std::int8_t, 32>& b_transposed,
                 const std::array<std::int32_t, 16>& init,
                 bool accumulate,
                 std::array<std::int32_t, 16>& c) {
    c = init;
    if (!accumulate) {
        c.fill(0);
    }
    y26_vmadot_4x4x8_scalar_s8s8s32(a.data(), b_transposed.data(), c.data(), accumulate);
}

void scalar_shifted_a(const std::array<std::int8_t, 64>& a,
                      const std::array<std::int8_t, 32>& b_transposed,
                      const std::array<std::int32_t, 16>& init,
                      bool accumulate,
                      int shift,
                      std::array<std::int32_t, 16>& c) {
    c = init;
    if (!accumulate) {
        c.fill(0);
    }
    for (int m = 0; m < 4; ++m) {
        for (int n = 0; n < 4; ++n) {
            std::int32_t acc = c[m * 4 + n];
            for (int k = 0; k < 8; ++k) {
                const int a_index = m * 8 + ((k + shift) & 7);
                const int b_index = n * 8 + k;
                acc += static_cast<std::int32_t>(a[a_index]) * static_cast<std::int32_t>(b_transposed[b_index]);
            }
            c[m * 4 + n] = acc;
        }
    }
}

void scalar_shifted_b(const std::array<std::int8_t, 64>& a,
                      const std::array<std::int8_t, 32>& b_transposed,
                      const std::array<std::int32_t, 16>& init,
                      bool accumulate,
                      int shift,
                      std::array<std::int32_t, 16>& c) {
    c = init;
    if (!accumulate) {
        c.fill(0);
    }
    for (int m = 0; m < 4; ++m) {
        for (int n = 0; n < 4; ++n) {
            std::int32_t acc = c[m * 4 + n];
            for (int k = 0; k < 8; ++k) {
                const int a_index = m * 8 + k;
                const int b_index = n * 8 + ((k + shift) & 7);
                acc += static_cast<std::int32_t>(a[a_index]) * static_cast<std::int32_t>(b_transposed[b_index]);
            }
            c[m * 4 + n] = acc;
        }
    }
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

long long checksum(const std::array<std::int32_t, 16>& values) {
    long long total = 0;
    for (const auto value : values) {
        total += value;
    }
    return total;
}

void print_i32_csv(const char* key, const std::array<std::int32_t, 16>& values) {
    std::printf("%s=", key);
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::printf("%s%d", i == 0 ? "" : ",", values[i]);
    }
    std::printf("\n");
}

int current_cpu() {
#if defined(__linux__)
    return sched_getcpu();
#else
    return -1;
#endif
}

struct HypothesisResult {
    const char* name;
    int mismatches;
    long long checksum_ref;
};

struct MapEntry {
    int a_index;
    int b_index;
    std::int32_t coefficient;
};

using OracleMap = std::array<std::vector<MapEntry>, 16>;

std::array<HypothesisResult, 7> compare_hypotheses(const Case& tc, const std::array<std::int32_t, 16>& actual) {
    std::array<HypothesisResult, 7> results {};
    std::array<std::int32_t, 16> ref {};
    scalar_gemm(tc.a, tc.b, tc.init, tc.accumulate, ref);
    results[0] = {"vmadot_base", mismatch_count(ref, actual), checksum(ref)};
    scalar_shifted_a(tc.a, tc.b, tc.init, tc.accumulate, 1, ref);
    results[1] = {"shift_a_1", mismatch_count(ref, actual), checksum(ref)};
    scalar_shifted_a(tc.a, tc.b, tc.init, tc.accumulate, 2, ref);
    results[2] = {"shift_a_2", mismatch_count(ref, actual), checksum(ref)};
    scalar_shifted_a(tc.a, tc.b, tc.init, tc.accumulate, 3, ref);
    results[3] = {"shift_a_3", mismatch_count(ref, actual), checksum(ref)};
    scalar_shifted_b(tc.a, tc.b, tc.init, tc.accumulate, 1, ref);
    results[4] = {"shift_b_1", mismatch_count(ref, actual), checksum(ref)};
    scalar_shifted_b(tc.a, tc.b, tc.init, tc.accumulate, 2, ref);
    results[5] = {"shift_b_2", mismatch_count(ref, actual), checksum(ref)};
    scalar_shifted_b(tc.a, tc.b, tc.init, tc.accumulate, 3, ref);
    results[6] = {"shift_b_3", mismatch_count(ref, actual), checksum(ref)};
    return results;
}

int run_variant_case(int variant, const Case& tc, std::array<std::int32_t, 16>& actual, int& trapped) {
    actual = tc.init;
    if (!tc.accumulate) {
        actual.fill(0);
    }
    trapped = 0;
    g_sigill_seen = 0;
    if (sigsetjmp(g_jump_env, 1) == 0) {
        const int status =
            y26_k1x_vmadot123_checked_cluster0_s8s8s32(variant, tc.a.data(), tc.b.data(), actual.data(), tc.accumulate);
        trapped = g_sigill_seen == 0 ? 0 : 1;
        return status;
    }
    trapped = 1;
    return Y26_VMADOT_STATUS_SIGILL_CAUGHT;
}

bool derive_oracle_map(int variant, OracleMap& map, int& status_failures, int& traps) {
    status_failures = 0;
    traps = 0;
    for (auto& lane_entries : map) {
        lane_entries.clear();
    }

    Case impulse {};
    impulse.name = "impulse";
    for (int ai = 0; ai < 64; ++ai) {
        for (int bi = 0; bi < 32; ++bi) {
            impulse.a.fill(0);
            impulse.b.fill(0);
            impulse.init.fill(0);
            impulse.a[static_cast<std::size_t>(ai)] = 1;
            impulse.b[static_cast<std::size_t>(bi)] = 1;
            std::array<std::int32_t, 16> actual {};
            int trapped = 0;
            const int status = run_variant_case(variant, impulse, actual, trapped);
            if (status != Y26_VMADOT_STATUS_SUCCESS) {
                ++status_failures;
            }
            traps += trapped;
            for (std::size_t lane = 0; lane < actual.size(); ++lane) {
                if (actual[lane] != 0) {
                    map[lane].push_back({ai, bi, actual[lane]});
                }
            }
        }
    }
    return status_failures == 0 && traps == 0;
}

void scalar_from_oracle_map(const OracleMap& map,
                            const Case& tc,
                            std::array<std::int32_t, 16>& out) {
    out = tc.init;
    if (!tc.accumulate) {
        out.fill(0);
    }
    for (std::size_t lane = 0; lane < out.size(); ++lane) {
        std::int32_t acc = out[lane];
        for (const auto& entry : map[lane]) {
            acc += entry.coefficient * static_cast<std::int32_t>(tc.a[static_cast<std::size_t>(entry.a_index)]) *
                   static_cast<std::int32_t>(tc.b[static_cast<std::size_t>(entry.b_index)]);
        }
        out[lane] = acc;
    }
}

int validate_oracle_map(int variant, const OracleMap& map, long long& checksum_actual_total, long long& checksum_ref_total) {
    int total_mismatches = 0;
    checksum_actual_total = 0;
    checksum_ref_total = 0;
    for (const auto& tc : make_cases()) {
        std::array<std::int32_t, 16> actual {};
        std::array<std::int32_t, 16> ref {};
        int trapped = 0;
        const int status = run_variant_case(variant, tc, actual, trapped);
        scalar_from_oracle_map(map, tc, ref);
        const int mismatches = mismatch_count(ref, actual);
        total_mismatches += mismatches;
        checksum_actual_total += checksum(actual);
        checksum_ref_total += checksum(ref);
        std::printf("oracle_validate\tvmadot%d\t%.*s\tstatus=%d\ttrapped=%d\tmismatches=%d\tchecksum_ref=%lld\tchecksum_actual=%lld\n",
                    variant,
                    static_cast<int>(tc.name.size()),
                    tc.name.data(),
                    status,
                    trapped,
                    mismatches,
                    checksum(ref),
                    checksum(actual));
    }
    return total_mismatches;
}

int run_derived_oracle_mode(int selected_variant) {
    int total_status_failures = 0;
    int total_traps = 0;
    int total_validation_mismatches = 0;
    std::printf("STAGE30_VMADOT123_DERIVED_ORACLE_BEGIN\n");
    std::printf("cpu_before=%d\n", current_cpu());
    for (int variant = Y26_VMADOT123_VARIANT_1; variant <= Y26_VMADOT123_VARIANT_3; ++variant) {
        if (selected_variant != 0 && selected_variant != variant) {
            continue;
        }
        OracleMap map {};
        int status_failures = 0;
        int traps = 0;
        derive_oracle_map(variant, map, status_failures, traps);
        total_status_failures += status_failures;
        total_traps += traps;
        std::size_t entries = 0;
        for (const auto& lane_entries : map) {
            entries += lane_entries.size();
        }
        std::printf("oracle_map\tvmadot%d\tentries=%zu\tstatus_failures=%d\ttraps=%d\n",
                    variant,
                    entries,
                    status_failures,
                    traps);
        std::printf("oracle_map_tsv_begin\tvmadot%d\n", variant);
        std::printf("variant\tlane\ta_index\tb_index\tcoefficient\n");
        for (std::size_t lane = 0; lane < map.size(); ++lane) {
            for (const auto& entry : map[lane]) {
                std::printf("vmadot%d\t%zu\t%d\t%d\t%d\n",
                            variant,
                            lane,
                            entry.a_index,
                            entry.b_index,
                            entry.coefficient);
            }
        }
        std::printf("oracle_map_tsv_end\tvmadot%d\n", variant);
        long long checksum_actual_total = 0;
        long long checksum_ref_total = 0;
        const int mismatches = validate_oracle_map(variant, map, checksum_actual_total, checksum_ref_total);
        total_validation_mismatches += mismatches;
        std::printf("oracle_validation_summary\tvmadot%d\tmismatches=%d\tchecksum_ref_total=%lld\tchecksum_actual_total=%lld\n",
                    variant,
                    mismatches,
                    checksum_ref_total,
                    checksum_actual_total);
    }
    std::printf("total_status_failures=%d\n", total_status_failures);
    std::printf("total_traps=%d\n", total_traps);
    std::printf("total_validation_mismatches=%d\n", total_validation_mismatches);
    std::printf("cpu_after=%d\n", current_cpu());
    std::printf("STAGE30_VMADOT123_DERIVED_ORACLE_END\n");
    return (total_status_failures == 0 && total_traps == 0 && total_validation_mismatches == 0) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    int selected_variant = 0;
    bool derive_oracle = false;
    if (argc == 3 && std::strcmp(argv[1], "--variant") == 0) {
        selected_variant = std::atoi(argv[2]);
    } else if (argc == 2 && std::strcmp(argv[1], "--derive-oracle") == 0) {
        derive_oracle = true;
    } else if (argc == 4 && std::strcmp(argv[1], "--derive-oracle") == 0 && std::strcmp(argv[2], "--variant") == 0) {
        derive_oracle = true;
        selected_variant = std::atoi(argv[3]);
    }

    struct sigaction old_action {};
    struct sigaction new_action {};
    new_action.sa_sigaction = on_sigill;
    new_action.sa_flags = SA_SIGINFO;
    sigemptyset(&new_action.sa_mask);
    if (sigaction(SIGILL, &new_action, &old_action) != 0) {
        std::perror("sigaction");
        return 2;
    }

    if (derive_oracle) {
        const int rc = run_derived_oracle_mode(selected_variant);
        sigaction(SIGILL, &old_action, nullptr);
        return rc;
    }

    std::printf("STAGE30_VMADOT123_PROBE_BEGIN\n");
    std::printf("buildtime_available=%d\n", y26_vmadot123_probe_available_buildtime() ? 1 : 0);
    std::printf("cpu_before=%d\n", current_cpu());
    std::printf("variant\tcase\tstatus\ttrapped\tchecksum_actual\tbest_hypothesis\tbest_mismatches\n");

    int total_status_failures = 0;
    int total_traps = 0;
    int total_best_mismatches = 0;
    for (int variant = Y26_VMADOT123_VARIANT_1; variant <= Y26_VMADOT123_VARIANT_3; ++variant) {
        if (selected_variant != 0 && selected_variant != variant) {
            continue;
        }
        for (const auto& tc : make_cases()) {
            std::array<std::int32_t, 16> actual {};
            int trapped = 0;
            const int status = run_variant_case(variant, tc, actual, trapped);
            if (status != Y26_VMADOT_STATUS_SUCCESS) {
                ++total_status_failures;
            }
            total_traps += trapped;
            const auto results = compare_hypotheses(tc, actual);
            const auto best = std::min_element(results.begin(), results.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.mismatches < rhs.mismatches;
            });
            total_best_mismatches += best->mismatches;
            std::printf("vmadot%d\t%.*s\t%d\t%d\t%lld\t%s\t%d\n",
                        variant,
                        static_cast<int>(tc.name.size()),
                        tc.name.data(),
                        status,
                        trapped,
                        checksum(actual),
                        best->name,
                        best->mismatches);
            print_i32_csv("actual", actual);
        }
    }

    std::printf("total_status_failures=%d\n", total_status_failures);
    std::printf("total_traps=%d\n", total_traps);
    std::printf("total_best_hypothesis_mismatches=%d\n", total_best_mismatches);
    std::printf("cpu_after=%d\n", current_cpu());
    std::printf("STAGE30_VMADOT123_PROBE_END\n");
    sigaction(SIGILL, &old_action, nullptr);
    return (total_status_failures == 0 && total_traps == 0 && total_best_mismatches == 0) ? 0 : 1;
}
