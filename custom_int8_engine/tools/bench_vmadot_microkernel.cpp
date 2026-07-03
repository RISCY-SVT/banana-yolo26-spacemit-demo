#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "y26_k1x_vmadot.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <csetjmp>
#include <csignal>
#include <vector>

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv) && defined(__linux__)
#include <sched.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
sigjmp_buf g_bench_sigill_jump;
volatile sig_atomic_t g_bench_sigill_seen = 0;

void bench_sigill_handler(int /*signo*/, siginfo_t* /*info*/, void* /*uctx*/) {
    g_bench_sigill_seen = 1;
    siglongjmp(g_bench_sigill_jump, 1);
}

bool bench_current_cpu_is_cluster0() {
#if defined(__linux__)
    const int cpu = sched_getcpu();
    return cpu >= 0 && cpu <= 3;
#else
    return false;
#endif
}

__attribute__((noinline)) void bench_ime_direct_4x4x8(const std::int8_t* a_4x8_row_major,
                                                      const std::int8_t* b_4x8_transposed_nk,
                                                      std::int32_t* c_4x4_row_major) {
    __asm__ volatile(
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vxor.vv      v28, v28, v28           \n\t"
        "vsetvli      t0, zero, e8, m1        \n\t"
        "vle8.v       v0, (%[A])              \n\t"
        "vle8.v       v1, (%[B])              \n\t"
        "smt.vmadot   v28, v0, v1             \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vse32.v      v28, (%[C])             \n\t"
        :
        : [A] "r"(a_4x8_row_major), [B] "r"(b_4x8_transposed_nk), [C] "r"(c_4x4_row_major)
        : "cc", "memory", "t0", "v0", "v1", "v28", "v29");
}
#endif

std::int64_t checksum(const std::array<std::int32_t, 16>& values) {
    std::int64_t total = 0;
    for (const auto value : values) {
        total += value;
    }
    return total;
}

template <typename Fn>
double run_timed(int iterations, Fn&& fn, std::int64_t& guard) {
    const auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        guard += fn(i);
    }
    const auto end = Clock::now();
    return std::chrono::duration<double, std::nano>(end - start).count() / static_cast<double>(iterations);
}

double stddev(const std::vector<double>& values, double mean) {
    if (values.size() < 2) {
        return 0.0;
    }
    double sum = 0.0;
    for (const auto value : values) {
        const double delta = value - mean;
        sum += delta * delta;
    }
    return std::sqrt(sum / static_cast<double>(values.size() - 1));
}

}  // namespace

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 100000;
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 5;
    const int warmup = 1000;

    std::array<std::int8_t, 32> a {};
    std::array<std::int8_t, 32> b {};
    for (std::size_t i = 0; i < 32; ++i) {
        a[i] = static_cast<std::int8_t>((static_cast<int>(i) * 7) % 251 - 125);
        b[i] = static_cast<std::int8_t>(123 - ((static_cast<int>(i) * 11) % 247));
    }

    std::array<std::int32_t, 16> c {};
    for (int i = 0; i < warmup; ++i) {
        y26_vmadot_4x4x8_scalar_s8s8s32(a.data(), b.data(), c.data(), false);
    }

    std::vector<double> scalar_runs;
    scalar_runs.reserve(static_cast<std::size_t>(repeats));
    std::int64_t scalar_guard = 0;
    for (int r = 0; r < repeats; ++r) {
        scalar_runs.push_back(run_timed(iterations, [&](int i) {
            c[0] = i;
            y26_vmadot_4x4x8_scalar_s8s8s32(a.data(), b.data(), c.data(), false);
            return checksum(c);
        }, scalar_guard));
    }

    double scalar_mean = 0.0;
    for (const auto value : scalar_runs) {
        scalar_mean += value;
    }
    scalar_mean /= static_cast<double>(scalar_runs.size());
    const double scalar_stddev = stddev(scalar_runs, scalar_mean);

    std::printf("benchmark_scope=microkernel_only_not_yolo26_inference\n");
    std::printf("iterations=%d\n", iterations);
    std::printf("warmup=%d\n", warmup);
    std::printf("repeats=%d\n", repeats);
    std::printf("packing_included=no\n");
    std::printf("public_guard_overhead_included_for_public_api=yes\n");
    std::printf("direct_ime_guard_overhead_included=no\n");
    std::printf("scalar_mean_ns_per_call=%.3f\n", scalar_mean);
    std::printf("scalar_stddev_ns_per_call=%.3f\n", scalar_stddev);
    std::printf("scalar_guard=%lld\n", static_cast<long long>(scalar_guard));

    if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
        std::printf("ime_status=not-built\n");
        return 0;
    }

    std::vector<double> ime_runs;
    ime_runs.reserve(static_cast<std::size_t>(repeats));
    std::int64_t ime_guard = 0;
    int last_status = Y26_VMADOT_STATUS_SUCCESS;
    for (int r = 0; r < repeats; ++r) {
        ime_runs.push_back(run_timed(iterations, [&](int i) {
            c[0] = i;
            last_status = y26_vmadot_4x4x8_ime_s8s8s32(a.data(), b.data(), c.data(), false);
            return checksum(c) + last_status;
        }, ime_guard));
        if (last_status != Y26_VMADOT_STATUS_SUCCESS) {
            break;
        }
    }

    double ime_mean = 0.0;
    for (const auto value : ime_runs) {
        ime_mean += value;
    }
    ime_mean /= static_cast<double>(ime_runs.size());
    const double ime_stddev = stddev(ime_runs, ime_mean);
    std::printf("ime_status=%d\n", last_status);
    std::printf("ime_mean_ns_per_call=%.3f\n", ime_mean);
    std::printf("ime_stddev_ns_per_call=%.3f\n", ime_stddev);
    std::printf("ime_guard=%lld\n", static_cast<long long>(ime_guard));
    if (last_status == Y26_VMADOT_STATUS_SUCCESS && ime_mean > 0.0) {
        std::printf("public_guarded_speedup_vs_scalar=%.3f\n", scalar_mean / ime_mean);
    }

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    if (!bench_current_cpu_is_cluster0()) {
        std::printf("ime_direct_status=runtime-safety-failed\n");
        return last_status == Y26_VMADOT_STATUS_SUCCESS ? 0 : 1;
    }

    struct sigaction old_action {};
    struct sigaction new_action {};
    new_action.sa_sigaction = bench_sigill_handler;
    new_action.sa_flags = SA_SIGINFO;
    sigemptyset(&new_action.sa_mask);
    if (sigaction(SIGILL, &new_action, &old_action) != 0) {
        std::printf("ime_direct_status=sigaction-failed\n");
        return 1;
    }

    volatile sig_atomic_t direct_smoke_ok = 0;
    if (sigsetjmp(g_bench_sigill_jump, 1) == 0) {
        bench_ime_direct_4x4x8(a.data(), b.data(), c.data());
        direct_smoke_ok = 1;
    }
    sigaction(SIGILL, &old_action, nullptr);

    int direct_status = (direct_smoke_ok != 0 && g_bench_sigill_seen == 0) ? 0 : Y26_VMADOT_STATUS_SIGILL_CAUGHT;
    std::printf("ime_direct_status=%d\n", direct_status);
    if (direct_status == 0) {
        for (int i = 0; i < warmup; ++i) {
            bench_ime_direct_4x4x8(a.data(), b.data(), c.data());
        }

        std::vector<double> direct_runs;
        direct_runs.reserve(static_cast<std::size_t>(repeats));
        std::int64_t direct_guard = 0;
        for (int r = 0; r < repeats; ++r) {
            direct_runs.push_back(run_timed(iterations, [&](int i) {
                c[0] = i;
                bench_ime_direct_4x4x8(a.data(), b.data(), c.data());
                return checksum(c);
            }, direct_guard));
        }

        double direct_mean = 0.0;
        for (const auto value : direct_runs) {
            direct_mean += value;
        }
        direct_mean /= static_cast<double>(direct_runs.size());
        const double direct_stddev = stddev(direct_runs, direct_mean);
        std::printf("ime_direct_mean_ns_per_call=%.3f\n", direct_mean);
        std::printf("ime_direct_stddev_ns_per_call=%.3f\n", direct_stddev);
        std::printf("ime_direct_guard=%lld\n", static_cast<long long>(direct_guard));
        if (direct_mean > 0.0) {
            std::printf("direct_speedup_vs_scalar=%.3f\n", scalar_mean / direct_mean);
        }
    }
#endif
    return last_status == Y26_VMADOT_STATUS_SUCCESS ? 0 : 1;
}
