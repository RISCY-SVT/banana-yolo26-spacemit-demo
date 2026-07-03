#include "y26_k1x_vmadot.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

std::int64_t checksum(const std::array<std::int32_t, 16>& values) {
    std::int64_t sum = 0;
    for (const auto value : values) {
        sum += value;
    }
    return sum;
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

double mean(const std::vector<double>& values) {
    double sum = 0.0;
    for (const auto value : values) {
        sum += value;
    }
    return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
}

double stddev(const std::vector<double>& values, double avg) {
    if (values.size() < 2) {
        return 0.0;
    }
    double sum = 0.0;
    for (const auto value : values) {
        const double delta = value - avg;
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
    std::array<std::int32_t, 16> c {};
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<std::int8_t>((static_cast<int>(i) * 5) % 253 - 126);
        b[i] = static_cast<std::int8_t>(127 - ((static_cast<int>(i) * 9) % 251));
    }

    std::printf("benchmark_scope=microkernel_hotpath_only_not_yolo26_inference\n");
    std::printf("iterations=%d\n", iterations);
    std::printf("warmup=%d\n", warmup);
    std::printf("repeats=%d\n", repeats);
    std::printf("packing_included=no\n");

    for (int i = 0; i < warmup; ++i) {
        y26_vmadot_4x4x8_scalar_s8s8s32(a.data(), b.data(), c.data(), false);
    }

    std::vector<double> scalar_runs;
    std::int64_t scalar_guard = 0;
    for (int r = 0; r < repeats; ++r) {
        scalar_runs.push_back(run_timed(iterations, [&](int i) {
            c[0] = i;
            y26_vmadot_4x4x8_scalar_s8s8s32(a.data(), b.data(), c.data(), false);
            return checksum(c);
        }, scalar_guard));
    }
    const double scalar_mean = mean(scalar_runs);
    std::printf("scalar_mean_ns_per_call=%.3f\n", scalar_mean);
    std::printf("scalar_stddev_ns_per_call=%.3f\n", stddev(scalar_runs, scalar_mean));
    std::printf("scalar_guard=%lld\n", static_cast<long long>(scalar_guard));

    if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
        std::printf("ime_status=not-built\n");
        return 0;
    }

    const int probe_status = y26_k1x_ime_probe_once();
    const bool hotpath_allowed = y26_k1x_ime_hotpath_allowed_on_current_cpu();
    const auto snapshot = y26_k1x_ime_runtime_state_snapshot();
    std::printf("probe_status=%d\n", probe_status);
    std::printf("probe_cpu=%d\n", snapshot.probe_cpu);
    std::printf("capability=%d\n", snapshot.capability);
    std::printf("hotpath_allowed=%d\n", hotpath_allowed ? 1 : 0);
    if (probe_status != Y26_VMADOT_STATUS_SUCCESS || !hotpath_allowed) {
        return 1;
    }

    std::vector<double> public_runs;
    std::int64_t public_guard = 0;
    int public_status = 0;
    for (int r = 0; r < repeats; ++r) {
        public_runs.push_back(run_timed(iterations, [&](int i) {
            c[0] = i;
            public_status = y26_vmadot_4x4x8_ime_s8s8s32(a.data(), b.data(), c.data(), false);
            return checksum(c) + public_status;
        }, public_guard));
    }
    const double public_mean = mean(public_runs);
    std::printf("public_cached_status=%d\n", public_status);
    std::printf("public_cached_mean_ns_per_call=%.3f\n", public_mean);
    std::printf("public_cached_stddev_ns_per_call=%.3f\n", stddev(public_runs, public_mean));
    std::printf("public_cached_guard=%lld\n", static_cast<long long>(public_guard));
    if (public_status == 0 && public_mean > 0.0) {
        std::printf("public_cached_speedup_vs_scalar=%.3f\n", scalar_mean / public_mean);
    }

    std::vector<double> unsafe_runs;
    std::int64_t unsafe_guard = 0;
    int unsafe_status = 0;
    for (int r = 0; r < repeats; ++r) {
        unsafe_runs.push_back(run_timed(iterations, [&](int i) {
            c[0] = i;
            unsafe_status = y26_k1x_vmadot_4x4x8_unsafe_cluster0_s8s8s32(a.data(), b.data(), c.data(), false);
            return checksum(c) + unsafe_status;
        }, unsafe_guard));
    }
    const double unsafe_mean = mean(unsafe_runs);
    std::printf("unsafe_cluster0_status=%d\n", unsafe_status);
    std::printf("unsafe_cluster0_mean_ns_per_call=%.3f\n", unsafe_mean);
    std::printf("unsafe_cluster0_stddev_ns_per_call=%.3f\n", stddev(unsafe_runs, unsafe_mean));
    std::printf("unsafe_cluster0_guard=%lld\n", static_cast<long long>(unsafe_guard));
    if (unsafe_status == 0 && unsafe_mean > 0.0) {
        std::printf("unsafe_cluster0_speedup_vs_scalar=%.3f\n", scalar_mean / unsafe_mean);
    }
    return (public_status == 0 && unsafe_status == 0) ? 0 : 1;
}
