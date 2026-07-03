#include "y26_k1x_conv_kernels.h"
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

std::uint32_t lcg_next(std::uint32_t state) {
    return state * 1664525U + 1013904223U;
}

std::int8_t sample_i8(std::uint32_t& state) {
    state = lcg_next(state);
    return static_cast<std::int8_t>(static_cast<int>((state >> 24U) & 0xFFU) - 128);
}

std::vector<std::int8_t> make_i8(std::size_t count, std::uint32_t seed) {
    std::vector<std::int8_t> values(count);
    for (auto& value : values) {
        value = sample_i8(seed);
    }
    return values;
}

std::vector<std::int32_t> make_bias(int output_c) {
    std::vector<std::int32_t> bias(static_cast<std::size_t>(output_c));
    for (int oc = 0; oc < output_c; ++oc) {
        bias[static_cast<std::size_t>(oc)] = oc * 13 - 19;
    }
    return bias;
}

std::int64_t checksum(const std::vector<std::int32_t>& values) {
    std::int64_t sum = 0;
    for (const auto value : values) {
        sum += value;
    }
    return sum;
}

int mismatches(const std::vector<std::int32_t>& lhs, const std::vector<std::int32_t>& rhs) {
    if (lhs.size() != rhs.size()) {
        return -1;
    }
    int count = 0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) {
            ++count;
        }
    }
    return count;
}

template <typename Fn>
double run_timed(int iterations, Fn&& fn, std::int64_t& guard) {
    const auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        guard += fn();
    }
    const auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / static_cast<double>(iterations);
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

void report_case(const char* name,
                 const Y26Conv2DParams& params,
                 int output_h,
                 int output_w,
                 int status,
                 int mismatch_count,
                 double scalar_mean_us,
                 double scalar_stddev_us,
                 double ime_mean_us,
                 double ime_stddev_us,
                 std::int64_t scalar_guard,
                 std::int64_t ime_guard) {
    std::printf("case=%s\n", name);
    std::printf("shape=input_%dx%dx%d_output_c_%d_output_%dx%d\n",
                params.input_h,
                params.input_w,
                params.input_c,
                params.output_c,
                output_h,
                output_w);
    std::printf("layout=NHWC_input_OC_major_weights_NHWC_int32_output\n");
    std::printf("packing_included=yes\n");
    std::printf("ime_status=%d\n", status);
    std::printf("mismatches=%d\n", mismatch_count);
    std::printf("scalar_mean_us_per_call=%.3f\n", scalar_mean_us);
    std::printf("scalar_stddev_us_per_call=%.3f\n", scalar_stddev_us);
    if (status == Y26_CONV_STATUS_SUCCESS) {
        std::printf("ime_mean_us_per_call=%.3f\n", ime_mean_us);
        std::printf("ime_stddev_us_per_call=%.3f\n", ime_stddev_us);
        if (ime_mean_us > 0.0) {
            std::printf("speedup_vs_scalar=%.3f\n", scalar_mean_us / ime_mean_us);
        }
    }
    std::printf("scalar_guard=%lld\n", static_cast<long long>(scalar_guard));
    std::printf("ime_guard=%lld\n", static_cast<long long>(ime_guard));
}

void report_tile_core(const char* name, int iterations, int repeats) {
    std::array<std::int8_t, 32> a {};
    std::array<std::int8_t, 32> b {};
    std::array<std::int32_t, 16> scalar {};
    std::array<std::int32_t, 16> ime {};
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<std::int8_t>((static_cast<int>(i) * 7) % 251 - 125);
        b[i] = static_cast<std::int8_t>(123 - ((static_cast<int>(i) * 11) % 247));
    }

    y26_vmadot_4x4x8_scalar_s8s8s32(a.data(), b.data(), scalar.data(), false);
    const int ime_status = y26_k1x_vmadot_4x4x8_unsafe_cluster0_s8s8s32(a.data(), b.data(), ime.data(), false);
    int mismatch_count = 0;
    for (std::size_t i = 0; i < scalar.size(); ++i) {
        if (scalar[i] != ime[i]) {
            ++mismatch_count;
        }
    }

    std::vector<double> scalar_runs;
    std::vector<double> ime_runs;
    std::int64_t scalar_guard = 0;
    std::int64_t ime_guard = 0;
    for (int r = 0; r < repeats; ++r) {
        scalar_runs.push_back(run_timed(iterations, [&]() {
            y26_vmadot_4x4x8_scalar_s8s8s32(a.data(), b.data(), scalar.data(), false);
            std::int64_t sum = 0;
            for (const auto value : scalar) {
                sum += value;
            }
            return sum;
        }, scalar_guard));
        if (ime_status == Y26_CONV_STATUS_SUCCESS) {
            ime_runs.push_back(run_timed(iterations, [&]() {
                y26_k1x_vmadot_4x4x8_unsafe_cluster0_s8s8s32(a.data(), b.data(), ime.data(), false);
                std::int64_t sum = 0;
                for (const auto value : ime) {
                    sum += value;
                }
                return sum;
            }, ime_guard));
        }
    }
    const double scalar_mean = mean(scalar_runs) * 1000.0;
    const double ime_mean = mean(ime_runs) * 1000.0;
    std::printf("case=%s\n", name);
    std::printf("shape=mmt4d_4x4x8\n");
    std::printf("layout=A_4x8_row_major_B_4x8_transposed_NK_C_4x4_s32\n");
    std::printf("packing_included=no\n");
    std::printf("ime_status=%d\n", ime_status);
    std::printf("mismatches=%d\n", mismatch_count);
    std::printf("scalar_mean_ns_per_call=%.3f\n", scalar_mean);
    if (ime_status == Y26_CONV_STATUS_SUCCESS) {
        std::printf("ime_mean_ns_per_call=%.3f\n", ime_mean);
        if (ime_mean > 0.0) {
            std::printf("speedup_vs_scalar=%.3f\n", scalar_mean / ime_mean);
        }
    }
    std::printf("scalar_guard=%lld\n", static_cast<long long>(scalar_guard));
    std::printf("ime_guard=%lld\n", static_cast<long long>(ime_guard));
}

}  // namespace

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 200;
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 5;

    std::printf("benchmark_scope=conv_kernel_only_not_yolo26_inference\n");
    std::printf("iterations=%d\n", iterations);
    std::printf("repeats=%d\n", repeats);
    std::printf("kernel_level_only=yes\n");

    const bool ime_build = y26_vmadot_4x4x8_ime_available_buildtime();
    const bool ime_ready = ime_build && y26_k1x_ime_hotpath_allowed_on_current_cpu();
    std::printf("ime_buildtime=%d\n", ime_build ? 1 : 0);
    std::printf("ime_ready=%d\n", ime_ready ? 1 : 0);
    if (ime_ready) {
        report_tile_core("conv1x1_mmt4d_tile_core", iterations * 100, repeats);
        report_tile_core("conv3x3_mmt4d_tile_core", iterations * 100, repeats);
    }

    {
        const Y26Conv2DParams params {8, 8, 16, 16, 1, 1, 0, 0};
        const int output_h = y26_conv1x1_output_h(&params);
        const int output_w = y26_conv1x1_output_w(&params);
        auto input = make_i8(static_cast<std::size_t>(params.input_h * params.input_w * params.input_c), 501U);
        auto weights = make_i8(static_cast<std::size_t>(params.output_c * params.input_c), 502U);
        auto bias = make_bias(params.output_c);
        std::vector<std::int32_t> scalar(static_cast<std::size_t>(output_h * output_w * params.output_c));
        std::vector<std::int32_t> ime(scalar.size());
        y26_conv1x1_i8s8s32_nhwc_scalar(input.data(), weights.data(), bias.data(), scalar.data(), &params);
        const int ime_status =
            y26_conv1x1_i8s8s32_nhwc_ime(input.data(), weights.data(), bias.data(), ime.data(), &params);

        std::vector<double> scalar_runs;
        std::vector<double> ime_runs;
        std::int64_t scalar_guard = 0;
        std::int64_t ime_guard = 0;
        for (int r = 0; r < repeats; ++r) {
            scalar_runs.push_back(run_timed(iterations, [&]() {
                y26_conv1x1_i8s8s32_nhwc_scalar(input.data(), weights.data(), bias.data(), scalar.data(), &params);
                return checksum(scalar);
            }, scalar_guard));
            if (ime_status == Y26_CONV_STATUS_SUCCESS) {
                ime_runs.push_back(run_timed(iterations, [&]() {
                    y26_conv1x1_i8s8s32_nhwc_ime(input.data(), weights.data(), bias.data(), ime.data(), &params);
                    return checksum(ime);
                }, ime_guard));
            }
        }
        const double scalar_mean = mean(scalar_runs);
        const double ime_mean = mean(ime_runs);
        report_case("conv1x1",
                    params,
                    output_h,
                    output_w,
                    ime_status,
                    ime_status == Y26_CONV_STATUS_SUCCESS ? mismatches(scalar, ime) : -1,
                    scalar_mean,
                    stddev(scalar_runs, scalar_mean),
                    ime_mean,
                    stddev(ime_runs, ime_mean),
                    scalar_guard,
                    ime_guard);
    }

    {
        const Y26Conv2DParams params {8, 8, 16, 16, 1, 1, 1, 1};
        const int output_h = y26_conv3x3_output_h(&params);
        const int output_w = y26_conv3x3_output_w(&params);
        auto input = make_i8(static_cast<std::size_t>(params.input_h * params.input_w * params.input_c), 601U);
        auto weights = make_i8(static_cast<std::size_t>(params.output_c * 3 * 3 * params.input_c), 602U);
        auto bias = make_bias(params.output_c);
        std::vector<std::int32_t> scalar(static_cast<std::size_t>(output_h * output_w * params.output_c));
        std::vector<std::int32_t> ime(scalar.size());
        y26_conv3x3_i8s8s32_nhwc_scalar(input.data(), weights.data(), bias.data(), scalar.data(), &params);
        const int ime_status =
            y26_conv3x3_i8s8s32_nhwc_ime(input.data(), weights.data(), bias.data(), ime.data(), &params);

        std::vector<double> scalar_runs;
        std::vector<double> ime_runs;
        std::int64_t scalar_guard = 0;
        std::int64_t ime_guard = 0;
        for (int r = 0; r < repeats; ++r) {
            scalar_runs.push_back(run_timed(iterations, [&]() {
                y26_conv3x3_i8s8s32_nhwc_scalar(input.data(), weights.data(), bias.data(), scalar.data(), &params);
                return checksum(scalar);
            }, scalar_guard));
            if (ime_status == Y26_CONV_STATUS_SUCCESS) {
                ime_runs.push_back(run_timed(iterations, [&]() {
                    y26_conv3x3_i8s8s32_nhwc_ime(input.data(), weights.data(), bias.data(), ime.data(), &params);
                    return checksum(ime);
                }, ime_guard));
            }
        }
        const double scalar_mean = mean(scalar_runs);
        const double ime_mean = mean(ime_runs);
        report_case("conv3x3",
                    params,
                    output_h,
                    output_w,
                    ime_status,
                    ime_status == Y26_CONV_STATUS_SUCCESS ? mismatches(scalar, ime) : -1,
                    scalar_mean,
                    stddev(scalar_runs, scalar_mean),
                    ime_mean,
                    stddev(ime_runs, ime_mean),
                    scalar_guard,
                    ime_guard);
    }

    return 0;
}
