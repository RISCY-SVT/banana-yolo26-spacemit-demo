#include "y26_k1x_model5_island.h"
#include "y26_k1x_conv_kernels.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double process_cpu_us() {
    timespec value {};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) != 0) {
        throw std::runtime_error("clock_gettime(CLOCK_PROCESS_CPUTIME_ID) failed");
    }
    return static_cast<double>(value.tv_sec) * 1.0e6 + static_cast<double>(value.tv_nsec) / 1.0e3;
}

struct Options {
    std::string mode = "ime";
    std::string model4_preact_path;
    std::string expected_model4_postact_path;
    std::string expected_model5_path;
    std::string weights_path;
    std::string weight_scales_path;
    std::string bias_path;
    std::string output_path;
    std::string corrected_output_path;
    std::string island_entry_path;
    int threads = 4;
    int warmup = 10;
    int runs = 100;
    int repeats = 5;
    int accumulators = 4;
    int pack_timing = 0;
    int dataflow_mode = Y26_MODEL5_DATAFLOW_STAGE43_R0;
};

template <typename T>
std::vector<T> read_binary(const std::string& path, std::size_t expected_count) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open " + path);
    }
    const std::streamsize bytes = stream.tellg();
    if (bytes != static_cast<std::streamsize>(expected_count * sizeof(T))) {
        throw std::runtime_error("unexpected byte size for " + path + ": " + std::to_string(bytes));
    }
    stream.seekg(0);
    std::vector<T> values(expected_count);
    if (!stream.read(reinterpret_cast<char*>(values.data()), bytes)) {
        throw std::runtime_error("cannot read " + path);
    }
    return values;
}

void write_binary(const std::string& path, const std::vector<std::uint8_t>& values) {
    if (path.empty()) {
        return;
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream || !stream.write(reinterpret_cast<const char*>(values.data()),
                                 static_cast<std::streamsize>(values.size()))) {
        throw std::runtime_error("cannot write " + path);
    }
}

void write_i32_binary(const std::string& path, const std::int32_t* values, std::size_t count) {
    if (path.empty()) {
        return;
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream || !stream.write(reinterpret_cast<const char*>(values),
                                 static_cast<std::streamsize>(count * sizeof(std::int32_t)))) {
        throw std::runtime_error("cannot write " + path);
    }
}

std::vector<std::int8_t> nchw_u8_to_nhwc_s8(const std::vector<std::uint8_t>& input,
                                             int height,
                                             int width,
                                             int channels) {
    std::vector<std::int8_t> output(input.size());
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                const std::size_t nchw = static_cast<std::size_t>((c * height + h) * width + w);
                const std::size_t nhwc = static_cast<std::size_t>((h * width + w) * channels + c);
                output[nhwc] = static_cast<std::int8_t>(static_cast<int>(input[nchw]) - 128);
            }
        }
    }
    return output;
}

void nchw_u8_to_nhwc_u8_into(const std::uint8_t* input,
                              std::uint8_t* output,
                              int height,
                              int width,
                              int channels) {
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                const std::size_t nchw = static_cast<std::size_t>((c * height + h) * width + w);
                const std::size_t nhwc = static_cast<std::size_t>((h * width + w) * channels + c);
                output[nhwc] = input[nchw];
            }
        }
    }
}

void nhwc_s8_to_nchw_u8_into(const std::int8_t* input,
                              std::uint8_t* output,
                              int height,
                              int width,
                              int channels) {
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                const std::size_t nhwc = static_cast<std::size_t>((h * width + w) * channels + c);
                const std::size_t nchw = static_cast<std::size_t>((c * height + h) * width + w);
                output[nchw] = static_cast<std::uint8_t>(static_cast<int>(input[nhwc]) + 128);
            }
        }
    }
}

std::vector<std::uint8_t> nchw_u8_to_nhwc_u8(const std::vector<std::uint8_t>& input,
                                              int height,
                                              int width,
                                              int channels) {
    std::vector<std::uint8_t> output(input.size());
    nchw_u8_to_nhwc_u8_into(input.data(), output.data(), height, width, channels);
    return output;
}

std::vector<std::uint8_t> nhwc_s8_to_nchw_u8(const std::vector<std::int8_t>& input,
                                              int height,
                                              int width,
                                              int channels) {
    std::vector<std::uint8_t> output(input.size());
    nhwc_s8_to_nchw_u8_into(input.data(), output.data(), height, width, channels);
    return output;
}

struct Difference {
    std::size_t mismatches = 0;
    int max_abs_diff = 0;
    std::size_t first_mismatch = static_cast<std::size_t>(-1);
};

Difference compare(const std::int8_t* actual, const std::int8_t* expected, std::size_t count) {
    Difference result;
    for (std::size_t index = 0; index < count; ++index) {
        if (actual[index] != expected[index]) {
            if (result.mismatches == 0) {
                result.first_mismatch = index;
            }
            ++result.mismatches;
            result.max_abs_diff = std::max(
                result.max_abs_diff,
                std::abs(static_cast<int>(actual[index]) - static_cast<int>(expected[index])));
        }
    }
    return result;
}

double percentile(std::vector<double> values, double quantile) {
    std::sort(values.begin(), values.end());
    if (values.empty()) {
        return 0.0;
    }
    const double position = quantile * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(values.size() - 1, lower + 1);
    const double fraction = position - static_cast<double>(lower);
    return values[lower] + fraction * (values[upper] - values[lower]);
}

struct Context {
    Options options;
    std::vector<std::int8_t> weights;
    std::vector<float> weight_scales;
    std::vector<std::int32_t> bias;
    std::vector<std::uint8_t> model4_preact_nhwc;
    std::vector<std::int8_t> expected_model4_postact_nhwc;
    std::vector<std::int8_t> expected_model5_nhwc;
    std::vector<std::int8_t> actual_model5_nhwc;
    Y26Model5IslandConfig config {};
    Y26Model5IslandWorkspace workspace {};
};

Y26Model5IslandConfig make_config(Context& context) {
    Y26Model5IslandConfig config {};
    config.model5_conv.node_name = "/model.5/conv/Conv";
    config.model5_conv.params = Y26Conv2DParams{80, 80, 128, 128, 2, 2, 1, 1};
    config.model5_conv.kernel_h = 3;
    config.model5_conv.kernel_w = 3;
    config.model5_conv.activation_zero_point_u8 = 9;
    config.model5_conv.input_storage_zero_point_s8 = -119;
    config.model5_conv.input_scale = 0.030298452824354172F;
    config.model5_conv.output_scale = 0.057099778205156326F;
    config.model5_conv.output_zero_point_u8 = 136;
    config.model5_conv.weight_scales = context.weight_scales.data();
    config.model5_conv.weight_scale_count = context.weight_scales.size();
    config.model5_conv.weights_ohwi_s8 = context.weights.data();
    config.model5_conv.weight_count = context.weights.size();
    config.model5_conv.bias_i32 = context.bias.data();
    config.model5_conv.bias_count = context.bias.size();
    config.model4_preact_scale = 0.066064663231372833F;
    config.model4_preact_zero_point_u8 = 142;
    config.model4_postact_scale = 0.030298452824354172F;
    config.model4_postact_zero_point_u8 = 9;
    config.model5_postact_scale = 0.027727888897061348F;
    config.model5_postact_zero_point_u8 = 10;
    config.ime_accumulator_groups = context.options.accumulators;
    config.dataflow_mode = context.options.dataflow_mode;
    return config;
}

Context make_context(const Options& options) {
    constexpr std::size_t model4_count = 80U * 80U * 128U;
    constexpr std::size_t model5_count = 40U * 40U * 128U;
    Context context;
    context.options = options;
    context.weights = read_binary<std::int8_t>(options.weights_path, 128U * 3U * 3U * 128U);
    context.weight_scales = read_binary<float>(options.weight_scales_path, 128);
    context.bias = read_binary<std::int32_t>(options.bias_path, 128);
    const auto preact = read_binary<std::uint8_t>(options.model4_preact_path, model4_count);
    const auto expected_post4 = read_binary<std::uint8_t>(options.expected_model4_postact_path, model4_count);
    const auto expected_model5 = read_binary<std::uint8_t>(options.expected_model5_path, model5_count);
    context.model4_preact_nhwc = nchw_u8_to_nhwc_u8(preact, 80, 80, 128);
    context.expected_model4_postact_nhwc = nchw_u8_to_nhwc_s8(expected_post4, 80, 80, 128);
    context.expected_model5_nhwc = nchw_u8_to_nhwc_s8(expected_model5, 40, 40, 128);
    context.actual_model5_nhwc.resize(model5_count);
    context.config = make_config(context);
    if (y26_model5_island_workspace_init(&context.workspace) != Y26_CONV_STATUS_SUCCESS) {
        throw std::runtime_error("model5 workspace init failed");
    }
    const int status = y26_model5_island_prepare(&context.config, options.threads, &context.workspace);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        throw std::runtime_error("model5 prepare failed status=" + std::to_string(status));
    }
    return context;
}

int run_once(Context& context, bool use_ime, Y26Model5IslandTimingUs* timing) {
    return use_ime ? y26_model5_island_run_ime_cluster0(&context.config,
                                                        &context.workspace,
                                                        context.model4_preact_nhwc.data(),
                                                        context.actual_model5_nhwc.data(),
                                                        timing)
                   : y26_model5_island_run_scalar(&context.config,
                                                  &context.workspace,
                                                  context.model4_preact_nhwc.data(),
                                                  context.actual_model5_nhwc.data(),
                                                  timing);
}

int validate(Context& context, bool use_ime, const char* label) {
    Y26Model5IslandTimingUs timing {};
    const int status = run_once(context, use_ime, &timing);
    const Difference post4 = compare(context.workspace.model4_postact_nhwc_s8,
                                     context.expected_model4_postact_nhwc.data(),
                                     context.expected_model4_postact_nhwc.size());
    const Difference output = compare(context.actual_model5_nhwc.data(),
                                      context.expected_model5_nhwc.data(),
                                      context.expected_model5_nhwc.size());
    std::cout << "stage43_model5_validation route=" << label << " status=" << status
              << " model4_postact_mismatches=" << post4.mismatches
              << " model4_postact_max_abs_diff=" << post4.max_abs_diff
              << " model5_mismatches=" << output.mismatches
              << " model5_max_abs_diff=" << output.max_abs_diff
              << " model5_first_mismatch=" << output.first_mismatch
              << " affinity_ok=" << y26_model5_island_worker_affinity_ok(&context.workspace)
              << " thread_count=" << y26_model5_island_thread_count(&context.workspace)
              << " persistent_workspace_bytes=" << context.workspace.workspace_bytes
              << " model4_postact_us=" << timing.model4_postact_us
              << " model5_conv_us=" << timing.model5_conv_us
              << " model5_postact_us=" << timing.model5_postact_us
              << " total_us=" << timing.total_us << "\n";
    write_binary(context.options.output_path, nhwc_s8_to_nchw_u8(context.actual_model5_nhwc, 40, 40, 128));
    write_i32_binary(context.options.corrected_output_path,
                     context.workspace.model5_corrected_nhwc_i32,
                     context.workspace.model5_element_count);
    if (status == Y26_CONV_STATUS_SUCCESS && post4.mismatches == 0 && output.mismatches == 0) {
        return 0;
    }
    return 1;
}

int benchmark(Context& context, bool use_ime, const char* label) {
    if (validate(context, use_ime, label) != 0) {
        return 1;
    }
    for (int iteration = 0; iteration < context.options.warmup; ++iteration) {
        if (run_once(context, use_ime, nullptr) != Y26_CONV_STATUS_SUCCESS) {
            return 1;
        }
    }
    std::vector<double> repeat_means;
    std::vector<double> process_cpu_repeat_means;
    for (int repeat = 0; repeat < context.options.repeats; ++repeat) {
        Y26Model5IslandTimingUs sums {};
        const auto begin = std::chrono::steady_clock::now();
        const double process_cpu_begin_us = process_cpu_us();
        for (int run = 0; run < context.options.runs; ++run) {
            Y26Model5IslandTimingUs timing {};
            if (run_once(context, use_ime, &timing) != Y26_CONV_STATUS_SUCCESS) {
                return 1;
            }
            sums.model4_postact_us += timing.model4_postact_us;
            sums.model5_conv_us += timing.model5_conv_us;
            sums.model5_im2col_pack_us += timing.model5_im2col_pack_us;
            sums.model5_compute_us += timing.model5_compute_us;
            sums.model5_correction_us += timing.model5_correction_us;
            sums.model5_thread_overhead_us += timing.model5_thread_overhead_us;
            sums.model5_postact_us += timing.model5_postact_us;
            sums.total_us += timing.total_us;
        }
        const double process_cpu_end_us = process_cpu_us();
        const auto end = std::chrono::steady_clock::now();
        const double divisor = static_cast<double>(context.options.runs);
        const double wall_mean = static_cast<double>(
                                     std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) /
                                 1000.0 / divisor;
        repeat_means.push_back(wall_mean);
        process_cpu_repeat_means.push_back((process_cpu_end_us - process_cpu_begin_us) / divisor);
        std::cout << "stage43_model5_repeat route=" << label << " repeat=" << repeat
                  << " runs=" << context.options.runs << " wall_mean_us=" << wall_mean
                  << " process_cpu_mean_us=" << process_cpu_repeat_means.back()
                  << " model4_postact_mean_us=" << sums.model4_postact_us / divisor
                  << " model5_conv_mean_us=" << sums.model5_conv_us / divisor
                  << " model5_im2col_pack_mean_us=" << sums.model5_im2col_pack_us / divisor
                  << " model5_compute_mean_us=" << sums.model5_compute_us / divisor
                  << " model5_correction_mean_us=" << sums.model5_correction_us / divisor
                  << " model5_thread_overhead_mean_us=" << sums.model5_thread_overhead_us / divisor
                  << " model5_postact_mean_us=" << sums.model5_postact_us / divisor
                  << " internal_total_mean_us=" << sums.total_us / divisor << "\n";
    }
    const double mean = std::accumulate(repeat_means.begin(), repeat_means.end(), 0.0) /
                        static_cast<double>(repeat_means.size());
    double variance = 0.0;
    for (const double value : repeat_means) {
        variance += (value - mean) * (value - mean);
    }
    variance /= static_cast<double>(repeat_means.size());
    std::cout << "stage43_model5_summary route=" << label << " mean_us=" << mean
              << " stddev_us=" << std::sqrt(variance)
              << " cv_pct=" << (mean > 0.0 ? 100.0 * std::sqrt(variance) / mean : 0.0)
              << " min_us=" << *std::min_element(repeat_means.begin(), repeat_means.end())
              << " max_us=" << *std::max_element(repeat_means.begin(), repeat_means.end())
              << " median_us=" << percentile(repeat_means, 0.5)
              << " p90_us=" << percentile(repeat_means, 0.9)
              << " p95_us=" << percentile(repeat_means, 0.95)
              << " process_cpu_mean_us="
              << std::accumulate(process_cpu_repeat_means.begin(), process_cpu_repeat_means.end(), 0.0) /
                     static_cast<double>(process_cpu_repeat_means.size())
              << " process_cpu_median_us=" << percentile(process_cpu_repeat_means, 0.5)
              << " pack_timing_enabled=" << context.options.pack_timing
              << " dataflow_mode=" << context.options.dataflow_mode << "\n";
    return 0;
}

int benchmark_dataflow_pair(Context& context) {
    context.config.dataflow_mode = Y26_MODEL5_DATAFLOW_STAGE43_R0;
    if (validate(context, true, "r0-paired-control") != 0) {
        return 1;
    }
    context.config.dataflow_mode = Y26_MODEL5_DATAFLOW_STAGE44_STRIDE2_FASTPACK;
    if (validate(context, true, "r2-fastpack-paired-control") != 0) {
        return 1;
    }
    auto run_mode = [&](int mode) -> double {
        context.config.dataflow_mode = mode;
        const auto begin = std::chrono::steady_clock::now();
        if (run_once(context, true, nullptr) != Y26_CONV_STATUS_SUCCESS) {
            throw std::runtime_error("paired dataflow run failed");
        }
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(end - begin).count();
    };
    for (int warmup = 0; warmup < context.options.warmup; ++warmup) {
        if ((warmup & 1) == 0) {
            (void)run_mode(Y26_MODEL5_DATAFLOW_STAGE43_R0);
            (void)run_mode(Y26_MODEL5_DATAFLOW_STAGE44_STRIDE2_FASTPACK);
        } else {
            (void)run_mode(Y26_MODEL5_DATAFLOW_STAGE44_STRIDE2_FASTPACK);
            (void)run_mode(Y26_MODEL5_DATAFLOW_STAGE43_R0);
        }
    }
    std::vector<double> r0_means;
    std::vector<double> r2_means;
    std::vector<double> deltas;
    for (int repeat = 0; repeat < context.options.repeats; ++repeat) {
        double r0_sum = 0.0;
        double r2_sum = 0.0;
        for (int run = 0; run < context.options.runs; ++run) {
            if (((repeat + run) & 1) == 0) {
                r0_sum += run_mode(Y26_MODEL5_DATAFLOW_STAGE43_R0);
                r2_sum += run_mode(Y26_MODEL5_DATAFLOW_STAGE44_STRIDE2_FASTPACK);
            } else {
                r2_sum += run_mode(Y26_MODEL5_DATAFLOW_STAGE44_STRIDE2_FASTPACK);
                r0_sum += run_mode(Y26_MODEL5_DATAFLOW_STAGE43_R0);
            }
        }
        const double divisor = static_cast<double>(context.options.runs);
        r0_means.push_back(r0_sum / divisor);
        r2_means.push_back(r2_sum / divisor);
        deltas.push_back(r2_means.back() - r0_means.back());
        std::cout << "stage44_model5_dataflow_pair_repeat repeat=" << repeat
                  << " r0_mean_us=" << r0_means.back()
                  << " r2_fastpack_mean_us=" << r2_means.back()
                  << " delta_us=" << deltas.back()
                  << " order=" << ((repeat & 1) == 0 ? "ABBA" : "BAAB") << "\n";
    }
    const auto mean_of = [](const std::vector<double>& values) {
        return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    };
    const double r0_mean = mean_of(r0_means);
    const double r2_mean = mean_of(r2_means);
    const double delta_mean = mean_of(deltas);
    double delta_variance = 0.0;
    for (const double delta : deltas) {
        delta_variance += (delta - delta_mean) * (delta - delta_mean);
    }
    delta_variance /= static_cast<double>(deltas.size());
    std::cout << "stage44_model5_dataflow_pair_summary"
              << " warmup=" << context.options.warmup
              << " runs=" << context.options.runs
              << " repeats=" << context.options.repeats
              << " r0_mean_us=" << r0_mean
              << " r2_fastpack_mean_us=" << r2_mean
              << " delta_mean_us=" << delta_mean
              << " delta_pct=" << (r0_mean > 0.0 ? 100.0 * delta_mean / r0_mean : 0.0)
              << " delta_stddev_us=" << std::sqrt(delta_variance)
              << " affinity_ok=" << y26_model5_island_worker_affinity_ok(&context.workspace)
              << " exact_controls=1"
              << " alternating_order=1\n";
    return y26_model5_island_worker_affinity_ok(&context.workspace) == 1 ? 0 : 1;
}

int benchmark_adapters(Context& context) {
    constexpr std::size_t entry_count = 80U * 80U * 64U;
    if (context.options.island_entry_path.empty()) {
        throw std::runtime_error("--island-entry is required for benchmark-adapters");
    }
    const std::vector<std::uint8_t> entry_nchw =
        read_binary<std::uint8_t>(context.options.island_entry_path, entry_count);
    std::vector<std::uint8_t> entry_nhwc(entry_count);
    std::vector<std::uint8_t> exit_nchw(context.actual_model5_nhwc.size());
    if (run_once(context, true, nullptr) != Y26_CONV_STATUS_SUCCESS) {
        return 1;
    }
    auto run_adapters = [&]() {
        const auto begin = std::chrono::steady_clock::now();
        nchw_u8_to_nhwc_u8_into(entry_nchw.data(), entry_nhwc.data(), 80, 80, 64);
        const auto middle = std::chrono::steady_clock::now();
        nhwc_s8_to_nchw_u8_into(context.actual_model5_nhwc.data(), exit_nchw.data(), 40, 40, 128);
        const auto end = std::chrono::steady_clock::now();
        return std::array<double, 2>{
            std::chrono::duration<double, std::micro>(middle - begin).count(),
            std::chrono::duration<double, std::micro>(end - middle).count(),
        };
    };
    for (int iteration = 0; iteration < context.options.warmup; ++iteration) {
        (void)run_adapters();
    }
    std::uint64_t checksum = 0;
    for (int repeat = 0; repeat < context.options.repeats; ++repeat) {
        double entry_sum = 0.0;
        double exit_sum = 0.0;
        for (int run = 0; run < context.options.runs; ++run) {
            const auto timing = run_adapters();
            entry_sum += timing[0];
            exit_sum += timing[1];
        }
        for (const std::uint8_t value : entry_nhwc) checksum += value;
        for (const std::uint8_t value : exit_nchw) checksum += value;
        std::cout << "stage43_adapter_repeat repeat=" << repeat << " runs=" << context.options.runs
                  << " entry_mean_us=" << entry_sum / context.options.runs
                  << " exit_mean_us=" << exit_sum / context.options.runs
                  << " entry_bytes_read_write=" << 2U * entry_count
                  << " exit_bytes_read_write=" << 2U * exit_nchw.size()
                  << " checksum=" << checksum << "\n";
    }
    return 0;
}

#if defined(__riscv)
unsigned read_frm() {
    unsigned value = 0;
    asm volatile("frrm %0" : "=r"(value));
    return value & 7U;
}

void set_frm(unsigned value) {
    switch (value) {
        case 0: asm volatile("fsrmi 0" ::: "memory"); break;
        case 1: asm volatile("fsrmi 1" ::: "memory"); break;
        case 2: asm volatile("fsrmi 2" ::: "memory"); break;
        case 3: asm volatile("fsrmi 3" ::: "memory"); break;
        case 4: asm volatile("fsrmi 4" ::: "memory"); break;
        default: asm volatile("fsrmi 0" ::: "memory"); break;
    }
}
#endif

int frm_sweep(Context& context) {
#if defined(__riscv)
    const unsigned saved = read_frm();
    int failures = 0;
    for (unsigned frm : {0U, 1U, 2U, 3U, 4U}) {
        set_frm(frm);
        Y26Model5IslandTimingUs timing {};
        const int run_status = run_once(context, true, &timing);
        const Difference post4 = compare(context.workspace.model4_postact_nhwc_s8,
                                         context.expected_model4_postact_nhwc.data(),
                                         context.expected_model4_postact_nhwc.size());
        const Difference output = compare(context.actual_model5_nhwc.data(),
                                          context.expected_model5_nhwc.data(),
                                          context.expected_model5_nhwc.size());
        const unsigned after = read_frm();
        set_frm(saved <= 4U ? saved : 0U);
        const int pass = run_status == Y26_CONV_STATUS_SUCCESS && post4.mismatches == 0 &&
                         output.mismatches == 0 && after == frm;
        std::cout << "stage43_model5_frm ambient=" << frm << " run_status=" << run_status
                  << " model4_postact_mismatches=" << post4.mismatches
                  << " model5_mismatches=" << output.mismatches << " after=" << after
                  << " pass=" << pass << "\n";
        failures += pass ? 0 : 1;
    }
    set_frm(saved <= 4U ? saved : 0U);
    return failures == 0 ? 0 : 1;
#else
    (void)context;
    std::cout << "stage43_model5_frm skipped_non_riscv\n";
    return 0;
#endif
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next = [&]() -> std::string {
            if (++index >= argc) {
                throw std::runtime_error("missing value after " + argument);
            }
            return argv[index];
        };
        if (argument == "--mode") options.mode = next();
        else if (argument == "--model4-preact") options.model4_preact_path = next();
        else if (argument == "--expected-model4-postact") options.expected_model4_postact_path = next();
        else if (argument == "--expected-model5") options.expected_model5_path = next();
        else if (argument == "--weights") options.weights_path = next();
        else if (argument == "--weight-scales") options.weight_scales_path = next();
        else if (argument == "--bias") options.bias_path = next();
        else if (argument == "--output") options.output_path = next();
        else if (argument == "--corrected-output") options.corrected_output_path = next();
        else if (argument == "--island-entry") options.island_entry_path = next();
        else if (argument == "--threads") options.threads = std::stoi(next());
        else if (argument == "--warmup") options.warmup = std::stoi(next());
        else if (argument == "--runs") options.runs = std::stoi(next());
        else if (argument == "--repeats") options.repeats = std::stoi(next());
        else if (argument == "--accumulators") options.accumulators = std::stoi(next());
        else if (argument == "--pack-timing") options.pack_timing = std::stoi(next());
        else if (argument == "--dataflow") {
            const std::string value = next();
            if (value == "r0") options.dataflow_mode = Y26_MODEL5_DATAFLOW_STAGE43_R0;
            else if (value == "r2-fastpack") {
                options.dataflow_mode = Y26_MODEL5_DATAFLOW_STAGE44_STRIDE2_FASTPACK;
            } else {
                throw std::runtime_error("--dataflow must be r0 or r2-fastpack");
            }
        }
        else throw std::runtime_error("unknown argument " + argument);
    }
    if (options.model4_preact_path.empty() || options.expected_model4_postact_path.empty() ||
        options.expected_model5_path.empty() || options.weights_path.empty() ||
        options.weight_scales_path.empty() || options.bias_path.empty()) {
        throw std::runtime_error("missing required tensor or asset argument");
    }
    if (options.pack_timing != 0 && options.pack_timing != 1) {
        throw std::runtime_error("--pack-timing must be 0 or 1");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        y26_conv_mmt4d_set_stage38_pack_timing_enabled(options.pack_timing);
        Context context = make_context(options);
        int result = 1;
        if (options.mode == "scalar") result = validate(context, false, "scalar");
        else if (options.mode == "ime") result = validate(context, true, "ime");
        else if (options.mode == "benchmark-scalar") result = benchmark(context, false, "scalar");
        else if (options.mode == "benchmark-ime") result = benchmark(context, true, "ime");
        else if (options.mode == "benchmark-dataflow-pair") result = benchmark_dataflow_pair(context);
        else if (options.mode == "frm-sweep") result = frm_sweep(context);
        else if (options.mode == "benchmark-adapters") result = benchmark_adapters(context);
        else throw std::runtime_error("unsupported mode " + options.mode);
        y26_model5_island_release(&context.workspace);
        y26_conv_mmt4d_set_stage38_pack_timing_enabled(0);
        return result;
    } catch (const std::exception& error) {
        std::cerr << "stage43_model5_error " << error.what() << "\n";
        return 2;
    }
}
