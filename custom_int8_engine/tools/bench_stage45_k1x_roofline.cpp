#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string mode = "all";
    int repeats = 5;
    double min_ms = 80.0;
};

struct Stats {
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;
    double median = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
};

volatile std::uint64_t g_sink = 0;

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[++i];
        };
        if (arg == "--mode") options.mode = next();
        else if (arg == "--repeats") options.repeats = std::max(1, std::atoi(next()));
        else if (arg == "--min-ms") options.min_ms = std::max(1.0, std::atof(next()));
        else if (arg == "--help") {
            std::cout << "usage: " << argv[0] << " [--mode all|primitives|microkernel]"
                      << " [--repeats N] [--min-ms N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.mode != "all" && options.mode != "primitives" && options.mode != "microkernel") {
        throw std::runtime_error("--mode must be all, primitives, or microkernel");
    }
    return options;
}

double percentile(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    const double pos = q * static_cast<double>(sorted.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    const double fraction = pos - static_cast<double>(lo);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * fraction;
}

Stats summarize(std::vector<double> values) {
    Stats result;
    if (values.empty()) return result;
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    result.min = *std::min_element(values.begin(), values.end());
    result.max = *std::max_element(values.begin(), values.end());
    if (values.size() > 1) {
        double sum = 0.0;
        for (double value : values) {
            const double delta = value - result.mean;
            sum += delta * delta;
        }
        result.stddev = std::sqrt(sum / static_cast<double>(values.size() - 1));
    }
    std::sort(values.begin(), values.end());
    result.median = percentile(values, 0.5);
    result.p90 = percentile(values, 0.9);
    result.p95 = percentile(values, 0.95);
    return result;
}

std::uint64_t hash_bytes(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

template <typename Fn>
std::pair<Stats, std::size_t> measure(const Options& options, Fn&& fn) {
    fn();
    std::size_t loops = 1;
    while (loops < (1U << 20)) {
        const auto begin = Clock::now();
        for (std::size_t i = 0; i < loops; ++i) fn();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        if (elapsed_ms >= options.min_ms * 0.2) {
            const double target = options.min_ms / std::max(0.001, elapsed_ms);
            loops = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(loops * target)));
            break;
        }
        loops *= 2;
    }
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(options.repeats));
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        const auto begin = Clock::now();
        for (std::size_t i = 0; i < loops; ++i) fn();
        const double elapsed_us = std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
        samples.push_back(elapsed_us / static_cast<double>(loops));
    }
    return {summarize(samples), loops};
}

void print_row(const char* category,
               const std::string& name,
               const std::string& working_set,
               std::uint64_t operations,
               std::uint64_t bytes,
               const Stats& stats,
               std::size_t loops,
               std::uint64_t checksum) {
    const double seconds = stats.mean * 1.0e-6;
    const double gops = operations == 0 || seconds == 0.0 ? 0.0 : static_cast<double>(operations) / seconds / 1.0e9;
    const double gbps = bytes == 0 || seconds == 0.0 ? 0.0 : static_cast<double>(bytes) / seconds / 1.0e9;
    const double cv = stats.mean == 0.0 ? 0.0 : stats.stddev / stats.mean * 100.0;
    std::cout << category << '\t' << name << '\t' << working_set << '\t'
              << operations << '\t' << bytes << '\t' << loops << '\t'
              << std::fixed << std::setprecision(6)
              << stats.mean << '\t' << stats.stddev << '\t' << cv << '\t'
              << stats.min << '\t' << stats.max << '\t' << stats.median << '\t'
              << stats.p90 << '\t' << stats.p95 << '\t' << gops << '\t' << gbps << '\t'
              << checksum << '\n';
}

std::int32_t requant_rne(std::int32_t value, std::int32_t multiplier, int shift) {
    const std::int64_t product = static_cast<std::int64_t>(value) * multiplier;
    if (shift <= 0) return static_cast<std::int32_t>(product << (-shift));
    const std::int64_t denominator = std::int64_t{1} << shift;
    std::int64_t quotient = product / denominator;
    std::int64_t remainder = product % denominator;
    if (remainder < 0) remainder = -remainder;
    const std::int64_t half = denominator / 2;
    if (remainder > half || (remainder == half && (quotient & 1) != 0)) quotient += product >= 0 ? 1 : -1;
    return static_cast<std::int32_t>(quotient);
}

void run_primitives(const Options& options) {
    std::cout << "category\tname\tworking_set\toperations\tcontract_bytes\tloops\tmean_us\tstddev_us\tcv_pct"
                 "\tmin_us\tmax_us\tmedian_us\tp90_us\tp95_us\tgops_or_gmacs\tgb_per_s\tchecksum\n";
    const std::array<std::size_t, 4> sizes = {32U << 10, 512U << 10, 8U << 20, 32U << 20};
    for (std::size_t size : sizes) {
        std::vector<std::uint8_t> source(size);
        std::vector<std::uint8_t> destination(size);
        for (std::size_t i = 0; i < size; ++i) source[i] = static_cast<std::uint8_t>((i * 131U + 17U) & 255U);
        const std::string label = std::to_string(size);

        std::uint64_t read_sum = 0;
        auto read = measure(options, [&]() {
            std::uint64_t local = 0;
            for (std::uint8_t value : source) local += value;
            read_sum ^= local;
            g_sink ^= local;
        });
        print_row("memory", "sequential_read", label, size, size, read.first, read.second, read_sum);

        std::uint8_t pattern = 0;
        auto write = measure(options, [&]() {
            std::fill(destination.begin(), destination.end(), ++pattern);
            g_sink ^= destination[size / 2];
        });
        print_row("memory", "sequential_write", label, size, size, write.first, write.second,
                  hash_bytes(destination.data(), destination.size()));

        auto copy = measure(options, [&]() {
            std::memcpy(destination.data(), source.data(), size);
            g_sink ^= destination[size / 3];
        });
        print_row("memory", "memcpy", label, size, size * 2U, copy.first, copy.second,
                  hash_bytes(destination.data(), destination.size()));
    }

    constexpr int c = 128;
    constexpr int h = 80;
    constexpr int w = 80;
    constexpr std::size_t count = static_cast<std::size_t>(c) * h * w;
    std::vector<std::uint8_t> nchw(count);
    std::vector<std::uint8_t> nhwc(count);
    for (std::size_t i = 0; i < count; ++i) nchw[i] = static_cast<std::uint8_t>((i * 29U + 3U) & 255U);
    auto transpose = measure(options, [&]() {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                for (int channel = 0; channel < c; ++channel) {
                    nhwc[(static_cast<std::size_t>(y) * w + x) * c + channel] =
                        nchw[(static_cast<std::size_t>(channel) * h + y) * w + x];
                }
            }
        }
        g_sink ^= nhwc[count / 2];
    });
    print_row("layout", "nchw_to_nhwc_u8", "1x128x80x80", count, count * 2U,
              transpose.first, transpose.second, hash_bytes(nhwc.data(), nhwc.size()));

    std::array<std::uint8_t, 256> lut {};
    for (std::size_t i = 0; i < lut.size(); ++i) lut[i] = static_cast<std::uint8_t>((i * 7U + 19U) & 255U);
    auto lut_result = measure(options, [&]() {
        for (std::size_t i = 0; i < count; ++i) nhwc[i] = lut[nchw[i]];
        g_sink ^= nhwc[count / 4];
    });
    print_row("activation", "uint8_lut", "819200_codes", count, count * 2U,
              lut_result.first, lut_result.second, hash_bytes(nhwc.data(), nhwc.size()));

    constexpr std::size_t requant_count = 128U * 40U * 40U;
    std::vector<std::int32_t> accum(requant_count);
    std::vector<std::uint8_t> requantized(requant_count);
    std::array<std::int32_t, 128> multipliers {};
    std::array<int, 128> shifts {};
    for (std::size_t i = 0; i < requant_count; ++i) accum[i] = static_cast<std::int32_t>((i * 7919U) % 16000001U) - 8000000;
    for (int channel = 0; channel < 128; ++channel) {
        multipliers[channel] = 1048576 + channel * 137;
        shifts[channel] = 22 + channel % 3;
    }
    auto requant = measure(options, [&]() {
        for (std::size_t i = 0; i < requant_count; ++i) {
            const int channel = static_cast<int>(i % 128U);
            const std::int32_t value = requant_rne(accum[i], multipliers[channel], shifts[channel]);
            requantized[i] = static_cast<std::uint8_t>(std::clamp(value + 128, 0, 255));
        }
        g_sink ^= requantized[requant_count / 2];
    });
    print_row("requant", "exact_fixed_rne_u8", "40x40x128", requant_count,
              requant_count * 5U, requant.first, requant.second,
              hash_bytes(requantized.data(), requantized.size()));

    constexpr int in_h = 80;
    constexpr int in_w = 80;
    constexpr int channels = 128;
    constexpr int out_h = 40;
    constexpr int out_w = 40;
    constexpr int kernel = 3;
    constexpr int tile_pixels = 4;
    constexpr std::size_t panel_size = tile_pixels * kernel * kernel * channels;
    std::vector<std::uint8_t> input(static_cast<std::size_t>(in_h) * in_w * channels);
    std::vector<std::uint8_t> panel(panel_size);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<std::uint8_t>((i * 31U + 11U) & 255U);
    std::uint64_t panel_checksum = 0;
    auto gather = measure(options, [&]() {
        std::size_t output_index = 0;
        for (int oy = 0; oy < out_h; ++oy) {
            for (int ox0 = 0; ox0 < out_w; ox0 += tile_pixels) {
                std::size_t cursor = 0;
                for (int pixel = 0; pixel < tile_pixels; ++pixel) {
                    const int ox = std::min(ox0 + pixel, out_w - 1);
                    for (int ky = 0; ky < kernel; ++ky) {
                        const int iy = oy * 2 + ky - 1;
                        for (int kx = 0; kx < kernel; ++kx) {
                            const int ix = ox * 2 + kx - 1;
                            if (iy < 0 || iy >= in_h || ix < 0 || ix >= in_w) {
                                std::fill_n(panel.data() + cursor, channels, std::uint8_t{128});
                            } else {
                                const auto* src = input.data() + (static_cast<std::size_t>(iy) * in_w + ix) * channels;
                                std::memcpy(panel.data() + cursor, src, channels);
                            }
                            cursor += channels;
                        }
                    }
                }
                panel_checksum ^= hash_bytes(panel.data(), panel.size()) + output_index++;
            }
        }
        g_sink ^= panel_checksum;
    });
    const std::uint64_t packed_bytes = static_cast<std::uint64_t>(out_h) * ((out_w + tile_pixels - 1) / tile_pixels) * panel_size;
    print_row("packing", "stride2_small_panel_reuse", "80x80x128_to_40x40", packed_bytes,
              packed_bytes * 2U, gather.first, gather.second, panel_checksum);
}

struct KernelShape {
    const char* name;
    int rows;
    int cols;
};

void scalar_block_grouped(const std::int8_t* a,
                          const std::int8_t* b,
                          int k_tiles,
                          int rows,
                          int cols,
                          std::int32_t* c) {
    std::vector<std::int32_t> row_major(static_cast<std::size_t>(rows) * cols, 0);
    const int column_groups = cols / 4;
    for (int kt = 0; kt < k_tiles; ++kt) {
        const std::int8_t* at = a + static_cast<std::size_t>(kt) * rows * 8;
        const std::int8_t* bt = b + static_cast<std::size_t>(kt) * cols * 8;
        for (int row = 0; row < rows; ++row) {
            for (int group = 0; group < column_groups; ++group) {
                const std::int8_t* bg = bt + group * 32;
                for (int column = 0; column < 4; ++column) {
                    std::int32_t sum = row_major[row * cols + group * 4 + column];
                    for (int kk = 0; kk < 8; ++kk) sum += static_cast<std::int32_t>(at[row * 8 + kk]) * bg[column * 8 + kk];
                    row_major[row * cols + group * 4 + column] = sum;
                }
            }
        }
    }
    std::size_t cursor = 0;
    for (int group = 0; group < column_groups; ++group) {
        for (int row_group = 0; row_group < rows / 4; ++row_group) {
            for (int row = 0; row < 4; ++row) {
                for (int column = 0; column < 4; ++column) {
                    c[cursor++] = row_major[(row_group * 4 + row) * cols + group * 4 + column];
                }
            }
        }
    }
}

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)

#define Y26_INIT_ACC(reg) "vxor.vv " #reg ", " #reg ", " #reg "\n\t"
#define Y26_DOT(acc, a, b) "smt.vmadot " #acc ", " #a ", " #b "\n\t"
#define Y26_STORE(reg) "vse32.v " #reg ", (t2)\n\t" "addi t2, t2, 64\n\t"

__attribute__((noinline)) void kernel_m4n16(const std::int8_t* a, const std::int8_t* b, int k_tiles, std::int32_t* c) {
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2\n\t"
        Y26_INIT_ACC(v16) Y26_INIT_ACC(v18) Y26_INIT_ACC(v20) Y26_INIT_ACC(v22)
        "vsetvli t0, zero, e8, m1\n\t"
        "mv t1, %[K]\n\t" "mv t3, %[A]\n\t" "mv t4, %[B]\n\t"
        "1:\n\t"
        "vle8.v v0, (t3)\n\t"
        "vle8.v v2, (t4)\n\t" "addi t5, t4, 32\n\t" "vle8.v v3, (t5)\n\t"
        "addi t5, t4, 64\n\t" "vle8.v v4, (t5)\n\t" "addi t5, t4, 96\n\t" "vle8.v v5, (t5)\n\t"
        Y26_DOT(v16, v0, v2) Y26_DOT(v18, v0, v3) Y26_DOT(v20, v0, v4) Y26_DOT(v22, v0, v5)
        "addi t3, t3, 32\n\t" "addi t4, t4, 128\n\t" "addi t1, t1, -1\n\t" "bnez t1, 1b\n\t"
        "vsetvli t0, zero, e32, m2\n\t" "mv t2, %[C]\n\t"
        Y26_STORE(v16) Y26_STORE(v18) Y26_STORE(v20) Y26_STORE(v22)
        : : [A] "r"(a), [B] "r"(b), [K] "r"(k_tiles), [C] "r"(c)
        : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5", "v0", "v2", "v3", "v4", "v5",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
}

__attribute__((noinline)) void kernel_m4n24(const std::int8_t* a, const std::int8_t* b, int k_tiles, std::int32_t* c) {
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2\n\t"
        Y26_INIT_ACC(v16) Y26_INIT_ACC(v18) Y26_INIT_ACC(v20) Y26_INIT_ACC(v22) Y26_INIT_ACC(v24) Y26_INIT_ACC(v26)
        "vsetvli t0, zero, e8, m1\n\t" "mv t1, %[K]\n\t" "mv t3, %[A]\n\t" "mv t4, %[B]\n\t"
        "1:\n\t" "vle8.v v0, (t3)\n\t"
        "vle8.v v2, (t4)\n\t" "addi t5, t4, 32\n\t" "vle8.v v3, (t5)\n\t"
        "addi t5, t4, 64\n\t" "vle8.v v4, (t5)\n\t" "addi t5, t4, 96\n\t" "vle8.v v5, (t5)\n\t"
        "addi t5, t4, 128\n\t" "vle8.v v6, (t5)\n\t" "addi t5, t4, 160\n\t" "vle8.v v7, (t5)\n\t"
        Y26_DOT(v16, v0, v2) Y26_DOT(v18, v0, v3) Y26_DOT(v20, v0, v4)
        Y26_DOT(v22, v0, v5) Y26_DOT(v24, v0, v6) Y26_DOT(v26, v0, v7)
        "addi t3, t3, 32\n\t" "addi t4, t4, 192\n\t" "addi t1, t1, -1\n\t" "bnez t1, 1b\n\t"
        "vsetvli t0, zero, e32, m2\n\t" "mv t2, %[C]\n\t"
        Y26_STORE(v16) Y26_STORE(v18) Y26_STORE(v20) Y26_STORE(v22) Y26_STORE(v24) Y26_STORE(v26)
        : : [A] "r"(a), [B] "r"(b), [K] "r"(k_tiles), [C] "r"(c)
        : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5", "v0", "v2", "v3", "v4", "v5", "v6", "v7",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
}

__attribute__((noinline)) void kernel_m8n16(const std::int8_t* a, const std::int8_t* b, int k_tiles, std::int32_t* c) {
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2\n\t"
        Y26_INIT_ACC(v16) Y26_INIT_ACC(v18) Y26_INIT_ACC(v20) Y26_INIT_ACC(v22)
        Y26_INIT_ACC(v24) Y26_INIT_ACC(v26) Y26_INIT_ACC(v28) Y26_INIT_ACC(v30)
        "vsetvli t0, zero, e8, m1\n\t" "mv t1, %[K]\n\t" "mv t3, %[A]\n\t" "mv t4, %[B]\n\t"
        "1:\n\t" "vle8.v v0, (t3)\n\t" "addi t5, t3, 32\n\t" "vle8.v v1, (t5)\n\t"
        "vle8.v v2, (t4)\n\t" "addi t5, t4, 32\n\t" "vle8.v v3, (t5)\n\t"
        "addi t5, t4, 64\n\t" "vle8.v v4, (t5)\n\t" "addi t5, t4, 96\n\t" "vle8.v v5, (t5)\n\t"
        Y26_DOT(v16, v0, v2) Y26_DOT(v18, v1, v2) Y26_DOT(v20, v0, v3) Y26_DOT(v22, v1, v3)
        Y26_DOT(v24, v0, v4) Y26_DOT(v26, v1, v4) Y26_DOT(v28, v0, v5) Y26_DOT(v30, v1, v5)
        "addi t3, t3, 64\n\t" "addi t4, t4, 128\n\t" "addi t1, t1, -1\n\t" "bnez t1, 1b\n\t"
        "vsetvli t0, zero, e32, m2\n\t" "mv t2, %[C]\n\t"
        Y26_STORE(v16) Y26_STORE(v18) Y26_STORE(v20) Y26_STORE(v22)
        Y26_STORE(v24) Y26_STORE(v26) Y26_STORE(v28) Y26_STORE(v30)
        : : [A] "r"(a), [B] "r"(b), [K] "r"(k_tiles), [C] "r"(c)
        : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5", "v0", "v1", "v2", "v3", "v4", "v5",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
          "v28", "v29", "v30", "v31");
}

__attribute__((noinline)) void kernel_m12n16(const std::int8_t* a, const std::int8_t* b, int k_tiles, std::int32_t* c) {
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2\n\t"
        Y26_INIT_ACC(v8) Y26_INIT_ACC(v10) Y26_INIT_ACC(v12) Y26_INIT_ACC(v14)
        Y26_INIT_ACC(v16) Y26_INIT_ACC(v18) Y26_INIT_ACC(v20) Y26_INIT_ACC(v22)
        Y26_INIT_ACC(v24) Y26_INIT_ACC(v26) Y26_INIT_ACC(v28) Y26_INIT_ACC(v30)
        "vsetvli t0, zero, e8, m1\n\t" "mv t1, %[K]\n\t" "mv t3, %[A]\n\t" "mv t4, %[B]\n\t"
        "1:\n\t" "vle8.v v0, (t3)\n\t" "addi t5, t3, 32\n\t" "vle8.v v1, (t5)\n\t"
        "addi t5, t3, 64\n\t" "vle8.v v2, (t5)\n\t"
        "vle8.v v4, (t4)\n\t" "addi t5, t4, 32\n\t" "vle8.v v5, (t5)\n\t"
        "addi t5, t4, 64\n\t" "vle8.v v6, (t5)\n\t" "addi t5, t4, 96\n\t" "vle8.v v7, (t5)\n\t"
        Y26_DOT(v8, v0, v4) Y26_DOT(v10, v1, v4) Y26_DOT(v12, v2, v4)
        Y26_DOT(v14, v0, v5) Y26_DOT(v16, v1, v5) Y26_DOT(v18, v2, v5)
        Y26_DOT(v20, v0, v6) Y26_DOT(v22, v1, v6) Y26_DOT(v24, v2, v6)
        Y26_DOT(v26, v0, v7) Y26_DOT(v28, v1, v7) Y26_DOT(v30, v2, v7)
        "addi t3, t3, 96\n\t" "addi t4, t4, 128\n\t" "addi t1, t1, -1\n\t" "bnez t1, 1b\n\t"
        "vsetvli t0, zero, e32, m2\n\t" "mv t2, %[C]\n\t"
        Y26_STORE(v8) Y26_STORE(v10) Y26_STORE(v12) Y26_STORE(v14)
        Y26_STORE(v16) Y26_STORE(v18) Y26_STORE(v20) Y26_STORE(v22)
        Y26_STORE(v24) Y26_STORE(v26) Y26_STORE(v28) Y26_STORE(v30)
        : : [A] "r"(a), [B] "r"(b), [K] "r"(k_tiles), [C] "r"(c)
        : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5",
          "v0", "v1", "v2", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

#undef Y26_INIT_ACC
#undef Y26_DOT
#undef Y26_STORE

#endif

void run_block(const KernelShape& shape, const std::int8_t* a, const std::int8_t* b, int k_tiles, std::int32_t* c) {
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    if (shape.rows == 4 && shape.cols == 16) kernel_m4n16(a, b, k_tiles, c);
    else if (shape.rows == 4 && shape.cols == 24) kernel_m4n24(a, b, k_tiles, c);
    else if (shape.rows == 8 && shape.cols == 16) kernel_m8n16(a, b, k_tiles, c);
    else if (shape.rows == 12 && shape.cols == 16) kernel_m12n16(a, b, k_tiles, c);
    else throw std::runtime_error("unsupported kernel shape");
#else
    scalar_block_grouped(a, b, k_tiles, shape.rows, shape.cols, c);
#endif
}

struct Workload {
    const char* name;
    int m;
    int n;
    int k;
};

void run_microkernels(const Options& options) {
    std::cout << "category\tname\tworking_set\toperations\tcontract_bytes\tloops\tmean_us\tstddev_us\tcv_pct"
                 "\tmin_us\tmax_us\tmedian_us\tp90_us\tp95_us\tgops_or_gmacs\tgb_per_s\tchecksum\n";
    const std::array<KernelShape, 4> shapes = {{{"m4n16", 4, 16}, {"m4n24", 4, 24},
                                                {"m8n16", 8, 16}, {"m12n16", 12, 16}}};
    const std::array<Workload, 4> workloads = {{{"l1_resident", 24, 48, 256},
                                                 {"l2_resident", 120, 96, 512},
                                                 {"streaming", 600, 96, 1024},
                                                 {"model5_geometry", 1600, 128, 1152}}};
    for (const KernelShape& shape : shapes) {
        std::vector<std::int8_t> validation_a(static_cast<std::size_t>(shape.rows) * 16);
        std::vector<std::int8_t> validation_b(static_cast<std::size_t>(shape.cols) * 16);
        std::vector<std::int32_t> validation_actual(static_cast<std::size_t>(shape.rows) * shape.cols);
        std::vector<std::int32_t> validation_expected(validation_actual.size());
        for (std::size_t i = 0; i < validation_a.size(); ++i) validation_a[i] = static_cast<std::int8_t>(static_cast<int>(i * 13U) % 31 - 15);
        for (std::size_t i = 0; i < validation_b.size(); ++i) validation_b[i] = static_cast<std::int8_t>(static_cast<int>(i * 19U) % 29 - 14);
        run_block(shape, validation_a.data(), validation_b.data(), 2, validation_actual.data());
        scalar_block_grouped(validation_a.data(), validation_b.data(), 2, shape.rows, shape.cols,
                             validation_expected.data());
        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < validation_actual.size(); ++i) {
            mismatches += validation_actual[i] != validation_expected[i];
        }
        if (mismatches != 0) {
            throw std::runtime_error(std::string("microkernel oracle mismatch for ") + shape.name +
                                     " mismatches=" + std::to_string(mismatches));
        }
        std::cout << "microkernel_validation=" << shape.name << ",mismatches=" << mismatches << "\n";
        for (const Workload& requested : workloads) {
            const int m = requested.m - requested.m % shape.rows;
            const int n = requested.n - requested.n % shape.cols;
            const int k = requested.k - requested.k % 8;
            const int k_tiles = k / 8;
            const int m_blocks = m / shape.rows;
            const int n_blocks = n / shape.cols;
            std::vector<std::int8_t> a(static_cast<std::size_t>(m_blocks) * k_tiles * shape.rows * 8);
            std::vector<std::int8_t> b(static_cast<std::size_t>(n_blocks) * k_tiles * shape.cols * 8);
            std::vector<std::int32_t> c(static_cast<std::size_t>(shape.rows) * shape.cols);
            for (std::size_t i = 0; i < a.size(); ++i) a[i] = static_cast<std::int8_t>(static_cast<int>(i * 17U + 3U) % 255 - 127);
            for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<std::int8_t>(static_cast<int>(i * 29U + 5U) % 255 - 127);
            std::uint64_t checksum = 0;
            auto fn = [&]() {
                std::uint64_t local = 0;
                for (int mb = 0; mb < m_blocks; ++mb) {
                    const auto* ap = a.data() + static_cast<std::size_t>(mb) * k_tiles * shape.rows * 8;
                    for (int nb = 0; nb < n_blocks; ++nb) {
                        const auto* bp = b.data() + static_cast<std::size_t>(nb) * k_tiles * shape.cols * 8;
                        run_block(shape, ap, bp, k_tiles, c.data());
                        local ^= static_cast<std::uint32_t>(c.front());
                        local ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(c.back())) << 32U;
                        local += static_cast<std::uint64_t>(mb * n_blocks + nb);
                    }
                }
                checksum ^= local;
                g_sink ^= local;
            };
            const auto measured = measure(options, fn);
            const std::uint64_t macs = static_cast<std::uint64_t>(m) * n * k;
            const std::uint64_t bytes_per_block = static_cast<std::uint64_t>(shape.rows) * k +
                                                  static_cast<std::uint64_t>(shape.cols) * k +
                                                  static_cast<std::uint64_t>(shape.rows) * shape.cols * sizeof(std::int32_t);
            const std::uint64_t bytes = static_cast<std::uint64_t>(m_blocks) * n_blocks * bytes_per_block;
            const std::string label = std::string(requested.name) + ":M" + std::to_string(m) + "N" +
                                      std::to_string(n) + "K" + std::to_string(k);
            print_row("vmadot_microkernel", shape.name, label, macs, bytes,
                      measured.first, measured.second, checksum);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
        std::cout << "ime_enabled=1\n";
#else
        std::cout << "ime_enabled=0\n";
#endif
        std::cout << "timing_source=std::chrono::steady_clock\n";
        std::cout << "rdcycle_used=0\n";
        if (options.mode == "all" || options.mode == "primitives") run_primitives(options);
        if (options.mode == "all" || options.mode == "microkernel") run_microkernels(options);
        std::cout << "guard=" << g_sink << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error=" << error.what() << '\n';
        return 2;
    }
}
