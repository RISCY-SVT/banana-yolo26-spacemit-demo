#include "y26_k1x_stage48_nchwc8.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string direction;
    std::filesystem::path input;
    std::filesystem::path expected;
    int n = 1;
    int c = 0;
    int h = 0;
    int w = 0;
    int warmup = 10;
    int runs = 100;
    int repeats = 5;
};

struct Statistics {
    double mean = 0.0;
    double stddev = 0.0;
    double cv = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    double median = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
};

int integer(const std::string& text) {
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size()) throw std::runtime_error("invalid integer: " + text);
    return value;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        if (index + 1 >= argc) throw std::runtime_error("missing option value");
        const std::string key = argv[index];
        const std::string value = argv[++index];
        if (key == "--direction") options.direction = value;
        else if (key == "--input") options.input = value;
        else if (key == "--expected") options.expected = value;
        else if (key == "--n") options.n = integer(value);
        else if (key == "--c") options.c = integer(value);
        else if (key == "--h") options.h = integer(value);
        else if (key == "--w") options.w = integer(value);
        else if (key == "--warmup") options.warmup = integer(value);
        else if (key == "--runs") options.runs = integer(value);
        else if (key == "--repeats") options.repeats = integer(value);
        else throw std::runtime_error("unknown option: " + key);
    }
    if ((options.direction != "entry" && options.direction != "exit") || options.input.empty() ||
        options.expected.empty() || options.n <= 0 || options.c <= 0 || options.c % 8 != 0 ||
        options.h <= 0 || options.w <= 0 || options.warmup < 0 || options.runs <= 0 || options.repeats <= 0) {
        throw std::runtime_error("invalid adapter contract");
    }
    return options;
}

std::vector<std::uint8_t> read(const std::filesystem::path& path, std::size_t bytes) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != static_cast<std::streamoff>(bytes)) {
        throw std::runtime_error("file size mismatch: " + path.string());
    }
    stream.seekg(0);
    std::vector<std::uint8_t> result(bytes);
    stream.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(bytes));
    if (!stream) throw std::runtime_error("file read failed: " + path.string());
    return result;
}

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

double process_cpu_us() {
    timespec value {};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) != 0) throw std::runtime_error("clock_gettime failed");
    return static_cast<double>(value.tv_sec) * 1.0e6 + static_cast<double>(value.tv_nsec) / 1.0e3;
}

double percentile(const std::vector<double>& values, double q) {
    const double position = q * static_cast<double>(values.size() - 1);
    const auto low = static_cast<std::size_t>(std::floor(position));
    const auto high = static_cast<std::size_t>(std::ceil(position));
    return values[low] + (values[high] - values[low]) * (position - static_cast<double>(low));
}

Statistics summarize(std::vector<double> values) {
    Statistics result;
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double squared = 0.0;
    for (double value : values) squared += (value - result.mean) * (value - result.mean);
    result.stddev = values.size() > 1 ? std::sqrt(squared / static_cast<double>(values.size() - 1)) : 0.0;
    result.cv = result.mean == 0.0 ? 0.0 : result.stddev * 100.0 / result.mean;
    std::sort(values.begin(), values.end());
    result.minimum = values.front();
    result.maximum = values.back();
    result.median = percentile(values, 0.5);
    result.p90 = percentile(values, 0.9);
    result.p95 = percentile(values, 0.95);
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        const std::size_t bytes = static_cast<std::size_t>(options.n) * options.c * options.h * options.w;
        const auto input = read(options.input, bytes);
        const auto expected = read(options.expected, bytes);
        std::vector<std::uint8_t> output(bytes);
        const auto execute = [&]() {
            if (options.direction == "entry") {
                y26::stage48::nchw_u8_to_nchwc8_s8(input.data(), reinterpret_cast<std::int8_t*>(output.data()),
                                                    options.n, options.c, options.h, options.w);
            } else {
                y26::stage48::nchwc8_s8_to_nchw_u8(reinterpret_cast<const std::int8_t*>(input.data()), output.data(),
                                                    options.n, options.c, options.h, options.w);
            }
        };
        for (int run = 0; run < options.warmup; ++run) execute();
        std::vector<double> wall;
        std::vector<double> cpu;
        std::vector<double> repeat_means;
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            std::vector<double> current;
            for (int run = 0; run < options.runs; ++run) {
                const double cpu_begin = process_cpu_us();
                const auto begin = Clock::now();
                execute();
                const auto end = Clock::now();
                const double cpu_end = process_cpu_us();
                const double sample = elapsed_us(begin, end);
                wall.push_back(sample);
                cpu.push_back(cpu_end - cpu_begin);
                current.push_back(sample);
                std::cout << "raw\trepeat=" << repeat << "\trun=" << run << "\twall_us=" << sample
                          << "\tprocess_cpu_us=" << (cpu_end - cpu_begin) << '\n';
            }
            repeat_means.push_back(summarize(current).mean);
        }
        const Statistics wall_stats = summarize(wall);
        const Statistics cpu_stats = summarize(cpu);
        const Statistics repeat_stats = summarize(repeat_means);
        const std::size_t mismatches = static_cast<std::size_t>(
            std::inner_product(output.begin(), output.end(), expected.begin(), 0,
                std::plus<>(), [](std::uint8_t left, std::uint8_t right) { return left != right; }));
        std::cout << std::fixed << std::setprecision(6)
                  << "direction=" << options.direction << "\nbytes=" << bytes << "\nmean_us=" << wall_stats.mean
                  << "\nstddev_us=" << wall_stats.stddev << "\ncv_pct=" << wall_stats.cv
                  << "\nmin_us=" << wall_stats.minimum << "\nmax_us=" << wall_stats.maximum
                  << "\nmedian_us=" << wall_stats.median << "\np90_us=" << wall_stats.p90
                  << "\np95_us=" << wall_stats.p95 << "\nprocess_cpu_mean_us=" << cpu_stats.mean
                  << "\nrepeat_mean_cv_pct=" << repeat_stats.cv << "\nmismatches=" << mismatches << '\n';
        return mismatches == 0 ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << "error=" << error.what() << '\n';
        return 64;
    }
}
