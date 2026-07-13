#include "y26_k1x_full_executor.h"
#include "y26_k1x_int8_v1.h"
#include "y26_k1x_package.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <pthread.h>
#include <sched.h>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::filesystem::path package;
    std::filesystem::path image;
    int threads = 4;
    int warmup = 10;
    int runs = 100;
    int repeats = 5;
};

struct Sample {
    double decode_us = 0.0;
    double resize_letterbox_us = 0.0;
    double color_convert_us = 0.0;
    double input_quantize_layout_us = 0.0;
    double pure_graph_us = 0.0;
    double executor_total_us = 0.0;
    double output_decode_us = 0.0;
    double pipeline_total_us = 0.0;
    std::uint64_t output_hash = 0;
    int detections = 0;
};

struct Stats {
    double mean = 0.0;
    double stddev = 0.0;
    double cv_pct = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    double median = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
};

Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next = [&]() -> std::string {
            if (++index >= argc) throw std::runtime_error("missing value for " + argument);
            return argv[index];
        };
        if (argument == "--package") options.package = next();
        else if (argument == "--image") options.image = next();
        else if (argument == "--threads") options.threads = std::stoi(next());
        else if (argument == "--warmup") options.warmup = std::stoi(next());
        else if (argument == "--runs") options.runs = std::stoi(next());
        else if (argument == "--repeats") options.repeats = std::stoi(next());
        else throw std::runtime_error("unknown argument: " + argument);
    }
    if (options.package.empty() || options.image.empty()) {
        throw std::runtime_error("--package and --image are required");
    }
    if (options.threads < 1 || options.threads > 4 || options.warmup < 0 ||
        options.runs < 1 || options.repeats < 1) {
        throw std::runtime_error("invalid numeric option");
    }
    return options;
}

void pin_controller() {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(4, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
        throw std::runtime_error("failed to pin controller to CPU4");
    }
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open image: " + path.string());
    const std::streamsize bytes = stream.tellg();
    if (bytes <= 0) throw std::runtime_error("empty image: " + path.string());
    stream.seekg(0);
    std::vector<std::uint8_t> result(static_cast<std::size_t>(bytes));
    if (!stream.read(reinterpret_cast<char*>(result.data()), bytes)) {
        throw std::runtime_error("failed to read image: " + path.string());
    }
    return result;
}

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

double percentile(const std::vector<double>& sorted, double fraction) {
    if (sorted.empty()) return 0.0;
    const double position = fraction * static_cast<double>(sorted.size() - 1U);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

Stats stats(std::vector<double> values) {
    if (values.empty()) return {};
    Stats result;
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) /
                  static_cast<double>(values.size());
    double squared = 0.0;
    for (double value : values) {
        const double delta = value - result.mean;
        squared += delta * delta;
    }
    result.stddev = values.size() > 1U
                        ? std::sqrt(squared / static_cast<double>(values.size() - 1U))
                        : 0.0;
    result.cv_pct = result.mean != 0.0 ? result.stddev / result.mean * 100.0 : 0.0;
    std::sort(values.begin(), values.end());
    result.minimum = values.front();
    result.maximum = values.back();
    result.median = percentile(values, 0.5);
    result.p90 = percentile(values, 0.9);
    result.p95 = percentile(values, 0.95);
    result.p99 = percentile(values, 0.99);
    return result;
}

Sample run_once(const std::vector<std::uint8_t>& encoded,
                y26::stage52::FullExecutor& executor,
                std::array<float, 1800>& output,
                cv::Mat& resized,
                cv::Mat& canvas,
                cv::Mat& rgb) {
    Sample sample;
    const auto total_begin = Clock::now();
    const auto decode_begin = total_begin;
    const cv::Mat bgr = cv::imdecode(encoded, cv::IMREAD_COLOR);
    const auto decode_end = Clock::now();
    if (bgr.empty()) throw std::runtime_error("OpenCV failed to decode the preloaded image bytes");

    const double ratio = std::min(640.0 / bgr.cols, 640.0 / bgr.rows);
    const int resized_width = static_cast<int>(std::nearbyint(bgr.cols * ratio));
    const int resized_height = static_cast<int>(std::nearbyint(bgr.rows * ratio));
    const double pad_x = (640.0 - resized_width) / 2.0;
    const double pad_y = (640.0 - resized_height) / 2.0;
    const int x0 = static_cast<int>(std::nearbyint(pad_x - 0.1));
    const int y0 = static_cast<int>(std::nearbyint(pad_y - 0.1));
    const auto resize_begin = decode_end;
    cv::resize(bgr, resized, cv::Size(resized_width, resized_height), 0.0, 0.0,
               cv::INTER_LINEAR);
    canvas.setTo(cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(x0, y0, resized_width, resized_height)));
    const auto resize_end = Clock::now();
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    const auto color_end = Clock::now();

    y26::stage52::RunTiming timing;
    if (executor.run_rgb(rgb.data, 640, 640, static_cast<int>(rgb.step), output.data(),
                         output.size(), &timing) != 0) {
        throw std::runtime_error("execution failed: " + executor.last_error());
    }
    const auto executor_end = Clock::now();

    int detections = 0;
    volatile double output_guard = 0.0;
    for (std::size_t row = 0; row < 300U; ++row) {
        const float* value = output.data() + row * 6U;
        if (value[4] > 0.001F && value[5] >= 0.0F && value[5] < 80.0F) ++detections;
        output_guard = output_guard + static_cast<double>(value[0]) +
                       static_cast<double>(value[4]) + static_cast<double>(value[5]);
    }
    (void)output_guard;
    const auto output_end = Clock::now();

    sample.decode_us = elapsed_us(decode_begin, decode_end);
    sample.resize_letterbox_us = elapsed_us(resize_begin, resize_end);
    sample.color_convert_us = elapsed_us(resize_end, color_end);
    sample.input_quantize_layout_us = timing.input_quantize_us;
    sample.executor_total_us = timing.total_us;
    sample.pure_graph_us = std::max(0.0, timing.total_us - timing.input_quantize_us);
    sample.output_decode_us = elapsed_us(executor_end, output_end);
    sample.pipeline_total_us = elapsed_us(total_begin, output_end);
    sample.output_hash = timing.output_hash;
    sample.detections = detections;
    return sample;
}

void print_summary(const char* phase, const std::vector<double>& values) {
    const Stats value = stats(values);
    std::cout << "summary\t" << phase << '\t' << value.mean << '\t' << value.stddev << '\t'
              << value.cv_pct << '\t' << value.minimum << '\t' << value.maximum << '\t'
              << value.median << '\t' << value.p90 << '\t' << value.p95 << '\t' << value.p99
              << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        pin_controller();
        const std::vector<std::uint8_t> encoded = read_bytes(options.image);

        y26::stage52::RunConfig config;
        config.workers = options.threads;
        config.worker_cpu_begin = 0;
        config.controller_cpu = 4;
        config.scheduler = y26::stage52::SchedulerMode::safe;
        config.compute = y26::stage52::ComputeMode::optimized;
        y26::stage52::FullExecutor executor;
        const std::string manifest =
            y26::int8_v1::sha256_file(options.package / "asset_hashes.tsv");
        if (executor.prepare(options.package, manifest, config) != 0) {
            throw std::runtime_error("prepare failed: " + executor.last_error());
        }

        cv::Mat resized;
        cv::Mat canvas(640, 640, CV_8UC3);
        cv::Mat rgb(640, 640, CV_8UC3);
        std::array<float, 1800> output {};
        for (int index = 0; index < options.warmup; ++index) {
            (void)run_once(encoded, executor, output, resized, canvas, rgb);
        }

        std::array<std::vector<double>, 8> values;
        for (auto& item : values) {
            item.reserve(static_cast<std::size_t>(options.runs * options.repeats));
        }
        std::uint64_t expected_hash = 0;
        int expected_detections = -1;
        std::cout << std::setprecision(12)
                  << "sample\trepeat\trun\tdecode_us\tresize_letterbox_us\tcolor_convert_us"
                     "\tinput_quantize_layout_us\tpure_graph_us\texecutor_total_us"
                     "\toutput_decode_us\tpipeline_total_us\toutput_hash\tdetections\n";
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            for (int run = 0; run < options.runs; ++run) {
                const Sample sample = run_once(encoded, executor, output, resized, canvas, rgb);
                if (expected_hash == 0) expected_hash = sample.output_hash;
                if (expected_detections < 0) expected_detections = sample.detections;
                if (sample.output_hash != expected_hash || sample.detections != expected_detections) {
                    throw std::runtime_error("nondeterministic pipeline output");
                }
                values[0].push_back(sample.decode_us);
                values[1].push_back(sample.resize_letterbox_us);
                values[2].push_back(sample.color_convert_us);
                values[3].push_back(sample.input_quantize_layout_us);
                values[4].push_back(sample.pure_graph_us);
                values[5].push_back(sample.executor_total_us);
                values[6].push_back(sample.output_decode_us);
                values[7].push_back(sample.pipeline_total_us);
                std::cout << "sample\t" << repeat << '\t' << run << '\t' << sample.decode_us << '\t'
                          << sample.resize_letterbox_us << '\t' << sample.color_convert_us << '\t'
                          << sample.input_quantize_layout_us << '\t' << sample.pure_graph_us << '\t'
                          << sample.executor_total_us << '\t' << sample.output_decode_us << '\t'
                          << sample.pipeline_total_us << "\t0x" << std::hex << sample.output_hash
                          << std::dec << '\t' << sample.detections << '\n';
            }
        }
        std::cout << "summary\tphase\tmean_us\tstddev_us\tcv_pct\tmin_us\tmax_us\tmedian_us"
                     "\tp90_us\tp95_us\tp99_us\n";
        constexpr std::array<const char*, 8> names = {
            "jpeg_decode", "resize_letterbox", "color_convert", "input_quantize_layout",
            "pure_graph", "executor_total", "output_decode", "preloaded_image_pipeline",
        };
        for (std::size_t index = 0; index < names.size(); ++index) {
            print_summary(names[index], values[index]);
        }
        std::cout << "metadata\tencoded_bytes\t" << encoded.size() << '\n'
                  << "metadata\toutput_hash\t0x" << std::hex << expected_hash << std::dec << '\n'
                  << "metadata\tdetections\t" << expected_detections << '\n'
                  << "metadata\tpackage_manifest_sha256\t" << executor.package_manifest_sha256()
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
