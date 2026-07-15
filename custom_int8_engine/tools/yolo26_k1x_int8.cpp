#include "y26_k1x_executor.h"
#include "y26_k1x_package.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

#if defined(Y26_K1X_HAVE_OPENCV)
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace {

struct Options {
    std::filesystem::path package;
    std::filesystem::path image;
    std::filesystem::path output_json;
    std::string input_mode = "preprocessed-f32";
    int threads = 4;
    int warmup = 0;
    int runs = 1;
    int repeats = 1;
    int inter_frame_gap_us = 0;
    bool benchmark = false;
    bool verify = false;
    bool version = false;
    bool dump_live_boundary = false;
    std::string dump_boundary;
    y26_scheduler scheduler = Y26_SCHEDULER_SAFE;
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
        else if (argument == "--output-json") options.output_json = next();
        else if (argument == "--input-mode") options.input_mode = next();
        else if (argument == "--threads") options.threads = std::stoi(next());
        else if (argument == "--warmup") options.warmup = std::stoi(next());
        else if (argument == "--runs") options.runs = std::stoi(next());
        else if (argument == "--repeats") options.repeats = std::stoi(next());
        else if (argument == "--inter-frame-gap-us") options.inter_frame_gap_us = std::stoi(next());
        else if (argument == "--scheduler") {
            const std::string value = next();
            if (value == "safe") options.scheduler = Y26_SCHEDULER_SAFE;
            else if (value == "rr20") options.scheduler = Y26_SCHEDULER_RR20;
            else throw std::runtime_error("scheduler must be safe or rr20");
        } else if (argument == "--pin") {
            if (next() != "0-3") throw std::runtime_error("only --pin 0-3 is supported");
        } else if (argument == "--benchmark") options.benchmark = true;
        else if (argument == "--verify") options.verify = true;
        else if (argument == "--version") options.version = true;
        else if (argument == "--dump-boundary") {
            options.dump_boundary = next();
        } else if (argument == "--dump-boundary-live") {
            options.dump_boundary = next();
            options.dump_live_boundary = true;
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    return options;
}

std::vector<float> read_f32(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open input: " + path.string());
    const std::streamsize bytes = stream.tellg();
    if (bytes != static_cast<std::streamsize>(Y26_K1X_EXECUTOR_INPUT_ELEMENTS * sizeof(float))) {
        throw std::runtime_error("preprocessed input must contain 1x3x640x640 float32 values");
    }
    stream.seekg(0);
    std::vector<float> result(Y26_K1X_EXECUTOR_INPUT_ELEMENTS);
    if (!stream.read(reinterpret_cast<char*>(result.data()), bytes)) throw std::runtime_error("input read failed");
    return result;
}

std::vector<std::uint8_t> read_rgb640(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open input: " + path.string());
    constexpr std::size_t kBytes = 640U * 640U * 3U;
    if (stream.tellg() != static_cast<std::streamsize>(kBytes)) {
        throw std::runtime_error("rgb640-u8 input must contain 640x640x3 interleaved bytes");
    }
    stream.seekg(0);
    std::vector<std::uint8_t> result(kBytes);
    if (!stream.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(result.size()))) {
        throw std::runtime_error("input read failed");
    }
    return result;
}

#if defined(Y26_K1X_HAVE_OPENCV)
std::vector<std::uint8_t> read_letterboxed_image(const std::filesystem::path& path) {
    const cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) throw std::runtime_error("cannot decode input image: " + path.string());
    const double ratio = std::min(640.0 / bgr.cols, 640.0 / bgr.rows);
    const int resized_width = static_cast<int>(std::nearbyint(bgr.cols * ratio));
    const int resized_height = static_cast<int>(std::nearbyint(bgr.rows * ratio));
    const double pad_x = (640.0 - resized_width) / 2.0;
    const double pad_y = (640.0 - resized_height) / 2.0;
    const int x0 = static_cast<int>(std::nearbyint(pad_x - 0.1));
    const int y0 = static_cast<int>(std::nearbyint(pad_y - 0.1));
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat canvas(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(x0, y0, resized_width, resized_height)));
    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) rgb = rgb.clone();
    return {rgb.data, rgb.data + 640U * 640U * 3U};
}
#endif

void dump_boundary(y26_executor* executor, const std::string& specification) {
    const std::size_t separator = specification.rfind('=');
    if (separator == std::string::npos || separator == 0 || separator + 1 == specification.size()) {
        throw std::runtime_error("--dump-boundary expects TENSOR_NAME=OUTPUT_FILE");
    }
    const std::string tensor_name = specification.substr(0, separator);
    const std::filesystem::path output_path = specification.substr(separator + 1);
    const int tensor_id = y26_executor_tensor_id(executor, tensor_name.c_str());
    const std::size_t bytes = y26_executor_tensor_bytes(executor, tensor_id);
    if (tensor_id < 0 || bytes == 0) throw std::runtime_error("unknown boundary tensor: " + tensor_name);
    std::vector<std::uint8_t> data(bytes);
    if (y26_executor_copy_boundary(executor, tensor_id, data.data(), data.size()) != Y26_STATUS_OK) {
        throw std::runtime_error("boundary copy failed");
    }
    std::ofstream stream(output_path, std::ios::binary);
    if (!stream || !stream.write(reinterpret_cast<const char*>(data.data()),
                                  static_cast<std::streamsize>(data.size()))) {
        throw std::runtime_error("boundary write failed: " + output_path.string());
    }
}

void write_json(const std::filesystem::path& path, const std::vector<float>& output) {
    std::ofstream stream(path.empty() ? "/dev/stdout" : path);
    if (!stream) throw std::runtime_error("cannot open output JSON");
    stream << std::setprecision(9) << "[\n";
    for (int detection = 0; detection < 300; ++detection) {
        const float* row = output.data() + static_cast<std::size_t>(detection) * 6U;
        stream << "  {\"box\":[" << row[0] << ',' << row[1] << ',' << row[2] << ',' << row[3]
               << "],\"score\":" << row[4] << ",\"class\":" << static_cast<int>(row[5]) << '}';
        stream << (detection == 299 ? "\n" : ",\n");
    }
    stream << "]\n";
}

double percentile(std::vector<double> values, double quantile) {
    std::sort(values.begin(), values.end());
    const double position = quantile * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(lower + 1, values.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return values[lower] + (values[upper] - values[lower]) * fraction;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        if (options.version) {
            std::cout << y26_executor_version() << '\n';
            return 0;
        }
        if (options.package.empty() || options.image.empty()) {
            throw std::runtime_error("--package and --image are required");
        }
        if (options.input_mode != "preprocessed-f32" && options.input_mode != "rgb640-u8" &&
            options.input_mode != "image") {
            throw std::runtime_error("--input-mode must be preprocessed-f32, rgb640-u8, or image");
        }
        if (options.threads < 1 || options.threads > 4 || options.warmup < 0 ||
            options.runs < 1 || options.repeats < 1 || options.inter_frame_gap_us < 0) {
            throw std::runtime_error("invalid numeric option");
        }
        const std::string manifest = y26::int8_v1::sha256_file(options.package / "asset_hashes.tsv");
        std::unique_ptr<y26_executor, decltype(&y26_executor_destroy)> executor(
            y26_executor_create(), y26_executor_destroy);
        if (!executor) throw std::runtime_error("executor allocation failed");
        y26_executor_options executor_options {};
        executor_options.struct_size = sizeof(executor_options);
        executor_options.abi_version = Y26_K1X_EXECUTOR_ABI_VERSION;
        executor_options.workers = options.threads;
        executor_options.worker_cpu_begin = 0;
        executor_options.controller_cpu = 4;
        executor_options.scheduler = options.scheduler;
        executor_options.flags = options.dump_boundary.empty() || options.dump_live_boundary
            ? Y26_EXECUTOR_FLAG_NONE : Y26_EXECUTOR_FLAG_CAPTURE_BOUNDARIES;
        if (y26_executor_prepare(executor.get(), options.package.c_str(), manifest.c_str(),
                                 &executor_options) != Y26_STATUS_OK) {
            const std::string error = y26_executor_last_error(executor.get());
            throw std::runtime_error("prepare failed: " + error);
        }
        std::vector<float> input;
        std::vector<std::uint8_t> rgb;
        if (options.input_mode == "preprocessed-f32") input = read_f32(options.image);
        else if (options.input_mode == "rgb640-u8") rgb = read_rgb640(options.image);
        else {
#if defined(Y26_K1X_HAVE_OPENCV)
            rgb = read_letterboxed_image(options.image);
#else
            throw std::runtime_error("this build has no OpenCV image decoder; use preprocessed-f32 or rgb640-u8");
#endif
        }
        std::vector<float> output(Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS);
        std::vector<float> expected_output;
        y26_run_timing timing {};
#if defined(__linux__)
        int trace_marker_fd = -1;
        if (const char* path = std::getenv("Y26_STAGE56_TRACE_MARKER");
            path != nullptr && path[0] != '\0') {
            trace_marker_fd = open(path, O_WRONLY | O_CLOEXEC);
            if (trace_marker_fd < 0) throw std::runtime_error("cannot open trace marker");
        }
        const auto trace_marker = [&](const char* phase, int repeat, int run) {
            if (trace_marker_fd < 0) return;
            char marker[96];
            const int bytes = std::snprintf(marker, sizeof(marker),
                "y26_%s repeat=%d run=%d\n", phase, repeat, run);
            if (bytes <= 0 || write(trace_marker_fd, marker,
                                     static_cast<std::size_t>(bytes)) != bytes) {
                throw std::runtime_error("trace marker write failed");
            }
        };
#else
        const auto trace_marker = [](const char*, int, int) {};
#endif
        const auto run_once = [&](y26_run_timing* run_timing) {
            const y26_status status = options.input_mode == "preprocessed-f32"
                ? y26_executor_run_preprocessed(executor.get(), input.data(), input.size(),
                                                output.data(), output.size(), run_timing)
                : y26_executor_run_rgb(executor.get(), rgb.data(), 640, 640, 640 * 3,
                                       output.data(), output.size(), run_timing);
            if (status != Y26_STATUS_OK) throw std::runtime_error(y26_executor_last_error(executor.get()));
        };
        for (int warmup = 0; warmup < options.warmup; ++warmup) {
            run_once(nullptr);
        }
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(options.runs * options.repeats));
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            for (int run = 0; run < options.runs; ++run) {
                trace_marker("begin", repeat, run);
                run_once(&timing);
                trace_marker("end", repeat, run);
                if (options.verify) {
                    if (expected_output.empty()) expected_output = output;
                    else if (std::memcmp(expected_output.data(), output.data(),
                                         output.size() * sizeof(float)) != 0) {
                        throw std::runtime_error("deterministic output verification failed");
                    }
                }
                samples.push_back(timing.total_us);
                if (options.benchmark) {
                    std::cout << "raw\trepeat=" << repeat << "\trun=" << run
                              << "\twall_us=" << timing.total_us
                              << "\tinput_us=" << timing.input_quantize_us
                              << "\tcore_us=" << timing.resident_core_us
                              << "\tdense_us=" << timing.dense_conv_us
                              << "\tdepthwise_us=" << timing.depthwise_us
                              << "\tattention_us=" << timing.attention_us
                              << "\tlut_us=" << timing.lut_us
                              << "\tconcat_us=" << timing.concat_us
                              << "\ttransform_us=" << timing.transform_us
                              << "\thead_us=" << timing.head_us
                              << "\tprocess_cpu_us=" << timing.process_cpu_us
                              << "\tvoluntary_cs=" << timing.voluntary_context_switches
                              << "\tinvoluntary_cs=" << timing.involuntary_context_switches
                              << "\taffinity_ok=" << timing.affinity_ok
                              << "\tcpu4_7_ime_count=" << timing.cpu4_7_ime_count
                              << "\thash=0x"
                              << std::hex << timing.output_hash << std::dec << '\n';
                }
                if (options.inter_frame_gap_us > 0 &&
                    (repeat + 1 != options.repeats || run + 1 != options.runs)) {
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(options.inter_frame_gap_us));
                }
            }
        }
        if (options.verify && samples.size() == 1) {
            run_once(nullptr);
            if (std::memcmp(expected_output.data(), output.data(), output.size() * sizeof(float)) != 0) {
                throw std::runtime_error("deterministic output verification failed");
            }
            std::vector<float> cached(output.size());
            if (y26_executor_get_output(executor.get(), cached.data(), cached.size()) != Y26_STATUS_OK ||
                std::memcmp(cached.data(), output.data(), output.size() * sizeof(float)) != 0) {
                throw std::runtime_error("cached output verification failed");
            }
        }
        write_json(options.output_json, output);
        if (!options.dump_boundary.empty()) dump_boundary(executor.get(), options.dump_boundary);
        if (options.benchmark) {
            const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
            std::cout << "mean_us=" << mean << '\n'
                      << "median_us=" << percentile(samples, 0.5) << '\n'
                      << "p95_us=" << percentile(samples, 0.95) << '\n'
                      << "p99_us=" << percentile(samples, 0.99) << '\n'
                      << "min_us=" << *std::min_element(samples.begin(), samples.end()) << '\n'
                      << "max_us=" << *std::max_element(samples.begin(), samples.end()) << '\n';
        }
#if defined(__linux__)
        if (trace_marker_fd >= 0) close(trace_marker_fd);
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
