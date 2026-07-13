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

constexpr std::array<int, 80> kCocoCategoryIds = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
};

struct Options {
    std::filesystem::path package;
    std::filesystem::path images;
    std::filesystem::path output;
    std::filesystem::path timing_tsv;
    int threads = 4;
    int limit = 0;
    int log_every = 100;
    float confidence = 0.001F;
};

struct Letterbox {
    cv::Mat rgb;
    int original_width = 0;
    int original_height = 0;
    double ratio = 1.0;
    double pad_x = 0.0;
    double pad_y = 0.0;
};

struct Prediction {
    int image_id = 0;
    int category_id = 0;
    float x = 0.0F;
    float y = 0.0F;
    float width = 0.0F;
    float height = 0.0F;
    float score = 0.0F;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next = [&]() -> std::string {
            if (++index >= argc) throw std::runtime_error("missing value for " + argument);
            return argv[index];
        };
        if (argument == "--package") options.package = next();
        else if (argument == "--images") options.images = next();
        else if (argument == "--output") options.output = next();
        else if (argument == "--timing-tsv") options.timing_tsv = next();
        else if (argument == "--threads") options.threads = std::stoi(next());
        else if (argument == "--limit") options.limit = std::stoi(next());
        else if (argument == "--log-every") options.log_every = std::stoi(next());
        else if (argument == "--conf") options.confidence = std::stof(next());
        else throw std::runtime_error("unknown argument: " + argument);
    }
    if (options.package.empty() || options.images.empty() || options.output.empty() ||
        options.timing_tsv.empty()) {
        throw std::runtime_error("--package, --images, --output, and --timing-tsv are required");
    }
    if (options.threads < 1 || options.threads > 4 || options.limit < 0 ||
        options.log_every < 0 || !(options.confidence >= 0.0F)) {
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

std::vector<std::filesystem::path> list_images(const std::filesystem::path& directory,
                                                int limit) {
    std::vector<std::filesystem::path> images;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        const std::string extension = entry.path().extension().string();
        if (extension == ".jpg" || extension == ".jpeg" || extension == ".JPG" ||
            extension == ".JPEG") {
            images.push_back(entry.path());
        }
    }
    std::sort(images.begin(), images.end());
    if (limit > 0 && static_cast<std::size_t>(limit) < images.size()) {
        images.resize(static_cast<std::size_t>(limit));
    }
    return images;
}

Letterbox letterbox(const std::filesystem::path& path) {
    const cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) throw std::runtime_error("failed to decode " + path.string());
    Letterbox result;
    result.original_width = bgr.cols;
    result.original_height = bgr.rows;
    result.ratio = std::min(640.0 / bgr.cols, 640.0 / bgr.rows);
    const int resized_width = static_cast<int>(std::nearbyint(bgr.cols * result.ratio));
    const int resized_height = static_cast<int>(std::nearbyint(bgr.rows * result.ratio));
    result.pad_x = (640.0 - resized_width) / 2.0;
    result.pad_y = (640.0 - resized_height) / 2.0;
    const int x0 = static_cast<int>(std::nearbyint(result.pad_x - 0.1));
    const int y0 = static_cast<int>(std::nearbyint(result.pad_y - 0.1));
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat canvas(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(x0, y0, resized_width, resized_height)));
    cv::cvtColor(canvas, result.rgb, cv::COLOR_BGR2RGB);
    return result;
}

int image_id(const std::filesystem::path& path) {
    std::size_t consumed = 0;
    const int id = std::stoi(path.stem().string(), &consumed);
    if (consumed != path.stem().string().size()) {
        throw std::runtime_error("non-numeric COCO image filename: " + path.filename().string());
    }
    return id;
}

std::vector<Prediction> decode(const std::array<float, 1800>& output, int id,
                               const Letterbox& transform, float confidence) {
    std::vector<Prediction> predictions;
    predictions.reserve(300);
    for (int row_index = 0; row_index < 300; ++row_index) {
        const float* row = output.data() + static_cast<std::size_t>(row_index) * 6U;
        const int class_id = static_cast<int>(row[5]);
        if (!(row[4] > confidence) || class_id < 0 || class_id >= 80) continue;
        const auto x_coordinate = [&](float value) {
            return static_cast<float>(std::clamp((value - transform.pad_x) / transform.ratio,
                                                 0.0, static_cast<double>(transform.original_width - 1)));
        };
        const auto y_coordinate = [&](float value) {
            return static_cast<float>(std::clamp((value - transform.pad_y) / transform.ratio,
                                                 0.0, static_cast<double>(transform.original_height - 1)));
        };
        const float x1 = x_coordinate(row[0]);
        const float y1 = y_coordinate(row[1]);
        const float x2 = x_coordinate(row[2]);
        const float y2 = y_coordinate(row[3]);
        if (!(x2 > x1 && y2 > y1)) continue;
        predictions.push_back({id, kCocoCategoryIds[static_cast<std::size_t>(class_id)],
                               x1, y1, x2 - x1, y2 - y1, row[4]});
    }
    return predictions;
}

void write_predictions(const std::filesystem::path& path,
                       const std::vector<Prediction>& predictions) {
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("cannot write " + path.string());
    stream << std::setprecision(9) << "[\n";
    for (std::size_t index = 0; index < predictions.size(); ++index) {
        const Prediction& value = predictions[index];
        stream << "  {\"image_id\":" << value.image_id
               << ",\"category_id\":" << value.category_id
               << ",\"bbox\":[" << value.x << ',' << value.y << ',' << value.width << ','
               << value.height << "],\"score\":" << value.score << '}';
        stream << (index + 1 == predictions.size() ? "\n" : ",\n");
    }
    stream << "]\n";
}

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        pin_controller();
        const auto images = list_images(options.images, options.limit);
        if (images.empty()) throw std::runtime_error("no COCO JPEG images found");

        y26::stage52::RunConfig config;
        config.workers = options.threads;
        config.worker_cpu_begin = 0;
        config.controller_cpu = 4;
        config.scheduler = y26::stage52::SchedulerMode::safe;
        config.compute = y26::stage52::ComputeMode::optimized;
        y26::stage52::FullExecutor executor;
        const std::string manifest = y26::int8_v1::sha256_file(options.package / "asset_hashes.tsv");
        if (executor.prepare(options.package, manifest, config) != 0) {
            throw std::runtime_error("prepare failed: " + executor.last_error());
        }

        std::ofstream timing(options.timing_tsv);
        if (!timing) throw std::runtime_error("cannot write " + options.timing_tsv.string());
        timing << "index\timage_id\twidth\theight\tdecode_letterbox_us\texecutor_us\tdecode_output_us"
                  "\tdetections\toutput_hash\n";
        std::vector<Prediction> all_predictions;
        std::array<float, 1800> output {};
        std::vector<double> executor_samples;
        executor_samples.reserve(images.size());
        const auto evaluation_begin = Clock::now();
        for (std::size_t index = 0; index < images.size(); ++index) {
            const auto preprocess_begin = Clock::now();
            const Letterbox transformed = letterbox(images[index]);
            const auto executor_begin = Clock::now();
            y26::stage52::RunTiming run_timing;
            if (executor.run_rgb(transformed.rgb.data, 640, 640,
                                 static_cast<int>(transformed.rgb.step), output.data(), output.size(),
                                 &run_timing) != 0) {
                throw std::runtime_error("execution failed: " + executor.last_error());
            }
            const auto executor_end = Clock::now();
            const int id = image_id(images[index]);
            auto predictions = decode(output, id, transformed, options.confidence);
            const auto decode_end = Clock::now();
            all_predictions.insert(all_predictions.end(), predictions.begin(), predictions.end());
            executor_samples.push_back(run_timing.total_us);
            timing << index << '\t' << id << '\t' << transformed.original_width << '\t'
                   << transformed.original_height << '\t'
                   << elapsed_us(preprocess_begin, executor_begin) << '\t' << run_timing.total_us << '\t'
                   << elapsed_us(executor_end, decode_end) << '\t' << predictions.size() << "\t0x"
                   << std::hex << run_timing.output_hash << std::dec << '\n';
            if (options.log_every > 0 && ((index + 1) % static_cast<std::size_t>(options.log_every) == 0 ||
                                          index + 1 == images.size())) {
                std::cerr << "progress=" << (index + 1) << '/' << images.size()
                          << " executor_us=" << run_timing.total_us
                          << " detections=" << all_predictions.size() << '\n';
            }
        }
        write_predictions(options.output, all_predictions);
        const double mean = std::accumulate(executor_samples.begin(), executor_samples.end(), 0.0) /
                            static_cast<double>(executor_samples.size());
        std::cout << "images=" << images.size() << '\n'
                  << "predictions=" << all_predictions.size() << '\n'
                  << "executor_mean_us=" << std::setprecision(12) << mean << '\n'
                  << "total_wall_us=" << elapsed_us(evaluation_begin, Clock::now()) << '\n'
                  << "package_manifest_sha256=" << executor.package_manifest_sha256() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
