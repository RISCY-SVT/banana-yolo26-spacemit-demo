/**
 * @file yolo26_coco_predict.cpp
 * @brief Batch COCO prediction harness for YOLO26 vendor-ORT rt204 R&D gates.
 */

#include "banana_demo/app/options.h"
#include "banana_demo/infer/detector.h"
#include "banana_demo/util/pinning.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kCocoCategoryIds[80] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
};

struct Cli {
    std::string model;
    std::string labels = "assets/coco80.txt";
    std::string images_dir;
    std::string output_json;
    std::string timing_tsv;
    std::string provider = "spacemit";
    std::string pin = "cluster0";
    int input_size = 640;
    int threads = 4;
    int limit = 0;
    int log_every = 100;
    float conf = 0.001f;
    float iou = 0.7f;
};

void Usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " --model <model.onnx> --images <val2017> --output <predictions.json> [options]\n"
              << "Options:\n"
              << "  --labels <coco80.txt>\n"
              << "  --timing-tsv <timing.tsv>\n"
              << "  --provider spacemit|cpu\n"
              << "  --input-size 640\n"
              << "  --threads 4\n"
              << "  --pin cluster0|none|list:<csv>\n"
              << "  --conf 0.001\n"
              << "  --iou 0.7\n"
              << "  --limit <N>\n"
              << "  --log-every <N>\n";
}

bool NeedValue(int i, int argc) { return i + 1 < argc; }

Cli Parse(int argc, char** argv) {
    Cli cli;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (!NeedValue(i, argc)) {
                throw std::runtime_error("missing value for " + arg);
            }
            return argv[++i];
        };
        if (arg == "--model") cli.model = next();
        else if (arg == "--labels") cli.labels = next();
        else if (arg == "--images") cli.images_dir = next();
        else if (arg == "--output") cli.output_json = next();
        else if (arg == "--timing-tsv") cli.timing_tsv = next();
        else if (arg == "--provider") cli.provider = next();
        else if (arg == "--pin") cli.pin = next();
        else if (arg == "--input-size") cli.input_size = std::stoi(next());
        else if (arg == "--threads") cli.threads = std::stoi(next());
        else if (arg == "--limit") cli.limit = std::stoi(next());
        else if (arg == "--log-every") cli.log_every = std::stoi(next());
        else if (arg == "--conf") cli.conf = std::stof(next());
        else if (arg == "--iou") cli.iou = std::stof(next());
        else if (arg == "--help" || arg == "-h") {
            Usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (cli.model.empty() || cli.images_dir.empty() || cli.output_json.empty()) {
        Usage(argv[0]);
        throw std::runtime_error("--model, --images, and --output are required");
    }
    return cli;
}

std::vector<fs::path> ListImages(const fs::path& dir, int limit) {
    std::vector<fs::path> images;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string ext = entry.path().extension().string();
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") images.push_back(entry.path());
    }
    std::sort(images.begin(), images.end());
    if (limit > 0 && static_cast<size_t>(limit) < images.size()) images.resize(static_cast<size_t>(limit));
    return images;
}

int ImageIdFromPath(const fs::path& path) {
    std::string stem = path.stem().string();
    size_t pos = 0;
    while (pos + 1 < stem.size() && stem[pos] == '0') ++pos;
    return std::stoi(stem.substr(pos));
}

void WriteJsonNumber(std::ostream& os, float value) {
    if (!std::isfinite(value)) value = 0.0f;
    os << std::fixed << std::setprecision(6) << value;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Cli cli = Parse(argc, argv);
        std::vector<int> pin_cpus;
        std::vector<int> cluster0;
        std::vector<int> cluster1;
        std::string error;
        if (!banana_demo::PreparePinCpus(cli.pin, pin_cpus, cluster0, cluster1, error)) {
            throw std::runtime_error(error);
        }
        if (!banana_demo::ApplyProcessAffinity(pin_cpus, error)) {
            throw std::runtime_error(error);
        }

        banana_demo::AppOptions options;
        options.model = cli.model;
        options.labels = cli.labels;
        options.input_size = cli.input_size;
        options.provider = cli.provider;
        options.pin = cli.pin;
        options.threads = cli.threads;
        options.conf_threshold = cli.conf;
        options.iou_threshold = cli.iou;
        options.display = 0;
        options.headless = 1;
        options.quiet = 1;
        options.benchmark_only = 0;
        options.preprocess_mode = "letterbox";
        options.source = "image:" + cli.images_dir;

        banana_demo::Yolo11Detector detector(options);
        const auto images = ListImages(cli.images_dir, cli.limit);
        fs::create_directories(fs::path(cli.output_json).parent_path());
        if (!cli.timing_tsv.empty()) fs::create_directories(fs::path(cli.timing_tsv).parent_path());
        std::ofstream json(cli.output_json);
        if (!json) throw std::runtime_error("failed to open output JSON: " + cli.output_json);
        std::ofstream timing;
        if (!cli.timing_tsv.empty()) {
            timing.open(cli.timing_tsv);
            if (!timing) throw std::runtime_error("failed to open timing TSV: " + cli.timing_tsv);
            timing << "image_id\tpath\tobjects\tpreprocess_ms\tinference_ms\tpostprocess_ms\ttotal_ms\toutput_sha256\tdetections_sha256\n";
        }

        std::cout << "provider=" << cli.provider << "\n";
        std::cout << "model=" << cli.model << "\n";
        std::cout << "images_dir=" << cli.images_dir << "\n";
        std::cout << "image_count=" << images.size() << "\n";
        std::cout << "affinity=" << banana_demo::FormatCpuList(banana_demo::CurrentAffinity()) << "\n";
        std::cout << detector.ProviderSummary() << "\n";

        json << "[\n";
        bool first_prediction = true;
        size_t total_detections = 0;
        std::vector<double> total_ms;
        total_ms.reserve(images.size());
        const auto begin_all = std::chrono::steady_clock::now();
        for (size_t idx = 0; idx < images.size(); ++idx) {
            const fs::path& image_path = images[idx];
            cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
            if (image.empty()) {
                std::cerr << "WARN failed_to_read=" << image_path << "\n";
                continue;
            }
            const int image_id = ImageIdFromPath(image_path);
            banana_demo::InferenceResult result = detector.ProcessImage(image, false);
            total_ms.push_back(result.metrics.total_ms);
            total_detections += result.detections.size();
            if (timing) {
                timing << image_id << '\t' << image_path.string() << '\t' << result.detections.size() << '\t'
                       << std::fixed << std::setprecision(6)
                       << result.metrics.preprocess_ms << '\t' << result.metrics.inference_ms << '\t'
                       << result.metrics.postprocess_ms << '\t' << result.metrics.total_ms << '\t'
                       << result.output_sha256 << '\t' << result.detections_sha256 << "\n";
            }
            for (const auto& det : result.detections) {
                if (det.class_id < 0 || det.class_id >= 80) continue;
                const float x = std::max(0.0f, det.x1);
                const float y = std::max(0.0f, det.y1);
                const float w = std::max(0.0f, det.x2 - det.x1);
                const float h = std::max(0.0f, det.y2 - det.y1);
                if (w <= 0.0f || h <= 0.0f || det.score <= 0.0f) continue;
                if (!first_prediction) json << ",\n";
                first_prediction = false;
                json << "  {\"image_id\":" << image_id << ",\"category_id\":" << kCocoCategoryIds[det.class_id]
                     << ",\"bbox\":[";
                WriteJsonNumber(json, x); json << ',';
                WriteJsonNumber(json, y); json << ',';
                WriteJsonNumber(json, w); json << ',';
                WriteJsonNumber(json, h); json << "],\"score\":";
                WriteJsonNumber(json, det.score);
                json << '}';
            }
            if (cli.log_every > 0 && ((idx + 1) % static_cast<size_t>(cli.log_every) == 0 || idx + 1 == images.size())) {
                std::cout << "progress=" << (idx + 1) << "/" << images.size()
                          << " last_image_id=" << image_id
                          << " objects=" << result.detections.size()
                          << " total_ms=" << std::fixed << std::setprecision(3) << result.metrics.total_ms << "\n";
            }
        }
        json << "\n]\n";
        const auto end_all = std::chrono::steady_clock::now();
        const double wall_s = std::chrono::duration<double>(end_all - begin_all).count();
        double mean_total = 0.0;
        if (!total_ms.empty()) mean_total = std::accumulate(total_ms.begin(), total_ms.end(), 0.0) / total_ms.size();
        std::cout << "done_images=" << total_ms.size() << "\n";
        std::cout << "total_detections=" << total_detections << "\n";
        std::cout << "mean_total_ms=" << std::fixed << std::setprecision(6) << mean_total << "\n";
        std::cout << "wall_seconds=" << std::fixed << std::setprecision(3) << wall_s << "\n";
        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "Ort::Exception: " << e.what() << "\n";
        return 10;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 2;
    }
}
