#include "banana_demo/infer/yolo26_executor_detector.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#if defined(Y26_DEMO_STAGE60_STATIC_PROFILE)
#include "y26_k1x_full_executor.h"
#include "y26_k1x_package.h"
#endif

namespace banana_demo {
namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

#if !defined(Y26_DEMO_STAGE60_STATIC_PROFILE)
std::string ExecutorError(y26_executor* executor, y26_status status, const char* action) {
    std::ostringstream out;
    out << action << " failed: " << y26_status_string(status);
    if (executor != nullptr) {
        const char* detail = y26_executor_last_error(executor);
        if (detail != nullptr && *detail != '\0') out << ": " << detail;
    }
    return out.str();
}
#endif

}  // namespace

LetterboxInfo ComputeLetterbox(int source_width, int source_height, int input_resolution) {
    if (source_width <= 0 || source_height <= 0) {
        throw std::invalid_argument("letterbox dimensions must be positive");
    }
    if (input_resolution <= 0) {
        throw std::invalid_argument("letterbox input resolution must be positive");
    }
    LetterboxInfo result;
    result.source_width = source_width;
    result.source_height = source_height;
    result.ratio = std::min(static_cast<double>(input_resolution) / source_width,
                            static_cast<double>(input_resolution) / source_height);
    result.resized_width = static_cast<int>(std::nearbyint(source_width * result.ratio));
    result.resized_height = static_cast<int>(std::nearbyint(source_height * result.ratio));
    result.pad_x = (static_cast<double>(input_resolution) - result.resized_width) / 2.0;
    result.pad_y = (static_cast<double>(input_resolution) - result.resized_height) / 2.0;
    result.paste_x = static_cast<int>(std::nearbyint(result.pad_x - 0.1));
    result.paste_y = static_cast<int>(std::nearbyint(result.pad_y - 0.1));
    return result;
}

std::vector<Detection> DecodeYolo26Output(const float* output, std::size_t elements,
                                          const LetterboxInfo& letterbox,
                                          float confidence_threshold,
                                          int class_count,
                                          int input_resolution) {
    if (output == nullptr || elements != Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS ||
        letterbox.source_width <= 0 || letterbox.source_height <= 0 ||
        !(letterbox.ratio > 0.0) || class_count <= 0 || input_resolution <= 0) {
        throw std::invalid_argument("invalid YOLO26 decode contract");
    }
    std::vector<Detection> detections;
    detections.reserve(300);
    for (std::size_t index = 0; index < 300U; ++index) {
        const float* row = output + index * 6U;
        if (!std::isfinite(row[0]) || !std::isfinite(row[1]) ||
            !std::isfinite(row[2]) || !std::isfinite(row[3]) ||
            !std::isfinite(row[4]) || !std::isfinite(row[5]) ||
            !(row[4] >= confidence_threshold)) {
            continue;
        }
        const int class_id = static_cast<int>(std::nearbyint(row[5]));
        if (class_id < 0 || class_id >= class_count ||
            std::fabs(row[5] - class_id) > 1.0e-4f) {
            continue;
        }
        const auto map_x = [&](float coordinate) {
            const double bounded = std::clamp(
                static_cast<double>(coordinate), 0.0, static_cast<double>(input_resolution));
            return static_cast<float>(std::clamp(
                (bounded - letterbox.pad_x) / letterbox.ratio,
                0.0, static_cast<double>(letterbox.source_width - 1)));
        };
        const auto map_y = [&](float coordinate) {
            const double bounded = std::clamp(
                static_cast<double>(coordinate), 0.0, static_cast<double>(input_resolution));
            return static_cast<float>(std::clamp(
                (bounded - letterbox.pad_y) / letterbox.ratio,
                0.0, static_cast<double>(letterbox.source_height - 1)));
        };
        Detection detection{map_x(row[0]), map_y(row[1]), map_x(row[2]), map_y(row[3]),
                            row[4], class_id};
        const float extent = static_cast<float>(input_resolution);
        detection.letterbox_x1 = std::clamp(row[0], 0.0f, extent);
        detection.letterbox_y1 = std::clamp(row[1], 0.0f, extent);
        detection.letterbox_x2 = std::clamp(row[2], 0.0f, extent);
        detection.letterbox_y2 = std::clamp(row[3], 0.0f, extent);
        if (detection.x2 > detection.x1 && detection.y2 > detection.y1) {
            detections.push_back(detection);
        }
    }
    return detections;
}

#if defined(Y26_DEMO_STAGE60_STATIC_PROFILE)
namespace stage60_detail {

class ResearchExecutor {
public:
    y26::stage52::FullExecutor executor;
};

}  // namespace stage60_detail
#endif

Yolo26ExecutorDetector::Yolo26ExecutorDetector(const AppOptions& options)
    : options_(options), output_(Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS) {
    y26_build_info_init(&build_info_);
    const y26_status info_status = y26_executor_get_build_info(&build_info_);
    if (info_status != Y26_STATUS_OK) {
        throw std::runtime_error("executor build-info query failed");
    }
    const std::uint32_t required = Y26_CAPABILITY_IME | Y26_CAPABILITY_RVV |
        Y26_CAPABILITY_FROZEN_PROFILE | Y26_CAPABILITY_RGB_INPUT;
    if ((build_info_.capability_flags & required) != required) {
        std::ostringstream error;
        error << "official demo requires IME/RVV/frozen-profile/RGB capabilities; got 0x"
              << std::hex << build_info_.capability_flags;
        throw std::runtime_error(error.str());
    }
#if !defined(Y26_DEMO_STAGE60_STATIC_PROFILE)
    if (options.expected_manifest_sha256 != build_info_.expected_package_manifest_sha256) {
        throw std::runtime_error("requested manifest does not match the frozen release profile");
    }
#endif
    LoadLabels(options.labels);
    Prepare(options);
}

Yolo26ExecutorDetector::~Yolo26ExecutorDetector() {
#if !defined(Y26_DEMO_STAGE60_STATIC_PROFILE)
    y26_executor_destroy(executor_);
#endif
}

void Yolo26ExecutorDetector::LoadLabels(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open labels: " + path);
    std::string label;
    while (std::getline(stream, label)) {
        if (!label.empty() && label.back() == '\r') label.pop_back();
        if (!label.empty()) labels_.push_back(label);
    }
    if (labels_.size() != 80U) {
        throw std::runtime_error("COCO label file must contain exactly 80 non-empty lines");
    }
}

void Yolo26ExecutorDetector::Prepare(const AppOptions& options) {
#if defined(Y26_DEMO_STAGE60_STATIC_PROFILE)
    research_executor_ = std::make_unique<stage60_detail::ResearchExecutor>();
    y26::stage52::RunConfig config;
    config.workers = 4;
    config.worker_cpu_begin = 0;
    config.controller_cpu = 4;
    config.scheduler = y26::stage52::SchedulerMode::safe;
    config.wake_policy = options.profile == "compatibility"
        ? y26::stage52::WakePolicy::condition_variable
        : y26::stage52::WakePolicy::frame_gated_spin;
    config.compute = y26::stage52::ComputeMode::optimized;
    config.allow_stage60_static_profiles = true;
    if (research_executor_->executor.prepare(
            options.package, options.expected_manifest_sha256, config) != 0) {
        throw std::runtime_error(
            "Stage60 prepare failed: " + research_executor_->executor.last_error());
    }
    input_resolution_ = research_executor_->executor.input_width();
#else
    executor_ = y26_executor_create();
    if (executor_ == nullptr) throw std::runtime_error("executor allocation failed");

    y26_executor_options executor_options;
    y26_executor_options_init(&executor_options);
    executor_options.wake_policy = options.profile == "compatibility"
        ? Y26_WAKE_CONDITION_VARIABLE : Y26_WAKE_FRAME_GATED_SPIN;
    const y26_status status = y26_executor_prepare(
        executor_, options.package.c_str(), options.expected_manifest_sha256.c_str(),
        &executor_options);
    if (status != Y26_STATUS_OK) throw std::runtime_error(ExecutorError(executor_, status, "prepare"));
#endif
}

InferenceResult Yolo26ExecutorDetector::Process(const cv::Mat& bgr) {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::runtime_error("demo input must be a non-empty CV_8UC3 BGR frame");
    }

    InferenceResult result;
    result.metrics.input_resolution = input_resolution_;
    result.letterbox = ComputeLetterbox(bgr.cols, bgr.rows, input_resolution_);

    const auto resize_begin = Clock::now();
    cv::Mat resized;
    cv::Mat canvas;
    cv::Mat rgb;
    if (options_.reuse_buffers) {
        resized_buffer_.create(result.letterbox.resized_height,
                               result.letterbox.resized_width, CV_8UC3);
        canvas_buffer_.create(input_resolution_, input_resolution_, CV_8UC3);
        rgb_buffer_.create(input_resolution_, input_resolution_, CV_8UC3);
        resized = resized_buffer_;
        canvas = canvas_buffer_;
        rgb = rgb_buffer_;
    }
    cv::resize(bgr, resized,
               cv::Size(result.letterbox.resized_width, result.letterbox.resized_height),
               0.0, 0.0, cv::INTER_LINEAR);
    if (canvas.empty()) canvas.create(input_resolution_, input_resolution_, CV_8UC3);
    canvas.setTo(cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(result.letterbox.paste_x, result.letterbox.paste_y,
                                  resized.cols, resized.rows)));
    const auto resize_end = Clock::now();

    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) rgb = rgb.clone();
    const auto color_end = Clock::now();

    double inference_us = 0.0;
    std::uint64_t output_hash = 0;
#if defined(Y26_DEMO_STAGE60_STATIC_PROFILE)
    y26::stage52::RunTiming timing;
    const int run_status = research_executor_->executor.run_rgb(
        rgb.data, input_resolution_, input_resolution_, static_cast<int>(rgb.step),
        output_.data(), output_.size(), &timing);
#else
    y26_run_timing timing{};
    const y26_status run_status = y26_executor_run_rgb(
        executor_, rgb.data, input_resolution_, input_resolution_, static_cast<int>(rgb.step), output_.data(),
        output_.size(), &timing);
#endif
    const auto inference_end = Clock::now();
#if defined(Y26_DEMO_STAGE60_STATIC_PROFILE)
    if (run_status != 0) {
        throw std::runtime_error(
            "Stage60 RGB inference failed: " + research_executor_->executor.last_error());
    }
    inference_us = timing.total_us;
    output_hash = timing.output_hash;
#else
    if (run_status != Y26_STATUS_OK) {
        throw std::runtime_error(ExecutorError(executor_, run_status, "RGB inference"));
    }
    inference_us = timing.total_us;
    output_hash = timing.output_hash;
#endif

    result.detections = DecodeYolo26Output(
        output_.data(), output_.size(), result.letterbox, options_.confidence_threshold,
        static_cast<int>(labels_.size()), input_resolution_);
    const auto postprocess_end = Clock::now();

    result.metrics.resize_letterbox_ms = ElapsedMs(resize_begin, resize_end);
    result.metrics.bgr_to_rgb_ms = ElapsedMs(resize_end, color_end);
    result.metrics.preprocess_ms = ElapsedMs(resize_begin, color_end);
    result.metrics.inference_ms = inference_us / 1000.0;
    result.metrics.postprocess_ms = ElapsedMs(inference_end, postprocess_end);
    result.metrics.total_ms = ElapsedMs(resize_begin, postprocess_end);
    result.metrics.output_hash = output_hash;
    result.metrics.objects = static_cast<int>(result.detections.size());
    return result;
}

const std::vector<std::string>& Yolo26ExecutorDetector::Labels() const noexcept {
    return labels_;
}

const y26_build_info& Yolo26ExecutorDetector::BuildInfo() const noexcept {
    return build_info_;
}

int Yolo26ExecutorDetector::InputResolution() const noexcept {
    return input_resolution_;
}

std::string Yolo26ExecutorDetector::BuildInfoSummary() const {
    std::ostringstream out;
    out << "release=" << build_info_.release_version
        << " abi=" << build_info_.abi_version
        << " source_commit=" << build_info_.source_commit
        << " contract=" << build_info_.integer_contract_id
        << " profile=" << build_info_.full_graph_profile_id
        << " capabilities=0x" << std::hex << build_info_.capability_flags << std::dec
        << " expected_manifest=" << build_info_.expected_package_manifest_sha256;
#if defined(Y26_DEMO_STAGE60_STATIC_PROFILE)
    out << " stage60_static_profile=1 input_resolution=" << input_resolution_
        << " package_manifest=" << research_executor_->executor.package_manifest_sha256();
#endif
    return out.str();
}

}  // namespace banana_demo
