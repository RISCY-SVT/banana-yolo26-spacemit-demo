#pragma once

#include <y26_k1x_executor.h>

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "banana_demo/app/options.h"

namespace banana_demo {

struct Detection {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;
    int class_id = -1;
    float letterbox_x1 = 0.0f;
    float letterbox_y1 = 0.0f;
    float letterbox_x2 = 0.0f;
    float letterbox_y2 = 0.0f;
};

struct LetterboxInfo {
    int source_width = 0;
    int source_height = 0;
    int resized_width = 0;
    int resized_height = 0;
    double ratio = 1.0;
    double pad_x = 0.0;
    double pad_y = 0.0;
    int paste_x = 0;
    int paste_y = 0;
};

struct FrameMetrics {
    double capture_ms = 0.0;
    double resize_letterbox_ms = 0.0;
    double bgr_to_rgb_ms = 0.0;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double render_ms = 0.0;
    double display_ms = 0.0;
    double record_ms = 0.0;
    double total_ms = 0.0;
    double read_to_display_ms = 0.0;
    double processed_fps = 0.0;
    double capture_fps = 0.0;
    std::uint64_t output_hash = 0;
    std::uint64_t source_sequence = 0;
    std::uint64_t dropped_frames = 0;
    int objects = 0;
};

struct InferenceResult {
    std::vector<Detection> detections;
    LetterboxInfo letterbox;
    FrameMetrics metrics;
};

LetterboxInfo ComputeLetterbox(int source_width, int source_height);
std::vector<Detection> DecodeYolo26Output(const float* output, std::size_t elements,
                                          const LetterboxInfo& letterbox,
                                          float confidence_threshold,
                                          int class_count);

class Yolo26ExecutorDetector {
public:
    explicit Yolo26ExecutorDetector(const AppOptions& options);
    ~Yolo26ExecutorDetector();

    Yolo26ExecutorDetector(const Yolo26ExecutorDetector&) = delete;
    Yolo26ExecutorDetector& operator=(const Yolo26ExecutorDetector&) = delete;

    InferenceResult Process(const cv::Mat& bgr);
    const std::vector<std::string>& Labels() const noexcept;
    std::string BuildInfoSummary() const;
    const y26_build_info& BuildInfo() const noexcept;

private:
    void LoadLabels(const std::string& path);
    void Prepare(const AppOptions& options);

    AppOptions options_;
    y26_executor* executor_ = nullptr;
    y26_build_info build_info_{};
    std::vector<std::string> labels_;
    std::vector<float> output_;
};

}  // namespace banana_demo
