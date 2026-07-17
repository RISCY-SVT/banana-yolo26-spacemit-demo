#pragma once

#include <string>

namespace banana_demo {

struct AppOptions {
    std::string package;
    std::string expected_manifest_sha256 =
        "fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be";
    std::string labels = "assets/coco80.txt";
    std::string source = "camera:auto";
    std::string profile = "low-latency";
    std::string flow = "latest-frame";
    float confidence_threshold = 0.25f;
    bool display = true;
    bool headless = false;
    bool quiet = false;
    int camera_width = 1280;
    int camera_height = 720;
    double camera_fps = 30.0;
    std::string camera_fourcc = "MJPG";
    int max_frames = 0;
    double duration_seconds = 0.0;
    int warmup_frames = 30;
    int opencv_threads = 1;
    int reconnect_attempts = 3;
    int capture_cpu = -1;
    bool reuse_buffers = true;
    std::string save_frame;
    std::string screenshot_dir = "/data/Screenshots";
    std::string record_path;
    std::string record_mode = "async";
    std::string metrics_tsv;
    std::string detections_tsv;
    std::string log_file;
    bool print_build_info = false;
};

enum class ParseResult {
    kRun,
    kHelp,
    kError,
};

ParseResult ParseAppOptions(int argc, char** argv, AppOptions& options, std::string& error);
std::string BuildUsage(const char* program);

}  // namespace banana_demo
