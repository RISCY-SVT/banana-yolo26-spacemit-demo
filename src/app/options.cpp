#include "banana_demo/app/options.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace banana_demo {
namespace {

bool NeedValue(int index, int argc) { return index + 1 < argc; }

bool ParseInt(const char* text, int& value) {
    if (text == nullptr) return false;
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') return false;
    value = static_cast<int>(parsed);
    return true;
}

bool ParseDouble(const char* text, double& value) {
    if (text == nullptr) return false;
    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !std::isfinite(parsed)) return false;
    value = parsed;
    return true;
}

bool ParseFloat(const char* text, float& value) {
    double parsed = 0.0;
    if (!ParseDouble(text, parsed)) return false;
    value = static_cast<float>(parsed);
    return std::isfinite(value);
}

bool ParseBool(const char* text, bool& value) {
    if (text == nullptr) return false;
    const std::string token(text);
    if (token == "1" || token == "true" || token == "yes") {
        value = true;
        return true;
    }
    if (token == "0" || token == "false" || token == "no") {
        value = false;
        return true;
    }
    return false;
}

bool SupportedModelResolution(int resolution) {
    switch (resolution) {
        case 256:
        case 320:
        case 352:
        case 384:
        case 416:
        case 448:
        case 512:
        case 640:
        case 768:
            return true;
        default:
            return false;
    }
}

}  // namespace

ParseResult ParseAppOptions(int argc, char** argv, AppOptions& options, std::string& error) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") return ParseResult::kHelp;
        if (arg == "--package" && NeedValue(i, argc)) options.package = argv[++i];
        else if (arg == "--expected-manifest-sha256" && NeedValue(i, argc))
            options.expected_manifest_sha256 = argv[++i];
        else if (arg == "--labels" && NeedValue(i, argc)) options.labels = argv[++i];
        else if (arg == "--source" && NeedValue(i, argc)) options.source = argv[++i];
        else if (arg == "--profile" && NeedValue(i, argc)) options.profile = argv[++i];
        else if (arg == "--model-resolution" && NeedValue(i, argc) &&
                 ParseInt(argv[++i], options.model_resolution)) {
            options.model_resolution_explicit = true;
        }
        else if (arg == "--flow" && NeedValue(i, argc)) options.flow = argv[++i];
        else if (arg == "--conf" && NeedValue(i, argc) && ParseFloat(argv[++i], options.confidence_threshold)) {}
        else if (arg == "--display" && NeedValue(i, argc) && ParseBool(argv[++i], options.display)) {}
        else if (arg == "--headless") { options.headless = true; options.display = false; }
        else if (arg == "--quiet") options.quiet = true;
        else if (arg == "--camera-width" && NeedValue(i, argc) && ParseInt(argv[++i], options.camera_width)) {}
        else if (arg == "--camera-height" && NeedValue(i, argc) && ParseInt(argv[++i], options.camera_height)) {}
        else if (arg == "--camera-fps" && NeedValue(i, argc) && ParseDouble(argv[++i], options.camera_fps)) {}
        else if (arg == "--camera-fourcc" && NeedValue(i, argc)) options.camera_fourcc = argv[++i];
        else if (arg == "--max-frames" && NeedValue(i, argc) && ParseInt(argv[++i], options.max_frames)) {}
        else if (arg == "--duration" && NeedValue(i, argc) && ParseDouble(argv[++i], options.duration_seconds)) {}
        else if (arg == "--warmup-frames" && NeedValue(i, argc) && ParseInt(argv[++i], options.warmup_frames)) {}
        else if (arg == "--opencv-threads" && NeedValue(i, argc) && ParseInt(argv[++i], options.opencv_threads)) {}
        else if (arg == "--reconnect-attempts" && NeedValue(i, argc) && ParseInt(argv[++i], options.reconnect_attempts)) {}
        else if (arg == "--capture-cpu" && NeedValue(i, argc) && ParseInt(argv[++i], options.capture_cpu)) {}
        else if (arg == "--reuse-buffers" && NeedValue(i, argc) && ParseBool(argv[++i], options.reuse_buffers)) {}
        else if (arg == "--save-frame" && NeedValue(i, argc)) options.save_frame = argv[++i];
        else if (arg == "--screenshot-dir" && NeedValue(i, argc)) options.screenshot_dir = argv[++i];
        else if (arg == "--record" && NeedValue(i, argc)) options.record_path = argv[++i];
        else if (arg == "--record-mode" && NeedValue(i, argc)) options.record_mode = argv[++i];
        else if (arg == "--metrics-tsv" && NeedValue(i, argc)) options.metrics_tsv = argv[++i];
        else if (arg == "--detections-tsv" && NeedValue(i, argc)) options.detections_tsv = argv[++i];
        else if (arg == "--log-file" && NeedValue(i, argc)) options.log_file = argv[++i];
        else if (arg == "--build-info") options.print_build_info = true;
        else if (arg == "--license") options.print_license = true;
        else if (arg == "--source-info") options.print_source_info = true;
        else {
            error = "unknown or invalid argument: " + arg;
            return ParseResult::kError;
        }
    }

    if (options.print_build_info || options.print_license || options.print_source_info)
        return ParseResult::kRun;
    if (options.package.empty()) error = "--package is required";
    else if (options.source.rfind("camera:", 0) != 0 &&
             options.source.rfind("image:", 0) != 0 &&
             options.source.rfind("video:", 0) != 0)
        error = "--source must be camera:, image:, or video:";
    else if (options.profile != "compatibility" && options.profile != "low-latency" &&
             options.profile != "low-latency-dedicated")
        error = "--profile must be compatibility|low-latency|low-latency-dedicated";
    else if (!SupportedModelResolution(options.model_resolution))
        error = "--model-resolution must be one of 256,320,352,384,416,448,512,640,768";
#if !defined(Y26_DEMO_STAGE60_STATIC_PROFILE)
    else if (options.model_resolution != 640)
        error = "non-R640 profiles require the integrated multiprofile research build";
#endif
    else if (options.model_resolution != 640 && !options.model_resolution_explicit)
        error = "non-R640 profiles require explicit --model-resolution";
    else if (options.model_resolution != 640 &&
             options.expected_manifest_sha256 ==
                 "fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be")
        error = "non-R640 profiles require their explicit --expected-manifest-sha256";
    else if (options.flow != "sequential" && options.flow != "latest-frame")
        error = "--flow must be sequential|latest-frame";
    else if (options.record_mode != "sync" && options.record_mode != "async")
        error = "--record-mode must be sync|async";
    else if (!(options.confidence_threshold >= 0.0f && options.confidence_threshold <= 1.0f))
        error = "--conf must be in [0,1]";
    else if (options.camera_width <= 0 || options.camera_height <= 0 || options.camera_fps <= 0.0)
        error = "camera width, height, and FPS must be positive";
    else if (options.max_frames < 0 || options.duration_seconds < 0.0 ||
             options.warmup_frames < 0 || options.opencv_threads < 1 || options.reconnect_attempts < 0)
        error = "frame, duration, thread, or reconnect count is invalid";
    else if (options.capture_cpu < -1 || options.capture_cpu > 7)
        error = "--capture-cpu must be -1 or a CPU in [0,7]";

    return error.empty() ? ParseResult::kRun : ParseResult::kError;
}

std::string BuildUsage(const char* program) {
    std::ostringstream out;
    out << "Usage: " << program << " --package DIR --source TYPE:VALUE [options]\n\n"
        << "Source:\n"
        << "  --source camera:auto|camera:/dev/videoN|camera:N\n"
        << "  --source image:FILE|video:FILE\n"
        << "  --camera-width N --camera-height N --camera-fps N --camera-fourcc MJPG|YUYV\n"
        << "  --flow sequential|latest-frame\n\n"
        << "Executor:\n"
        << "  --profile compatibility|low-latency|low-latency-dedicated\n"
        << "  --model-resolution 640 (other accepted Q0 profiles require research build and explicit opt-in)\n"
        << "  --expected-manifest-sha256 HEX --labels FILE --conf FLOAT\n"
        << "  --build-info --license --source-info\n\n"
        << "Output and measurement:\n"
        << "  --display 0|1 --headless --record FILE --record-mode sync|async\n"
        << "  --save-frame FILE\n"
        << "  --screenshot-dir DIR --metrics-tsv FILE --detections-tsv FILE --log-file FILE\n"
        << "  --warmup-frames N --max-frames N --duration SECONDS\n"
        << "  --opencv-threads N --reconnect-attempts N --capture-cpu -1|0..7\n"
        << "  --reuse-buffers 0|1 --quiet\n\n"
        << "GUI keys: q/Esc exit, s save PNG, r toggle recording, space pause.\n";
#if defined(Y26_DEMO_STAGE60_STATIC_PROFILE)
    out << "Multiprofile research mode requires an explicit non-R640 resolution and matching package hash; no second NMS is run.\n";
#else
    out << "The model input is always exact 640x640 RGB8 letterbox; no second NMS is run.\n";
#endif
    return out.str();
}

}  // namespace banana_demo
