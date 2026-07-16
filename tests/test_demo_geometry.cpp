#include "banana_demo/infer/yolo26_executor_detector.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>

namespace {
int failures = 0;

void Check(bool value, const char* name) {
    std::cout << name << '\t' << (value ? "pass" : "fail") << '\n';
    if (!value) ++failures;
}

bool Near(double left, double right, double tolerance = 1.0e-6) {
    return std::fabs(left - right) <= tolerance;
}
}  // namespace

int main() {
    const auto hd = banana_demo::ComputeLetterbox(1280, 720);
    Check(Near(hd.ratio, 0.5) && hd.resized_width == 640 && hd.resized_height == 360 &&
          Near(hd.pad_x, 0.0) && Near(hd.pad_y, 140.0) && hd.paste_x == 0 && hd.paste_y == 140,
          "letterbox_1280x720");
    const auto vga = banana_demo::ComputeLetterbox(640, 480);
    Check(Near(vga.ratio, 1.0) && vga.resized_width == 640 && vga.resized_height == 480 &&
          Near(vga.pad_y, 80.0), "letterbox_640x480");
    const auto portrait = banana_demo::ComputeLetterbox(1088, 1920);
    Check(portrait.resized_width == 363 && portrait.resized_height == 640 &&
          Near(portrait.pad_x, 138.5) && portrait.paste_x == 138,
          "letterbox_1088x1920");

    std::array<float, Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS> output{};
    output[0] = 0.0f;
    output[1] = 140.0f;
    output[2] = 640.0f;
    output[3] = 500.0f;
    output[4] = 0.75f;
    output[5] = 0.0f;
    output[6 + 4] = 0.9f;
    output[6 + 5] = 1.5f;
    output[12] = std::numeric_limits<float>::quiet_NaN();
    output[12 + 4] = 0.9f;
    const auto detections = banana_demo::DecodeYolo26Output(
        output.data(), output.size(), hd, 0.25f, 80);
    Check(detections.size() == 1U, "decode_filters_invalid_rows");
    if (!detections.empty()) {
        const auto& detection = detections.front();
        Check(Near(detection.x1, 0.0) && Near(detection.y1, 0.0) &&
              Near(detection.x2, 1279.0) && Near(detection.y2, 719.0) &&
              detection.class_id == 0 && Near(detection.score, 0.75),
              "decode_deletterbox_clamp");
        Check(Near(detection.letterbox_x1, 0.0) && Near(detection.letterbox_y1, 140.0) &&
              Near(detection.letterbox_x2, 640.0) && Near(detection.letterbox_y2, 500.0),
              "decode_preserves_letterbox_box");
    }
    std::cout << "failures\t" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
