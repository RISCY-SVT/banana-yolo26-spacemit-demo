/**
 * @file renderer.cpp
 * @brief Frame annotation and display helpers.
 */

#include "banana_demo/render/renderer.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace banana_demo {

namespace {

/** @brief Pick a stable class color from a short palette. */
cv::Scalar ColorForClass(int class_id)
{
    static const std::array<cv::Scalar, 10> kPalette = {
        cv::Scalar(255, 99, 71), cv::Scalar(60, 179, 113), cv::Scalar(30, 144, 255),
        cv::Scalar(255, 215, 0), cv::Scalar(138, 43, 226), cv::Scalar(255, 105, 180),
        cv::Scalar(0, 206, 209), cv::Scalar(255, 140, 0), cv::Scalar(46, 139, 87),
        cv::Scalar(70, 130, 180),
    };
    return kPalette[static_cast<size_t>(class_id >= 0 ? class_id : 0) % kPalette.size()];
}

/** @brief Format the visible label text for one detection. */
std::string LabelForDetection(const Detection& det, const std::vector<std::string>& labels)
{
    std::ostringstream oss;
    if (det.class_id >= 0 && det.class_id < static_cast<int>(labels.size()))
        oss << labels[det.class_id];
    else
        oss << "class_" << det.class_id;
    oss.setf(std::ios::fixed);
    oss.precision(2);
    oss << " " << det.score;
    return oss.str();
}

}  // namespace

Renderer::Renderer() = default;

cv::Mat Renderer::Draw(const cv::Mat& image, const std::vector<Detection>& detections,
                       const std::vector<std::string>& labels, const FrameMetrics& metrics,
                       const std::string& profile, const std::string& flow,
                       const std::string& camera_format) const
{
    cv::Mat out = DrawDetections(image, detections, labels);
    DrawOverlay(out, metrics, profile, flow, camera_format);
    return out;
}

cv::Mat Renderer::DrawDetections(const cv::Mat& image,
                                 const std::vector<Detection>& detections,
                                 const std::vector<std::string>& labels) const
{
    cv::Mat out = image.clone();
    for (const auto& det : detections)
    {
        const cv::Scalar color = ColorForClass(det.class_id);
        const cv::Point p1(static_cast<int>(det.x1), static_cast<int>(det.y1));
        const cv::Point p2(static_cast<int>(det.x2), static_cast<int>(det.y2));
        cv::rectangle(out, p1, p2, color, 2);
        const std::string text = LabelForDetection(det, labels);
        cv::putText(out, text, cv::Point(p1.x, std::max(18, p1.y - 6)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv::LINE_AA);
    }

    return out;
}

void Renderer::DrawOverlay(cv::Mat& out, const FrameMetrics& metrics,
                           const std::string& profile, const std::string& flow,
                           const std::string& camera_format) const
{
    std::vector<std::string> lines;
    lines.reserve(6);
    {
        std::ostringstream text;
        text.setf(std::ios::fixed);
        text.precision(2);
        text << "objects=" << metrics.objects << " capture=" << metrics.capture_ms
             << " ms preprocess=" << metrics.preprocess_ms << " ms";
        lines.push_back(text.str());
    }
    {
        std::ostringstream text;
        text.setf(std::ios::fixed);
        text.precision(2);
        text << "inference=" << metrics.inference_ms << " ms postprocess="
             << metrics.postprocess_ms << " ms";
        lines.push_back(text.str());
    }
    {
        std::ostringstream text;
        text.setf(std::ios::fixed);
        text.precision(2);
        text << "render=" << metrics.render_ms << " ms display=" << metrics.display_ms
             << " ms total=" << metrics.total_ms << " ms";
        lines.push_back(text.str());
    }
    {
        std::ostringstream text;
        text.setf(std::ios::fixed);
        text.precision(2);
        text << "processed_fps=" << metrics.processed_fps << " capture_fps="
             << metrics.capture_fps << " dropped=" << metrics.dropped_frames;
        lines.push_back(text.str());
    }
    lines.push_back("profile=" + profile + " flow=" + flow + " input=640x640");
    lines.push_back("camera=" + camera_format);

    double font_scale = 0.50;
    const int available_width = std::max(1, out.cols - 36);
    for (const std::string& line : lines) {
        int baseline = 0;
        const cv::Size size = cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX,
                                              font_scale, 1, &baseline);
        if (size.width > available_width) {
            font_scale = std::max(0.36, font_scale * static_cast<double>(available_width) /
                                                   static_cast<double>(size.width));
        }
    }
    const int line_height = 22;
    const int overlay_height = 14 + static_cast<int>(lines.size()) * line_height;
    cv::rectangle(out, cv::Rect(8, 8, std::max(1, out.cols - 16), overlay_height),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    for (std::size_t index = 0; index < lines.size(); ++index) {
        cv::putText(out, lines[index], cv::Point(18, 29 + static_cast<int>(index) * line_height),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 255, 255), 1,
                    cv::LINE_AA);
    }
}

bool Renderer::TryShow(const std::string& window_name, const cv::Mat& image, std::string& error)
{
    try
    {
        if (!window_created_) {
            cv::namedWindow(window_name, cv::WINDOW_NORMAL);
            const double fit = std::min(
                {1.0, 1280.0 / static_cast<double>(image.cols),
                 720.0 / static_cast<double>(image.rows)});
            cv::resizeWindow(window_name,
                             static_cast<int>(std::nearbyint(image.cols * fit)),
                             static_cast<int>(std::nearbyint(image.rows * fit)));
            cv::moveWindow(window_name, 0, 0);
            window_created_ = true;
        }
        cv::imshow(window_name, image);
        return true;
    }
    catch (const cv::Exception& e)
    {
        error = e.what();
        return false;
    }
}

}  // namespace banana_demo
