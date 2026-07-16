/**
 * @file renderer.h
 * @brief Rendering helpers for annotated image and live camera output.
 */

#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "banana_demo/infer/yolo26_executor_detector.h"

namespace banana_demo {

/**
 * @brief Draws detections and status overlays on frames.
 */
class Renderer
{
public:
    /** @brief Construct the stateless renderer. */
    Renderer();

    /**
     * @brief Draw detections and timing metadata on a frame.
     * @param image Input frame.
     * @param detections Detections in source-image coordinates.
     * @param labels Class labels for @p detections.
     * @param metrics Per-frame timing metrics to overlay.
     * @return Annotated image.
     */
    cv::Mat Draw(const cv::Mat& image, const std::vector<Detection>& detections,
                 const std::vector<std::string>& labels, const FrameMetrics& metrics,
                 const std::string& profile, const std::string& flow,
                 const std::string& camera_format) const;

    cv::Mat DrawDetections(const cv::Mat& image, const std::vector<Detection>& detections,
                           const std::vector<std::string>& labels) const;

    void DrawOverlay(cv::Mat& image, const FrameMetrics& metrics, const std::string& profile,
                     const std::string& flow, const std::string& camera_format) const;
    /**
     * @brief Attempt to show a frame in an OpenCV HighGUI window.
     * @param window_name Stable window title.
     * @param image Frame to show.
     * @param error Human-readable backend failure text.
     * @return `true` on success.
     */
    bool TryShow(const std::string& window_name, const cv::Mat& image, std::string& error);

private:
    bool window_created_ = false;
};

}  // namespace banana_demo
