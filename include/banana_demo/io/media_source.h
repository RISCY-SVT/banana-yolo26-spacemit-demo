/**
 * @file media_source.h
 * @brief Image and camera input abstraction for the demo application.
 * @details Handles source parsing, camera target resolution, and lightweight
 * capture telemetry shared by image mode and live camera mode.
 */

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <string>

#include "banana_demo/app/options.h"

namespace banana_demo {

/**
 * @brief Media input wrapper for one image or one live camera stream.
 */
class MediaSource
{
public:
    /**
     * @brief Construct a media source from application options.
     * @param options Immutable application options snapshot.
     */
    explicit MediaSource(const AppOptions& options);
    /** @brief Release any open capture handle. */
    ~MediaSource();

    /**
     * @brief Open the configured media source.
     * @param error Human-readable error message on failure.
     * @return `true` on success.
     */
    bool Open(std::string& error);
    /** @brief Return whether the source is a still image. */
    bool IsImage() const;
    /** @brief Return whether the source is a live camera stream. */
    bool IsCamera() const;
    /** @brief Return whether the source is a finite video file. */
    bool IsVideo() const;
    /** @brief Return a short human-readable source description. */
    std::string Describe() const;
    /**
     * @brief Read the next frame.
     * @param frame Output image matrix.
     * @return `true` when a non-empty frame was produced.
     */
    bool Read(cv::Mat& frame);
    /** @brief Reopen a failed camera or video source. */
    bool Reopen(std::string& error);
    /** @brief Return the last frame read time in milliseconds. */
    double LastReadMs() const;
    /** @brief Return the resolved frame width. */
    int FrameWidth() const;
    /** @brief Return the resolved frame height. */
    int FrameHeight() const;
    /** @brief Return the runtime-reported camera FPS. */
    double Fps() const;
    /** @brief Return the effective pixel format reported by OpenCV. */
    std::string PixelFormat() const;
    /** @brief Return the capture open strategy that succeeded. */
    std::string OpenMethod() const;
    /** @brief Return the active OpenCV capture backend name if available. */
    std::string BackendName() const;
    /** @brief Return the canonical camera node or source file. */
    std::string ResolvedPath() const;
    /** @brief Return effective width, height, FPS, and FOURCC. */
    std::string EffectiveFormat() const;

private:
    /** @brief Open a still image source. */
    bool OpenImage(std::string& error);
    /** @brief Open a camera source with deterministic fallback attempts. */
    bool OpenCamera(std::string& error);
    /** @brief Open a video file. */
    bool OpenVideo(std::string& error);
    /** @brief Return the preferred OpenCV API for camera capture. */
    int ResolveCameraApi() const;
    /** @brief Resolve `camera:auto` to a concrete device path or index. */
    bool ResolveCameraTarget(std::string& error);
    /** @brief Apply user-requested camera properties after open. */
    void ApplyCameraProperties();
    /** @brief Resolve the requested FOURCC code for camera capture. */
    int ResolveFourcc() const;
    /** @brief Parse `/dev/videoN` to a numeric OpenCV camera index. */
    static int ParseVideoIndex(const std::string& path);
    /** @brief Convert a FOURCC integer to a printable short string. */
    static std::string FourccToString(int fourcc);

    /** Immutable per-run configuration snapshot. */
    AppOptions options_;
    /** Whether the opened source is a still image. */
    bool is_image_ = false;
    /** Whether the opened source is a live camera stream. */
    bool is_camera_ = false;
    /** Whether the opened source is a video file. */
    bool is_video_ = false;
    /** Whether the single still image was already returned. */
    bool image_consumed_ = false;
    /** Resolved image path for still-image mode. */
    std::string image_path_;
    /** Resolved video file path. */
    std::string video_path_;
    /** Original camera path or index argument. */
    std::string camera_path_;
    /** Canonical resolved camera path, usually `/dev/videoN`. */
    std::string camera_resolved_path_;
    /** Human-readable camera description used in logs. */
    std::string camera_display_name_;
    /** OpenCV-reported pixel format after camera open. */
    std::string camera_pixfmt_actual_ = "unknown";
    /** Human-readable name of the successful open strategy. */
    std::string camera_open_method_ = "unopened";
    /** OpenCV backend name reported by the active capture handle. */
    std::string camera_backend_name_ = "unknown";
    /** Parsed numeric camera index when available. */
    int camera_index_ = -1;
    /** Cached still-image contents. */
    cv::Mat image_;
    /** Active OpenCV capture handle for live mode. */
    cv::VideoCapture capture_;
    /** Last measured frame read latency in milliseconds. */
    double last_read_ms_ = 0.0;
};

}  // namespace banana_demo
