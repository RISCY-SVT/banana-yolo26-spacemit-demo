/**
 * @file media_source.cpp
 * @brief Media input handling for still images and live V4L2 cameras.
 */

#include "banana_demo/io/media_source.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <system_error>
#include <vector>

#include <opencv2/imgcodecs.hpp>

namespace banana_demo {

namespace {

/** @brief Query the active OpenCV backend name without throwing. */
std::string SafeBackendName(const cv::VideoCapture& capture)
{
    try
    {
        if (!capture.isOpened())
            return "unknown";
        return capture.getBackendName();
    }
    catch (const cv::Exception&)
    {
        return "unknown";
    }
}

}  // namespace

MediaSource::MediaSource(const AppOptions& options) : options_(options) {}

MediaSource::~MediaSource()
{
    if (capture_.isOpened())
        capture_.release();
}

bool MediaSource::Open(std::string& error)
{
    if (options_.source.rfind("image:", 0) == 0)
    {
        is_image_ = true;
        image_path_ = options_.source.substr(6);
        return OpenImage(error);
    }

    if (options_.source.rfind("camera:", 0) == 0)
    {
        is_camera_ = true;
        camera_path_ = options_.source.substr(7);
        return OpenCamera(error);
    }

    if (options_.source.rfind("video:", 0) == 0)
    {
        is_video_ = true;
        video_path_ = options_.source.substr(6);
        return OpenVideo(error);
    }

    error = "unsupported source: " + options_.source;
    return false;
}

bool MediaSource::IsImage() const
{
    return is_image_;
}

bool MediaSource::IsCamera() const
{
    return is_camera_;
}

bool MediaSource::IsVideo() const
{
    return is_video_;
}

std::string MediaSource::Describe() const
{
    if (is_image_)
        return "image:" + image_path_;
    if (is_camera_)
        return "camera:" + (camera_display_name_.empty() ? camera_path_ : camera_display_name_);
    if (is_video_)
        return "video:" + video_path_;
    return options_.source;
}

bool MediaSource::Read(cv::Mat& frame)
{
    const auto start = std::chrono::steady_clock::now();
    if (is_image_)
    {
        if (image_consumed_)
            return false;
        frame = image_.clone();
        image_consumed_ = true;
    }
    else if (is_camera_ || is_video_)
    {
        if (!capture_.read(frame))
            return false;
    }
    else
    {
        return false;
    }
    const auto end = std::chrono::steady_clock::now();
    last_read_ms_ = std::chrono::duration<double, std::milli>(end - start).count();
    return !frame.empty();
}

bool MediaSource::Reopen(std::string& error)
{
    capture_.release();
    if (is_camera_)
        return OpenCamera(error);
    if (is_video_)
        return OpenVideo(error);
    error = "source cannot be reopened";
    return false;
}

double MediaSource::LastReadMs() const
{
    return last_read_ms_;
}

int MediaSource::FrameWidth() const
{
    if (is_image_)
        return image_.cols;
    if (is_camera_ || is_video_)
        return static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
    return 0;
}

int MediaSource::FrameHeight() const
{
    if (is_image_)
        return image_.rows;
    if (is_camera_ || is_video_)
        return static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
    return 0;
}

double MediaSource::Fps() const
{
    if (is_camera_ || is_video_)
        return capture_.get(cv::CAP_PROP_FPS);
    return 0.0;
}

std::string MediaSource::PixelFormat() const
{
    return camera_pixfmt_actual_;
}

std::string MediaSource::OpenMethod() const
{
    return camera_open_method_;
}

std::string MediaSource::BackendName() const
{
    return camera_backend_name_;
}

std::string MediaSource::ResolvedPath() const
{
    if (is_camera_)
        return camera_resolved_path_;
    if (is_video_)
        return video_path_;
    return image_path_;
}

std::string MediaSource::EffectiveFormat() const
{
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out << FrameWidth() << 'x' << FrameHeight() << '@' << std::setprecision(3) << Fps();
    if (is_camera_)
        out << ' ' << camera_pixfmt_actual_;
    return out.str();
}

bool MediaSource::OpenImage(std::string& error)
{
    image_consumed_ = false;
    image_ = cv::imread(image_path_, cv::IMREAD_COLOR);
    if (image_.empty())
    {
        error = "failed to read image: " + image_path_;
        return false;
    }
    return true;
}

bool MediaSource::OpenVideo(std::string& error)
{
    capture_.release();
    if (!capture_.open(video_path_))
    {
        error = "failed to open video: " + video_path_;
        return false;
    }
    camera_open_method_ = "video-file-auto";
    camera_backend_name_ = SafeBackendName(capture_);
    return true;
}

bool MediaSource::OpenCamera(std::string& error)
{
    if (!ResolveCameraTarget(error))
        return false;

    const int api = ResolveCameraApi();
    capture_.release();
    camera_open_method_ = "unopened";
    camera_backend_name_ = "unknown";

    // Prefer explicit V4L2 opens first, then fall back to generic OpenCV capture only once per target.
    const auto try_open = [&](const std::string& method_name, auto&& opener) -> bool {
        capture_.release();
        if (!opener())
            return false;
        camera_open_method_ = method_name;
        camera_backend_name_ = SafeBackendName(capture_);
        return true;
    };

    bool opened = false;
    if (camera_index_ >= 0)
        opened = try_open("index-v4l2", [&] { return capture_.open(camera_index_, api); });

    if (!opened && !camera_resolved_path_.empty())
        opened = try_open("resolved-path-v4l2", [&] { return capture_.open(camera_resolved_path_, api); });

    if (!opened && !camera_path_.empty())
        opened = try_open("requested-path-v4l2", [&] { return capture_.open(camera_path_, api); });

    if (!opened && camera_index_ >= 0)
        opened = try_open("index-auto", [&] { return capture_.open(camera_index_); });

    if (!opened && !camera_resolved_path_.empty())
        opened = try_open("resolved-path-auto", [&] { return capture_.open(camera_resolved_path_); });

    if (!opened && !camera_path_.empty())
        opened = try_open("requested-path-auto", [&] { return capture_.open(camera_path_); });

    if (!opened)
    {
        error = "failed to open camera: " + camera_display_name_;
        return false;
    }

    ApplyCameraProperties();
    camera_pixfmt_actual_ = FourccToString(static_cast<int>(capture_.get(cv::CAP_PROP_FOURCC)));
    camera_backend_name_ = SafeBackendName(capture_);
    return true;
}

int MediaSource::ResolveCameraApi() const
{
    return cv::CAP_V4L2;
}

void MediaSource::ApplyCameraProperties()
{
    // Keep capture latency low by preferring the smallest backend queue the driver accepts.
    capture_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    const int fourcc = ResolveFourcc();
    if (fourcc != 0)
        capture_.set(cv::CAP_PROP_FOURCC, fourcc);
    capture_.set(cv::CAP_PROP_FRAME_WIDTH, options_.camera_width);
    capture_.set(cv::CAP_PROP_FRAME_HEIGHT, options_.camera_height);
    capture_.set(cv::CAP_PROP_FPS, options_.camera_fps);
}

int MediaSource::ResolveFourcc() const
{
    std::string format = options_.camera_fourcc;
    std::transform(format.begin(), format.end(), format.begin(),
                   [](unsigned char value) { return static_cast<char>(std::toupper(value)); });
    if (format == "MJPG")
        return cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    if (format == "YUYV")
        return cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V');
    return 0;
}

bool MediaSource::ResolveCameraTarget(std::string& error)
{
    camera_resolved_path_.clear();
    camera_display_name_.clear();
    camera_index_ = -1;

    std::string target = camera_path_;
    if (target.empty() || target == "auto")
    {
        static const char* kGlobs[] = {
            "/dev/v4l/by-id",
            "/dev/v4l/by-path",
        };
        for (const char* base : kGlobs)
        {
            std::error_code ec;
            if (!std::filesystem::exists(base, ec))
                continue;
            std::vector<std::filesystem::path> candidates;
            for (const auto& entry : std::filesystem::directory_iterator(base, ec))
            {
                const std::string candidate = entry.path().filename().string();
                if (candidate.find("video-index0") == std::string::npos)
                    continue;
                candidates.push_back(entry.path());
            }
            std::sort(candidates.begin(), candidates.end());
            if (!candidates.empty()) target = candidates.front().string();
            if (!target.empty() && target != "auto")
                break;
        }
        if (target.empty() || target == "auto")
        {
            error = "failed to auto-select camera: no stable /dev/v4l/by-id or /dev/v4l/by-path capture node found";
            return false;
        }
    }

    if (!target.empty() && std::all_of(target.begin(), target.end(), [](unsigned char ch) { return std::isdigit(ch); }))
    {
        camera_index_ = std::stoi(target);
        camera_resolved_path_ = "/dev/video" + target;
        camera_display_name_ = camera_resolved_path_ + " (index=" + target + ")";
        camera_path_ = target;
        return true;
    }

    camera_path_ = target;
    camera_display_name_ = target;
    camera_resolved_path_ = target;

    std::error_code ec;
    if (std::filesystem::exists(target, ec))
    {
        const std::filesystem::path canonical = std::filesystem::canonical(target, ec);
        if (!ec)
        {
            camera_resolved_path_ = canonical.string();
            camera_display_name_ = target == camera_resolved_path_ ? target : target + " -> " + camera_resolved_path_;
        }
    }

    const int parsed_index = ParseVideoIndex(camera_resolved_path_);
    if (parsed_index >= 0)
        camera_index_ = parsed_index;

    return true;
}

int MediaSource::ParseVideoIndex(const std::string& path)
{
    const std::filesystem::path fs_path(path);
    const std::string name = fs_path.filename().string();
    if (name.rfind("video", 0) != 0 || name.size() <= 5)
        return -1;

    for (size_t i = 5; i < name.size(); ++i)
    {
        if (!std::isdigit(static_cast<unsigned char>(name[i])))
            return -1;
    }

    return std::stoi(name.substr(5));
}

std::string MediaSource::FourccToString(int fourcc)
{
    if (fourcc == 0)
        return "auto";

    std::string out(4, '\0');
    out[0] = static_cast<char>(fourcc & 0xff);
    out[1] = static_cast<char>((fourcc >> 8) & 0xff);
    out[2] = static_cast<char>((fourcc >> 16) & 0xff);
    out[3] = static_cast<char>((fourcc >> 24) & 0xff);
    for (char& ch : out)
    {
        if (!std::isprint(static_cast<unsigned char>(ch)))
            ch = '?';
    }
    return out;
}

}  // namespace banana_demo
