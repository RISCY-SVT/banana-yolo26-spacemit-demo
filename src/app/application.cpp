#include "banana_demo/app/application.h"

#include "banana_demo/infer/yolo26_executor_detector.h"
#include "banana_demo/io/media_source.h"
#include "banana_demo/render/renderer.h"
#include "banana_demo/util/logger.h"

#include <y26_k1x_executor.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace banana_demo {
namespace {

using Clock = std::chrono::steady_clock;
volatile sig_atomic_t g_stop_signal = 0;

void StopSignal(int) { g_stop_signal = 1; }

bool StopRequested() noexcept { return g_stop_signal != 0; }

void RequestStop() noexcept { g_stop_signal = 1; }

bool InstallSignalHandlers() noexcept {
    struct sigaction action {};
    sigemptyset(&action.sa_mask);
    action.sa_handler = StopSignal;
    action.sa_flags = 0;
    return sigaction(SIGINT, &action, nullptr) == 0 &&
        sigaction(SIGTERM, &action, nullptr) == 0 &&
        sigaction(SIGHUP, &action, nullptr) == 0;
}

double ElapsedMs(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::uint64_t SteadyNs(Clock::time_point value) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(value.time_since_epoch()).count());
}

bool DisplayPossible(const AppOptions& options) {
    if (!options.display || options.headless) return false;
    const char* display = std::getenv("DISPLAY");
    const char* wayland = std::getenv("WAYLAND_DISPLAY");
    return (display != nullptr && *display != '\0') ||
        (wayland != nullptr && *wayland != '\0');
}

std::string TimestampForPath() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t value = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
    localtime_r(&value, &local);
    std::ostringstream out;
    out << std::put_time(&local, "%Y%m%d-%H%M%S");
    return out.str();
}

void EnsureParent(const std::string& path) {
    if (path.empty()) return;
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
}

double Mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size());
}

double Percentile(std::vector<double> values, double quantile) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = quantile * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(lower + 1, values.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return values[lower] + fraction * (values[upper] - values[lower]);
}

struct FramePacket {
    cv::Mat frame;
    std::uint64_t sequence = 0;
    double capture_ms = 0.0;
    Clock::time_point read_return{};
};

struct CaptureSnapshot {
    Clock::time_point time{};
    std::uint64_t captured = 0;
    std::uint64_t replacements = 0;
};

class LatestFrameCapture {
public:
    LatestFrameCapture(MediaSource& source, int reconnect_attempts, int capture_cpu,
                       Logger& logger)
        : source_(source), reconnect_attempts_(reconnect_attempts),
          capture_cpu_(capture_cpu), logger_(logger) {}

    ~LatestFrameCapture() { Stop(); }

    void Start() {
        capture_start_ = Clock::now();
        thread_ = std::thread([this] { CaptureLoop(); });
    }

    void Stop() {
        stop_.store(true, std::memory_order_release);
        condition_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    bool Wait(FramePacket& packet) {
        std::unique_lock lock(mutex_);
        const auto ready = [&] {
            return slot_.has_value() || finished_ || stop_.load(std::memory_order_acquire) ||
                StopRequested();
        };
        while (!ready()) (void)condition_.wait_for(lock, std::chrono::milliseconds(100));
        if (!slot_.has_value()) return false;
        packet = std::move(*slot_);
        slot_.reset();
        return true;
    }

    std::uint64_t Captured() const noexcept { return captured_.load(std::memory_order_acquire); }
    std::uint64_t Replacements() const noexcept {
        return replacements_.load(std::memory_order_acquire);
    }
    CaptureSnapshot Snapshot() const {
        std::lock_guard lock(mutex_);
        return {
            Clock::now(),
            captured_.load(std::memory_order_acquire),
            replacements_.load(std::memory_order_acquire),
        };
    }
    double DecodedFrameFps() const {
        const double seconds = std::chrono::duration<double>(Clock::now() - capture_start_).count();
        return seconds > 0.0 ? static_cast<double>(Captured()) / seconds : 0.0;
    }

private:
    void CaptureLoop() {
        cpu_set_t housekeeping;
        CPU_ZERO(&housekeeping);
        if (capture_cpu_ >= 0) {
            CPU_SET(capture_cpu_, &housekeeping);
        } else {
            CPU_SET(5, &housekeeping);
            CPU_SET(6, &housekeeping);
            CPU_SET(7, &housekeeping);
        }
        const int affinity_status = pthread_setaffinity_np(
            pthread_self(), sizeof(housekeeping), &housekeeping);
        if (affinity_status == 0)
            logger_.Info("latest-frame capture affinity=" +
                         (capture_cpu_ >= 0 ? std::to_string(capture_cpu_) : "5-7"));
        else
            logger_.Warn("latest-frame capture affinity=effective-cgroup-mask status=" +
                         std::to_string(affinity_status));
        int reconnects = 0;
        while (!stop_.load(std::memory_order_acquire) &&
               !StopRequested()) {
            cv::Mat frame;
            if (!source_.Read(frame)) {
                if (!source_.IsCamera() || reconnects >= reconnect_attempts_) break;
                ++reconnects;
                std::string error;
                logger_.Warn("camera read failed; reconnect attempt=" + std::to_string(reconnects));
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                if (!source_.Reopen(error)) logger_.Warn("camera reconnect failed: " + error);
                continue;
            }
            reconnects = 0;
            FramePacket packet;
            packet.frame = std::move(frame);
            packet.capture_ms = source_.LastReadMs();
            packet.read_return = Clock::now();
            {
                std::lock_guard lock(mutex_);
                packet.sequence = captured_.fetch_add(1, std::memory_order_acq_rel) + 1;
                if (slot_.has_value()) replacements_.fetch_add(1, std::memory_order_acq_rel);
                slot_ = std::move(packet);
            }
            condition_.notify_one();
        }
        {
            std::lock_guard lock(mutex_);
            finished_ = true;
        }
        condition_.notify_all();
    }

    MediaSource& source_;
    int reconnect_attempts_ = 0;
    int capture_cpu_ = -1;
    Logger& logger_;
    std::atomic<bool> stop_{false};
    std::atomic<std::uint64_t> captured_{0};
    std::atomic<std::uint64_t> replacements_{0};
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::optional<FramePacket> slot_;
    bool finished_ = false;
    Clock::time_point capture_start_{};
    std::thread thread_;
};

class MetricsWriter {
public:
    explicit MetricsWriter(std::string path) : path_(std::move(path)) {
        if (path_.empty()) return;
        buffer_ << "# metrics_schema_version=2\n";
        buffer_ << "processed_index\tmeasured\tsource_sequence\tcaptured_total"
                   "\tcaptured_measured\tapplication_slot_replacements_total"
                   "\tapplication_slot_replacements_measured\tcapture_ms\twait_for_slot_ms"
                   "\tresize_letterbox_ms\tbgr_to_rgb_ms\tpreprocess_ms\texecutor_ms"
                   "\tpostprocess_ms\trender_ms\tdisplay_ms\trecord_ms\ttotal_ms"
                   "\tconsumer_loop_ms\tdecoded_read_return_to_display_call_ms"
                   "\tmeasured_window_start_ns\tframe_done_ns\tobjects\toutput_hash\n";
    }

    ~MetricsWriter() { (void)Flush(); }

    void Write(std::uint64_t index, bool measured, const FrameMetrics& metrics) {
        if (path_.empty()) return;
        buffer_ << std::fixed << std::setprecision(6)
                << index << '\t' << (measured ? 1 : 0) << '\t' << metrics.source_sequence
                << '\t' << metrics.captured_total << '\t' << metrics.captured_measured
                << '\t' << metrics.application_slot_replacements_total
                << '\t' << metrics.application_slot_replacements_measured
                << '\t' << metrics.capture_ms << '\t' << metrics.wait_for_slot_ms
                << '\t' << metrics.resize_letterbox_ms << '\t' << metrics.bgr_to_rgb_ms
                << '\t' << metrics.preprocess_ms << '\t' << metrics.inference_ms
                << '\t' << metrics.postprocess_ms << '\t' << metrics.render_ms
                << '\t' << metrics.display_ms << '\t' << metrics.record_ms
                << '\t' << metrics.total_ms << '\t' << metrics.consumer_loop_ms
                << '\t' << metrics.decoded_read_return_to_display_call_ms
                << '\t' << metrics.measured_window_start_ns << '\t' << metrics.frame_done_ns
                << '\t' << metrics.objects << "\t0x" << std::hex << std::setw(16)
                << std::setfill('0') << metrics.output_hash << std::dec << std::setfill(' ')
                << '\n';
    }

    bool Flush() noexcept {
        if (path_.empty() || flushed_) return true;
        try {
            EnsureParent(path_);
            std::ofstream stream(path_, std::ios::out | std::ios::trunc);
            if (!stream) return false;
            stream << buffer_.str();
            flushed_ = stream.good();
            return flushed_;
        } catch (...) {
            return false;
        }
    }

private:
    std::string path_;
    std::ostringstream buffer_;
    bool flushed_ = false;
};

class DetectionWriter {
public:
    explicit DetectionWriter(std::string path) : path_(std::move(path)) {
        if (path_.empty()) return;
        buffer_ << "processed_index\tmeasured\tsource_sequence\tclass_id\tclass_name"
                   "\tconfidence\toriginal_x1\toriginal_y1\toriginal_x2\toriginal_y2"
                   "\toriginal_width\toriginal_height\toriginal_area"
                   "\tletterbox_x1\tletterbox_y1\tletterbox_x2\tletterbox_y2"
                   "\tletterbox_width\tletterbox_height\tletterbox_area\n";
    }

    ~DetectionWriter() { (void)Flush(); }

    void Write(std::uint64_t index, bool measured, std::uint64_t sequence,
               const std::vector<Detection>& detections,
               const std::vector<std::string>& labels) {
        if (path_.empty()) return;
        for (const Detection& detection : detections) {
            const double original_width = detection.x2 - detection.x1;
            const double original_height = detection.y2 - detection.y1;
            const double letterbox_width = detection.letterbox_x2 - detection.letterbox_x1;
            const double letterbox_height = detection.letterbox_y2 - detection.letterbox_y1;
            const std::string& label = labels.at(static_cast<std::size_t>(detection.class_id));
            buffer_ << std::fixed << std::setprecision(6)
                    << index << '\t' << (measured ? 1 : 0) << '\t' << sequence << '\t'
                    << detection.class_id << '\t' << label << '\t' << detection.score << '\t'
                    << detection.x1 << '\t' << detection.y1 << '\t'
                    << detection.x2 << '\t' << detection.y2 << '\t'
                    << original_width << '\t' << original_height << '\t'
                    << original_width * original_height << '\t'
                    << detection.letterbox_x1 << '\t' << detection.letterbox_y1 << '\t'
                    << detection.letterbox_x2 << '\t' << detection.letterbox_y2 << '\t'
                    << letterbox_width << '\t' << letterbox_height << '\t'
                    << letterbox_width * letterbox_height << '\n';
        }
    }

    bool Flush() noexcept {
        if (path_.empty() || flushed_) return true;
        try {
            EnsureParent(path_);
            std::ofstream stream(path_, std::ios::out | std::ios::trunc);
            if (!stream) return false;
            stream << buffer_.str();
            flushed_ = stream.good();
            return flushed_;
        } catch (...) {
            return false;
        }
    }

private:
    std::string path_;
    std::ostringstream buffer_;
    bool flushed_ = false;
};

class Recorder {
public:
    Recorder(Logger& logger, bool asynchronous)
        : logger_(logger), asynchronous_(asynchronous) {}

    ~Recorder() { Stop(); }

    bool Start(const std::string& path, double fps, const cv::Size& size) {
        Stop();
        path_ = path;
        fps_ = std::max(1.0, fps);
        size_ = size;
        EnsureParent(path);
        frame_index_.store(0, std::memory_order_release);
        replacements_.store(0, std::memory_order_release);
        failures_.store(0, std::memory_order_release);
        if (asynchronous_) {
            {
                std::lock_guard lock(mutex_);
                stop_requested_ = false;
                ready_ = false;
                active_.store(false, std::memory_order_release);
                queue_.clear();
            }
            thread_ = std::thread([this] { WriterThreadEntry(); });
            std::unique_lock lock(mutex_);
            ready_condition_.wait(lock, [this] { return ready_; });
            return active_.load(std::memory_order_acquire);
        }
        const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        if (sync_writer_.open(path, fourcc, fps_, size)) {
            mode_ = "sync-mjpg-avi";
            active_.store(true, std::memory_order_release);
            logger_.Info("recording started mode=sync path=" + path +
                         " backend=" + BackendName(sync_writer_));
            return true;
        }
        fallback_dir_ = path + ".frames";
        std::filesystem::create_directories(fallback_dir_);
        mode_ = "sync-png-sequence";
        active_.store(true, std::memory_order_release);
        logger_.Warn("MJPG writer unavailable; using PNG sequence: " + fallback_dir_);
        return true;
    }

    void Stop() {
        if (asynchronous_ && thread_.joinable()) {
            {
                std::lock_guard lock(mutex_);
                stop_requested_ = true;
            }
            condition_.notify_all();
            thread_.join();
        }
        if (sync_writer_.isOpened()) sync_writer_.release();
        FlushSyncMetadata();
        if (!path_.empty()) {
            logger_.Info("recording stopped path=" + path_ +
                         " frames=" + std::to_string(Frames()) +
                         " queue_replacements=" + std::to_string(Replacements()) +
                         " failures=" + std::to_string(Failures()));
        }
        active_.store(false, std::memory_order_release);
        path_.clear();
        fallback_dir_.clear();
        mode_ = "off";
    }

    bool Active() const noexcept { return active_.load(std::memory_order_acquire); }
    std::uint64_t Frames() const noexcept { return frame_index_.load(std::memory_order_acquire); }
    std::uint64_t Replacements() const noexcept {
        return replacements_.load(std::memory_order_acquire);
    }
    std::uint64_t Failures() const noexcept { return failures_.load(std::memory_order_acquire); }
    std::string Mode() const { return mode_; }

    bool Write(const cv::Mat& frame, std::uint64_t source_sequence) {
        if (!Active()) return false;
        if (asynchronous_) {
            RecordItem item{frame.clone(), source_sequence, SteadyNs(Clock::now())};
            std::lock_guard lock(mutex_);
            if (stop_requested_) return false;
            if (queue_.size() == kQueueDepth) {
                queue_.pop_front();
                replacements_.fetch_add(1, std::memory_order_acq_rel);
            }
            queue_.push_back(std::move(item));
            condition_.notify_one();
            return true;
        }
        bool result = false;
        try {
            result = WriteFrame(sync_writer_, fallback_dir_, frame, source_sequence,
                                SteadyNs(Clock::now()), sync_metadata_);
        } catch (const std::exception& error) {
            logger_.Error("synchronous recording failure: " + std::string(error.what()));
            active_.store(false, std::memory_order_release);
        }
        if (!result) failures_.fetch_add(1, std::memory_order_acq_rel);
        return result;
    }

private:
    struct RecordItem {
        cv::Mat frame;
        std::uint64_t source_sequence = 0;
        std::uint64_t enqueue_ns = 0;
    };

    static constexpr std::size_t kQueueDepth = 2;

    bool WriteFrame(cv::VideoWriter& writer, const std::string& fallback_dir,
                    const cv::Mat& frame, std::uint64_t source_sequence,
                    std::uint64_t enqueue_ns,
                    std::ostringstream& metadata) {
        bool wrote = false;
        const std::uint64_t output_index = frame_index_.load(std::memory_order_acquire);
        if (writer.isOpened()) {
            writer.write(frame);
            wrote = true;
        }
        if (!fallback_dir.empty()) {
            std::ostringstream name;
            name << fallback_dir << "/frame-" << std::setw(8) << std::setfill('0')
                 << output_index << ".png";
            wrote = cv::imwrite(name.str(), frame);
        }
        if (wrote) {
            frame_index_.fetch_add(1, std::memory_order_acq_rel);
            metadata << output_index << '\t' << source_sequence << '\t' << enqueue_ns << '\t'
                     << SteadyNs(Clock::now()) << '\n';
        }
        return wrote;
    }

    static std::string BackendName(const cv::VideoWriter& writer) {
        try { return writer.getBackendName(); }
        catch (const cv::Exception&) { return "unknown"; }
    }

    void WriterThreadEntry() noexcept {
        try {
            WriterLoop();
        } catch (const std::exception& error) {
            failures_.fetch_add(1, std::memory_order_acq_rel);
            logger_.Error("async recording failure: " + std::string(error.what()));
            {
                std::lock_guard lock(mutex_);
                queue_.clear();
                ready_ = true;
                stop_requested_ = true;
                active_.store(false, std::memory_order_release);
            }
            ready_condition_.notify_all();
            condition_.notify_all();
        } catch (...) {
            failures_.fetch_add(1, std::memory_order_acq_rel);
            logger_.Error("async recording failure: unknown exception");
            {
                std::lock_guard lock(mutex_);
                queue_.clear();
                ready_ = true;
                stop_requested_ = true;
                active_.store(false, std::memory_order_release);
            }
            ready_condition_.notify_all();
            condition_.notify_all();
        }
    }

    void WriterLoop() {
        cpu_set_t cpu;
        CPU_ZERO(&cpu);
        CPU_SET(6, &cpu);
        const int affinity_status = pthread_setaffinity_np(pthread_self(), sizeof(cpu), &cpu);
        if (affinity_status != 0) {
            logger_.Warn("async recorder CPU6 affinity failed status=" +
                         std::to_string(affinity_status));
        }

        cv::VideoWriter writer;
        std::string fallback_dir;
        const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        if (writer.open(path_, fourcc, fps_, size_)) {
            mode_ = "async-mjpg-avi";
            logger_.Info("recording started mode=async queue_depth=2 cpu=6 path=" + path_ +
                         " backend=" + BackendName(writer));
        } else {
            fallback_dir = path_ + ".frames";
            std::filesystem::create_directories(fallback_dir);
            mode_ = "async-png-sequence";
            logger_.Warn("async MJPG writer unavailable; using PNG sequence: " + fallback_dir);
        }
        std::ostringstream metadata;
        metadata << "record_index\tsource_sequence\tenqueue_ns\twrite_done_ns\n";
        {
            std::lock_guard lock(mutex_);
            active_.store(writer.isOpened() || !fallback_dir.empty(), std::memory_order_release);
            ready_ = true;
        }
        ready_condition_.notify_one();

        for (;;) {
            RecordItem item;
            {
                std::unique_lock lock(mutex_);
                condition_.wait(lock, [this] { return stop_requested_ || !queue_.empty(); });
                if (queue_.empty() && stop_requested_) break;
                item = std::move(queue_.front());
                queue_.pop_front();
            }
            if (!WriteFrame(writer, fallback_dir, item.frame, item.source_sequence,
                            item.enqueue_ns, metadata)) {
                failures_.fetch_add(1, std::memory_order_acq_rel);
            }
        }
        if (writer.isOpened()) writer.release();
        try {
            std::ofstream stream(path_ + ".frames.tsv", std::ios::out | std::ios::trunc);
            stream << metadata.str();
            if (!stream.good()) failures_.fetch_add(1, std::memory_order_acq_rel);
        } catch (...) {
            failures_.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    void FlushSyncMetadata() noexcept {
        if (asynchronous_ || sync_metadata_.str().empty() || path_.empty()) return;
        try {
            std::ofstream stream(path_ + ".frames.tsv", std::ios::out | std::ios::trunc);
            stream << "record_index\tsource_sequence\tenqueue_ns\twrite_done_ns\n"
                   << sync_metadata_.str();
            if (!stream.good()) failures_.fetch_add(1, std::memory_order_acq_rel);
        } catch (...) {
            failures_.fetch_add(1, std::memory_order_acq_rel);
        }
        sync_metadata_.str("");
        sync_metadata_.clear();
    }

    Logger& logger_;
    bool asynchronous_ = false;
    cv::VideoWriter sync_writer_;
    std::string path_;
    std::string fallback_dir_;
    std::string mode_ = "off";
    double fps_ = 1.0;
    cv::Size size_;
    std::atomic<bool> active_{false};
    std::atomic<std::uint64_t> frame_index_{0};
    std::atomic<std::uint64_t> replacements_{0};
    std::atomic<std::uint64_t> failures_{0};
    std::mutex mutex_;
    std::condition_variable condition_;
    std::condition_variable ready_condition_;
    std::deque<RecordItem> queue_;
    bool stop_requested_ = false;
    bool ready_ = false;
    std::thread thread_;
    std::ostringstream sync_metadata_;
};

std::string SaveInteractiveFrame(const cv::Mat& frame, const AppOptions& options) {
    std::filesystem::create_directories(options.screenshot_dir);
    const std::string path = options.screenshot_dir + "/yolo26-stage59-" +
        TimestampForPath() + ".png";
    if (!cv::imwrite(path, frame)) throw std::runtime_error("failed to save screenshot: " + path);
    return path;
}

}  // namespace

Application::Application(AppOptions options) : options_(std::move(options)) {}

int Application::Run() {
    g_stop_signal = 0;
    if (!InstallSignalHandlers()) {
        std::cerr << "failed to install SIGINT/SIGTERM/SIGHUP handlers\n";
        return 2;
    }
    if (options_.print_license) return PrintLicense();
    if (options_.print_source_info) return PrintSourceInfo();
    if (options_.print_build_info) return PrintBuildInfo();
    cv::setNumThreads(options_.opencv_threads);
    return options_.source.rfind("image:", 0) == 0 ? RunImage() : RunStream();
}

int Application::PrintLicense() const {
    std::cout
        << "Project license: GNU Affero General Public License v3.0 or later\n"
        << "License text: LICENSE and LICENSES/AGPL-3.0.txt\n"
        << "Third-party terms: THIRD_PARTY_NOTICES.md\n"
        << "No warranty: NO_WARRANTY.md\n";
    return 0;
}

int Application::PrintSourceInfo() const {
    std::cout
        << "Preferred source form and build instructions: SOURCE_ACCESS.md\n"
        << "Model provenance and license evidence: MODEL_LICENSE_AND_PROVENANCE.md\n"
        << "Stable R640 reference: d0e3611c8d99dfade049bd261cb557509222a456\n"
        << "Stage61 Q0 reference: fa668ccaf7938336bd10313455ab81557b33e020\n"
        << "Default profile: K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001\n"
        << "Non-R640 Q0 profiles are experimental and require explicit selection.\n";
    return 0;
}

int Application::PrintBuildInfo() const {
    y26_build_info info;
    y26_build_info_init(&info);
    const y26_status status = y26_executor_get_build_info(&info);
    if (status != Y26_STATUS_OK) {
        std::cerr << "build-info query failed: " << y26_status_string(status) << '\n';
        return 1;
    }
    std::cout << "release_version=" << info.release_version << '\n'
              << "abi_version=" << info.abi_version << '\n'
              << "source_commit=" << info.source_commit << '\n'
              << "integer_contract_id=" << info.integer_contract_id << '\n'
              << "full_graph_profile_id=" << info.full_graph_profile_id << '\n'
              << "capability_flags=0x" << std::hex << info.capability_flags << std::dec << '\n'
              << "ime_enabled=" << !!(info.capability_flags & Y26_CAPABILITY_IME) << '\n'
              << "rvv_enabled=" << !!(info.capability_flags & Y26_CAPABILITY_RVV) << '\n'
              << "frozen_profile_enabled=" << !!(info.capability_flags & Y26_CAPABILITY_FROZEN_PROFILE) << '\n'
              << "rgb_api_enabled=" << !!(info.capability_flags & Y26_CAPABILITY_RGB_INPUT) << '\n'
              << "expected_package_manifest_sha256=" << info.expected_package_manifest_sha256 << '\n';
    return 0;
}

int Application::RunImage() {
    Logger logger(options_.quiet, options_.log_file);
    MediaSource source(options_);
    std::string error;
    if (!source.Open(error)) {
        logger.Error(error);
        return 2;
    }
    Yolo26ExecutorDetector detector(options_);
    logger.Info(detector.BuildInfoSummary());
    cv::Mat frame;
    const auto loop_begin = Clock::now();
    if (!source.Read(frame)) {
        logger.Error("image read failed");
        return 2;
    }
    const auto read_return = Clock::now();
    InferenceResult result = detector.Process(frame);
    result.metrics.capture_ms = source.LastReadMs();
    Renderer renderer;
    const auto render_begin = Clock::now();
    cv::Mat annotated = renderer.DrawDetections(frame, result.detections, detector.Labels());
    result.metrics.render_ms = ElapsedMs(render_begin, Clock::now());
    result.metrics.total_ms = ElapsedMs(loop_begin, Clock::now());
    renderer.DrawOverlay(annotated, result.metrics, options_.profile, "sequential",
                         source.EffectiveFormat());
    const auto render_end = Clock::now();
    result.metrics.render_ms = ElapsedMs(render_begin, render_end);
    result.metrics.total_ms = ElapsedMs(loop_begin, render_end);
    result.metrics.consumer_loop_ms = ElapsedMs(read_return, render_end);
    result.metrics.decoded_read_return_to_display_call_ms = result.metrics.consumer_loop_ms;
    result.metrics.captured_total = 1;
    result.metrics.captured_measured = 1;
    result.metrics.measured_window_start_ns = SteadyNs(loop_begin);
    result.metrics.frame_done_ns = SteadyNs(render_end);

    if (!options_.save_frame.empty()) {
        EnsureParent(options_.save_frame);
        if (!cv::imwrite(options_.save_frame, annotated)) {
            logger.Error("failed to save annotated image: " + options_.save_frame);
            return 2;
        }
        logger.Info("saved_frame=" + options_.save_frame);
    }
    MetricsWriter metrics(options_.metrics_tsv);
    DetectionWriter detections(options_.detections_tsv);
    metrics.Write(1, true, result.metrics);
    detections.Write(1, true, 1, result.detections, detector.Labels());
    if (!metrics.Flush() || !detections.Flush()) {
        logger.Error("failed to write image evidence TSV");
        return 2;
    }
    if (DisplayPossible(options_)) {
        if (renderer.TryShow("YOLO26 K1X INT8", annotated, error)) cv::waitKey(0);
        else logger.Warn("display failed: " + error);
    }
    std::cout << std::fixed << std::setprecision(6)
              << "SUMMARY source=image processed_frames=1 objects=" << result.metrics.objects
              << " preprocess_ms=" << result.metrics.preprocess_ms
              << " inference_ms=" << result.metrics.inference_ms
              << " postprocess_ms=" << result.metrics.postprocess_ms
              << " render_ms=" << result.metrics.render_ms
              << " total_ms=" << result.metrics.total_ms
              << " output_hash=0x" << std::hex << result.metrics.output_hash << std::dec << '\n';
    return 0;
}

int Application::RunStream() {
    Logger logger(options_.quiet, options_.log_file);
    MediaSource source(options_);
    std::string error;
    if (!source.Open(error)) {
        logger.Error(error);
        return 2;
    }
    if (source.IsVideo() && options_.flow == "latest-frame") {
        logger.Warn("video source uses sequential flow to preserve file order");
        options_.flow = "sequential";
    }
    Yolo26ExecutorDetector detector(options_);
    logger.Info(detector.BuildInfoSummary());
    logger.Info("source_requested=" + options_.source + " source_resolved=" + source.ResolvedPath());
    logger.Info("capture_backend=" + source.BackendName() + " open_method=" + source.OpenMethod());
    logger.Info("camera_requested=" + std::to_string(options_.camera_width) + "x" +
                std::to_string(options_.camera_height) + "@" + std::to_string(options_.camera_fps) +
                " " + options_.camera_fourcc + " effective=" + source.EffectiveFormat());
    logger.Info("profile=" + options_.profile + " flow=" + options_.flow +
                " opencv_threads=" + std::to_string(options_.opencv_threads));

    const bool display_enabled = DisplayPossible(options_);
    if (options_.display && !display_enabled) {
        logger.Warn("display requested but DISPLAY/WAYLAND_DISPLAY is unavailable; running headless");
    }

    std::unique_ptr<LatestFrameCapture> latest;
    if (options_.flow == "latest-frame") {
        latest = std::make_unique<LatestFrameCapture>(
            source, options_.reconnect_attempts, options_.capture_cpu, logger);
        latest->Start();
    }

    MetricsWriter metrics_writer(options_.metrics_tsv);
    DetectionWriter detection_writer(options_.detections_tsv);
    Renderer renderer;
    Recorder recorder(logger, options_.record_mode == "async");
    cv::Mat last_annotated;
    const bool recording_requested = !options_.record_path.empty();

    std::vector<double> consumer_times;
    std::vector<double> decoded_call_latencies;
    std::vector<double> wait_times;
    std::vector<double> inference;
    std::uint64_t processed = 0;
    std::uint64_t measured = 0;
    std::uint64_t measured_displayed = 0;
    std::uint64_t sequential_sequence = 0;
    bool paused = false;
    double previous_render_ms = 0.0;
    double previous_display_ms = 0.0;
    double previous_consumer_loop_ms = 0.0;
    const auto run_begin = Clock::now();
    std::optional<Clock::time_point> measured_begin;
    std::uint64_t captured_at_measured_begin = 0;
    std::uint64_t replacements_at_measured_begin = 0;

    while (!StopRequested()) {
        FramePacket packet;
        const auto wait_begin = Clock::now();
        const bool next_frame_is_measured =
            processed >= static_cast<std::uint64_t>(options_.warmup_frames);
        if (next_frame_is_measured && !measured_begin.has_value()) {
            if (latest) {
                const CaptureSnapshot snapshot = latest->Snapshot();
                measured_begin = snapshot.time;
                captured_at_measured_begin = snapshot.captured;
                replacements_at_measured_begin = snapshot.replacements;
            } else {
                measured_begin = wait_begin;
                captured_at_measured_begin = sequential_sequence;
                replacements_at_measured_begin = 0;
            }
        }
        if (latest) {
            if (!latest->Wait(packet)) break;
        } else {
            if (!source.Read(packet.frame)) {
                if (source.IsCamera()) {
                    bool reopened = false;
                    for (int attempt = 1; attempt <= options_.reconnect_attempts; ++attempt) {
                        logger.Warn("camera read failed; reconnect attempt=" + std::to_string(attempt));
                        std::this_thread::sleep_for(std::chrono::milliseconds(250));
                        if (source.Reopen(error)) { reopened = true; break; }
                    }
                    if (reopened) continue;
                }
                break;
            }
            packet.sequence = ++sequential_sequence;
            packet.capture_ms = source.LastReadMs();
            packet.read_return = Clock::now();
        }
        const auto consumer_begin = Clock::now();

        InferenceResult result = detector.Process(packet.frame);
        result.metrics.capture_ms = packet.capture_ms;
        result.metrics.wait_for_slot_ms = ElapsedMs(wait_begin, consumer_begin);
        result.metrics.source_sequence = packet.sequence;
        const CaptureSnapshot producer_snapshot = latest
            ? latest->Snapshot()
            : CaptureSnapshot{consumer_begin, sequential_sequence, 0};
        result.metrics.captured_total = producer_snapshot.captured;
        result.metrics.application_slot_replacements_total = producer_snapshot.replacements;
        result.metrics.captured_measured = measured_begin.has_value()
            ? result.metrics.captured_total - captured_at_measured_begin : 0;
        result.metrics.application_slot_replacements_measured = measured_begin.has_value()
            ? result.metrics.application_slot_replacements_total - replacements_at_measured_begin : 0;
        const double measured_capture_elapsed_ms = measured_begin.has_value()
            ? ElapsedMs(*measured_begin, producer_snapshot.time) : 0.0;
        result.metrics.opencv_decoded_frame_fps = measured_capture_elapsed_ms > 0.0
            ? 1000.0 * static_cast<double>(result.metrics.captured_measured) /
                measured_capture_elapsed_ms
            : (latest ? latest->DecodedFrameFps() : source.Fps());
        result.metrics.measured_window_start_ns = measured_begin.has_value()
            ? SteadyNs(*measured_begin) : 0;

        const auto render_begin = Clock::now();
        result.metrics.processed_fps = measured > 0 && measured_begin.has_value()
            ? 1000.0 * static_cast<double>(measured) /
                ElapsedMs(*measured_begin, render_begin) : 0.0;
        cv::Mat annotated = renderer.DrawDetections(packet.frame, result.detections,
                                                    detector.Labels());
        result.metrics.previous_render_ms = previous_render_ms;
        result.metrics.previous_display_ms = previous_display_ms;
        result.metrics.previous_consumer_loop_ms = previous_consumer_loop_ms;
        renderer.DrawOverlay(annotated, result.metrics, options_.profile, options_.flow,
                             source.EffectiveFormat());
        const auto render_end = Clock::now();
        const double current_render_ms = ElapsedMs(render_begin, render_end);

        if (recording_requested && !recorder.Active()) {
            const double first_frame_ms = std::max(1.0, ElapsedMs(consumer_begin, render_end));
            const double record_fps = std::clamp(1000.0 / first_frame_ms, 1.0,
                                                 std::max(1.0, source.Fps()));
            recorder.Start(options_.record_path, record_fps, annotated.size());
        }

        const auto record_begin = Clock::now();
        if (recorder.Active() && !recorder.Write(annotated, packet.sequence)) {
            logger.Warn("recording write failed");
        }
        const auto record_end = Clock::now();
        result.metrics.record_ms = recorder.Active() ? ElapsedMs(record_begin, record_end) : 0.0;

        int key = -1;
        const auto display_begin = Clock::now();
        if (display_enabled) {
            if (!renderer.TryShow("YOLO26 K1X INT8 Camera", annotated, error)) {
                logger.Error("display failed: " + error);
                break;
            }
            key = cv::waitKey(1);
        }
        const auto display_end = Clock::now();
        result.metrics.display_ms = display_enabled ? ElapsedMs(display_begin, display_end) : 0.0;
        result.metrics.render_ms = current_render_ms;
        result.metrics.consumer_loop_ms = ElapsedMs(consumer_begin, display_end);
        result.metrics.total_ms = ElapsedMs(wait_begin, display_end);
        result.metrics.decoded_read_return_to_display_call_ms =
            ElapsedMs(packet.read_return, display_end);
        result.metrics.frame_done_ns = SteadyNs(display_end);
        previous_render_ms = result.metrics.render_ms;
        previous_display_ms = result.metrics.display_ms;
        previous_consumer_loop_ms = result.metrics.consumer_loop_ms;

        ++processed;
        const bool is_measured = next_frame_is_measured;
        if (is_measured) {
            ++measured;
            consumer_times.push_back(result.metrics.consumer_loop_ms);
            decoded_call_latencies.push_back(
                result.metrics.decoded_read_return_to_display_call_ms);
            wait_times.push_back(result.metrics.wait_for_slot_ms);
            inference.push_back(result.metrics.inference_ms);
            if (display_enabled) ++measured_displayed;
        }
        metrics_writer.Write(processed, is_measured, result.metrics);
        detection_writer.Write(processed, is_measured, packet.sequence,
                               result.detections, detector.Labels());
        last_annotated = std::move(annotated);

        if (key == 27 || key == 'q' || key == 'Q') break;
        if (key == 's' || key == 'S') logger.Info("saved_frame=" + SaveInteractiveFrame(last_annotated, options_));
        if (key == 'r' || key == 'R') {
            if (recorder.Active()) recorder.Stop();
            else {
                const double current_measured_ms = measured_begin.has_value()
                    ? ElapsedMs(*measured_begin, Clock::now()) : 0.0;
                const double fps = current_measured_ms > 0.0
                    ? 1000.0 * static_cast<double>(measured) / current_measured_ms
                    : std::max(1.0, std::min(source.Fps(), 7.0));
                recorder.Start(options_.screenshot_dir + "/yolo26-stage59-" + TimestampForPath() + ".avi",
                               fps, last_annotated.size());
            }
        }
        if (key == ' ') paused = !paused;
        while (paused && display_enabled && !StopRequested()) {
            const int pause_key = cv::waitKey(30);
            if (pause_key == ' ') paused = false;
            else if (pause_key == 27 || pause_key == 'q' || pause_key == 'Q') {
                RequestStop();
            }
            else if (pause_key == 's' || pause_key == 'S') {
                logger.Info("saved_frame=" + SaveInteractiveFrame(last_annotated, options_));
            }
        }

        if (measured == 1 || (measured > 0 && measured % 100 == 0)) {
            logger.Info("progress measured_frames=" + std::to_string(measured) +
                        " source_sequence=" + std::to_string(packet.sequence) +
                        " slot_replacements_measured=" +
                            std::to_string(result.metrics.application_slot_replacements_measured) +
                        " executor_ms=" + std::to_string(result.metrics.inference_ms));
        }
        if (options_.max_frames > 0 && measured >= static_cast<std::uint64_t>(options_.max_frames)) break;
        if (options_.duration_seconds > 0.0 && measured_begin.has_value() &&
            std::chrono::duration<double>(Clock::now() - *measured_begin).count() >= options_.duration_seconds) break;
    }

    const CaptureSnapshot capture_snapshot = latest
        ? latest->Snapshot()
        : CaptureSnapshot{Clock::now(), sequential_sequence, 0};
    const auto measured_end = capture_snapshot.time;
    const std::uint64_t captured_total_at_window_end = capture_snapshot.captured;
    const std::uint64_t replacements_total_at_window_end = capture_snapshot.replacements;
    if (latest) latest->Stop();
    const std::string recording_mode = recorder.Mode();
    recorder.Stop();
    const std::uint64_t recorded_frames = recorder.Frames();
    const std::uint64_t recording_queue_replacements = recorder.Replacements();
    const std::uint64_t recording_failures = recorder.Failures();
    if (!metrics_writer.Flush() || !detection_writer.Flush()) {
        logger.Error("failed to write stream evidence TSV");
        return 2;
    }
    if (!options_.save_frame.empty() && !last_annotated.empty()) {
        EnsureParent(options_.save_frame);
        if (!cv::imwrite(options_.save_frame, last_annotated)) {
            logger.Error("failed to save final annotated frame: " + options_.save_frame);
            return 2;
        }
        logger.Info("saved_frame=" + options_.save_frame);
    }
    const auto run_end = Clock::now();
    const double measured_elapsed_ms = measured_begin.has_value()
        ? ElapsedMs(*measured_begin, measured_end) : 0.0;
    const double processed_fps = measured_elapsed_ms > 0.0
        ? 1000.0 * static_cast<double>(measured) / measured_elapsed_ms : 0.0;
    const double displayed_fps = measured_elapsed_ms > 0.0
        ? 1000.0 * static_cast<double>(measured_displayed) / measured_elapsed_ms : 0.0;
    const std::uint64_t captured_total = captured_total_at_window_end;
    const std::uint64_t replacements_total = replacements_total_at_window_end;
    const std::uint64_t captured_measured = measured_begin.has_value()
        ? captured_total - captured_at_measured_begin : 0;
    const std::uint64_t replacements_measured = measured_begin.has_value()
        ? replacements_total - replacements_at_measured_begin : 0;
    const double run_elapsed_ms = ElapsedMs(run_begin, run_end);
    const double decoded_fps = measured_elapsed_ms > 0.0
        ? 1000.0 * static_cast<double>(captured_measured) / measured_elapsed_ms : 0.0;
    const double recording_fps = run_elapsed_ms > 0.0
        ? 1000.0 * static_cast<double>(recorded_frames) / run_elapsed_ms : 0.0;
    const double replacement_pct = captured_measured > 0
        ? 100.0 * static_cast<double>(replacements_measured) /
            static_cast<double>(captured_measured) : 0.0;

    std::cout << std::fixed << std::setprecision(6)
              << "SUMMARY metrics_schema_version=2 source="
              << (source.IsCamera() ? "camera" : "video")
              << " profile=" << options_.profile << " flow=" << options_.flow
              << " effective_format=\"" << source.EffectiveFormat() << "\""
              << " requested_fps=" << options_.camera_fps
              << " backend_reported_fps=" << source.Fps()
              << " warmup_frames=" << options_.warmup_frames
              << " measured_frames=" << measured
              << " measured_window_start_ns="
              << (measured_begin.has_value() ? SteadyNs(*measured_begin) : 0)
              << " measured_window_end_ns=" << SteadyNs(measured_end)
              << " captured_total=" << captured_total
              << " captured_measured=" << captured_measured
              << " application_slot_replacements_total=" << replacements_total
              << " application_slot_replacements_measured=" << replacements_measured
              << " application_slot_replacement_pct=" << replacement_pct
              << " opencv_decoded_frame_fps=" << decoded_fps
              << " processed_fps=" << processed_fps
              << " displayed_fps=" << displayed_fps
              << " recorded_frames=" << recorded_frames
              << " recording_fps=" << recording_fps
              << " recording_mode=" << recording_mode
              << " recording_queue_replacements=" << recording_queue_replacements
              << " recording_failures=" << recording_failures
              << " shutdown_finalize_ms=" << ElapsedMs(measured_end, run_end)
              << " wait_for_slot_mean_ms=" << Mean(wait_times)
              << " wait_for_slot_p95_ms=" << Percentile(wait_times, 0.95)
              << " consumer_loop_mean_ms=" << Mean(consumer_times)
              << " consumer_loop_p95_ms=" << Percentile(consumer_times, 0.95)
              << " decoded_read_return_to_display_call_mean_ms="
              << Mean(decoded_call_latencies)
              << " decoded_read_return_to_display_call_p95_ms="
              << Percentile(decoded_call_latencies, 0.95)
              << " executor_mean_ms=" << Mean(inference)
              << '\n';
    return measured > 0 || source.IsVideo() ? 0 : 2;
}

}  // namespace banana_demo
