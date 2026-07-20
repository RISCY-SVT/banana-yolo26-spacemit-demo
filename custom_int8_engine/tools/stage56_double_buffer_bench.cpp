#include "y26_k1x_full_executor.h"
#include "y26_k1x_int8_v1.h"
#include "y26_k1x_package.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::filesystem::path package;
    std::filesystem::path image;
    int warmup = 10;
    int runs = 100;
    int repeats = 5;
};

struct Slot {
    cv::Mat rgb;
    double prepare_us = 0.0;
    bool ready = false;
    bool requested = false;
};

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next = [&]() -> std::string {
            if (++index >= argc) throw std::runtime_error("missing value for " + argument);
            return argv[index];
        };
        if (argument == "--package") options.package = next();
        else if (argument == "--image") options.image = next();
        else if (argument == "--warmup") options.warmup = std::stoi(next());
        else if (argument == "--runs") options.runs = std::stoi(next());
        else if (argument == "--repeats") options.repeats = std::stoi(next());
        else throw std::runtime_error("unknown argument: " + argument);
    }
    if (options.package.empty() || options.image.empty() || options.warmup < 0 ||
        options.runs < 1 || options.repeats < 1) {
        throw std::runtime_error("invalid options");
    }
    return options;
}

void pin_controller() {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(4, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
        throw std::runtime_error("cannot pin pipeline controller to CPU4");
    }
}

void pin_preprocessor() {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(5, &set);
    CPU_SET(6, &set);
    CPU_SET(7, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
        throw std::runtime_error("cannot pin preprocessor to CPU5-7");
    }
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open image: " + path.string());
    const std::streamsize size = stream.tellg();
    if (size <= 0) throw std::runtime_error("empty image: " + path.string());
    stream.seekg(0);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error("cannot read image: " + path.string());
    }
    return bytes;
}

double prepare_rgb(const std::vector<std::uint8_t>& encoded, cv::Mat& rgb, int resolution) {
    const auto begin = Clock::now();
    const cv::Mat bgr = cv::imdecode(encoded, cv::IMREAD_COLOR);
    if (bgr.empty()) throw std::runtime_error("OpenCV image decode failed");
    const double ratio = std::min(static_cast<double>(resolution) / bgr.cols,
                                  static_cast<double>(resolution) / bgr.rows);
    const int width = static_cast<int>(std::nearbyint(bgr.cols * ratio));
    const int height = static_cast<int>(std::nearbyint(bgr.rows * ratio));
    const int x0 = static_cast<int>(std::nearbyint(
        (static_cast<double>(resolution) - width) / 2.0 - 0.1));
    const int y0 = static_cast<int>(std::nearbyint(
        (static_cast<double>(resolution) - height) / 2.0 - 0.1));
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat canvas(resolution, resolution, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(x0, y0, width, height)));
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    return elapsed_us(begin, Clock::now());
}

class Preprocessor {
public:
    Preprocessor(const std::vector<std::uint8_t>& encoded, std::array<Slot, 2>& slots,
                 int resolution)
        : encoded_(encoded), slots_(slots), resolution_(resolution),
          thread_([this]() { loop(); }) {}

    ~Preprocessor() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        request_cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    void request(int index) {
        std::lock_guard lock(mutex_);
        Slot& slot = slots_[static_cast<std::size_t>(index)];
        if (pending_ >= 0 || slot.ready || slot.requested) {
            throw std::runtime_error("invalid double-buffer prepare request");
        }
        slot.requested = true;
        pending_ = index;
        request_cv_.notify_one();
    }

    double consume(int index) {
        std::unique_lock lock(mutex_);
        ready_cv_.wait(lock, [&]() {
            return failure_ || slots_[static_cast<std::size_t>(index)].ready;
        });
        if (failure_) throw std::runtime_error(failure_message_);
        Slot& slot = slots_[static_cast<std::size_t>(index)];
        const double value = slot.prepare_us;
        slot.ready = false;
        return value;
    }

private:
    void loop() noexcept {
        try {
            pin_preprocessor();
            cv::setNumThreads(3);
            for (;;) {
                int index = -1;
                {
                    std::unique_lock lock(mutex_);
                    request_cv_.wait(lock, [&]() { return stopping_ || pending_ >= 0; });
                    if (stopping_) return;
                    index = pending_;
                    pending_ = -1;
                }
                const double duration = prepare_rgb(
                    encoded_, slots_[static_cast<std::size_t>(index)].rgb, resolution_);
                {
                    std::lock_guard lock(mutex_);
                    Slot& slot = slots_[static_cast<std::size_t>(index)];
                    slot.prepare_us = duration;
                    slot.requested = false;
                    slot.ready = true;
                }
                ready_cv_.notify_all();
            }
        } catch (const std::exception& error) {
            {
                std::lock_guard lock(mutex_);
                failure_ = true;
                failure_message_ = error.what();
            }
            ready_cv_.notify_all();
        }
    }

    const std::vector<std::uint8_t>& encoded_;
    std::array<Slot, 2>& slots_;
    int resolution_ = 0;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable request_cv_;
    std::condition_variable ready_cv_;
    int pending_ = -1;
    bool stopping_ = false;
    bool failure_ = false;
    std::string failure_message_;
};

double percentile(std::vector<double> values, double probability) {
    std::sort(values.begin(), values.end());
    const double position = probability * static_cast<double>(values.size() - 1U);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

void print_summary(const char* name, const std::vector<double>& values) {
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) /
                        static_cast<double>(values.size());
    double sum_squared = 0.0;
    for (double value : values) sum_squared += (value - mean) * (value - mean);
    const double stddev = std::sqrt(sum_squared / static_cast<double>(values.size()));
    std::cout << "summary\t" << name << '\t' << values.size() << '\t' << mean << '\t'
              << stddev << '\t' << percentile(values, 0.5) << '\t'
              << percentile(values, 0.9) << '\t' << percentile(values, 0.95) << '\t'
              << percentile(values, 0.99) << '\t'
              << *std::max_element(values.begin(), values.end()) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        pin_controller();
        const std::vector<std::uint8_t> encoded = read_bytes(options.image);

        y26::stage52::RunConfig config;
        config.workers = 4;
        config.worker_cpu_begin = 0;
        config.controller_cpu = 4;
        config.scheduler = y26::stage52::SchedulerMode::safe;
        config.wake_policy = y26::stage52::WakePolicy::frame_gated_spin;
        config.compute = y26::stage52::ComputeMode::optimized;
        config.allow_stage60_static_profiles = true;
        y26::stage52::FullExecutor executor;
        const std::string manifest = y26::int8_v1::sha256_file(options.package / "asset_hashes.tsv");
        if (executor.prepare(options.package, manifest, config) != 0) {
            throw std::runtime_error("prepare failed: " + executor.last_error());
        }

        std::array<Slot, 2> slots;
        for (Slot& slot : slots) {
            slot.rgb.create(executor.input_height(), executor.input_width(), CV_8UC3);
        }
        std::array<float, 1800> output {};
        for (int index = 0; index < options.warmup; ++index) {
            (void)prepare_rgb(encoded, slots[0].rgb, executor.input_width());
            y26::stage52::RunTiming timing;
            if (executor.run_rgb(slots[0].rgb.data, executor.input_width(), executor.input_height(),
                                 static_cast<int>(slots[0].rgb.step), output.data(), output.size(),
                                 &timing) != 0) {
                throw std::runtime_error("warmup execution failed: " + executor.last_error());
            }
            if (timing.affinity_ok != 1 || timing.cpu4_7_ime_count != 0) {
                throw std::runtime_error("warmup CPU affinity or IME ownership contract failed");
            }
        }

        Preprocessor preprocessor(encoded, slots, executor.input_width());
        std::vector<double> prepare_values;
        std::vector<double> executor_values;
        std::vector<double> interval_values;
        const std::size_t sample_count = static_cast<std::size_t>(options.runs * options.repeats);
        prepare_values.reserve(sample_count);
        executor_values.reserve(sample_count);
        interval_values.reserve(sample_count);
        std::uint64_t expected_hash = 0;
        int expected_detections = -1;
        std::cout << std::setprecision(12)
                  << "sample\trepeat\trun\tprepare_us\texecutor_us\tinterval_us\toutput_hash\tdetections\n";

        slots[0].prepare_us = prepare_rgb(encoded, slots[0].rgb, executor.input_width());
        slots[0].ready = true;
        preprocessor.request(1);
        int current = 0;
        const auto throughput_begin = Clock::now();
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            for (int run = 0; run < options.runs; ++run) {
                const auto interval_begin = Clock::now();
                const double prepare_us = preprocessor.consume(current);
                y26::stage52::RunTiming timing;
                if (executor.run_rgb(slots[static_cast<std::size_t>(current)].rgb.data,
                                     executor.input_width(), executor.input_height(),
                                     static_cast<int>(slots[static_cast<std::size_t>(current)].rgb.step),
                                     output.data(), output.size(), &timing) != 0) {
                    throw std::runtime_error("pipeline execution failed: " + executor.last_error());
                }
                if (timing.affinity_ok != 1 || timing.cpu4_7_ime_count != 0) {
                    throw std::runtime_error("pipeline CPU affinity or IME ownership contract failed");
                }
                int detections = 0;
                for (std::size_t row = 0; row < 300U; ++row) {
                    const float* value = output.data() + row * 6U;
                    if (value[4] > 0.001F && value[5] >= 0.0F && value[5] < 80.0F) ++detections;
                }
                if (expected_hash == 0) expected_hash = timing.output_hash;
                if (expected_detections < 0) expected_detections = detections;
                if (timing.output_hash != expected_hash || detections != expected_detections) {
                    throw std::runtime_error("double-buffer output is nondeterministic");
                }
                const int consumed = current;
                current = 1 - current;
                const bool last = repeat == options.repeats - 1 && run == options.runs - 1;
                if (!last) preprocessor.request(consumed);
                const double interval_us = elapsed_us(interval_begin, Clock::now());
                prepare_values.push_back(prepare_us);
                executor_values.push_back(timing.total_us);
                interval_values.push_back(interval_us);
                std::cout << "sample\t" << repeat << '\t' << run << '\t' << prepare_us << '\t'
                          << timing.total_us << '\t' << interval_us << "\t0x" << std::hex
                          << timing.output_hash << std::dec << '\t' << detections << '\n';
            }
        }
        const double total_us = elapsed_us(throughput_begin, Clock::now());
        std::cout << "summary\tphase\tsamples\tmean_us\tstddev_us\tmedian_us\tp90_us\tp95_us\tp99_us\tmax_us\n";
        print_summary("preprocessor", prepare_values);
        print_summary("executor", executor_values);
        print_summary("pipeline_interval", interval_values);
        std::cout << "metadata\ttotal_elapsed_us\t" << total_us << '\n'
                  << "metadata\tresolution\t" << executor.input_width() << '\n'
                  << "metadata\tpackage_manifest_sha256\t"
                  << executor.package_manifest_sha256() << '\n'
                  << "metadata\tsteady_state_fps\t" << sample_count * 1.0e6 / total_us << '\n'
                  << "metadata\toutput_hash\t0x" << std::hex << expected_hash << std::dec << '\n'
                  << "metadata\tdetections\t" << expected_detections << '\n'
                  << "metadata\tpreprocessor_cpus\t5-7\n"
                  << "metadata\texecutor_cpus\t0-4\n"
                  << "metadata\tcpu4_7_ime_count\t0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
