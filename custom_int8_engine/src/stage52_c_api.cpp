#include "y26_k1x_executor.h"

#include "y26_k1x_full_executor.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <new>
#include <string>
#include <string_view>

#ifndef Y26_K1X_RELEASE_VERSION
#define Y26_K1X_RELEASE_VERSION "research"
#endif
#ifndef Y26_K1X_SOURCE_COMMIT
#define Y26_K1X_SOURCE_COMMIT "unknown"
#endif
#ifndef Y26_K1X_BUILD_HAS_IME
#define Y26_K1X_BUILD_HAS_IME 0
#endif
#ifndef Y26_K1X_BUILD_HAS_RVV
#define Y26_K1X_BUILD_HAS_RVV 0
#endif
#ifndef Y26_K1X_BUILD_FROZEN_PROFILE
#define Y26_K1X_BUILD_FROZEN_PROFILE 0
#endif
#ifndef Y26_K1X_VERSION_STRING
#define Y26_K1X_VERSION_STRING "research/K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001/abi1"
#endif

struct y26_executor {
    y26::stage52::FullExecutor implementation;
    mutable std::mutex error_mutex;
    std::string error;
    std::atomic<bool> busy {false};
    std::atomic<bool> prepared {false};
};

namespace {

constexpr std::size_t kAbi1LegacyOptionsSize = offsetof(y26_executor_options, wake_policy);
constexpr const char* kReleaseVersion = Y26_K1X_RELEASE_VERSION;
constexpr const char* kSourceCommit = Y26_K1X_SOURCE_COMMIT;
constexpr const char* kIntegerContract = "K1X_INT8_V1";
constexpr const char* kFullGraphProfile = "K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001";
constexpr const char* kExpectedPackageManifest =
    "fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be";

constexpr uint32_t build_capabilities() noexcept {
    uint32_t flags = Y26_CAPABILITY_RGB_INPUT;
#if Y26_K1X_BUILD_HAS_IME
    flags |= Y26_CAPABILITY_IME;
#endif
#if Y26_K1X_BUILD_HAS_RVV
    flags |= Y26_CAPABILITY_RVV;
#endif
#if Y26_K1X_BUILD_FROZEN_PROFILE
    flags |= Y26_CAPABILITY_FROZEN_PROFILE;
#endif
    return flags;
}

y26_status map_run_status(int status) noexcept {
    if (status == 0) return Y26_STATUS_OK;
    if (status == 1) return Y26_STATUS_INVALID_ARGUMENT;
    if (status == 4) return Y26_STATUS_UNSUPPORTED;
    return Y26_STATUS_RUNTIME_ERROR;
}

y26_status map_prepare_status(int status, std::string_view error) noexcept {
    if (status == 0) return Y26_STATUS_OK;
    if (status == 1) return Y26_STATUS_INVALID_ARGUMENT;
    if (error.find("package") != std::string_view::npos ||
        error.find("asset") != std::string_view::npos ||
        error.find("manifest") != std::string_view::npos) {
        return Y26_STATUS_PACKAGE_ERROR;
    }
    return Y26_STATUS_RUNTIME_ERROR;
}

void copy_timing(const y26::stage52::RunTiming& source, y26_run_timing* target) noexcept {
    if (target == nullptr) return;
    target->input_quantize_us = source.input_quantize_us;
    target->pure_executor_us = source.total_us - source.input_quantize_us;
    target->resident_core_us = source.resident_core_us;
    target->dense_conv_us = source.dense_conv_us;
    target->depthwise_us = source.depthwise_us;
    target->attention_us = source.attention_us;
    target->lut_us = source.lut_us;
    target->concat_us = source.concat_us;
    target->transform_us = source.transform_us;
    target->head_us = source.head_us;
    target->total_us = source.total_us;
    target->process_cpu_us = source.process_cpu_us;
    target->voluntary_context_switches = source.voluntary_context_switches;
    target->involuntary_context_switches = source.involuntary_context_switches;
    target->output_hash = source.output_hash;
    target->affinity_ok = source.affinity_ok;
    target->cpu4_7_ime_count = source.cpu4_7_ime_count;
}

void set_error(y26_executor* executor, std::string_view message) noexcept {
    if (executor == nullptr) return;
    try {
        std::lock_guard lock(executor->error_mutex);
        executor->error.assign(message);
    } catch (...) {
    }
}

class BusyGuard {
public:
    explicit BusyGuard(y26_executor* executor) : executor_(executor) {
        bool expected = false;
        acquired_ = executor_ != nullptr &&
            executor_->busy.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
    }

    ~BusyGuard() {
        if (acquired_) executor_->busy.store(false, std::memory_order_release);
    }

    bool acquired() const noexcept { return acquired_; }

private:
    y26_executor* executor_ = nullptr;
    bool acquired_ = false;
};

bool valid_manifest(const char* value) noexcept {
    if (value == nullptr) return false;
    for (std::size_t index = 0; index < 64U; ++index) {
        const char character = value[index];
        const bool hexadecimal = (character >= '0' && character <= '9') ||
            (character >= 'a' && character <= 'f') ||
            (character >= 'A' && character <= 'F');
        if (!hexadecimal) return false;
    }
    return value[64] == '\0';
}

bool ranges_overlap(const void* left, std::size_t left_bytes,
                    const void* right, std::size_t right_bytes) noexcept {
    if (left == nullptr || right == nullptr || left_bytes == 0 || right_bytes == 0) return false;
    const auto left_begin = reinterpret_cast<std::uintptr_t>(left);
    const auto right_begin = reinterpret_cast<std::uintptr_t>(right);
    if (left_begin > UINTPTR_MAX - left_bytes || right_begin > UINTPTR_MAX - right_bytes) return true;
    return left_begin < right_begin + right_bytes && right_begin < left_begin + left_bytes;
}

y26_status reject(y26_executor* executor, y26_status status, std::string_view detail) noexcept {
    set_error(executor, detail.empty() ? std::string_view(y26_status_string(status)) : detail);
    return status;
}

}  // namespace

extern "C" void y26_executor_options_init(y26_executor_options* options) {
    if (options == nullptr) return;
    *options = y26_executor_options {};
    options->struct_size = sizeof(*options);
    options->abi_version = Y26_K1X_EXECUTOR_ABI_VERSION;
    options->workers = 4;
    options->worker_cpu_begin = 0;
    options->controller_cpu = 4;
    options->scheduler = Y26_SCHEDULER_SAFE;
    options->flags = Y26_EXECUTOR_FLAG_NONE;
    options->wake_policy = Y26_WAKE_CONDITION_VARIABLE;
}

extern "C" void y26_build_info_init(y26_build_info* info) {
    if (info == nullptr) return;
    *info = y26_build_info {};
    info->struct_size = sizeof(*info);
    info->info_version = Y26_K1X_BUILD_INFO_VERSION;
}

extern "C" y26_status y26_executor_get_build_info(y26_build_info* info) {
    if (info == nullptr || info->struct_size < sizeof(y26_build_info) ||
        info->info_version != Y26_K1X_BUILD_INFO_VERSION) {
        return Y26_STATUS_INVALID_ARGUMENT;
    }
    const uint32_t struct_size = info->struct_size;
    *info = y26_build_info {};
    info->struct_size = struct_size;
    info->info_version = Y26_K1X_BUILD_INFO_VERSION;
    info->abi_version = Y26_K1X_EXECUTOR_ABI_VERSION;
    info->capability_flags = build_capabilities();
    info->release_version = kReleaseVersion;
    info->source_commit = kSourceCommit;
    info->integer_contract_id = kIntegerContract;
    info->full_graph_profile_id = kFullGraphProfile;
    info->expected_package_manifest_sha256 = kExpectedPackageManifest;
    return Y26_STATUS_OK;
}

extern "C" const char* y26_status_string(y26_status status) {
    switch (status) {
        case Y26_STATUS_OK: return "ok";
        case Y26_STATUS_INVALID_ARGUMENT: return "invalid argument";
        case Y26_STATUS_PACKAGE_ERROR: return "package error";
        case Y26_STATUS_RUNTIME_ERROR: return "runtime error";
        case Y26_STATUS_UNSUPPORTED: return "unsupported";
        case Y26_STATUS_INVALID_STATE: return "invalid state";
        case Y26_STATUS_BUSY: return "executor busy";
    }
    return "unknown status";
}

extern "C" y26_executor* y26_executor_create(void) {
    return new (std::nothrow) y26_executor;
}

extern "C" y26_status y26_executor_prepare(y26_executor* executor, const char* package_dir,
                                             const char* trusted_manifest_sha256,
                                             const y26_executor_options* options) {
    if (executor == nullptr) return Y26_STATUS_INVALID_ARGUMENT;
    BusyGuard guard(executor);
    if (!guard.acquired()) return reject(executor, Y26_STATUS_BUSY, "prepare while executor is busy");
    if (executor->prepared.load(std::memory_order_acquire)) {
        return reject(executor, Y26_STATUS_INVALID_STATE, "executor is already prepared");
    }
    if (package_dir == nullptr || package_dir[0] == '\0' || !valid_manifest(trusted_manifest_sha256) ||
        options == nullptr || options->struct_size < kAbi1LegacyOptionsSize ||
        options->abi_version != Y26_K1X_EXECUTOR_ABI_VERSION ||
        options->workers < 1 || options->workers > 4 || options->worker_cpu_begin != 0 ||
        options->controller_cpu != 4 ||
        (options->scheduler != Y26_SCHEDULER_SAFE && options->scheduler != Y26_SCHEDULER_RR20) ||
        (options->flags & ~static_cast<uint32_t>(Y26_EXECUTOR_FLAG_CAPTURE_BOUNDARIES)) != 0U) {
        return reject(executor, Y26_STATUS_INVALID_ARGUMENT, "invalid prepare argument or CPU topology");
    }
    const int wake_policy = options->struct_size >= sizeof(y26_executor_options)
        ? options->wake_policy : Y26_WAKE_CONDITION_VARIABLE;
    if (wake_policy != Y26_WAKE_CONDITION_VARIABLE &&
        wake_policy != Y26_WAKE_FRAME_GATED_SPIN) {
        return reject(executor, Y26_STATUS_INVALID_ARGUMENT, "invalid wake policy");
    }
    try {
        y26::stage52::RunConfig config;
        config.workers = options->workers;
        config.worker_cpu_begin = options->worker_cpu_begin;
        config.controller_cpu = options->controller_cpu;
        config.scheduler = options->scheduler == Y26_SCHEDULER_RR20
            ? y26::stage52::SchedulerMode::rr20 : y26::stage52::SchedulerMode::safe;
        config.wake_policy = wake_policy == Y26_WAKE_FRAME_GATED_SPIN
            ? y26::stage52::WakePolicy::frame_gated_spin
            : y26::stage52::WakePolicy::condition_variable;
        config.capture_boundaries =
            (options->flags & static_cast<uint32_t>(Y26_EXECUTOR_FLAG_CAPTURE_BOUNDARIES)) != 0U;
        const int status = executor->implementation.prepare(
            package_dir, trusted_manifest_sha256, config);
        const std::string error = executor->implementation.last_error();
        const y26_status mapped = map_prepare_status(status, error);
        if (mapped != Y26_STATUS_OK) return reject(executor, mapped, error);
        executor->prepared.store(true, std::memory_order_release);
        set_error(executor, "");
        return Y26_STATUS_OK;
    } catch (const std::exception& error) {
        return reject(executor, Y26_STATUS_RUNTIME_ERROR, error.what());
    } catch (...) {
        return reject(executor, Y26_STATUS_RUNTIME_ERROR, "unknown prepare exception");
    }
}

extern "C" y26_status y26_executor_run_preprocessed(y26_executor* executor,
                                                      const float* input, size_t input_elements,
                                                      float* output, size_t output_elements,
                                                      y26_run_timing* timing) {
    if (executor == nullptr) return Y26_STATUS_INVALID_ARGUMENT;
    BusyGuard guard(executor);
    if (!guard.acquired()) return reject(executor, Y26_STATUS_BUSY, "concurrent use of one executor handle");
    if (!executor->prepared.load(std::memory_order_acquire)) {
        return reject(executor, Y26_STATUS_INVALID_STATE, "run before prepare");
    }
    if (input == nullptr || output == nullptr ||
        input_elements != Y26_K1X_EXECUTOR_INPUT_ELEMENTS ||
        output_elements != Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS ||
        ranges_overlap(input, input_elements * sizeof(float),
                       output, output_elements * sizeof(float))) {
        return reject(executor, Y26_STATUS_INVALID_ARGUMENT, "invalid or overlapping preprocessed buffers");
    }
    try {
        y26::stage52::RunTiming internal;
        const int status = executor->implementation.run_preprocessed(
            input, input_elements, output, output_elements, timing == nullptr ? nullptr : &internal);
        const y26_status mapped = map_run_status(status);
        if (mapped != Y26_STATUS_OK) {
            return reject(executor, mapped, executor->implementation.last_error());
        }
        copy_timing(internal, timing);
        set_error(executor, "");
        return Y26_STATUS_OK;
    } catch (const std::exception& error) {
        return reject(executor, Y26_STATUS_RUNTIME_ERROR, error.what());
    } catch (...) {
        return reject(executor, Y26_STATUS_RUNTIME_ERROR, "unknown run exception");
    }
}

extern "C" y26_status y26_executor_run_rgb(y26_executor* executor, const uint8_t* rgb,
                                             int width, int height, int row_stride_bytes,
                                             float* output, size_t output_elements,
                                             y26_run_timing* timing) {
    if (executor == nullptr) return Y26_STATUS_INVALID_ARGUMENT;
    BusyGuard guard(executor);
    if (!guard.acquired()) return reject(executor, Y26_STATUS_BUSY, "concurrent use of one executor handle");
    if (!executor->prepared.load(std::memory_order_acquire)) {
        return reject(executor, Y26_STATUS_INVALID_STATE, "run before prepare");
    }
    const std::size_t rgb_bytes = width == 640 && height == 640 && row_stride_bytes >= 640 * 3
        ? static_cast<std::size_t>(row_stride_bytes) * 640U : 0U;
    if (rgb == nullptr || output == nullptr || rgb_bytes == 0 ||
        output_elements != Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS ||
        ranges_overlap(rgb, rgb_bytes, output, output_elements * sizeof(float))) {
        return reject(executor, Y26_STATUS_INVALID_ARGUMENT, "invalid or overlapping RGB buffers");
    }
    try {
        y26::stage52::RunTiming internal;
        const int status = executor->implementation.run_rgb(
            rgb, width, height, row_stride_bytes, output, output_elements,
            timing == nullptr ? nullptr : &internal);
        const y26_status mapped = map_run_status(status);
        if (mapped != Y26_STATUS_OK) {
            return reject(executor, mapped, executor->implementation.last_error());
        }
        copy_timing(internal, timing);
        set_error(executor, "");
        return Y26_STATUS_OK;
    } catch (const std::exception& error) {
        return reject(executor, Y26_STATUS_RUNTIME_ERROR, error.what());
    } catch (...) {
        return reject(executor, Y26_STATUS_RUNTIME_ERROR, "unknown RGB run exception");
    }
}

extern "C" y26_status y26_executor_get_output(const y26_executor* executor,
                                                float* output, size_t output_elements) {
    if (executor == nullptr) return Y26_STATUS_INVALID_ARGUMENT;
    auto* mutable_executor = const_cast<y26_executor*>(executor);
    BusyGuard guard(mutable_executor);
    if (!guard.acquired()) return reject(mutable_executor, Y26_STATUS_BUSY, "output read while executor is busy");
    if (!executor->prepared.load(std::memory_order_acquire)) {
        return reject(mutable_executor, Y26_STATUS_INVALID_STATE, "output read before prepare");
    }
    if (output == nullptr || output_elements != Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS) {
        return reject(mutable_executor, Y26_STATUS_INVALID_ARGUMENT, "invalid output buffer");
    }
    try {
        const y26_status status = map_run_status(
            executor->implementation.copy_output(output, output_elements));
        if (status != Y26_STATUS_OK) return reject(mutable_executor, status, "output copy failed");
        set_error(mutable_executor, "");
        return status;
    } catch (...) {
        return reject(mutable_executor, Y26_STATUS_RUNTIME_ERROR, "output copy exception");
    }
}

extern "C" int y26_executor_tensor_id(const y26_executor* executor, const char* tensor_name) {
    if (executor == nullptr || tensor_name == nullptr ||
        !executor->prepared.load(std::memory_order_acquire) ||
        executor->busy.load(std::memory_order_acquire)) return -1;
    try {
        return executor->implementation.tensor_id_for_name(tensor_name);
    } catch (...) {
        return -1;
    }
}

extern "C" size_t y26_executor_tensor_bytes(const y26_executor* executor, int tensor_id) {
    if (executor == nullptr || !executor->prepared.load(std::memory_order_acquire) ||
        executor->busy.load(std::memory_order_acquire)) return 0;
    try {
        return executor->implementation.tensor_bytes(tensor_id);
    } catch (...) {
        return 0;
    }
}

extern "C" y26_status y26_executor_copy_boundary(const y26_executor* executor, int tensor_id,
                                                   uint8_t* output, size_t output_bytes) {
    if (executor == nullptr) return Y26_STATUS_INVALID_ARGUMENT;
    auto* mutable_executor = const_cast<y26_executor*>(executor);
    BusyGuard guard(mutable_executor);
    if (!guard.acquired()) return reject(mutable_executor, Y26_STATUS_BUSY, "boundary read while executor is busy");
    if (!executor->prepared.load(std::memory_order_acquire)) {
        return reject(mutable_executor, Y26_STATUS_INVALID_STATE, "boundary read before prepare");
    }
    if (output == nullptr) return reject(mutable_executor, Y26_STATUS_INVALID_ARGUMENT, "null boundary output");
    try {
        const y26_status status = map_run_status(
            executor->implementation.copy_boundary(tensor_id, output, output_bytes));
        if (status != Y26_STATUS_OK) return reject(mutable_executor, status, "boundary copy failed");
        set_error(mutable_executor, "");
        return status;
    } catch (...) {
        return reject(mutable_executor, Y26_STATUS_RUNTIME_ERROR, "boundary copy exception");
    }
}

extern "C" void y26_executor_destroy(y26_executor* executor) {
    delete executor;
}

extern "C" const char* y26_executor_last_error(const y26_executor* executor) {
    if (executor == nullptr) return "invalid executor";
    thread_local std::string snapshot;
    try {
        std::lock_guard lock(executor->error_mutex);
        snapshot = executor->error;
        return snapshot.c_str();
    } catch (...) {
        return "cannot read executor error";
    }
}

extern "C" const char* y26_executor_version(void) {
    return Y26_K1X_VERSION_STRING;
}
