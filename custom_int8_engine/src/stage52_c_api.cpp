#include "y26_k1x_executor.h"

#include "y26_k1x_full_executor.h"

#include <exception>
#include <new>
#include <string>

struct y26_executor {
    y26::stage52::FullExecutor implementation;
    std::string error;
};

namespace {

y26_status map_status(int status) noexcept {
    if (status == 0) return Y26_STATUS_OK;
    if (status == 1) return Y26_STATUS_INVALID_ARGUMENT;
    if (status == 2) return Y26_STATUS_PACKAGE_ERROR;
    if (status == 4) return Y26_STATUS_UNSUPPORTED;
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

void set_error(y26_executor* executor, const char* message) noexcept {
    if (executor == nullptr) return;
    try {
        executor->error = message == nullptr ? "unknown error" : message;
    } catch (...) {
    }
}

}  // namespace

extern "C" y26_executor* y26_executor_create(void) {
    return new (std::nothrow) y26_executor;
}

extern "C" y26_status y26_executor_prepare(y26_executor* executor, const char* package_dir,
                                             const char* trusted_manifest_sha256,
                                             const y26_executor_options* options) {
    if (executor == nullptr || package_dir == nullptr || trusted_manifest_sha256 == nullptr ||
        options == nullptr || options->struct_size != sizeof(y26_executor_options) ||
        options->abi_version != Y26_K1X_EXECUTOR_ABI_VERSION ||
        (options->scheduler != Y26_SCHEDULER_SAFE && options->scheduler != Y26_SCHEDULER_RR20) ||
        (options->flags & ~static_cast<uint32_t>(Y26_EXECUTOR_FLAG_CAPTURE_BOUNDARIES)) != 0U) {
        return Y26_STATUS_INVALID_ARGUMENT;
    }
    try {
        y26::stage52::RunConfig config;
        config.workers = options->workers;
        config.worker_cpu_begin = options->worker_cpu_begin;
        config.controller_cpu = options->controller_cpu;
        config.scheduler = options->scheduler == Y26_SCHEDULER_RR20
            ? y26::stage52::SchedulerMode::rr20 : y26::stage52::SchedulerMode::safe;
        config.capture_boundaries =
            (options->flags & static_cast<uint32_t>(Y26_EXECUTOR_FLAG_CAPTURE_BOUNDARIES)) != 0U;
        const int status = executor->implementation.prepare(package_dir, trusted_manifest_sha256, config);
        set_error(executor, executor->implementation.last_error().c_str());
        return map_status(status);
    } catch (const std::exception& error) {
        set_error(executor, error.what());
        return Y26_STATUS_RUNTIME_ERROR;
    } catch (...) {
        set_error(executor, "unknown prepare exception");
        return Y26_STATUS_RUNTIME_ERROR;
    }
}

extern "C" y26_status y26_executor_run_preprocessed(y26_executor* executor,
                                                      const float* input, size_t input_elements,
                                                      float* output, size_t output_elements,
                                                      y26_run_timing* timing) {
    if (executor == nullptr) return Y26_STATUS_INVALID_ARGUMENT;
    try {
        y26::stage52::RunTiming internal;
        const int status = executor->implementation.run_preprocessed(
            input, input_elements, output, output_elements, timing == nullptr ? nullptr : &internal);
        set_error(executor, executor->implementation.last_error().c_str());
        if (status == 0) copy_timing(internal, timing);
        return map_status(status);
    } catch (const std::exception& error) {
        set_error(executor, error.what());
        return Y26_STATUS_RUNTIME_ERROR;
    } catch (...) {
        set_error(executor, "unknown run exception");
        return Y26_STATUS_RUNTIME_ERROR;
    }
}

extern "C" y26_status y26_executor_run_rgb(y26_executor* executor, const uint8_t* rgb,
                                             int width, int height, int row_stride_bytes,
                                             float* output, size_t output_elements,
                                             y26_run_timing* timing) {
    if (executor == nullptr) return Y26_STATUS_INVALID_ARGUMENT;
    try {
        y26::stage52::RunTiming internal;
        const int status = executor->implementation.run_rgb(
            rgb, width, height, row_stride_bytes, output, output_elements,
            timing == nullptr ? nullptr : &internal);
        set_error(executor, executor->implementation.last_error().c_str());
        if (status == 0) copy_timing(internal, timing);
        return map_status(status);
    } catch (const std::exception& error) {
        set_error(executor, error.what());
        return Y26_STATUS_RUNTIME_ERROR;
    } catch (...) {
        set_error(executor, "unknown RGB run exception");
        return Y26_STATUS_RUNTIME_ERROR;
    }
}

extern "C" y26_status y26_executor_get_output(const y26_executor* executor,
                                                 float* output, size_t output_elements) {
    if (executor == nullptr) return Y26_STATUS_INVALID_ARGUMENT;
    try {
        return map_status(executor->implementation.copy_output(output, output_elements));
    } catch (...) {
        return Y26_STATUS_RUNTIME_ERROR;
    }
}

extern "C" int y26_executor_tensor_id(const y26_executor* executor, const char* tensor_name) {
    if (executor == nullptr || tensor_name == nullptr) return -1;
    try {
        return executor->implementation.tensor_id_for_name(tensor_name);
    } catch (...) {
        return -1;
    }
}

extern "C" size_t y26_executor_tensor_bytes(const y26_executor* executor, int tensor_id) {
    if (executor == nullptr) return 0;
    try {
        return executor->implementation.tensor_bytes(tensor_id);
    } catch (...) {
        return 0;
    }
}

extern "C" y26_status y26_executor_copy_boundary(const y26_executor* executor, int tensor_id,
                                                    uint8_t* output, size_t output_bytes) {
    if (executor == nullptr) return Y26_STATUS_INVALID_ARGUMENT;
    try {
        return map_status(executor->implementation.copy_boundary(tensor_id, output, output_bytes));
    } catch (...) {
        return Y26_STATUS_RUNTIME_ERROR;
    }
}

extern "C" void y26_executor_destroy(y26_executor* executor) {
    delete executor;
}

extern "C" const char* y26_executor_last_error(const y26_executor* executor) {
    return executor == nullptr ? "invalid executor" : executor->error.c_str();
}

extern "C" const char* y26_executor_version(void) {
    return "K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001/abi1";
}
