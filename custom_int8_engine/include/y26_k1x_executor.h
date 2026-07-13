#ifndef Y26_K1X_EXECUTOR_H
#define Y26_K1X_EXECUTOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Y26_K1X_EXECUTOR_ABI_VERSION 1u
#define Y26_K1X_EXECUTOR_INPUT_ELEMENTS (3u * 640u * 640u)
#define Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS (300u * 6u)

typedef struct y26_executor y26_executor;

typedef enum y26_status {
    Y26_STATUS_OK = 0,
    Y26_STATUS_INVALID_ARGUMENT = 1,
    Y26_STATUS_PACKAGE_ERROR = 2,
    Y26_STATUS_RUNTIME_ERROR = 3,
    Y26_STATUS_UNSUPPORTED = 4
} y26_status;

typedef enum y26_scheduler {
    Y26_SCHEDULER_SAFE = 0,
    Y26_SCHEDULER_RR20 = 1
} y26_scheduler;

typedef enum y26_executor_flag {
    Y26_EXECUTOR_FLAG_NONE = 0,
    Y26_EXECUTOR_FLAG_CAPTURE_BOUNDARIES = 1u << 0
} y26_executor_flag;

typedef struct y26_executor_options {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t workers;
    int32_t worker_cpu_begin;
    int32_t controller_cpu;
    int32_t scheduler;
    uint32_t flags;
} y26_executor_options;

typedef struct y26_run_timing {
    double input_quantize_us;
    double pure_executor_us;
    double resident_core_us;
    double dense_conv_us;
    double depthwise_us;
    double attention_us;
    double lut_us;
    double concat_us;
    double transform_us;
    double head_us;
    double total_us;
    double process_cpu_us;
    uint64_t voluntary_context_switches;
    uint64_t involuntary_context_switches;
    uint64_t output_hash;
    int32_t affinity_ok;
    int32_t cpu4_7_ime_count;
} y26_run_timing;

y26_executor* y26_executor_create(void);
y26_status y26_executor_prepare(y26_executor* executor, const char* package_dir,
                                const char* trusted_manifest_sha256,
                                const y26_executor_options* options);
y26_status y26_executor_run_preprocessed(y26_executor* executor,
                                         const float* nchw_rgb_0_to_1,
                                         size_t input_elements,
                                         float* output_1x300x6,
                                         size_t output_elements,
                                         y26_run_timing* timing);
y26_status y26_executor_run_rgb(y26_executor* executor, const uint8_t* rgb,
                                int width, int height, int row_stride_bytes,
                                float* output_1x300x6, size_t output_elements,
                                y26_run_timing* timing);
y26_status y26_executor_get_output(const y26_executor* executor,
                                   float* output_1x300x6, size_t output_elements);
int y26_executor_tensor_id(const y26_executor* executor, const char* tensor_name);
size_t y26_executor_tensor_bytes(const y26_executor* executor, int tensor_id);
y26_status y26_executor_copy_boundary(const y26_executor* executor, int tensor_id,
                                      uint8_t* output, size_t output_bytes);
void y26_executor_destroy(y26_executor* executor);
const char* y26_executor_last_error(const y26_executor* executor);
const char* y26_executor_version(void);

#ifdef __cplusplus
}
#endif

#endif
