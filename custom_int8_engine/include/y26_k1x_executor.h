#ifndef Y26_K1X_EXECUTOR_H
#define Y26_K1X_EXECUTOR_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(Y26_K1X_BUILDING_LIBRARY)
#define Y26_K1X_API __declspec(dllexport)
#else
#define Y26_K1X_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
#define Y26_K1X_API __attribute__((visibility("default")))
#else
#define Y26_K1X_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define Y26_K1X_EXECUTOR_ABI_VERSION 1u
#define Y26_K1X_BUILD_INFO_VERSION 1u
#define Y26_K1X_EXECUTOR_INPUT_ELEMENTS (3u * 640u * 640u)
#define Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS (300u * 6u)

typedef struct y26_executor y26_executor;

typedef enum y26_status {
    Y26_STATUS_OK = 0,
    Y26_STATUS_INVALID_ARGUMENT = 1,
    Y26_STATUS_PACKAGE_ERROR = 2,
    Y26_STATUS_RUNTIME_ERROR = 3,
    Y26_STATUS_UNSUPPORTED = 4,
    Y26_STATUS_INVALID_STATE = 5,
    Y26_STATUS_BUSY = 6
} y26_status;

typedef enum y26_scheduler {
    Y26_SCHEDULER_SAFE = 0,
    Y26_SCHEDULER_RR20 = 1
} y26_scheduler;

typedef enum y26_wake_policy {
    Y26_WAKE_CONDITION_VARIABLE = 0,
    Y26_WAKE_FRAME_GATED_SPIN = 1
} y26_wake_policy;

typedef enum y26_executor_flag {
    Y26_EXECUTOR_FLAG_NONE = 0,
    Y26_EXECUTOR_FLAG_CAPTURE_BOUNDARIES = 1u << 0
} y26_executor_flag;

typedef enum y26_capability_flag {
    Y26_CAPABILITY_NONE = 0,
    Y26_CAPABILITY_IME = 1u << 0,
    Y26_CAPABILITY_RVV = 1u << 1,
    Y26_CAPABILITY_FROZEN_PROFILE = 1u << 2,
    Y26_CAPABILITY_RGB_INPUT = 1u << 3
} y26_capability_flag;

typedef struct y26_executor_options {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t workers;
    int32_t worker_cpu_begin;
    int32_t controller_cpu;
    int32_t scheduler;
    uint32_t flags;
    int32_t wake_policy;
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

/*
 * Additive ABI1 build metadata. Call y26_build_info_init() before querying.
 * Pointers returned in this structure refer to immutable library-owned strings.
 */
typedef struct y26_build_info {
    uint32_t struct_size;
    uint32_t info_version;
    uint32_t abi_version;
    uint32_t capability_flags;
    const char* release_version;
    const char* source_commit;
    const char* integer_contract_id;
    const char* full_graph_profile_id;
    const char* expected_package_manifest_sha256;
} y26_build_info;

Y26_K1X_API void y26_executor_options_init(y26_executor_options* options);
Y26_K1X_API void y26_build_info_init(y26_build_info* info);
Y26_K1X_API y26_status y26_executor_get_build_info(y26_build_info* info);
Y26_K1X_API const char* y26_status_string(y26_status status);
Y26_K1X_API y26_executor* y26_executor_create(void);
Y26_K1X_API y26_status y26_executor_prepare(y26_executor* executor, const char* package_dir,
                                            const char* trusted_manifest_sha256,
                                            const y26_executor_options* options);
Y26_K1X_API y26_status y26_executor_run_preprocessed(y26_executor* executor,
                                                     const float* nchw_rgb_0_to_1,
                                                     size_t input_elements,
                                                     float* output_1x300x6,
                                                     size_t output_elements,
                                                     y26_run_timing* timing);
Y26_K1X_API y26_status y26_executor_run_rgb(y26_executor* executor, const uint8_t* rgb,
                                            int width, int height, int row_stride_bytes,
                                            float* output_1x300x6, size_t output_elements,
                                            y26_run_timing* timing);
Y26_K1X_API y26_status y26_executor_get_output(const y26_executor* executor,
                                               float* output_1x300x6,
                                               size_t output_elements);
Y26_K1X_API int y26_executor_tensor_id(const y26_executor* executor,
                                       const char* tensor_name);
Y26_K1X_API size_t y26_executor_tensor_bytes(const y26_executor* executor, int tensor_id);
Y26_K1X_API y26_status y26_executor_copy_boundary(const y26_executor* executor, int tensor_id,
                                                  uint8_t* output, size_t output_bytes);
Y26_K1X_API void y26_executor_destroy(y26_executor* executor);
Y26_K1X_API const char* y26_executor_last_error(const y26_executor* executor);
Y26_K1X_API const char* y26_executor_version(void);

#ifdef __cplusplus
}
#endif

#endif
