#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace y26::stage52 {

inline constexpr const char* kFullGraphProfileId =
    "K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001";

enum class SchedulerMode { safe, rr20 };
enum class ComputeMode { scalar, optimized };

struct RunConfig {
    int workers = 4;
    int worker_cpu_begin = 0;
    int controller_cpu = 4;
    SchedulerMode scheduler = SchedulerMode::safe;
    ComputeMode compute = ComputeMode::optimized;
    bool capture_boundaries = false;
};

struct RunTiming {
    double input_quantize_us = 0.0;
    double prefix_us = 0.0;
    double resident_core_us = 0.0;
    double dense_conv_us = 0.0;
    double attention_us = 0.0;
    double depthwise_us = 0.0;
    double lut_us = 0.0;
    double concat_us = 0.0;
    double transform_us = 0.0;
    double head_us = 0.0;
    double total_us = 0.0;
    double process_cpu_us = 0.0;
    std::uint64_t voluntary_context_switches = 0;
    std::uint64_t involuntary_context_switches = 0;
    std::uint64_t output_hash = 0;
    int affinity_ok = 0;
    int cpu4_7_ime_count = 0;
};

struct DiagnosticConvShapeResult {
    int operation_index = -1;
    std::string operation_name;
    std::string operation_kind;
    int output_h = 0;
    int output_w = 0;
    int output_c = 0;
    int input_c = 0;
    int kernel_h = 0;
    int kernel_w = 0;
    int stride_h = 0;
    int stride_w = 0;
    int m = 0;
    int n = 0;
    int k = 0;
    std::size_t working_set_bytes = 0;
    std::size_t packed_weight_bytes = 0;
    double mean_us = 0.0;
    double median_us = 0.0;
    double p95_us = 0.0;
    double maximum_us = 0.0;
    std::uint64_t output_hash = 0;
    bool deterministic = false;
};

class FullExecutor {
public:
    struct Impl;

    FullExecutor();
    ~FullExecutor();
    FullExecutor(FullExecutor&&) noexcept;
    FullExecutor& operator=(FullExecutor&&) noexcept;
    FullExecutor(const FullExecutor&) = delete;
    FullExecutor& operator=(const FullExecutor&) = delete;

    int prepare(const std::filesystem::path& package_dir,
                const std::string& trusted_manifest_sha256,
                const RunConfig& config);

    int run_preprocessed(const float* nchw_rgb_0_to_1, std::size_t element_count,
                         float* output_1x300x6, std::size_t output_count,
                         RunTiming* timing);

    int run_rgb(const std::uint8_t* rgb, int width, int height, int row_stride_bytes,
                float* output_1x300x6, std::size_t output_count,
                RunTiming* timing);

    // Diagnostic-only exact-shape timing. Dimensions may only shrink a prepared
    // package operation, so all arithmetic assets and selected kernels remain unchanged.
    int diagnostic_benchmark_conv_shape(int operation_index, int output_h,
                                        int output_w, int output_channels,
                                        int warmup, int runs,
                                        DiagnosticConvShapeResult* result);

    int copy_boundary(int tensor_id, std::uint8_t* output, std::size_t bytes) const;
    int copy_output(float* output_1x300x6, std::size_t output_count) const;
    int tensor_id_for_name(const std::string& name) const noexcept;
    std::size_t tensor_bytes(int tensor_id) const noexcept;
    int operation_count() const noexcept;
    int tensor_count() const noexcept;
    std::size_t arena_bytes() const noexcept;
    std::size_t packed_weight_bytes() const noexcept;
    const std::string& package_manifest_sha256() const noexcept;
    const std::string& last_error() const noexcept;

private:
    int run_input_surface(const float* nchw_rgb_0_to_1,
                          const std::uint8_t* rgb640,
                          int rgb_row_stride_bytes,
                          float* output_1x300x6,
                          std::size_t output_count,
                          RunTiming* timing);

    std::unique_ptr<Impl> impl_;
};

const char* scheduler_mode_name(SchedulerMode value) noexcept;
const char* compute_mode_name(ComputeMode value) noexcept;

}  // namespace y26::stage52
