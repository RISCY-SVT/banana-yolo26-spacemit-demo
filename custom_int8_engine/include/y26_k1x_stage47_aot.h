#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace y26::stage47 {

enum class KernelShape {
    scalar,
    m4n16,
    m8n16,
    m12n16,
};

enum class PartitionPolicy {
    spatial,
    output_channel,
};

struct TensorSpec {
    int h = 0;
    int w = 0;
    int c = 0;
    float scale = 0.0f;
    int zero_point_u8 = 0;
};

struct OutputSegmentSpec {
    int channel_begin = 0;
    int channel_count = 0;
    TensorSpec output;
    bool silu = false;
};

struct ConvSpec {
    TensorSpec input;
    int output_h = 0;
    int output_w = 0;
    int output_c = 0;
    int kernel_h = 0;
    int kernel_w = 0;
    int stride_h = 0;
    int stride_w = 0;
    int pad_h = 0;
    int pad_w = 0;
    int group = 1;
    float conv_output_scale = 0.0f;
    int conv_output_zero_point_u8 = 0;
    const std::int8_t* weights_ohwi_s8 = nullptr;
    std::size_t weight_count = 0;
    const float* weight_scales = nullptr;
    std::size_t weight_scale_count = 0;
    const std::int32_t* bias_i32 = nullptr;
    std::size_t bias_count = 0;
    std::vector<OutputSegmentSpec> segments;
};

struct IntegratedTiming {
    double gather_pack_us = 0.0;
    double vmadot_us = 0.0;
    double fused_epilogue_us = 0.0;
    double barrier_us = 0.0;
    double total_us = 0.0;
    double max_worker_us = 0.0;
    double min_worker_us = 0.0;
    int workers = 0;
    int affinity_ok = 0;
};

struct RunOptions {
    KernelShape kernel = KernelShape::m12n16;
    PartitionPolicy partition = PartitionPolicy::spatial;
    int workers = 4;
    int stop_after_operation = -1;
    bool profile_phases = false;
};

class WorkerPool {
public:
    explicit WorkerPool(int workers);
    ~WorkerPool();
    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    int capacity() const noexcept;
    bool affinity_ok() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class IntegratedConv;
};

class IntegratedConv {
public:
    struct Impl;

    IntegratedConv();
    ~IntegratedConv();
    IntegratedConv(IntegratedConv&&) noexcept;
    IntegratedConv& operator=(IntegratedConv&&) noexcept;
    IntegratedConv(const IntegratedConv&) = delete;
    IntegratedConv& operator=(const IntegratedConv&) = delete;

    int prepare(const ConvSpec& spec);
    int run(WorkerPool& pool,
            const std::int8_t* input_nhwc_s8,
            const std::array<std::int8_t*, 2>& outputs_nhwc_s8,
            std::size_t output_count,
            const RunOptions& options,
            IntegratedTiming* timing) const;
    std::size_t prepared_weight_bytes() const noexcept;
    std::size_t per_worker_workspace_bytes(KernelShape shape) const noexcept;
    std::uint64_t macs() const noexcept;

private:
    std::unique_ptr<Impl> impl_;
};

struct OperationTiming {
    int operation_index = -1;
    std::string name;
    std::string kind;
    double total_us = 0.0;
    double gather_pack_us = 0.0;
    double vmadot_us = 0.0;
    double fused_epilogue_us = 0.0;
};

struct ExecutorTiming {
    double total_us = 0.0;
    double conv_us = 0.0;
    double lut_us = 0.0;
    double add_us = 0.0;
    double concat_us = 0.0;
    std::vector<OperationTiming> operations;
};

class AotExecutor {
public:
    struct Impl;

    AotExecutor();
    ~AotExecutor();
    AotExecutor(AotExecutor&&) noexcept;
    AotExecutor& operator=(AotExecutor&&) noexcept;
    AotExecutor(const AotExecutor&) = delete;
    AotExecutor& operator=(const AotExecutor&) = delete;

    int prepare(const std::filesystem::path& package_dir, int worker_capacity);
    int run(const std::int8_t* input_nhwc_s8,
            std::int8_t* output_nhwc_s8,
            const RunOptions& options,
            ExecutorTiming* timing);
    int set_input(const std::int8_t* source, std::size_t bytes);
    int copy_tensor(int tensor_id, std::int8_t* destination, std::size_t bytes) const;
    const TensorSpec* tensor_spec(int tensor_id) const noexcept;
    std::size_t tensor_bytes(int tensor_id) const noexcept;
    std::size_t arena_bytes() const noexcept;
    std::size_t packed_weight_bytes() const noexcept;
    int operation_count() const noexcept;
    int input_tensor_id() const noexcept;
    int output_tensor_id() const noexcept;
    bool worker_affinity_ok() const noexcept;
    const std::string& last_error() const noexcept;

private:
    std::unique_ptr<Impl> impl_;
};

void nchw_u8_to_nhwc_s8(const std::uint8_t* input,
                         std::int8_t* output,
                         int h,
                         int w,
                         int c);

void nhwc_s8_to_nchw_u8(const std::int8_t* input,
                         std::uint8_t* output,
                         int h,
                         int w,
                         int c);

const char* kernel_shape_name(KernelShape shape) noexcept;
const char* partition_policy_name(PartitionPolicy policy) noexcept;

}  // namespace y26::stage47
