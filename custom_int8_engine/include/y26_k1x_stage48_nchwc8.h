#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace y26::stage48 {

enum class ComputeRoute {
    scalar,
    ime,
};

enum class MBlock {
    m4 = 4,
    m8 = 8,
    m12 = 12,
};

enum class LoadStrategy {
    c8_u64,
    rvv_vlse64,
    rvv_vlseg2e64,
};

enum class PartitionPolicy {
    spatial,
    output_channel,
};

struct RunOptions {
    ComputeRoute route = ComputeRoute::ime;
    MBlock m_block = MBlock::m12;
    LoadStrategy load_strategy = LoadStrategy::rvv_vlse64;
    PartitionPolicy partition = PartitionPolicy::spatial;
    int workers = 4;
    bool profile_phases = false;
};

struct Timing {
    double direct_a_delivery_us = 0.0;
    double vmadot_us = 0.0;
    double scalar_epilogue_us = 0.0;
    double barrier_us = 0.0;
    double total_us = 0.0;
    double min_worker_us = 0.0;
    double max_worker_us = 0.0;
    int workers = 0;
    int affinity_ok = 0;
    std::uint64_t vector_groups = 0;
    std::uint64_t scalar_c8_groups = 0;
    std::uint64_t border_chunks = 0;
};

class Model5DirectConv {
public:
    struct Impl;

    Model5DirectConv();
    ~Model5DirectConv();
    Model5DirectConv(Model5DirectConv&&) noexcept;
    Model5DirectConv& operator=(Model5DirectConv&&) noexcept;
    Model5DirectConv(const Model5DirectConv&) = delete;
    Model5DirectConv& operator=(const Model5DirectConv&) = delete;

    int prepare(const std::filesystem::path& package_dir, int worker_capacity);
    int run(const std::int8_t* input_nchwc8_s8,
            std::int8_t* output_nchwc8_s8,
            const RunOptions& options,
            Timing* timing);
    int debug_pack_a(const std::int8_t* input_nchwc8_s8,
                     int m_begin,
                     MBlock m_block,
                     LoadStrategy strategy,
                     std::int8_t* panel,
                     std::size_t panel_bytes) const;

    std::size_t input_bytes() const noexcept;
    std::size_t output_bytes() const noexcept;
    std::size_t packed_weight_bytes() const noexcept;
    std::size_t per_worker_workspace_bytes() const noexcept;
    std::uint64_t macs() const noexcept;
    bool affinity_ok() const noexcept;
    const std::string& last_error() const noexcept;

private:
    std::unique_ptr<Impl> impl_;
};

void nchw_u8_to_nchwc8_s8(const std::uint8_t* input,
                           std::int8_t* output,
                           int batches,
                           int channels,
                           int height,
                           int width);

void nchwc8_s8_to_nchw_u8(const std::int8_t* input,
                           std::uint8_t* output,
                           int batches,
                           int channels,
                           int height,
                           int width);

const char* compute_route_name(ComputeRoute route) noexcept;
const char* m_block_name(MBlock block) noexcept;
const char* load_strategy_name(LoadStrategy strategy) noexcept;
const char* partition_policy_name(PartitionPolicy policy) noexcept;

}  // namespace y26::stage48
