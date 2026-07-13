#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace y26::stage49 {

enum class ComputeRoute { scalar, ime };
enum class KernelShape { m4n16 = 4, m8n16 = 8, m12n16 = 12 };
enum class LoadStrategy {
    four_u64,
    vlse64,
    vlseg2_even,
    vlseg2_pair_vlse,
    vlseg2_pair_shift,
};
enum class EpilogueStrategy { generic_scalar, inline_scalar, rvv_q62 };
enum class PartitionPolicy { spatial, output_channel };
enum class NonConvStrategy { serial_scalar, parallel_scalar, explicit_rvv_lut };
enum class SchedulerStrategy { all_workers_complete, active_workers_complete };

struct RunOptions {
    ComputeRoute route = ComputeRoute::ime;
    KernelShape kernel = KernelShape::m12n16;
    LoadStrategy load = LoadStrategy::vlseg2_even;
    EpilogueStrategy epilogue = EpilogueStrategy::inline_scalar;
    PartitionPolicy partition = PartitionPolicy::spatial;
    NonConvStrategy nonconv = NonConvStrategy::serial_scalar;
    SchedulerStrategy scheduler = SchedulerStrategy::all_workers_complete;
    int workers = 4;
    bool profile_phases = false;
    bool capture_intermediates = false;
    std::string counter_event;
};

struct WorkerCounter {
    int worker = -1;
    std::string event;
    std::string status;
    int error_number = 0;
    std::uint64_t count = 0;
    std::uint64_t time_enabled = 0;
    std::uint64_t time_running = 0;
};

struct OperationTiming {
    int operation_index = -1;
    std::string kind;
    std::string name;
    double wall_us = 0.0;
    double delivery_worker_sum_us = 0.0;
    double vmadot_worker_sum_us = 0.0;
    double epilogue_worker_sum_us = 0.0;
};

struct SliceTiming {
    double total_us = 0.0;
    double conv_us = 0.0;
    double lut_us = 0.0;
    double add_us = 0.0;
    double concat_us = 0.0;
    double min_worker_us = 0.0;
    double max_worker_us = 0.0;
    int affinity_ok = 0;
    std::vector<OperationTiming> operations;
    std::vector<WorkerCounter> worker_counters;
};

class PersistentSlice {
public:
    struct Impl;

    PersistentSlice();
    ~PersistentSlice();
    PersistentSlice(PersistentSlice&&) noexcept;
    PersistentSlice& operator=(PersistentSlice&&) noexcept;
    PersistentSlice(const PersistentSlice&) = delete;
    PersistentSlice& operator=(const PersistentSlice&) = delete;

    int prepare(const std::filesystem::path& package_dir,
                const std::string& trusted_manifest_sha256,
                int worker_capacity);
    int prepare_with_contract(const std::filesystem::path& package_dir,
                              const std::string& trusted_manifest_sha256,
                              int worker_capacity,
                              const std::string& expected_contract_id,
                              const std::string& expected_profile_id);

    int run_model5(const std::int8_t* model4_postactivation_nchwc8_s8,
                   std::int8_t* model5_output_nchwc8_s8,
                   const RunOptions& options,
                   SliceTiming* timing);

    int run_slice(const std::int8_t* model4_preactivation_nchwc8_s8,
                  std::int8_t* slice_output_nchwc8_s8,
                  const RunOptions& options,
                  SliceTiming* timing);

    int load_tensor(int tensor_id, const std::int8_t* source, std::size_t bytes);
    int run_model5_resident(const RunOptions& options, SliceTiming* timing);
    int run_slice_resident(const RunOptions& options, SliceTiming* timing);
    int run_operation_resident(int operation_index, const RunOptions& options,
                               SliceTiming* timing);

    int copy_tensor(int tensor_id, std::int8_t* destination, std::size_t bytes) const;
    int copy_captured_tensor(int tensor_id, std::int8_t* destination, std::size_t bytes) const;
    std::size_t tensor_bytes(int tensor_id) const noexcept;
    int tensor_count() const noexcept;
    int operation_count() const noexcept;
    int input_tensor_id() const noexcept;
    int model5_output_tensor_id() const noexcept;
    int model6_output_tensor_id() const noexcept;
    int output_tensor_id() const noexcept;
    int operation_input_tensor_id(int operation_index, int slot) const noexcept;
    int operation_output_tensor_id(int operation_index, int slot) const noexcept;
    std::size_t arena_bytes() const noexcept;
    std::size_t packed_weight_bytes() const noexcept;
    bool worker_affinity_ok() const noexcept;
    const std::string& manifest_sha256() const noexcept;
    const std::string& last_error() const noexcept;

private:
    std::unique_ptr<Impl> impl_;
};

const char* compute_route_name(ComputeRoute value) noexcept;
const char* kernel_shape_name(KernelShape value) noexcept;
const char* load_strategy_name(LoadStrategy value) noexcept;
const char* epilogue_strategy_name(EpilogueStrategy value) noexcept;
const char* partition_policy_name(PartitionPolicy value) noexcept;
const char* nonconv_strategy_name(NonConvStrategy value) noexcept;
const char* scheduler_strategy_name(SchedulerStrategy value) noexcept;

}  // namespace y26::stage49
