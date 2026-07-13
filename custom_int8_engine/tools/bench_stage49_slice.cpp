#include "y26_k1x_conv_kernels.h"
#include "y26_k1x_package.h"
#include "y26_k1x_stage49_slice.h"

#include <algorithm>
#include <array>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
using y26::stage49::ComputeRoute;
using y26::stage49::EpilogueStrategy;
using y26::stage49::KernelShape;
using y26::stage49::LoadStrategy;
using y26::stage49::NonConvStrategy;
using y26::stage49::PartitionPolicy;
using y26::stage49::PersistentSlice;
using y26::stage49::RunOptions;
using y26::stage49::SchedulerStrategy;
using y26::stage49::SliceTiming;

struct Options {
    std::string mode = "verify";
    std::filesystem::path package;
    std::string trusted_manifest;
    std::string fixture = "F0";
    RunOptions run;
    int warmup = 10;
    int runs = 100;
    int repeats = 5;
    int controller_cpu = -1;
    int operation_index = -1;
    std::string integer_contract = "v1";
};

struct Statistics {
    double mean = 0.0;
    double stddev = 0.0;
    double cv_pct = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    double median = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
};

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

int parse_int(const std::string& value, const char* option) {
    std::size_t consumed = 0;
    const long parsed = std::stol(value, &consumed);
    if (consumed != value.size() || parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        fail(std::string("invalid ") + option + ": " + value);
    }
    return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (index + 1 >= argc) fail("missing value for " + key);
        const std::string value = argv[++index];
        if (key == "--mode") options.mode = value;
        else if (key == "--package") options.package = value;
        else if (key == "--trusted-manifest-sha256") options.trusted_manifest = value;
        else if (key == "--fixture") options.fixture = value;
        else if (key == "--integer-contract") {
            if (value != "v1" && value != "q31") fail("invalid integer contract: " + value);
            options.integer_contract = value;
        }
        else if (key == "--route") {
            if (value == "scalar") options.run.route = ComputeRoute::scalar;
            else if (value == "ime") options.run.route = ComputeRoute::ime;
            else fail("invalid route: " + value);
        } else if (key == "--kernel") {
            if (value == "m4n16") options.run.kernel = KernelShape::m4n16;
            else if (value == "m8n16") options.run.kernel = KernelShape::m8n16;
            else if (value == "m12n16") options.run.kernel = KernelShape::m12n16;
            else fail("invalid kernel: " + value);
        } else if (key == "--load") {
            if (value == "four_u64") options.run.load = LoadStrategy::four_u64;
            else if (value == "vlse64") options.run.load = LoadStrategy::vlse64;
            else if (value == "vlseg2_even") options.run.load = LoadStrategy::vlseg2_even;
            else if (value == "vlseg2_pair_vlse") options.run.load = LoadStrategy::vlseg2_pair_vlse;
            else if (value == "vlseg2_pair_shift") options.run.load = LoadStrategy::vlseg2_pair_shift;
            else fail("invalid load strategy: " + value);
        } else if (key == "--epilogue") {
            if (value == "e0") options.run.epilogue = EpilogueStrategy::generic_scalar;
            else if (value == "e1") options.run.epilogue = EpilogueStrategy::inline_scalar;
            else if (value == "e2") options.run.epilogue = EpilogueStrategy::rvv_q62;
            else fail("invalid epilogue: " + value);
        } else if (key == "--partition") {
            if (value == "spatial") options.run.partition = PartitionPolicy::spatial;
            else if (value == "output_channel") options.run.partition = PartitionPolicy::output_channel;
            else fail("invalid partition: " + value);
        } else if (key == "--nonconv") {
            if (value == "serial") options.run.nonconv = NonConvStrategy::serial_scalar;
            else if (value == "parallel") options.run.nonconv = NonConvStrategy::parallel_scalar;
            else if (value == "rvv_lut") options.run.nonconv = NonConvStrategy::explicit_rvv_lut;
            else fail("invalid non-Conv strategy: " + value);
        } else if (key == "--scheduler") {
            if (value == "all") options.run.scheduler = SchedulerStrategy::all_workers_complete;
            else if (value == "active") options.run.scheduler = SchedulerStrategy::active_workers_complete;
            else fail("invalid scheduler strategy: " + value);
        } else if (key == "--counter-event") options.run.counter_event = value;
        else if (key == "--workers") options.run.workers = parse_int(value, "workers");
        else if (key == "--warmup") options.warmup = parse_int(value, "warmup");
        else if (key == "--runs") options.runs = parse_int(value, "runs");
        else if (key == "--repeats") options.repeats = parse_int(value, "repeats");
        else if (key == "--controller-cpu") options.controller_cpu = parse_int(value, "controller-cpu");
        else if (key == "--operation-index") options.operation_index = parse_int(value, "operation-index");
        else fail("unknown option: " + key);
    }
    if (options.package.empty() || options.trusted_manifest.size() != 64) {
        fail("--package and a 64-character --trusted-manifest-sha256 are required");
    }
    if (options.fixture.size() != 2 || options.fixture[0] != 'F' || options.fixture[1] < '0' || options.fixture[1] > '7') {
        fail("fixture must be F0-F7");
    }
    if (options.run.workers < 1 || options.run.workers > 4 || options.warmup < 0 ||
        options.runs < 1 || options.repeats < 1 || (options.controller_cpu != -1 && options.controller_cpu != 4)) {
        fail("invalid worker/timing/controller option");
    }
    return options;
}

std::vector<std::int8_t> read_bytes(const std::filesystem::path& path, std::size_t expected) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) fail("cannot open " + path.string());
    const auto size = stream.tellg();
    if (size < 0 || static_cast<std::size_t>(size) != expected) fail("size mismatch: " + path.string());
    std::vector<std::int8_t> bytes(expected);
    stream.seekg(0);
    if (expected != 0) stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(expected));
    if (!stream) fail("read failure: " + path.string());
    return bytes;
}

std::filesystem::path oracle_path(const Options& options, int tensor_id) {
    std::ostringstream name;
    name << "tensor_" << std::setw(3) << std::setfill('0') << tensor_id << "_nchwc8_s8.bin";
    return options.package / "oracles" / options.fixture / name.str();
}

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

double process_cpu_us() {
    timespec value {};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) != 0) fail("clock_gettime failed");
    return static_cast<double>(value.tv_sec) * 1.0e6 + static_cast<double>(value.tv_nsec) / 1.0e3;
}

double percentile(const std::vector<double>& sorted, double quantile) {
    if (sorted.empty()) return 0.0;
    const double position = quantile * static_cast<double>(sorted.size() - 1);
    const auto low = static_cast<std::size_t>(std::floor(position));
    const auto high = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(low);
    return sorted[low] * (1.0 - fraction) + sorted[high] * fraction;
}

Statistics statistics(std::vector<double> values) {
    if (values.empty()) fail("empty timing vector");
    Statistics result;
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    double squared = 0.0;
    for (double value : values) squared += (value - result.mean) * (value - result.mean);
    result.stddev = values.size() > 1 ? std::sqrt(squared / static_cast<double>(values.size() - 1)) : 0.0;
    result.cv_pct = result.mean != 0.0 ? result.stddev * 100.0 / result.mean : 0.0;
    std::sort(values.begin(), values.end());
    result.minimum = values.front();
    result.maximum = values.back();
    result.median = percentile(values, 0.5);
    result.p90 = percentile(values, 0.9);
    result.p95 = percentile(values, 0.95);
    return result;
}

std::size_t mismatch_count(const std::vector<std::int8_t>& actual, const std::vector<std::int8_t>& expected) {
    if (actual.size() != expected.size()) return std::max(actual.size(), expected.size());
    std::size_t mismatches = 0;
    for (std::size_t index = 0; index < actual.size(); ++index) mismatches += actual[index] != expected[index];
    return mismatches;
}

std::uint64_t fnv1a64(const std::vector<std::int8_t>& bytes) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::int8_t byte : bytes) {
        hash ^= static_cast<std::uint8_t>(byte);
        hash *= 1099511628211ULL;
    }
    return hash;
}

void print_contract(const Options& options, const PersistentSlice& executor) {
    const bool q31 = options.integer_contract == "q31";
    std::cout << "contract_id=" << (q31 ? "K1X_INT8_V2_Q31_CANDIDATE" : "K1X_INT8_V1") << '\n'
              << "profile_id=" << (q31 ? "K1X_INT8_V2_Q31_CANDIDATE_GENERAL" : "K1X_INT8_V1_GENERAL") << '\n'
              << "layout_id=NCHWc8_SPATIAL_INNER_V1\n"
              << "manifest_sha256=" << executor.manifest_sha256() << '\n'
              << "fixture=" << options.fixture << '\n'
              << "route=" << y26::stage49::compute_route_name(options.run.route) << '\n'
              << "kernel=" << y26::stage49::kernel_shape_name(options.run.kernel) << '\n'
              << "load=" << y26::stage49::load_strategy_name(options.run.load) << '\n'
              << "epilogue=" << y26::stage49::epilogue_strategy_name(options.run.epilogue) << '\n'
              << "partition=" << y26::stage49::partition_policy_name(options.run.partition) << '\n'
              << "nonconv=" << y26::stage49::nonconv_strategy_name(options.run.nonconv) << '\n'
              << "scheduler=" << y26::stage49::scheduler_strategy_name(options.run.scheduler) << '\n'
              << "workers=" << options.run.workers << '\n'
              << "controller_cpu=" << options.controller_cpu << '\n'
              << "arena_bytes=" << executor.arena_bytes() << '\n'
              << "packed_weight_bytes=" << executor.packed_weight_bytes() << '\n'
              << "tensor_count=" << executor.tensor_count() << '\n'
              << "operation_count=" << executor.operation_count() << '\n';
}

bool pin_controller(int cpu) {
    if (cpu < 0) return true;
#if defined(__linux__)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) == 0;
#else
    (void)cpu;
    return false;
#endif
}

int run_once(PersistentSlice& executor, const std::string& surface, const RunOptions& options,
             const std::vector<std::int8_t>& input, std::vector<std::int8_t>& output, SliceTiming* timing) {
    if (surface == "model5") return executor.run_model5(input.data(), output.data(), options, timing);
    return executor.run_slice(input.data(), output.data(), options, timing);
}

int validate(PersistentSlice& executor, const Options& options, bool slice, bool boundaries) {
    const int input_id = slice ? executor.input_tensor_id() : executor.operation_input_tensor_id(1, 0);
    const int output_id = slice ? executor.output_tensor_id() : executor.model5_output_tensor_id();
    auto input = read_bytes(oracle_path(options, input_id), executor.tensor_bytes(input_id));
    auto expected = read_bytes(oracle_path(options, output_id), executor.tensor_bytes(output_id));
    std::vector<std::int8_t> actual(expected.size());
    RunOptions run = options.run;
    run.capture_intermediates = boundaries;
    SliceTiming timing;
    const int status = run_once(executor, slice ? "slice" : "model5", run, input, actual, &timing);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "execution_status=" << status << " error=" << executor.last_error() << '\n';
        return 2;
    }
    std::size_t mismatches = mismatch_count(actual, expected);
    if (boundaries) {
        for (int tensor_id = 0; tensor_id < executor.tensor_count(); ++tensor_id) {
            std::vector<std::int8_t> captured(executor.tensor_bytes(tensor_id));
            const int copy_status = executor.copy_captured_tensor(tensor_id, captured.data(), captured.size());
            if (copy_status != Y26_CONV_STATUS_SUCCESS) {
                std::cout << "boundary=" << tensor_id << " status=not-captured\n";
                ++mismatches;
                continue;
            }
            const auto boundary_expected = read_bytes(oracle_path(options, tensor_id), captured.size());
            const auto boundary_mismatches = mismatch_count(captured, boundary_expected);
            std::cout << "boundary=" << tensor_id << " mismatches=" << boundary_mismatches
                      << " hash_fnv1a64=0x" << std::hex << fnv1a64(captured) << std::dec << '\n';
            mismatches += boundary_mismatches;
        }
    }
    std::cout << "execution_status=0\nmismatches=" << mismatches
              << "\noutput_hash_fnv1a64=0x" << std::hex << fnv1a64(actual) << std::dec
              << "\naffinity_ok=" << timing.affinity_ok << '\n';
    return mismatches == 0 ? 0 : 3;
}

int benchmark(PersistentSlice& executor, const Options& options, bool slice) {
    const int input_id = slice ? executor.input_tensor_id() : executor.operation_input_tensor_id(1, 0);
    const int output_id = slice ? executor.output_tensor_id() : executor.model5_output_tensor_id();
    auto input = read_bytes(oracle_path(options, input_id), executor.tensor_bytes(input_id));
    auto expected = read_bytes(oracle_path(options, output_id), executor.tensor_bytes(output_id));
    std::vector<std::int8_t> output(expected.size());
    const auto execute_resident = [&]() -> int {
        if (executor.load_tensor(input_id, input.data(), input.size()) != Y26_CONV_STATUS_SUCCESS) {
            return Y26_CONV_STATUS_INVALID_ARGUMENT;
        }
        return slice ? executor.run_slice_resident(options.run, nullptr)
                     : executor.run_model5_resident(options.run, nullptr);
    };
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        if (execute_resident() != Y26_CONV_STATUS_SUCCESS) {
            return 2;
        }
    }
    std::vector<double> wall_samples;
    std::vector<double> cpu_samples;
    std::vector<double> repeat_means;
    wall_samples.reserve(static_cast<std::size_t>(options.runs * options.repeats));
    cpu_samples.reserve(wall_samples.capacity());
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        std::vector<double> current;
        current.reserve(static_cast<std::size_t>(options.runs));
        for (int run = 0; run < options.runs; ++run) {
            if (executor.load_tensor(input_id, input.data(), input.size()) != Y26_CONV_STATUS_SUCCESS) return 2;
            const double cpu_begin = process_cpu_us();
            const auto begin = Clock::now();
            const int status = slice ? executor.run_slice_resident(options.run, nullptr)
                                     : executor.run_model5_resident(options.run, nullptr);
            const auto end = Clock::now();
            const double cpu_end = process_cpu_us();
            if (status != Y26_CONV_STATUS_SUCCESS) return 2;
            const double wall = elapsed_us(begin, end);
            wall_samples.push_back(wall);
            cpu_samples.push_back(cpu_end - cpu_begin);
            current.push_back(wall);
            std::cout << "raw\trepeat=" << repeat << "\trun=" << run << "\twall_us=" << wall
                      << "\tprocess_cpu_us=" << (cpu_end - cpu_begin) << '\n';
        }
        const auto current_stats = statistics(current);
        repeat_means.push_back(current_stats.mean);
        std::cout << "repeat_summary\trepeat=" << repeat << "\tmean_us=" << current_stats.mean
                  << "\tstddev_us=" << current_stats.stddev << "\tp95_us=" << current_stats.p95 << '\n';
    }
    const auto wall = statistics(wall_samples);
    const auto cpu = statistics(cpu_samples);
    const auto repeats = statistics(repeat_means);
    if (executor.copy_tensor(output_id, output.data(), output.size()) != Y26_CONV_STATUS_SUCCESS) return 2;
    const auto mismatches = mismatch_count(output, expected);
    std::cout << std::fixed << std::setprecision(6)
              << "mean_us=" << wall.mean << "\nstddev_us=" << wall.stddev << "\ncv_pct=" << wall.cv_pct
              << "\nmin_us=" << wall.minimum << "\nmax_us=" << wall.maximum << "\nmedian_us=" << wall.median
              << "\np90_us=" << wall.p90 << "\np95_us=" << wall.p95
              << "\nprocess_cpu_mean_us=" << cpu.mean << "\nrepeat_mean_cv_pct=" << repeats.cv_pct
              << "\nmismatches=" << mismatches << "\noutput_hash_fnv1a64=0x" << std::hex << fnv1a64(output) << std::dec
              << "\naffinity_ok=" << (executor.worker_affinity_ok() ? 1 : 0) << '\n';
    return mismatches == 0 ? 0 : 3;
}

int benchmark_operation(PersistentSlice& executor, const Options& options) {
    if (options.operation_index < 0 || options.operation_index >= executor.operation_count()) {
        fail("--operation-index is required for benchmark-operation");
    }
    struct Input {
        int id = -1;
        std::vector<std::int8_t> bytes;
    };
    std::vector<Input> inputs;
    for (int slot = 0; slot < 3; ++slot) {
        const int id = executor.operation_input_tensor_id(options.operation_index, slot);
        if (id < 0 || std::any_of(inputs.begin(), inputs.end(), [id](const Input& input) { return input.id == id; })) {
            continue;
        }
        inputs.push_back({id, read_bytes(oracle_path(options, id), executor.tensor_bytes(id))});
    }
    const int output_id = executor.operation_output_tensor_id(options.operation_index, 0);
    if (inputs.empty() || output_id < 0) fail("invalid operation descriptor");
    const auto expected = read_bytes(oracle_path(options, output_id), executor.tensor_bytes(output_id));
    const auto load_inputs = [&]() {
        for (const Input& input : inputs) {
            if (executor.load_tensor(input.id, input.bytes.data(), input.bytes.size()) != Y26_CONV_STATUS_SUCCESS) {
                return false;
            }
        }
        return true;
    };
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        if (!load_inputs() || executor.run_operation_resident(options.operation_index, options.run, nullptr) !=
                                  Y26_CONV_STATUS_SUCCESS) return 2;
    }
    std::vector<double> wall_samples;
    std::vector<double> cpu_samples;
    std::vector<double> repeat_means;
    wall_samples.reserve(static_cast<std::size_t>(options.runs * options.repeats));
    cpu_samples.reserve(wall_samples.capacity());
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        std::vector<double> current;
        current.reserve(static_cast<std::size_t>(options.runs));
        for (int run = 0; run < options.runs; ++run) {
            if (!load_inputs()) return 2;
            const double cpu_begin = process_cpu_us();
            const auto begin = Clock::now();
            const int status = executor.run_operation_resident(options.operation_index, options.run, nullptr);
            const auto end = Clock::now();
            const double cpu_end = process_cpu_us();
            if (status != Y26_CONV_STATUS_SUCCESS) return 2;
            const double wall = elapsed_us(begin, end);
            wall_samples.push_back(wall);
            cpu_samples.push_back(cpu_end - cpu_begin);
            current.push_back(wall);
            std::cout << "raw\trepeat=" << repeat << "\trun=" << run << "\twall_us=" << wall
                      << "\tprocess_cpu_us=" << (cpu_end - cpu_begin) << '\n';
        }
        const auto current_stats = statistics(current);
        repeat_means.push_back(current_stats.mean);
        std::cout << "repeat_summary\trepeat=" << repeat << "\tmean_us=" << current_stats.mean
                  << "\tstddev_us=" << current_stats.stddev << "\tp95_us=" << current_stats.p95 << '\n';
    }
    std::vector<std::int8_t> output(expected.size());
    if (executor.copy_tensor(output_id, output.data(), output.size()) != Y26_CONV_STATUS_SUCCESS) return 2;
    const auto wall = statistics(wall_samples);
    const auto cpu = statistics(cpu_samples);
    const auto repeats = statistics(repeat_means);
    const auto mismatches = mismatch_count(output, expected);
    std::cout << std::fixed << std::setprecision(6)
              << "operation_index=" << options.operation_index
              << "\nmean_us=" << wall.mean << "\nstddev_us=" << wall.stddev << "\ncv_pct=" << wall.cv_pct
              << "\nmin_us=" << wall.minimum << "\nmax_us=" << wall.maximum << "\nmedian_us=" << wall.median
              << "\np90_us=" << wall.p90 << "\np95_us=" << wall.p95
              << "\nprocess_cpu_mean_us=" << cpu.mean << "\nrepeat_mean_cv_pct=" << repeats.cv_pct
              << "\nmismatches=" << mismatches << "\noutput_hash_fnv1a64=0x" << std::hex << fnv1a64(output)
              << std::dec << "\naffinity_ok=" << (executor.worker_affinity_ok() ? 1 : 0) << '\n';
    return mismatches == 0 ? 0 : 3;
}

int profile(PersistentSlice& executor, const Options& options, bool slice) {
    const int input_id = slice ? executor.input_tensor_id() : executor.operation_input_tensor_id(1, 0);
    const int output_id = slice ? executor.output_tensor_id() : executor.model5_output_tensor_id();
    auto input = read_bytes(oracle_path(options, input_id), executor.tensor_bytes(input_id));
    auto expected = read_bytes(oracle_path(options, output_id), executor.tensor_bytes(output_id));
    std::vector<std::int8_t> output(expected.size());
    RunOptions run = options.run;
    run.profile_phases = true;
    SliceTiming timing;
    const int status = run_once(executor, slice ? "slice" : "model5", run, input, output, &timing);
    if (status != Y26_CONV_STATUS_SUCCESS) return 2;
    for (const auto& operation : timing.operations) {
        std::cout << "operation=" << operation.operation_index << "\tkind=" << operation.kind
                  << "\tname=" << operation.name << "\twall_us=" << operation.wall_us
                  << "\tdelivery_worker_sum_us=" << operation.delivery_worker_sum_us
                  << "\tvmadot_worker_sum_us=" << operation.vmadot_worker_sum_us
                  << "\tepilogue_worker_sum_us=" << operation.epilogue_worker_sum_us << '\n';
    }
    for (const auto& counter : timing.worker_counters) {
        const double scale = counter.time_running == 0
                                 ? 0.0
                                 : static_cast<double>(counter.time_enabled) /
                                       static_cast<double>(counter.time_running);
        std::cout << "worker_counter=" << counter.worker << "\tevent=" << counter.event
                  << "\tstatus=" << counter.status << "\terrno=" << counter.error_number
                  << "\tcount=" << counter.count << "\ttime_enabled=" << counter.time_enabled
                  << "\ttime_running=" << counter.time_running << "\tscale=" << scale << '\n';
    }
    std::cout << "total_us=" << timing.total_us << "\nconv_us=" << timing.conv_us << "\nlut_us=" << timing.lut_us
              << "\nadd_us=" << timing.add_us << "\nconcat_us=" << timing.concat_us
              << "\nmin_worker_us=" << timing.min_worker_us << "\nmax_worker_us=" << timing.max_worker_us
              << "\nmismatches=" << mismatch_count(output, expected) << "\naffinity_ok=" << timing.affinity_ok << '\n';
    return mismatch_count(output, expected) == 0 ? 0 : 3;
}

int frm_sweep(PersistentSlice& executor, const Options& options) {
#if defined(__riscv)
    const auto read_frm = []() noexcept {
        unsigned value = 0;
        asm volatile("csrr %0, frm" : "=r"(value));
        return value;
    };
    const auto write_frm = [&](unsigned value) noexcept {
        asm volatile("csrw frm, %0" : : "r"(value));
        return read_frm() == value;
    };
    const unsigned original = read_frm();
    constexpr std::array<const char*, 5> names {"RNE", "RTZ", "RDN", "RUP", "RMM"};
    for (unsigned mode = 0; mode < names.size(); ++mode) {
        if (!write_frm(mode) || validate(executor, options, true, false) != 0 || read_frm() != mode) {
            write_frm(original);
            return 3;
        }
        std::cout << "frm=" << names[mode] << " status=exact\n";
    }
    const bool restored = write_frm(original);
    std::cout << "frm_restored=" << (restored && read_frm() == original ? 1 : 0) << '\n';
    return restored ? 0 : 3;
#else
    const int original = std::fegetround();
    const std::array<int, 4> modes {FE_TONEAREST, FE_TOWARDZERO, FE_DOWNWARD, FE_UPWARD};
    for (int mode : modes) {
        if (std::fesetround(mode) != 0 || validate(executor, options, true, false) != 0) {
            std::fesetround(original);
            return 3;
        }
        std::cout << "frm=" << mode << " status=exact\n";
    }
    const int restore_status = std::fesetround(original);
    std::cout << "frm_restored=" << (restore_status == 0 && std::fegetround() == original ? 1 : 0) << '\n';
    return restore_status == 0 ? 0 : 3;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (!pin_controller(options.controller_cpu)) fail("controller affinity failed");
        PersistentSlice executor;
        const bool q31 = options.integer_contract == "q31";
        const int prepare_status = q31
            ? executor.prepare_with_contract(options.package, options.trusted_manifest, 4,
                  "K1X_INT8_V2_Q31_CANDIDATE", "K1X_INT8_V2_Q31_CANDIDATE_GENERAL")
            : executor.prepare(options.package, options.trusted_manifest, 4);
        if (prepare_status != Y26_CONV_STATUS_SUCCESS) {
            std::cerr << "prepare_status=" << prepare_status << " error=" << executor.last_error() << '\n';
            return 2;
        }
        print_contract(options, executor);
        if (options.mode == "verify") return 0;
        if (options.mode == "validate-model5") return validate(executor, options, false, false);
        if (options.mode == "validate-slice") return validate(executor, options, true, false);
        if (options.mode == "validate-boundaries") return validate(executor, options, true, true);
        if (options.mode == "benchmark-model5") return benchmark(executor, options, false);
        if (options.mode == "benchmark-slice") return benchmark(executor, options, true);
        if (options.mode == "benchmark-operation") return benchmark_operation(executor, options);
        if (options.mode == "profile-model5") return profile(executor, options, false);
        if (options.mode == "profile-slice") return profile(executor, options, true);
        if (options.mode == "frm-sweep") return frm_sweep(executor, options);
        fail("invalid mode: " + options.mode);
    } catch (const std::exception& error) {
        std::cerr << "error=" << error.what() << '\n';
        return 64;
    }
}
