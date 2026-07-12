#include "y26_k1x_stage47_aot.h"

#include "y26_k1x_conv_kernels.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cfenv>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using y26::stage47::AotExecutor;
using y26::stage47::ConvSpec;
using y26::stage47::ExecutorTiming;
using y26::stage47::IntegratedConv;
using y26::stage47::IntegratedTiming;
using y26::stage47::KernelShape;
using y26::stage47::OutputSegmentSpec;
using y26::stage47::PartitionPolicy;
using y26::stage47::RunOptions;
using y26::stage47::TensorSpec;
using y26::stage47::WorkerPool;

struct Options {
    std::string mode;
    std::filesystem::path package;
    std::filesystem::path generated_root;
    KernelShape kernel = KernelShape::scalar;
    PartitionPolicy partition = PartitionPolicy::spatial;
    int workers = 1;
    int case_id = -1;
    int warmup = 10;
    int runs = 100;
    int repeats = 5;
    bool profile_phases = false;
    std::string fixture;
};

struct Stats {
    double mean = 0.0;
    double stddev = 0.0;
    double cv_pct = 0.0;
    double min = 0.0;
    double max = 0.0;
    double median = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
};

std::vector<std::string> split_tsv(const std::string& line) {
    std::vector<std::string> values;
    std::size_t begin = 0;
    for (;;) {
        const std::size_t end = line.find('\t', begin);
        values.push_back(line.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return values;
}

std::vector<std::unordered_map<std::string, std::string>> read_tsv(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open TSV: " + path.string());
    std::string line;
    if (!std::getline(stream, line)) throw std::runtime_error("empty TSV: " + path.string());
    const auto header = split_tsv(line);
    std::vector<std::unordered_map<std::string, std::string>> rows;
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        const auto fields = split_tsv(line);
        if (fields.size() != header.size()) throw std::runtime_error("malformed TSV: " + path.string());
        auto& row = rows.emplace_back();
        for (std::size_t index = 0; index < header.size(); ++index) row.emplace(header[index], fields[index]);
    }
    return rows;
}

const std::string& field(const std::unordered_map<std::string, std::string>& row, const char* key) {
    const auto found = row.find(key);
    if (found == row.end()) throw std::runtime_error(std::string("missing field: ") + key);
    return found->second;
}

int integer(const std::unordered_map<std::string, std::string>& row, const char* key) {
    const std::string& value = field(row, key);
    int result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size()) {
        throw std::runtime_error(std::string("invalid integer: ") + key);
    }
    return result;
}

std::uint64_t unsigned_integer(const std::unordered_map<std::string, std::string>& row, const char* key) {
    const std::string& value = field(row, key);
    std::uint64_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size()) {
        throw std::runtime_error(std::string("invalid unsigned integer: ") + key);
    }
    return result;
}

float floating(const std::unordered_map<std::string, std::string>& row, const char* key) {
    std::size_t consumed = 0;
    const std::string& value = field(row, key);
    const float result = std::stof(value, &consumed);
    if (consumed != value.size() || !std::isfinite(result)) throw std::runtime_error(std::string("invalid float: ") + key);
    return result;
}

template <typename T>
std::vector<T> read_binary(const std::filesystem::path& path, std::size_t expected_count = 0) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open binary: " + path.string());
    const std::streamsize bytes = stream.tellg();
    if (bytes < 0 || static_cast<std::size_t>(bytes) % sizeof(T) != 0) throw std::runtime_error("binary size invalid");
    const std::size_t count = static_cast<std::size_t>(bytes) / sizeof(T);
    if (expected_count != 0 && count != expected_count) throw std::runtime_error("binary count mismatch: " + path.string());
    stream.seekg(0);
    std::vector<T> result(count);
    if (bytes != 0 && !stream.read(reinterpret_cast<char*>(result.data()), bytes)) throw std::runtime_error("binary read failed");
    return result;
}

std::uint64_t hash_bytes(const void* data, std::size_t bytes) {
    const auto* values = static_cast<const std::uint8_t*>(data);
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t index = 0; index < bytes; ++index) {
        hash ^= values[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

double percentile(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    const double position = q * static_cast<double>(sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
}

Stats summarize(std::vector<double> values) {
    Stats result;
    if (values.empty()) return result;
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    result.min = *std::min_element(values.begin(), values.end());
    result.max = *std::max_element(values.begin(), values.end());
    if (values.size() > 1) {
        double sum = 0.0;
        for (double value : values) sum += (value - result.mean) * (value - result.mean);
        result.stddev = std::sqrt(sum / static_cast<double>(values.size() - 1));
    }
    result.cv_pct = result.mean == 0.0 ? 0.0 : result.stddev / result.mean * 100.0;
    std::sort(values.begin(), values.end());
    result.median = percentile(values, 0.5);
    result.p90 = percentile(values, 0.9);
    result.p95 = percentile(values, 0.95);
    return result;
}

KernelShape parse_kernel(const std::string& value) {
    if (value == "scalar") return KernelShape::scalar;
    if (value == "m4n16") return KernelShape::m4n16;
    if (value == "m8n16") return KernelShape::m8n16;
    if (value == "m12n16") return KernelShape::m12n16;
    throw std::runtime_error("invalid kernel: " + value);
}

PartitionPolicy parse_partition(const std::string& value) {
    if (value == "spatial") return PartitionPolicy::spatial;
    if (value == "output_channel") return PartitionPolicy::output_channel;
    throw std::runtime_error("invalid partition: " + value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next = [&]() -> std::string {
            if (++index >= argc) throw std::runtime_error("missing value for " + argument);
            return argv[index];
        };
        if (argument == "--mode") options.mode = next();
        else if (argument == "--package") options.package = next();
        else if (argument == "--generated-root") options.generated_root = next();
        else if (argument == "--kernel") options.kernel = parse_kernel(next());
        else if (argument == "--partition") options.partition = parse_partition(next());
        else if (argument == "--workers") options.workers = std::stoi(next());
        else if (argument == "--case-id") options.case_id = std::stoi(next());
        else if (argument == "--warmup") options.warmup = std::stoi(next());
        else if (argument == "--runs") options.runs = std::stoi(next());
        else if (argument == "--repeats") options.repeats = std::stoi(next());
        else if (argument == "--profile-phases") options.profile_phases = std::stoi(next()) != 0;
        else if (argument == "--fixture") options.fixture = next();
        else if (argument == "--help") {
            std::cout << "usage: bench_stage47_executor --mode aot-validate|aot-benchmark|kernel-validate|kernel-benchmark|frm-sweep"
                         " --package PATH --generated-root PATH [--kernel scalar|m4n16|m8n16|m12n16]"
                         " [--partition spatial|output_channel] [--workers 1..4] [--case-id N]"
                         " [--warmup N --runs N --repeats N --profile-phases 0|1]\n";
            std::exit(0);
        } else throw std::runtime_error("unknown option: " + argument);
    }
    if (options.mode.empty() || options.workers < 1 || options.workers > 4 || options.warmup < 0 ||
        options.runs < 1 || options.repeats < 1) throw std::runtime_error("invalid options");
    return options;
}

void print_stats(const char* table,
                 const std::string& surface,
                 const Options& options,
                 const Stats& stats,
                 std::uint64_t macs,
                 std::uint64_t checksum) {
    const double gmacs = stats.mean > 0.0 ? static_cast<double>(macs) / (stats.mean * 1000.0) : 0.0;
    std::cout << table << '\t' << surface << '\t' << y26::stage47::kernel_shape_name(options.kernel) << '\t'
              << y26::stage47::partition_policy_name(options.partition) << '\t' << options.workers << '\t'
              << options.warmup << '\t' << options.runs << '\t' << options.repeats << '\t' << macs << '\t'
              << std::fixed << std::setprecision(6) << stats.mean << '\t' << stats.stddev << '\t' << stats.cv_pct << '\t'
              << stats.min << '\t' << stats.max << '\t' << stats.median << '\t' << stats.p90 << '\t' << stats.p95 << '\t'
              << gmacs << '\t' << checksum << '\n';
}

struct PreparedCase {
    int id = -1;
    std::string name;
    TensorSpec input_spec;
    TensorSpec output_spec;
    std::vector<std::int8_t> weights;
    std::vector<float> scales;
    std::vector<std::int32_t> bias;
    std::vector<std::uint8_t> input_nchw;
    std::vector<std::uint8_t> expected_nchw;
    std::vector<std::int8_t> input_nhwc;
    std::vector<std::int8_t> output_nhwc;
    std::vector<std::uint8_t> output_nchw;
    IntegratedConv conv;
    std::uint64_t macs = 0;
};

PreparedCase load_case(const Options& options, const std::unordered_map<std::string, std::string>& row) {
    PreparedCase result;
    result.id = integer(row, "case_id");
    result.name = field(row, "node_name");
    result.input_spec = {integer(row, "input_h"), integer(row, "input_w"), integer(row, "input_c"),
                         floating(row, "input_scale"), integer(row, "input_zero_point")};
    result.output_spec = {integer(row, "output_h"), integer(row, "output_w"), integer(row, "output_c"),
                          floating(row, "output_scale"), integer(row, "output_zero_point")};
    const auto root = options.generated_root;
    result.weights = read_binary<std::int8_t>(root / field(row, "weights_file"));
    result.scales = read_binary<float>(root / field(row, "weight_scales_file"), result.output_spec.c);
    result.bias = read_binary<std::int32_t>(root / field(row, "bias_file"), result.output_spec.c);
    result.input_nchw = read_binary<std::uint8_t>(root / field(row, "input_file"));
    result.expected_nchw = read_binary<std::uint8_t>(root / field(row, "expected_file"));
    result.input_nhwc.resize(result.input_nchw.size());
    result.output_nhwc.resize(result.expected_nchw.size());
    result.output_nchw.resize(result.expected_nchw.size());
    y26::stage47::nchw_u8_to_nhwc_s8(result.input_nchw.data(), result.input_nhwc.data(),
                                      result.input_spec.h, result.input_spec.w, result.input_spec.c);
    ConvSpec spec;
    spec.input = result.input_spec;
    spec.output_h = result.output_spec.h;
    spec.output_w = result.output_spec.w;
    spec.output_c = result.output_spec.c;
    spec.kernel_h = integer(row, "kernel_h");
    spec.kernel_w = integer(row, "kernel_w");
    spec.stride_h = integer(row, "stride_h");
    spec.stride_w = integer(row, "stride_w");
    spec.pad_h = integer(row, "pad_h");
    spec.pad_w = integer(row, "pad_w");
    spec.group = integer(row, "group");
    spec.conv_output_scale = floating(row, "conv_output_scale");
    spec.conv_output_zero_point_u8 = integer(row, "conv_output_zero_point");
    spec.weights_ohwi_s8 = result.weights.data();
    spec.weight_count = result.weights.size();
    spec.weight_scales = result.scales.data();
    spec.weight_scale_count = result.scales.size();
    spec.bias_i32 = result.bias.data();
    spec.bias_count = result.bias.size();
    spec.segments.push_back(OutputSegmentSpec{0, result.output_spec.c, result.output_spec, field(row, "activation") == "silu"});
    const int status = result.conv.prepare(spec);
    if (status != Y26_CONV_STATUS_SUCCESS) throw std::runtime_error("case prepare failed: " + result.name);
    result.macs = unsigned_integer(row, "macs");
    return result;
}

std::pair<std::size_t, int> compare_output(PreparedCase& test_case) {
    y26::stage47::nhwc_s8_to_nchw_u8(test_case.output_nhwc.data(), test_case.output_nchw.data(),
                                      test_case.output_spec.h, test_case.output_spec.w, test_case.output_spec.c);
    std::size_t mismatches = 0;
    int max_abs = 0;
    for (std::size_t index = 0; index < test_case.expected_nchw.size(); ++index) {
        const int difference = std::abs(static_cast<int>(test_case.output_nchw[index]) - test_case.expected_nchw[index]);
        mismatches += difference != 0;
        max_abs = std::max(max_abs, difference);
    }
    return {mismatches, max_abs};
}

unsigned read_frm() noexcept {
#if defined(__riscv)
    unsigned value = 0;
    asm volatile("csrr %0, frm" : "=r"(value));
    return value;
#else
    switch (std::fegetround()) {
        case FE_TOWARDZERO: return 1;
        case FE_DOWNWARD: return 2;
        case FE_UPWARD: return 3;
        default: return 0;
    }
#endif
}

bool write_frm(unsigned value) noexcept {
#if defined(__riscv)
    asm volatile("csrw frm, %0" : : "r"(value));
    return read_frm() == value;
#else
    constexpr int host_modes[] = {FE_TONEAREST, FE_TOWARDZERO, FE_DOWNWARD, FE_UPWARD};
    return value < std::size(host_modes) && std::fesetround(host_modes[value]) == 0;
#endif
}

void run_frm_sweep(const Options& options) {
    const auto rows = read_tsv(options.generated_root / "integrated_kernel_cases/cases.tsv");
    const auto row = std::find_if(rows.begin(), rows.end(), [&](const auto& candidate) {
        return integer(candidate, "case_id") == options.case_id;
    });
    if (row == rows.end()) throw std::runtime_error("FRM sweep case not found");
    PreparedCase test_case = load_case(options, *row);
    const std::array<std::int8_t*, 2> outputs {test_case.output_nhwc.data(), nullptr};
    const RunOptions run_options{options.kernel, options.partition, options.workers, -1, false};
    constexpr const char* names[] = {"RNE", "RTZ", "RDN", "RUP", "RMM"};
    const unsigned original = read_frm();
    for (unsigned mode = 0; mode < std::size(names); ++mode) {
        if (!write_frm(mode)) {
            std::cout << "frm_sweep\t" << names[mode] << "\tunavailable\t" << read_frm() << '\n';
            continue;
        }
        WorkerPool pool(options.workers);
        const int status = test_case.conv.run(pool, test_case.input_nhwc.data(), outputs, 1, run_options, nullptr);
        const auto comparison = compare_output(test_case);
        std::cout << "frm_sweep\t" << names[mode] << '\t' << status << '\t' << read_frm() << '\t'
                  << comparison.first << '\t' << comparison.second << '\t'
                  << hash_bytes(test_case.output_nchw.data(), test_case.output_nchw.size()) << '\t'
                  << pool.affinity_ok() << '\n';
    }
    const bool restored = write_frm(original);
    std::cout << "frm_restore\t" << original << '\t' << read_frm() << '\t' << restored << '\n';
}

void run_kernel_mode(const Options& options, bool benchmark) {
    const auto rows = read_tsv(options.generated_root / "integrated_kernel_cases/cases.tsv");
    WorkerPool pool(options.workers);
    for (const auto& row : rows) {
        if (options.case_id >= 0 && integer(row, "case_id") != options.case_id) continue;
        PreparedCase test_case = load_case(options, row);
        RunOptions run_options{options.kernel, options.partition, options.workers, -1, options.profile_phases};
        const std::array<std::int8_t*, 2> outputs {test_case.output_nhwc.data(), nullptr};
        IntegratedTiming timing;
        const int status = test_case.conv.run(pool, test_case.input_nhwc.data(), outputs, 1, run_options, &timing);
        if (status != Y26_CONV_STATUS_SUCCESS) {
            std::cout << "kernel_status\t" << test_case.id << '\t' << test_case.name << '\t' << status << '\n';
            continue;
        }
        const auto comparison = compare_output(test_case);
        std::cout << "kernel_correctness\t" << test_case.id << '\t' << test_case.name << '\t'
                  << y26::stage47::kernel_shape_name(options.kernel) << '\t' << options.workers << '\t'
                  << comparison.first << '\t' << comparison.second << '\t'
                  << hash_bytes(test_case.output_nchw.data(), test_case.output_nchw.size()) << '\t'
                  << timing.affinity_ok << '\n';
        if (comparison.first != 0 || !benchmark) continue;
        for (int index = 0; index < options.warmup; ++index) {
            if (test_case.conv.run(pool, test_case.input_nhwc.data(), outputs, 1, run_options, nullptr) != Y26_CONV_STATUS_SUCCESS)
                throw std::runtime_error("kernel warmup failed");
        }
        std::vector<double> repeats;
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            const auto begin = Clock::now();
            for (int run = 0; run < options.runs; ++run) {
                if (test_case.conv.run(pool, test_case.input_nhwc.data(), outputs, 1, run_options, nullptr) != Y26_CONV_STATUS_SUCCESS)
                    throw std::runtime_error("kernel benchmark failed");
            }
            const double mean_us = std::chrono::duration<double, std::micro>(Clock::now() - begin).count() / options.runs;
            repeats.push_back(mean_us);
            std::cout << "kernel_raw\t" << test_case.id << '\t' << repeat << '\t' << std::fixed << std::setprecision(6)
                      << mean_us << '\n';
        }
        print_stats("kernel_summary", std::to_string(test_case.id) + ":" + test_case.name, options,
                    summarize(repeats), test_case.macs,
                    hash_bytes(test_case.output_nchw.data(), test_case.output_nchw.size()));
        if (options.profile_phases) {
            IntegratedTiming profile;
            test_case.conv.run(pool, test_case.input_nhwc.data(), outputs, 1, run_options, &profile);
            std::cout << "kernel_profile\t" << test_case.id << '\t' << profile.gather_pack_us << '\t'
                      << profile.vmadot_us << '\t' << profile.fused_epilogue_us << '\t' << profile.barrier_us << '\t'
                      << profile.total_us << '\t' << profile.min_worker_us << '\t' << profile.max_worker_us << '\n';
        }
    }
}

struct OracleRow {
    std::string fixture;
    int tensor = -1;
    std::filesystem::path path;
};

std::vector<OracleRow> oracle_rows(const Options& options) {
    std::vector<OracleRow> result;
    for (const auto& row : read_tsv(options.package / "oracle_manifest.tsv")) {
        result.push_back({field(row, "fixture_id"), integer(row, "tensor_id"), options.package / field(row, "path")});
    }
    return result;
}

std::unordered_map<int, int> tensor_producers(const Options& options) {
    std::unordered_map<int, int> result;
    for (const auto& row : read_tsv(options.package / "tensors.tsv")) {
        result.emplace(integer(row, "id"), integer(row, "first_op"));
    }
    return result;
}

void run_aot_validate(const Options& options) {
    AotExecutor executor;
    if (executor.prepare(options.package, options.workers) != Y26_CONV_STATUS_SUCCESS)
        throw std::runtime_error("AOT prepare failed: " + executor.last_error());
    const auto manifest = oracle_rows(options);
    const auto producers = tensor_producers(options);
    for (int fixture_index = 0; fixture_index < 8; ++fixture_index) {
        const std::string fixture = "F" + std::to_string(fixture_index);
        if (!options.fixture.empty() && options.fixture != fixture) continue;
        const auto input_row = std::find_if(manifest.begin(), manifest.end(), [&](const OracleRow& row) {
            return row.fixture == fixture && row.tensor == 0;
        });
        if (input_row == manifest.end()) throw std::runtime_error("missing AOT input oracle");
        const TensorSpec* input_spec = executor.tensor_spec(0);
        auto input_nchw = read_binary<std::uint8_t>(input_row->path, executor.tensor_bytes(0));
        std::vector<std::int8_t> input_nhwc(input_nchw.size());
        y26::stage47::nchw_u8_to_nhwc_s8(input_nchw.data(), input_nhwc.data(), input_spec->h, input_spec->w, input_spec->c);
        if (executor.set_input(input_nhwc.data(), input_nhwc.size()) != Y26_CONV_STATUS_SUCCESS) throw std::runtime_error("set input failed");
        for (const OracleRow& expected : manifest) {
            if (expected.fixture != fixture || expected.tensor == 0) continue;
            const int tensor_id = expected.tensor;
            const auto producer = producers.find(tensor_id);
            if (producer == producers.end()) throw std::runtime_error("missing tensor producer");
            RunOptions run_options{options.kernel, options.partition, options.workers, producer->second, false};
            const int status = executor.run(nullptr, nullptr, run_options, nullptr);
            const TensorSpec* tensor_spec = executor.tensor_spec(tensor_id);
            std::vector<std::int8_t> actual_nhwc(executor.tensor_bytes(tensor_id));
            std::vector<std::uint8_t> actual_nchw(actual_nhwc.size());
            auto expected_nchw = read_binary<std::uint8_t>(expected.path, actual_nhwc.size());
            if (status == Y26_CONV_STATUS_SUCCESS)
                executor.copy_tensor(tensor_id, actual_nhwc.data(), actual_nhwc.size());
            y26::stage47::nhwc_s8_to_nchw_u8(actual_nhwc.data(), actual_nchw.data(), tensor_spec->h, tensor_spec->w, tensor_spec->c);
            std::size_t mismatches = 0;
            int max_abs = 0;
            std::size_t first_mismatch = actual_nchw.size();
            int first_actual = -1;
            int first_expected = -1;
            for (std::size_t index = 0; index < actual_nchw.size(); ++index) {
                const int difference = std::abs(static_cast<int>(actual_nchw[index]) - expected_nchw[index]);
                if (difference != 0) {
                    if (first_mismatch == actual_nchw.size()) {
                        first_mismatch = index;
                        first_actual = actual_nchw[index];
                        first_expected = expected_nchw[index];
                    }
                    ++mismatches;
                }
                max_abs = std::max(max_abs, difference);
            }
            std::cout << "aot_correctness\t" << fixture << '\t' << tensor_id << '\t' << status << '\t'
                      << mismatches << '\t' << max_abs << '\t'
                      << hash_bytes(actual_nchw.data(), actual_nchw.size()) << '\t' << executor.worker_affinity_ok() << '\t'
                      << first_mismatch << '\t' << first_actual << '\t' << first_expected << '\n';
        }
    }
}

void run_aot_benchmark(const Options& options) {
    AotExecutor executor;
    if (executor.prepare(options.package, options.workers) != Y26_CONV_STATUS_SUCCESS)
        throw std::runtime_error("AOT prepare failed: " + executor.last_error());
    const auto manifest = oracle_rows(options);
    const auto input_row = std::find_if(manifest.begin(), manifest.end(), [](const OracleRow& row) {
        return row.fixture == "F0" && row.tensor == 0;
    });
    if (input_row == manifest.end()) throw std::runtime_error("missing F0 input");
    auto input_nchw = read_binary<std::uint8_t>(input_row->path, executor.tensor_bytes(0));
    std::vector<std::int8_t> input_nhwc(input_nchw.size());
    const TensorSpec* input_spec = executor.tensor_spec(0);
    const auto entry_begin = Clock::now();
    y26::stage47::nchw_u8_to_nhwc_s8(input_nchw.data(), input_nhwc.data(), input_spec->h, input_spec->w, input_spec->c);
    const double entry_us = std::chrono::duration<double, std::micro>(Clock::now() - entry_begin).count();
    executor.set_input(input_nhwc.data(), input_nhwc.size());
    RunOptions run_options{options.kernel, options.partition, options.workers, -1, false};
    for (int index = 0; index < options.warmup; ++index) {
        if (executor.run(nullptr, nullptr, run_options, nullptr) != Y26_CONV_STATUS_SUCCESS) throw std::runtime_error("AOT warmup failed");
    }
    std::vector<double> repeats;
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        double elapsed = 0.0;
        for (int run = 0; run < options.runs; ++run) {
            const auto begin = Clock::now();
            if (executor.run(nullptr, nullptr, run_options, nullptr) != Y26_CONV_STATUS_SUCCESS) throw std::runtime_error("AOT run failed");
            elapsed += std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
        }
        repeats.push_back(elapsed / options.runs);
        std::cout << "aot_raw\t" << repeat << '\t' << std::fixed << std::setprecision(6) << repeats.back() << '\n';
    }
    std::vector<std::int8_t> output_nhwc(executor.tensor_bytes(executor.output_tensor_id()));
    std::vector<std::uint8_t> output_nchw(output_nhwc.size());
    executor.copy_tensor(executor.output_tensor_id(), output_nhwc.data(), output_nhwc.size());
    const TensorSpec* output_spec = executor.tensor_spec(executor.output_tensor_id());
    const auto exit_begin = Clock::now();
    y26::stage47::nhwc_s8_to_nchw_u8(output_nhwc.data(), output_nchw.data(), output_spec->h, output_spec->w, output_spec->c);
    const double exit_us = std::chrono::duration<double, std::micro>(Clock::now() - exit_begin).count();
    print_stats("aot_summary", "model4preact_to_model8", options, summarize(repeats), 0,
                hash_bytes(output_nchw.data(), output_nchw.size()));
    std::cout << "aot_contract\t" << executor.operation_count() << '\t' << executor.arena_bytes() << '\t'
              << executor.packed_weight_bytes() << '\t' << entry_us << '\t' << exit_us << '\t'
              << executor.worker_affinity_ok() << '\n';
    RunOptions profile_options = run_options;
    profile_options.profile_phases = true;
    ExecutorTiming profile;
    if (executor.run(nullptr, nullptr, profile_options, &profile) == Y26_CONV_STATUS_SUCCESS) {
        for (const auto& operation : profile.operations) {
            std::cout << "aot_profile\t" << operation.operation_index << '\t' << operation.kind << '\t'
                      << operation.name << '\t' << operation.total_us << '\t' << operation.gather_pack_us << '\t'
                      << operation.vmadot_us << '\t' << operation.fused_epilogue_us << '\n';
        }
        std::cout << "aot_profile_total\t" << profile.total_us << '\t' << profile.conv_us << '\t'
                  << profile.lut_us << '\t' << profile.add_us << '\t' << profile.concat_us << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::cout << "stage47_executor_v1\n";
        std::cout << "timing_source=steady_clock\nrdcycle_used=0\n";
        if (options.mode == "kernel-validate") run_kernel_mode(options, false);
        else if (options.mode == "kernel-benchmark") run_kernel_mode(options, true);
        else if (options.mode == "frm-sweep") run_frm_sweep(options);
        else if (options.mode == "aot-validate") run_aot_validate(options);
        else if (options.mode == "aot-benchmark") run_aot_benchmark(options);
        else throw std::runtime_error("invalid mode: " + options.mode);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error=" << error.what() << '\n';
        return 2;
    }
}
