#include "y26_k1x_stage48_nchwc8.h"

#include "y26_k1x_conv_kernels.h"
#include "y26_k1x_int8_v1.h"

#include <algorithm>
#include <array>
#include <cfenv>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using y26::stage48::ComputeRoute;
using y26::stage48::LoadStrategy;
using y26::stage48::MBlock;
using y26::stage48::Model5DirectConv;
using y26::stage48::PartitionPolicy;
using y26::stage48::RunOptions;
using y26::stage48::Timing;

struct Options {
    std::string mode;
    std::filesystem::path package;
    std::filesystem::path dump_output;
    std::string fixture = "F0";
    ComputeRoute route = ComputeRoute::scalar;
    MBlock m_block = MBlock::m12;
    LoadStrategy load = LoadStrategy::c8_u64;
    PartitionPolicy partition = PartitionPolicy::spatial;
    int workers = 1;
    int warmup = 10;
    int runs = 100;
    int repeats = 5;
};

struct Stats {
    double mean = 0.0;
    double stddev = 0.0;
    double cv_pct = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    double median = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
};

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

double process_cpu_us() {
    timespec value {};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) != 0) {
        return -1.0;
    }
    return static_cast<double>(value.tv_sec) * 1.0e6 + static_cast<double>(value.tv_nsec) / 1.0e3;
}

double percentile(const std::vector<double>& sorted, double quantile) {
    if (sorted.empty()) {
        return 0.0;
    }
    const double position = quantile * static_cast<double>(sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
}

Stats summarize(std::vector<double> values) {
    Stats result;
    if (values.empty()) {
        return result;
    }
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    result.minimum = *std::min_element(values.begin(), values.end());
    result.maximum = *std::max_element(values.begin(), values.end());
    if (values.size() > 1) {
        double variance = 0.0;
        for (double value : values) {
            variance += (value - result.mean) * (value - result.mean);
        }
        result.stddev = std::sqrt(variance / static_cast<double>(values.size() - 1));
    }
    result.cv_pct = result.mean == 0.0 ? 0.0 : result.stddev / result.mean * 100.0;
    std::sort(values.begin(), values.end());
    result.median = percentile(values, 0.5);
    result.p90 = percentile(values, 0.9);
    result.p95 = percentile(values, 0.95);
    return result;
}

std::uint64_t fnv1a(const void* data, std::size_t bytes) {
    const auto* values = static_cast<const std::uint8_t*>(data);
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t index = 0; index < bytes; ++index) {
        hash ^= values[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

template <typename T>
std::vector<T> read_binary(const std::filesystem::path& path, std::size_t expected_count = 0) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open binary: " + path.string());
    }
    const std::streamsize bytes = stream.tellg();
    if (bytes < 0 || static_cast<std::size_t>(bytes) % sizeof(T) != 0) {
        throw std::runtime_error("invalid binary size: " + path.string());
    }
    const std::size_t count = static_cast<std::size_t>(bytes) / sizeof(T);
    if (expected_count != 0 && count != expected_count) {
        throw std::runtime_error("binary element count mismatch: " + path.string());
    }
    stream.seekg(0);
    std::vector<T> values(count);
    if (bytes != 0 && !stream.read(reinterpret_cast<char*>(values.data()), bytes)) {
        throw std::runtime_error("binary read failed: " + path.string());
    }
    return values;
}

void write_binary(const std::filesystem::path& path, const void* data, std::size_t bytes) {
    if (path.empty()) {
        return;
    }
    std::ofstream stream(path, std::ios::binary);
    if (!stream || (bytes != 0 && !stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes)))) {
        throw std::runtime_error("cannot write output: " + path.string());
    }
}

std::vector<std::string> split_tsv(const std::string& line) {
    std::vector<std::string> values;
    std::size_t begin = 0;
    for (;;) {
        const std::size_t end = line.find('\t', begin);
        values.push_back(line.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return values;
}

std::vector<std::unordered_map<std::string, std::string>> read_tsv(const std::filesystem::path& path) {
    std::ifstream stream(path);
    std::string line;
    if (!stream || !std::getline(stream, line)) {
        throw std::runtime_error("cannot read TSV: " + path.string());
    }
    const auto header = split_tsv(line);
    std::vector<std::unordered_map<std::string, std::string>> rows;
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        const auto values = split_tsv(line);
        if (values.size() != header.size()) {
            throw std::runtime_error("malformed TSV: " + path.string());
        }
        auto& row = rows.emplace_back();
        for (std::size_t index = 0; index < header.size(); ++index) {
            row.emplace(header[index], values[index]);
        }
    }
    return rows;
}

const std::string& field(const std::unordered_map<std::string, std::string>& row, const char* name) {
    const auto found = row.find(name);
    if (found == row.end()) {
        throw std::runtime_error(std::string("missing TSV field: ") + name);
    }
    return found->second;
}

std::int64_t parse_i64(std::string_view value) {
    std::int64_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size()) {
        throw std::runtime_error("invalid integer");
    }
    return result;
}

ComputeRoute parse_route(const std::string& value) {
    if (value == "scalar") return ComputeRoute::scalar;
    if (value == "ime") return ComputeRoute::ime;
    throw std::runtime_error("invalid route: " + value);
}

MBlock parse_m_block(const std::string& value) {
    if (value == "m4n16") return MBlock::m4;
    if (value == "m8n16") return MBlock::m8;
    if (value == "m12n16") return MBlock::m12;
    throw std::runtime_error("invalid M block: " + value);
}

LoadStrategy parse_load(const std::string& value) {
    if (value == "u64") return LoadStrategy::c8_u64;
    if (value == "vlse64") return LoadStrategy::rvv_vlse64;
    if (value == "vlseg2e64") return LoadStrategy::rvv_vlseg2e64;
    throw std::runtime_error("invalid load strategy: " + value);
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
            if (++index >= argc) {
                throw std::runtime_error("missing value for " + argument);
            }
            return argv[index];
        };
        if (argument == "--mode") options.mode = next();
        else if (argument == "--package") options.package = next();
        else if (argument == "--fixture") options.fixture = next();
        else if (argument == "--route") options.route = parse_route(next());
        else if (argument == "--kernel") options.m_block = parse_m_block(next());
        else if (argument == "--load") options.load = parse_load(next());
        else if (argument == "--partition") options.partition = parse_partition(next());
        else if (argument == "--workers") options.workers = std::stoi(next());
        else if (argument == "--warmup") options.warmup = std::stoi(next());
        else if (argument == "--runs") options.runs = std::stoi(next());
        else if (argument == "--repeats") options.repeats = std::stoi(next());
        else if (argument == "--dump-output") options.dump_output = next();
        else if (argument == "--help") {
            std::cout << "usage: bench_stage48_nchwc8 --mode validate|benchmark|profile|conversion-benchmark|byte-order|oracle-vectors|frm-sweep"
                         " --package PATH [--fixture F0] [--route scalar|ime]"
                         " [--kernel m4n16|m8n16|m12n16] [--load u64|vlse64|vlseg2e64]"
                         " [--partition spatial|output_channel] [--workers 1..4]"
                         " [--warmup 10 --runs 100 --repeats 5] [--dump-output PATH]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (options.mode.empty() || options.package.empty() || options.workers < 1 || options.workers > 4 ||
        options.warmup < 0 || options.runs < 1 || options.repeats < 1) {
        throw std::runtime_error("invalid or missing options");
    }
    return options;
}

RunOptions run_options(const Options& options, bool profile) {
    RunOptions result;
    result.route = options.route;
    result.m_block = options.m_block;
    result.load_strategy = options.load;
    result.partition = options.partition;
    result.workers = options.workers;
    result.profile_phases = profile;
    return result;
}

struct Fixture {
    std::vector<std::int8_t> input;
    std::vector<std::int8_t> expected;
    std::vector<std::int8_t> output;
};

Fixture load_fixture(const Options& options, const Model5DirectConv& conv) {
    const auto root = options.package / "fixtures" / options.fixture;
    Fixture fixture;
    fixture.input = read_binary<std::int8_t>(root / "input_nchwc8_s8.bin", conv.input_bytes());
    fixture.expected = read_binary<std::int8_t>(root / "expected_nchwc8_s8.bin", conv.output_bytes());
    fixture.output.resize(conv.output_bytes());
    return fixture;
}

int compare(const std::vector<std::int8_t>& actual,
            const std::vector<std::int8_t>& expected,
            std::size_t* first_mismatch,
            int* maximum_difference) {
    int mismatches = 0;
    *first_mismatch = actual.size();
    *maximum_difference = 0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const int difference = std::abs(static_cast<int>(actual[index]) - static_cast<int>(expected[index]));
        if (difference != 0) {
            if (mismatches == 0) {
                *first_mismatch = index;
            }
            ++mismatches;
            *maximum_difference = std::max(*maximum_difference, difference);
        }
    }
    return mismatches;
}

int run_validate(const Options& options) {
    Model5DirectConv conv;
    const int prepare = conv.prepare(options.package, options.workers);
    if (prepare != Y26_CONV_STATUS_SUCCESS) {
        throw std::runtime_error("prepare failed: " + conv.last_error());
    }
    auto fixture = load_fixture(options, conv);
    Timing timing;
    const int status = conv.run(fixture.input.data(), fixture.output.data(), run_options(options, true), &timing);
    std::size_t first = 0;
    int maximum = 0;
    const int mismatches = compare(fixture.output, fixture.expected, &first, &maximum);
    write_binary(options.dump_output, fixture.output.data(), fixture.output.size());
    std::cout << "mode\tfixture\troute\tkernel\tload\tpartition\tworkers\tstatus\tmismatches\tmax_abs_diff"
                 "\tfirst_mismatch\tactual_hash64\texpected_hash64\taffinity_ok\ttotal_us\tdirect_a_us\tvmadot_us"
                 "\tepilogue_us\tvector_groups\tscalar_c8_groups\tborder_chunks\n";
    std::cout << "validate\t" << options.fixture << '\t' << y26::stage48::compute_route_name(options.route) << '\t'
              << y26::stage48::m_block_name(options.m_block) << '\t'
              << y26::stage48::load_strategy_name(options.load) << '\t'
              << y26::stage48::partition_policy_name(options.partition) << '\t' << options.workers << '\t'
              << status << '\t' << mismatches << '\t' << maximum << '\t' << first << '\t'
              << fnv1a(fixture.output.data(), fixture.output.size()) << '\t'
              << fnv1a(fixture.expected.data(), fixture.expected.size()) << '\t' << timing.affinity_ok << '\t'
              << std::fixed << std::setprecision(6) << timing.total_us << '\t' << timing.direct_a_delivery_us << '\t'
              << timing.vmadot_us << '\t' << timing.scalar_epilogue_us << '\t' << timing.vector_groups << '\t'
              << timing.scalar_c8_groups << '\t' << timing.border_chunks << '\n';
    return status == Y26_CONV_STATUS_SUCCESS && mismatches == 0 ? 0 : 2;
}

int run_benchmark(const Options& options) {
    Model5DirectConv conv;
    if (conv.prepare(options.package, options.workers) != Y26_CONV_STATUS_SUCCESS) {
        throw std::runtime_error("prepare failed: " + conv.last_error());
    }
    auto fixture = load_fixture(options, conv);
    const RunOptions selected = run_options(options, false);
    for (int index = 0; index < options.warmup; ++index) {
        if (conv.run(fixture.input.data(), fixture.output.data(), selected, nullptr) != Y26_CONV_STATUS_SUCCESS) {
            throw std::runtime_error("warmup failed");
        }
    }
    std::vector<double> all_wall;
    std::vector<double> all_cpu;
    std::cout << "kind\trepeat\trun\twall_us\tprocess_cpu_us\troute\tkernel\tload\tpartition\tworkers\toutput_hash64\n";
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        std::vector<double> repeat_wall;
        std::vector<double> repeat_cpu;
        for (int run = 0; run < options.runs; ++run) {
            const double cpu_begin = process_cpu_us();
            const auto begin = Clock::now();
            const int status = conv.run(fixture.input.data(), fixture.output.data(), selected, nullptr);
            const auto end = Clock::now();
            const double cpu_end = process_cpu_us();
            if (status != Y26_CONV_STATUS_SUCCESS) {
                throw std::runtime_error("benchmark run failed");
            }
            const double wall = elapsed_us(begin, end);
            const double cpu = cpu_begin < 0.0 || cpu_end < 0.0 ? -1.0 : cpu_end - cpu_begin;
            repeat_wall.push_back(wall);
            repeat_cpu.push_back(cpu);
            all_wall.push_back(wall);
            all_cpu.push_back(cpu);
            std::cout << "sample\t" << repeat << '\t' << run << '\t' << std::fixed << std::setprecision(6)
                      << wall << '\t' << cpu << '\t' << y26::stage48::compute_route_name(options.route) << '\t'
                      << y26::stage48::m_block_name(options.m_block) << '\t'
                      << y26::stage48::load_strategy_name(options.load) << '\t'
                      << y26::stage48::partition_policy_name(options.partition) << '\t' << options.workers << '\t'
                      << fnv1a(fixture.output.data(), fixture.output.size()) << '\n';
        }
        const Stats wall = summarize(repeat_wall);
        const Stats cpu = summarize(repeat_cpu);
        std::cout << "repeat_summary\t" << repeat << "\t-1\t" << wall.mean << '\t' << cpu.mean << '\t'
                  << y26::stage48::compute_route_name(options.route) << '\t'
                  << y26::stage48::m_block_name(options.m_block) << '\t'
                  << y26::stage48::load_strategy_name(options.load) << '\t'
                  << y26::stage48::partition_policy_name(options.partition) << '\t' << options.workers << '\t'
                  << fnv1a(fixture.output.data(), fixture.output.size()) << '\n';
    }
    std::size_t first = 0;
    int maximum = 0;
    const int mismatches = compare(fixture.output, fixture.expected, &first, &maximum);
    const Stats wall = summarize(all_wall);
    const Stats cpu = summarize(all_cpu);
    const double gmacs = static_cast<double>(conv.macs()) / wall.mean / 1000.0;
    std::cout << "summary\trepeat\truns\trepeats\tmean_us\tstddev_us\tcv_pct\tmin_us\tmax_us\tmedian_us"
                 "\tp90_us\tp95_us\tprocess_cpu_mean_us\tgmacs\tmismatches\tmax_abs_diff\tfirst_mismatch"
                 "\taffinity_ok\tinput_bytes\toutput_bytes\tpacked_weight_bytes\tworker_workspace_bytes\n";
    std::cout << "summary\t-1\t" << options.runs << '\t' << options.repeats << '\t' << wall.mean << '\t'
              << wall.stddev << '\t' << wall.cv_pct << '\t' << wall.minimum << '\t' << wall.maximum << '\t'
              << wall.median << '\t' << wall.p90 << '\t' << wall.p95 << '\t' << cpu.mean << '\t' << gmacs << '\t'
              << mismatches << '\t' << maximum << '\t' << first << '\t' << (conv.affinity_ok() ? 1 : 0) << '\t'
              << conv.input_bytes() << '\t' << conv.output_bytes() << '\t' << conv.packed_weight_bytes() << '\t'
              << conv.per_worker_workspace_bytes() << '\n';
    write_binary(options.dump_output, fixture.output.data(), fixture.output.size());
    return mismatches == 0 ? 0 : 2;
}

int run_profile(const Options& options) {
    Model5DirectConv conv;
    if (conv.prepare(options.package, options.workers) != Y26_CONV_STATUS_SUCCESS) {
        throw std::runtime_error("prepare failed: " + conv.last_error());
    }
    auto fixture = load_fixture(options, conv);
    Timing timing;
    const int status = conv.run(fixture.input.data(), fixture.output.data(), run_options(options, true), &timing);
    std::size_t first = 0;
    int maximum = 0;
    const int mismatches = compare(fixture.output, fixture.expected, &first, &maximum);
    std::cout << "route\tkernel\tload\tpartition\tworkers\tstatus\tmismatches\ttotal_us\tdirect_a_worker_sum_us"
                 "\tvmadot_worker_sum_us\tepilogue_worker_sum_us\tbarrier_us\tmin_worker_us\tmax_worker_us"
                 "\tvector_groups\tscalar_c8_groups\tborder_chunks\taffinity_ok\n";
    std::cout << y26::stage48::compute_route_name(options.route) << '\t'
              << y26::stage48::m_block_name(options.m_block) << '\t'
              << y26::stage48::load_strategy_name(options.load) << '\t'
              << y26::stage48::partition_policy_name(options.partition) << '\t' << options.workers << '\t'
              << status << '\t' << mismatches << '\t' << std::fixed << std::setprecision(6) << timing.total_us << '\t'
              << timing.direct_a_delivery_us << '\t' << timing.vmadot_us << '\t' << timing.scalar_epilogue_us << '\t'
              << timing.barrier_us << '\t' << timing.min_worker_us << '\t' << timing.max_worker_us << '\t'
              << timing.vector_groups << '\t' << timing.scalar_c8_groups << '\t' << timing.border_chunks << '\t'
              << timing.affinity_ok << '\n';
    return status == Y26_CONV_STATUS_SUCCESS && mismatches == 0 ? 0 : 2;
}

int run_conversion_benchmark(const Options& options) {
    constexpr std::size_t input_count = 1U * 128U * 80U * 80U;
    constexpr std::size_t output_count = 1U * 128U * 40U * 40U;
    const auto root = options.package / "fixtures" / options.fixture;
    const auto input_nchw = read_binary<std::uint8_t>(root / "input_nchw_u8.bin", input_count);
    const auto input_expected = read_binary<std::int8_t>(root / "input_nchwc8_s8.bin", input_count);
    const auto output_blocked = read_binary<std::int8_t>(root / "expected_nchwc8_s8.bin", output_count);
    const auto output_expected = read_binary<std::uint8_t>(root / "expected_nchw_u8.bin", output_count);
    std::vector<std::int8_t> input_actual(input_count);
    std::vector<std::uint8_t> output_actual(output_count);

    auto run_entry = [&]() {
        y26::stage48::nchw_u8_to_nchwc8_s8(input_nchw.data(), input_actual.data(), 1, 128, 80, 80);
    };
    auto run_exit = [&]() {
        y26::stage48::nchwc8_s8_to_nchw_u8(output_blocked.data(), output_actual.data(), 1, 128, 40, 40);
    };
    for (int index = 0; index < options.warmup; ++index) {
        run_entry();
        run_exit();
    }

    std::vector<double> entry_values;
    std::vector<double> exit_values;
    std::cout << "kind\trepeat\trun\tentry_us\texit_us\n";
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        for (int run = 0; run < options.runs; ++run) {
            const auto entry_begin = Clock::now();
            run_entry();
            const auto entry_end = Clock::now();
            run_exit();
            const auto exit_end = Clock::now();
            const double entry_us = elapsed_us(entry_begin, entry_end);
            const double exit_us = elapsed_us(entry_end, exit_end);
            entry_values.push_back(entry_us);
            exit_values.push_back(exit_us);
            std::cout << "sample\t" << repeat << '\t' << run << '\t' << std::fixed << std::setprecision(6)
                      << entry_us << '\t' << exit_us << '\n';
        }
    }
    const Stats entry = summarize(entry_values);
    const Stats exit = summarize(exit_values);
    const int entry_mismatches = static_cast<int>(
        std::inner_product(input_actual.begin(), input_actual.end(), input_expected.begin(), std::size_t{0},
                           std::plus<>(), std::not_equal_to<>()));
    const int exit_mismatches = static_cast<int>(
        std::inner_product(output_actual.begin(), output_actual.end(), output_expected.begin(), std::size_t{0},
                           std::plus<>(), std::not_equal_to<>()));
    std::cout << "surface\tmean_us\tstddev_us\tcv_pct\tmin_us\tmax_us\tmedian_us\tp90_us\tp95_us"
                 "\tbytes_read\tbytes_written\teffective_read_write_gbps\tmismatches\thash64\n";
    const auto emit = [](const char* name, const Stats& stats, std::size_t bytes, int mismatches, std::uint64_t hash) {
        const double gbps = stats.mean == 0.0 ? 0.0 : (2.0 * static_cast<double>(bytes)) / stats.mean / 1000.0;
        std::cout << name << '\t' << stats.mean << '\t' << stats.stddev << '\t' << stats.cv_pct << '\t'
                  << stats.minimum << '\t' << stats.maximum << '\t' << stats.median << '\t' << stats.p90 << '\t'
                  << stats.p95 << '\t' << bytes << '\t' << bytes << '\t' << gbps << '\t' << mismatches << '\t'
                  << hash << '\n';
    };
    emit("nchw_u8_to_nchwc8_s8", entry, input_count, entry_mismatches,
         fnv1a(input_actual.data(), input_actual.size()));
    emit("nchwc8_s8_to_nchw_u8", exit, output_count, exit_mismatches,
         fnv1a(output_actual.data(), output_actual.size()));
    return entry_mismatches == 0 && exit_mismatches == 0 ? 0 : 2;
}

void reference_pack(const std::vector<std::int8_t>& input, int m_begin, int rows, std::int8_t* panel) {
    constexpr int input_zero_point = 9;
    constexpr int input_h = 80;
    constexpr int input_w = 80;
    constexpr int output_h = 40;
    constexpr int output_w = 40;
    constexpr int input_c = 128;
    constexpr int channel_blocks = input_c / 8;
    const std::int8_t padding = y26::int8_v1::signed_storage(input_zero_point);
    for (int tile = 0; tile < 3 * 3 * channel_blocks; ++tile) {
        const int block = tile % channel_blocks;
        const int position = tile / channel_blocks;
        const int kernel_y = position / 3;
        const int kernel_x = position % 3;
        for (int row = 0; row < rows; ++row) {
            const int flat = m_begin + row;
            const int output_y = flat / output_w;
            const int output_x = flat % output_w;
            const int input_y = output_y * 2 + kernel_y - 1;
            const int input_x = output_x * 2 + kernel_x - 1;
            for (int lane = 0; lane < 8; ++lane) {
                std::int8_t value = padding;
                if (flat < output_h * output_w && input_y >= 0 && input_y < input_h && input_x >= 0 && input_x < input_w) {
                    const std::size_t offset =
                        (((static_cast<std::size_t>(block) * input_h + input_y) * input_w + input_x) * 8U) + lane;
                    value = input[offset];
                }
                panel[(static_cast<std::size_t>(tile) * rows + row) * 8U + lane] = value;
            }
        }
    }
}

int run_byte_order(const Options& options) {
    Model5DirectConv conv;
    if (conv.prepare(options.package, 1) != Y26_CONV_STATUS_SUCCESS) {
        throw std::runtime_error("prepare failed: " + conv.last_error());
    }
    auto fixture = load_fixture(options, conv);
    constexpr std::array<int, 3> positions{0, 4, 40};
    constexpr std::array<LoadStrategy, 3> strategies{
        LoadStrategy::c8_u64, LoadStrategy::rvv_vlse64, LoadStrategy::rvv_vlseg2e64};
    int total_mismatches = 0;
    std::cout << "m_begin\tkernel\tstrategy\tmismatches\tpanel_hash64\tfirst_32_center_tile_hex\n";
    for (int m_begin : positions) {
        for (LoadStrategy strategy : strategies) {
            const int rows = static_cast<int>(options.m_block);
            std::vector<std::int8_t> actual(static_cast<std::size_t>(rows) * 3 * 3 * 128);
            std::vector<std::int8_t> expected(actual.size());
            if (conv.debug_pack_a(fixture.input.data(), m_begin, options.m_block, strategy, actual.data(), actual.size()) !=
                Y26_CONV_STATUS_SUCCESS) {
                throw std::runtime_error("debug pack failed");
            }
            reference_pack(fixture.input, m_begin, rows, expected.data());
            int mismatches = 0;
            for (std::size_t index = 0; index < actual.size(); ++index) {
                mismatches += actual[index] != expected[index];
            }
            total_mismatches += mismatches;
            constexpr int center_tile = 4 * (128 / 8);
            const std::size_t offset = static_cast<std::size_t>(center_tile) * rows * 8U;
            std::cout << m_begin << '\t' << y26::stage48::m_block_name(options.m_block) << '\t'
                      << y26::stage48::load_strategy_name(strategy) << '\t' << mismatches << '\t'
                      << fnv1a(actual.data(), actual.size()) << '\t';
            for (int index = 0; index < 32; ++index) {
                std::cout << std::hex << std::setw(2) << std::setfill('0')
                          << static_cast<int>(static_cast<std::uint8_t>(actual[offset + index]));
            }
            std::cout << std::dec << std::setfill(' ') << '\n';
        }
    }
    return total_mismatches == 0 ? 0 : 2;
}

int run_oracle_vectors(const Options& options) {
    const auto rows = read_tsv(options.package / "adversarial_requant.tsv");
    int mismatches = 0;
    std::cout << "case_id\tname\tstatus\tactual_rounded\texpected_rounded\tactual_code\texpected_code\n";
    for (const auto& row : rows) {
        const std::int64_t accumulator = parse_i64(field(row, "accumulator"));
        const std::int64_t multiplier = parse_i64(field(row, "multiplier"));
        const auto right_shift = static_cast<std::int32_t>(parse_i64(field(row, "right_shift")));
        const int zero_point = static_cast<int>(parse_i64(field(row, "output_zero_point")));
        const std::int64_t expected_rounded = parse_i64(field(row, "rounded"));
        const int expected_code = static_cast<int>(parse_i64(field(row, "output_code")));
        std::int64_t actual_rounded = 0;
        const bool rounded_ok = y26::int8_v1::round_product_right_even(
            accumulator, multiplier, right_shift, &actual_rounded);
        y26::int8_v1::RequantAsset asset{multiplier, right_shift, zero_point, 0, 255};
        std::uint8_t actual_code = 0;
        const bool code_ok = y26::int8_v1::requantize_u8(accumulator, asset, &actual_code);
        const bool exact = rounded_ok && code_ok && actual_rounded == expected_rounded && actual_code == expected_code;
        mismatches += !exact;
        std::cout << field(row, "case_id") << '\t' << field(row, "name") << '\t'
                  << (exact ? "pass" : "fail") << '\t' << actual_rounded << '\t' << expected_rounded << '\t'
                  << static_cast<int>(actual_code) << '\t' << expected_code << '\n';
    }
    return mismatches == 0 ? 0 : 2;
}

unsigned read_frm() noexcept {
#if defined(__riscv)
    unsigned value = 0;
    asm volatile("csrr %0, frm" : "=r"(value));
    return value & 7U;
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

int run_frm_sweep(const Options& options) {
    Model5DirectConv conv;
    if (conv.prepare(options.package, options.workers) != Y26_CONV_STATUS_SUCCESS) {
        throw std::runtime_error("prepare failed: " + conv.last_error());
    }
    auto fixture = load_fixture(options, conv);
    const unsigned original = read_frm();
    constexpr std::array<const char*, 5> names{{"RNE", "RTZ", "RDN", "RUP", "RMM"}};
    int failures = 0;
    std::cout << "mode\tset_status\trun_status\tmismatches\toutput_hash64\tambient_after\n";
    for (unsigned mode = 0; mode < names.size(); ++mode) {
        const bool set_status = write_frm(mode);
        const int run_status = conv.run(fixture.input.data(), fixture.output.data(), run_options(options, false), nullptr);
        std::size_t first = 0;
        int maximum = 0;
        const int mismatches = compare(fixture.output, fixture.expected, &first, &maximum);
        failures += !set_status || run_status != Y26_CONV_STATUS_SUCCESS || mismatches != 0 || read_frm() != mode;
        std::cout << names[mode] << '\t' << set_status << '\t' << run_status << '\t' << mismatches << '\t'
                  << fnv1a(fixture.output.data(), fixture.output.size()) << '\t' << read_frm() << '\n';
    }
    const bool restore = write_frm(original);
    std::cout << "restore\t" << restore << "\toriginal\t" << original << "\tfinal\t" << read_frm() << '\n';
    return failures == 0 && restore && read_frm() == original ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.mode == "validate") return run_validate(options);
        if (options.mode == "benchmark") return run_benchmark(options);
        if (options.mode == "profile") return run_profile(options);
        if (options.mode == "conversion-benchmark") return run_conversion_benchmark(options);
        if (options.mode == "byte-order") return run_byte_order(options);
        if (options.mode == "oracle-vectors") return run_oracle_vectors(options);
        if (options.mode == "frm-sweep") return run_frm_sweep(options);
        throw std::runtime_error("invalid mode: " + options.mode);
    } catch (const std::exception& error) {
        std::cerr << "stage48_error=" << error.what() << '\n';
        return 2;
    }
}
