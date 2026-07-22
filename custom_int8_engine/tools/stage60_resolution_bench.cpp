#include "y26_k1x_full_executor.h"
#include "y26_k1x_package.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::filesystem::path package;
    std::filesystem::path input;
    std::filesystem::path output;
    std::string surface = "preprocessed";
    int warmup = 10;
    int runs = 100;
    int repeats = 5;
    bool condition_variable = false;
};

struct Sample {
    int repeat = 0;
    int run = 0;
    y26::stage52::RunTiming timing;
};

Options parse(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next = [&]() -> std::string {
            if (++index >= argc) throw std::runtime_error("missing value for " + argument);
            return argv[index];
        };
        if (argument == "--package") result.package = next();
        else if (argument == "--input") result.input = next();
        else if (argument == "--output") result.output = next();
        else if (argument == "--surface") result.surface = next();
        else if (argument == "--warmup") result.warmup = std::stoi(next());
        else if (argument == "--runs") result.runs = std::stoi(next());
        else if (argument == "--repeats") result.repeats = std::stoi(next());
        else if (argument == "--wake") {
            const std::string wake = next();
            if (wake == "condition-variable") result.condition_variable = true;
            else if (wake != "frame-gated-spin") throw std::runtime_error("invalid wake policy");
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (result.package.empty() || result.input.empty() || result.output.empty() ||
        (result.surface != "preprocessed" && result.surface != "rgb") ||
        result.warmup < 0 || result.runs < 1 || result.repeats < 1) {
        throw std::runtime_error("invalid Stage60 benchmark arguments");
    }
    return result;
}

template <typename Value>
std::vector<Value> read_exact(const std::filesystem::path& path, std::size_t count) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != static_cast<std::streamsize>(count * sizeof(Value))) {
        throw std::runtime_error("input size does not match prepared static profile");
    }
    stream.seekg(0);
    std::vector<Value> result(count);
    if (!stream.read(reinterpret_cast<char*>(result.data()),
                     static_cast<std::streamsize>(result.size() * sizeof(Value)))) {
        throw std::runtime_error("cannot read benchmark input");
    }
    return result;
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1U);
    const auto lower = static_cast<std::size_t>(position);
    const auto upper = std::min(lower + 1U, values.size() - 1U);
    const double weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        y26::stage52::RunConfig config;
        config.workers = 4;
        config.worker_cpu_begin = 0;
        config.controller_cpu = 4;
        config.scheduler = y26::stage52::SchedulerMode::safe;
        config.wake_policy = options.condition_variable
            ? y26::stage52::WakePolicy::condition_variable
            : y26::stage52::WakePolicy::frame_gated_spin;
        config.compute = y26::stage52::ComputeMode::optimized;
        config.allow_stage60_static_profiles = true;
        y26::stage52::FullExecutor executor;
        const std::string manifest = y26::int8_v1::sha256_file(
            options.package / "asset_hashes.tsv");
        if (executor.prepare(options.package, manifest, config) != 0) {
            throw std::runtime_error("prepare failed: " + executor.last_error());
        }

        const auto input = options.surface == "preprocessed"
            ? read_exact<float>(options.input, executor.input_elements())
            : std::vector<float> {};
        const auto rgb = options.surface == "rgb"
            ? read_exact<std::uint8_t>(options.input,
                static_cast<std::size_t>(executor.input_width()) * executor.input_height() * 3U)
            : std::vector<std::uint8_t> {};
        std::vector<float> output(300U * 6U);
        auto run_once = [&]() {
            y26::stage52::RunTiming timing;
            const int status = options.surface == "preprocessed"
                ? executor.run_preprocessed(input.data(), input.size(), output.data(), output.size(), &timing)
                : executor.run_rgb(rgb.data(), executor.input_width(), executor.input_height(),
                                   executor.input_width() * 3, output.data(), output.size(), &timing);
            if (status != 0) throw std::runtime_error("run failed: " + executor.last_error());
            if (timing.affinity_ok != 1 || timing.cpu4_7_ime_count != 0) {
                throw std::runtime_error("CPU affinity or IME ownership contract failed");
            }
            return timing;
        };
        for (int index = 0; index < options.warmup; ++index) (void)run_once();

        std::vector<Sample> samples;
        samples.reserve(static_cast<std::size_t>(options.runs) * options.repeats);
        std::uint64_t expected_hash = 0;
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            for (int run = 0; run < options.runs; ++run) {
                const auto timing = run_once();
                if (expected_hash == 0) expected_hash = timing.output_hash;
                if (timing.output_hash != expected_hash) {
                    throw std::runtime_error("non-deterministic Stage60 output hash");
                }
                samples.push_back({repeat, run, timing});
            }
        }

        if (!options.output.parent_path().empty()) {
            std::filesystem::create_directories(options.output.parent_path());
        }
        std::ofstream stream(options.output);
        if (!stream) throw std::runtime_error("cannot write benchmark TSV");
        stream << "repeat\trun\tresolution\tsurface\twake_policy\ttotal_us\tinput_us"
                  "\tresident_core_us\tdense_us\tattention_us\tdepthwise_us\tlut_us"
                  "\tconcat_us\ttransform_us\thead_us\tprocess_cpu_us\tvoluntary_cs"
                  "\tinvoluntary_cs\taffinity_ok\tcpu4_7_ime_count\toutput_hash\tmanifest_sha256\n";
        stream << std::setprecision(15);
        std::vector<double> walls;
        walls.reserve(samples.size());
        for (const Sample& sample : samples) {
            const auto& timing = sample.timing;
            walls.push_back(timing.total_us);
            stream << sample.repeat << '\t' << sample.run << '\t' << executor.input_width()
                   << '\t' << options.surface << '\t'
                   << y26::stage52::wake_policy_name(config.wake_policy)
                   << '\t' << timing.total_us << '\t' << timing.input_quantize_us
                   << '\t' << timing.resident_core_us << '\t' << timing.dense_conv_us
                   << '\t' << timing.attention_us << '\t' << timing.depthwise_us
                   << '\t' << timing.lut_us << '\t' << timing.concat_us
                   << '\t' << timing.transform_us << '\t' << timing.head_us
                   << '\t' << timing.process_cpu_us << '\t' << timing.voluntary_context_switches
                   << '\t' << timing.involuntary_context_switches << '\t' << timing.affinity_ok
                   << '\t' << timing.cpu4_7_ime_count << "\t0x" << std::hex
                   << timing.output_hash << std::dec << '\t' << manifest << '\n';
        }
        const double mean = std::accumulate(walls.begin(), walls.end(), 0.0) /
            static_cast<double>(walls.size());
        std::cout << std::setprecision(15)
                  << "resolution=" << executor.input_width() << '\n'
                  << "samples=" << walls.size() << '\n'
                  << "mean_us=" << mean << '\n'
                  << "median_us=" << percentile(walls, 0.5) << '\n'
                  << "p95_us=" << percentile(walls, 0.95) << '\n'
                  << "p99_us=" << percentile(walls, 0.99) << '\n'
                  << "max_us=" << *std::max_element(walls.begin(), walls.end()) << '\n'
                  << "output_hash=0x" << std::hex << expected_hash << std::dec << '\n'
                  << "manifest_sha256=" << manifest << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
