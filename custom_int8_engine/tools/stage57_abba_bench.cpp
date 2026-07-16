#include "y26_k1x_full_executor.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::filesystem::path package;
    std::filesystem::path fixture;
    std::string manifest;
    std::string candidate;
    std::string surface = "preprocessed";
    int warmup = 10;
    int runs = 100;
    int blocks = 10;
    std::uint64_t expected_hash = 0xd43f5e018b415631ULL;
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
        else if (argument == "--fixture") result.fixture = next();
        else if (argument == "--manifest") result.manifest = next();
        else if (argument == "--candidate") result.candidate = next();
        else if (argument == "--surface") result.surface = next();
        else if (argument == "--warmup") result.warmup = std::stoi(next());
        else if (argument == "--runs") result.runs = std::stoi(next());
        else if (argument == "--blocks") result.blocks = std::stoi(next());
        else if (argument == "--expected-hash") result.expected_hash = std::stoull(next(), nullptr, 0);
        else throw std::runtime_error("unknown argument: " + argument);
    }
    if (result.package.empty() || result.fixture.empty() || result.manifest.size() != 64U ||
        result.candidate.empty() ||
        (result.surface != "preprocessed" && result.surface != "rgb") ||
        result.warmup < 0 || result.runs < 1 || result.blocks < 2) {
        throw std::runtime_error("invalid Stage57 ABBA arguments");
    }
    return result;
}

std::vector<std::uint8_t> read_rgb_fixture(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    constexpr std::size_t kBytes = 640U * 640U * 3U;
    if (!stream || stream.tellg() != static_cast<std::streamsize>(kBytes)) {
        throw std::runtime_error("invalid RGB fixture");
    }
    stream.seekg(0);
    std::vector<std::uint8_t> input(kBytes);
    if (!stream.read(reinterpret_cast<char*>(input.data()),
                     static_cast<std::streamsize>(input.size()))) {
        throw std::runtime_error("cannot read RGB fixture");
    }
    return input;
}

std::vector<float> read_fixture(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    constexpr std::size_t kElements = 3U * 640U * 640U;
    if (!stream || stream.tellg() != static_cast<std::streamsize>(kElements * sizeof(float))) {
        throw std::runtime_error("invalid preprocessed fixture");
    }
    stream.seekg(0);
    std::vector<float> input(kElements);
    if (!stream.read(reinterpret_cast<char*>(input.data()),
                     static_cast<std::streamsize>(input.size() * sizeof(float)))) {
        throw std::runtime_error("cannot read preprocessed fixture");
    }
    return input;
}

void set_stage56_control() {
    static constexpr std::array<const char*, 15> enabled {
        "Y26_STAGE54_E2C3", "Y26_STAGE55_E2C4", "Y26_STAGE55_DENSE_FAMILY_A",
        "Y26_STAGE54_DIRECT_1X1", "Y26_STAGE54_DENSE_PACK_RVV",
        "Y26_STAGE53_FUSED_LUT", "Y26_STAGE54_DEPTHWISE_V2",
        "Y26_STAGE54_DEPTHWISE_X2", "Y26_STAGE54_DEPTHWISE_BORDER_V2",
        "Y26_STAGE54_INPUT_RVV_V2", "Y26_STAGE54_INPUT_COMPACT_C3",
        "Y26_STAGE54_LUT2_RVV", "Y26_STAGE54_ATTENTION_V2",
        "Y26_STAGE56_HEAD_PRODUCER_REDUCTION", "Y26_STAGE56_ATTENTION_DIRECT_PACK",
    };
    for (const char* name : enabled) setenv(name, "1", 1);
    unsetenv("Y26_STAGE54_HEAD_V2");
    unsetenv("Y26_STAGE55_DEPTHWISE_E2C4");
}

void clear_stage57_candidates() {
    static constexpr std::array<const char*, 5> names {
        "Y26_STAGE57_E2C5", "Y26_STAGE57_ATTENTION_MATMUL_C8",
        "Y26_STAGE57_ATTENTION_SOFTMAX_CACHE", "Y26_STAGE57_HEAD_BUCKET",
        "Y26_STAGE57_RGB_COPY_RVV",
    };
    for (const char* name : names) unsetenv(name);
}

void enable_candidate(std::string_view candidate) {
    if (candidate == "e2c5") setenv("Y26_STAGE57_E2C5", "1", 1);
    else if (candidate == "attention-matmul") setenv("Y26_STAGE57_ATTENTION_MATMUL_C8", "1", 1);
    else if (candidate == "attention-softmax") setenv("Y26_STAGE57_ATTENTION_SOFTMAX_CACHE", "1", 1);
    else if (candidate == "head-bucket") setenv("Y26_STAGE57_HEAD_BUCKET", "1", 1);
    else if (candidate == "rgb-copy") setenv("Y26_STAGE57_RGB_COPY_RVV", "1", 1);
    else if (candidate == "selected-bundle") {
        setenv("Y26_STAGE57_E2C5", "1", 1);
        setenv("Y26_STAGE57_ATTENTION_MATMUL_C8", "1", 1);
    } else {
        throw std::runtime_error("unsupported Stage57 candidate");
    }
}

void prepare(y26::stage52::FullExecutor& executor, const Options& options) {
    y26::stage52::RunConfig config;
    config.workers = 4;
    config.worker_cpu_begin = 0;
    config.controller_cpu = 4;
    config.scheduler = y26::stage52::SchedulerMode::safe;
    config.wake_policy = y26::stage52::WakePolicy::frame_gated_spin;
    if (executor.prepare(options.package, options.manifest, config) != 0) {
        throw std::runtime_error("prepare failed: " + executor.last_error());
    }
}

void run_once(y26::stage52::FullExecutor& executor, const std::vector<float>& input,
              const std::vector<std::uint8_t>& rgb,
              std::vector<float>& output, const Options& options,
              int cycle, int position, char arm, int block, int run,
              std::uint64_t& observed_hash, bool emit = true) {
    y26::stage52::RunTiming timing;
    const int status = options.surface == "preprocessed"
        ? executor.run_preprocessed(input.data(), input.size(), output.data(), output.size(), &timing)
        : executor.run_rgb(rgb.data(), 640, 640, 640 * 3,
                           output.data(), output.size(), &timing);
    if (status != 0) {
        throw std::runtime_error("run failed: " + executor.last_error());
    }
    if (observed_hash == 0) observed_hash = timing.output_hash;
    if (timing.output_hash != observed_hash || timing.cpu4_7_ime_count != 0 ||
        timing.affinity_ok != 1) {
        throw std::runtime_error("Stage57 ABBA identity or CPU contract mismatch");
    }
    if (!emit) return;
    std::cout << "raw\tcycle=" << cycle << "\tposition=" << position
              << "\tarm=" << arm << "\tblock=" << block << "\trun=" << run
              << "\twall_us=" << timing.total_us
              << "\tinput_us=" << timing.input_quantize_us
              << "\tcore_us=" << timing.resident_core_us
              << "\tdense_us=" << timing.dense_conv_us
              << "\tdepthwise_us=" << timing.depthwise_us
              << "\tattention_us=" << timing.attention_us
              << "\tlut_us=" << timing.lut_us
              << "\tconcat_us=" << timing.concat_us
              << "\ttransform_us=" << timing.transform_us
              << "\thead_us=" << timing.head_us
              << "\tprocess_cpu_us=" << timing.process_cpu_us
              << "\tvoluntary_cs=" << timing.voluntary_context_switches
              << "\tinvoluntary_cs=" << timing.involuntary_context_switches
              << "\taffinity_ok=" << timing.affinity_ok
              << "\tcpu4_7_ime_count=" << timing.cpu4_7_ime_count
              << "\thash=0x" << std::hex << timing.output_hash << std::dec << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        const std::vector<float> input = options.surface == "preprocessed"
            ? read_fixture(options.fixture) : std::vector<float> {};
        const std::vector<std::uint8_t> rgb = options.surface == "rgb"
            ? read_rgb_fixture(options.fixture) : std::vector<std::uint8_t> {};
        std::vector<float> output(300U * 6U);
        std::uint64_t observed_hash = options.expected_hash;
        set_stage56_control();
        clear_stage57_candidates();
        y26::stage52::FullExecutor control;
        prepare(control, options);
        enable_candidate(options.candidate);
        y26::stage52::FullExecutor candidate;
        prepare(candidate, options);
        clear_stage57_candidates();

        for (int index = 0; index < options.warmup; ++index) {
            run_once(control, input, rgb, output, options, -1, 0, 'A', -1, index,
                     observed_hash, false);
            run_once(candidate, input, rgb, output, options, -1, 1, 'B', -1, index,
                     observed_hash, false);
        }
        int block = 0;
        for (int cycle = 0; cycle < options.blocks; ++cycle) {
            const std::array<char, 2> order = cycle % 2 == 0
                ? std::array<char, 2> {'A', 'B'} : std::array<char, 2> {'B', 'A'};
            for (int position = 0; position < 2; ++position, ++block) {
                const char arm = order[static_cast<std::size_t>(position)];
                std::cout << "block\t" << cycle << '\t' << position << '\t' << arm
                          << '\t' << block << '\n';
                y26::stage52::FullExecutor& executor = arm == 'A' ? control : candidate;
                for (int run = 0; run < options.runs; ++run) {
                    run_once(executor, input, rgb, output, options, cycle, position, arm,
                             block, run, observed_hash);
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
