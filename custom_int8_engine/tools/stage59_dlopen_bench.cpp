#include "y26_k1x_executor.h"

#include <dlfcn.h>
#include <sched.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename Function>
Function LoadSymbol(void* library, const char* name) {
    dlerror();
    void* symbol = dlsym(library, name);
    const char* error = dlerror();
    if (error != nullptr || symbol == nullptr) {
        throw std::runtime_error(std::string("cannot resolve ") + name + ": " +
                                 (error == nullptr ? "unknown error" : error));
    }
    return reinterpret_cast<Function>(symbol);
}

std::vector<float> ReadInput(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open input: " + path);
    const auto bytes = stream.tellg();
    const auto expected = static_cast<std::streamoff>(
        Y26_K1X_EXECUTOR_INPUT_ELEMENTS * sizeof(float));
    if (bytes != expected) throw std::runtime_error("unexpected input size: " + path);
    stream.seekg(0);
    std::vector<float> input(Y26_K1X_EXECUTOR_INPUT_ELEMENTS);
    if (!stream.read(reinterpret_cast<char*>(input.data()), bytes)) {
        throw std::runtime_error("cannot read input: " + path);
    }
    return input;
}

void PinController() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(4, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
        throw std::runtime_error(std::string("cannot pin controller: ") + std::strerror(errno));
    }
}

struct Api {
    using OptionsInit = void (*)(y26_executor_options*);
    using Create = y26_executor* (*)(void);
    using Prepare = y26_status (*)(y26_executor*, const char*, const char*,
                                   const y26_executor_options*);
    using Run = y26_status (*)(y26_executor*, const float*, size_t, float*, size_t,
                               y26_run_timing*);
    using LastError = const char* (*)(const y26_executor*);
    using Destroy = void (*)(y26_executor*);
    using Version = const char* (*)(void);

    void* library = nullptr;
    OptionsInit options_init = nullptr;
    Create create = nullptr;
    Prepare prepare = nullptr;
    Run run = nullptr;
    LastError last_error = nullptr;
    Destroy destroy = nullptr;
    Version version = nullptr;

    explicit Api(const std::string& path) {
        library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (library == nullptr) throw std::runtime_error(dlerror());
        options_init = LoadSymbol<OptionsInit>(library, "y26_executor_options_init");
        create = LoadSymbol<Create>(library, "y26_executor_create");
        prepare = LoadSymbol<Prepare>(library, "y26_executor_prepare");
        run = LoadSymbol<Run>(library, "y26_executor_run_preprocessed");
        last_error = LoadSymbol<LastError>(library, "y26_executor_last_error");
        destroy = LoadSymbol<Destroy>(library, "y26_executor_destroy");
        version = LoadSymbol<Version>(library, "y26_executor_version");
    }

    Api(const Api&) = delete;
    Api& operator=(const Api&) = delete;

    ~Api() {
        if (library != nullptr) dlclose(library);
    }
};

class Arm {
public:
    Arm(std::string label, const std::string& library, const std::string& package,
        const std::string& manifest, const std::vector<float>& input)
        : label_(std::move(label)), api_(library), input_(input),
          output_(Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS) {
        executor_ = api_.create();
        if (executor_ == nullptr) throw std::runtime_error(label_ + ": create failed");
        y26_executor_options options;
        api_.options_init(&options);
        options.workers = 4;
        options.worker_cpu_begin = 0;
        options.controller_cpu = 4;
        options.scheduler = Y26_SCHEDULER_SAFE;
        options.wake_policy = Y26_WAKE_FRAME_GATED_SPIN;
        const y26_status status = api_.prepare(executor_, package.c_str(), manifest.c_str(), &options);
        if (status != Y26_STATUS_OK) {
            throw std::runtime_error(label_ + ": prepare failed: " + api_.last_error(executor_));
        }
        std::cerr << "prepared arm=" << label_ << " version=" << api_.version() << '\n';
    }

    Arm(const Arm&) = delete;
    Arm& operator=(const Arm&) = delete;

    ~Arm() {
        if (executor_ != nullptr) api_.destroy(executor_);
    }

    y26_run_timing Run() {
        y26_run_timing timing {};
        const y26_status status = api_.run(executor_, input_.data(), input_.size(),
                                           output_.data(), output_.size(), &timing);
        if (status != Y26_STATUS_OK) {
            throw std::runtime_error(label_ + ": run failed: " + api_.last_error(executor_));
        }
        return timing;
    }

    const std::string& label() const noexcept { return label_; }

private:
    std::string label_;
    Api api_;
    const std::vector<float>& input_;
    std::vector<float> output_;
    y26_executor* executor_ = nullptr;
};

void Emit(const Arm& arm, int block, int cycle, int position, std::uint64_t sample,
          const y26_run_timing& timing) {
    std::cout << arm.label() << '\t' << block << '\t' << cycle << '\t' << position << '\t'
              << sample << '\t' << timing.total_us << '\t' << timing.pure_executor_us << '\t'
              << timing.process_cpu_us << '\t' << timing.voluntary_context_switches << '\t'
              << timing.involuntary_context_switches << "\t0x" << std::hex << timing.output_hash
              << std::dec << '\t' << timing.affinity_ok << '\t' << timing.cpu4_7_ime_count << '\n';
}

int ParsePositive(const char* value, const char* name) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > 1000000) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return static_cast<int>(parsed);
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc != 10) {
        std::cerr << "usage: " << argv[0]
                  << " LIB_A LIB_B PACKAGE INPUT MANIFEST WARMUP BLOCKS CYCLES_PER_BLOCK LABEL_A,LABEL_B\n";
        return 2;
    }
    PinController();
    const std::vector<float> input = ReadInput(argv[4]);
    const int warmup = ParsePositive(argv[6], "warmup");
    const int blocks = ParsePositive(argv[7], "blocks");
    const int cycles = ParsePositive(argv[8], "cycles");
    const std::string labels = argv[9];
    const std::size_t comma = labels.find(',');
    if (comma == std::string::npos) throw std::runtime_error("labels must be LABEL_A,LABEL_B");
    Arm a(labels.substr(0, comma), argv[1], argv[3], argv[5], input);
    Arm b(labels.substr(comma + 1), argv[2], argv[3], argv[5], input);

    for (int index = 0; index < warmup; ++index) {
        (void)a.Run();
        (void)b.Run();
    }

    std::cout << "arm\tblock\tcycle\tposition\tsample\ttotal_us\tpure_executor_us"
                 "\tprocess_cpu_us\tvoluntary_cs\tinvoluntary_cs\toutput_hash\taffinity_ok"
                 "\tcpu4_7_ime_count\n";
    std::uint64_t sample_a = 0;
    std::uint64_t sample_b = 0;
    for (int block = 0; block < blocks; ++block) {
        for (int cycle = 0; cycle < cycles; ++cycle) {
            Arm* order[4] = {&a, &b, &b, &a};
            if (((block + cycle) & 1) != 0) {
                order[0] = &b;
                order[1] = &a;
                order[2] = &a;
                order[3] = &b;
            }
            for (int position = 0; position < 4; ++position) {
                Arm& arm = *order[position];
                const bool is_a = &arm == &a;
                const std::uint64_t sample = is_a ? ++sample_a : ++sample_b;
                Emit(arm, block, cycle, position, sample, arm.Run());
            }
        }
        std::cout.flush();
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "stage59_dlopen_bench: " << error.what() << '\n';
    return 1;
}
