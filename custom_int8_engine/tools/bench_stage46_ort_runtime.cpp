#include <onnxruntime_cxx_api.h>
#include <spacemit_ort_env.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string provider = "cpu";
    std::string model;
    std::string input;
    std::string output;
    std::string input_name;
    std::string output_name;
    std::string optimization = "disable";
    std::string execution_mode = "sequential";
    std::string profile_prefix;
    std::string custom_op_library;
    std::string plugin_counter_symbol;
    int intra_threads = 1;
    int inter_threads = 1;
    int memory_pattern = 1;
    int cpu_arena = 1;
    int thread_spinning = 0;
    int log_severity = 2;
    int log_verbosity = 0;
    int warmup = 0;
    int runs = 1;
    int repeats = 1;
    std::vector<std::pair<std::string, std::string>> provider_options;
};

struct Stats {
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;
    double median = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double cv_pct = 0.0;
};

std::string require_value(int& index, int argc, char** argv, const char* option) {
    if (index + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + option);
    }
    return argv[++index];
}

int parse_bool(const std::string& value, const char* option) {
    if (value == "0") {
        return 0;
    }
    if (value == "1") {
        return 1;
    }
    throw std::runtime_error(std::string(option) + " must be 0 or 1");
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--provider") {
            options.provider = require_value(index, argc, argv, "--provider");
        } else if (arg == "--model") {
            options.model = require_value(index, argc, argv, "--model");
        } else if (arg == "--input") {
            options.input = require_value(index, argc, argv, "--input");
        } else if (arg == "--output") {
            options.output = require_value(index, argc, argv, "--output");
        } else if (arg == "--input-name") {
            options.input_name = require_value(index, argc, argv, "--input-name");
        } else if (arg == "--output-name") {
            options.output_name = require_value(index, argc, argv, "--output-name");
        } else if (arg == "--opt-level") {
            options.optimization = require_value(index, argc, argv, "--opt-level");
        } else if (arg == "--execution-mode") {
            options.execution_mode = require_value(index, argc, argv, "--execution-mode");
        } else if (arg == "--intra-threads") {
            options.intra_threads = std::max(1, std::stoi(require_value(index, argc, argv, "--intra-threads")));
        } else if (arg == "--inter-threads") {
            options.inter_threads = std::max(1, std::stoi(require_value(index, argc, argv, "--inter-threads")));
        } else if (arg == "--memory-pattern") {
            options.memory_pattern = parse_bool(require_value(index, argc, argv, "--memory-pattern"), "--memory-pattern");
        } else if (arg == "--cpu-arena") {
            options.cpu_arena = parse_bool(require_value(index, argc, argv, "--cpu-arena"), "--cpu-arena");
        } else if (arg == "--thread-spinning") {
            options.thread_spinning =
                parse_bool(require_value(index, argc, argv, "--thread-spinning"), "--thread-spinning");
        } else if (arg == "--log-severity") {
            options.log_severity = std::stoi(require_value(index, argc, argv, "--log-severity"));
        } else if (arg == "--log-verbosity") {
            options.log_verbosity = std::stoi(require_value(index, argc, argv, "--log-verbosity"));
        } else if (arg == "--warmup") {
            options.warmup = std::max(0, std::stoi(require_value(index, argc, argv, "--warmup")));
        } else if (arg == "--runs") {
            options.runs = std::max(1, std::stoi(require_value(index, argc, argv, "--runs")));
        } else if (arg == "--repeats") {
            options.repeats = std::max(1, std::stoi(require_value(index, argc, argv, "--repeats")));
        } else if (arg == "--profile-prefix") {
            options.profile_prefix = require_value(index, argc, argv, "--profile-prefix");
        } else if (arg == "--custom-op-library") {
            options.custom_op_library = require_value(index, argc, argv, "--custom-op-library");
        } else if (arg == "--plugin-counter-symbol") {
            options.plugin_counter_symbol = require_value(index, argc, argv, "--plugin-counter-symbol");
        } else if (arg == "--provider-option") {
            const std::string value = require_value(index, argc, argv, "--provider-option");
            const std::size_t split = value.find('=');
            if (split == std::string::npos || split == 0) {
                throw std::runtime_error("--provider-option requires key=value");
            }
            options.provider_options.emplace_back(value.substr(0, split), value.substr(split + 1));
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.provider != "cpu" && options.provider != "spacemit") {
        throw std::runtime_error("--provider must be cpu or spacemit");
    }
    if (options.model.empty() || options.input.empty() || options.output.empty()) {
        throw std::runtime_error("--model, --input, and --output are required");
    }
    const std::vector<std::string> levels = {"disable", "basic", "extended", "all"};
    if (std::find(levels.begin(), levels.end(), options.optimization) == levels.end()) {
        throw std::runtime_error("--opt-level must be disable, basic, extended, or all");
    }
    if (options.execution_mode != "sequential" && options.execution_mode != "parallel") {
        throw std::runtime_error("--execution-mode must be sequential or parallel");
    }
    return options;
}

GraphOptimizationLevel graph_optimization_level(const std::string& value) {
    if (value == "disable") return ORT_DISABLE_ALL;
    if (value == "basic") return ORT_ENABLE_BASIC;
    if (value == "extended") return ORT_ENABLE_EXTENDED;
    return ORT_ENABLE_ALL;
}

std::size_t element_size(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return 8;
        default:
            return 0;
    }
}

std::size_t checked_element_count(const std::vector<std::int64_t>& shape) {
    std::size_t result = 1;
    for (std::int64_t dimension : shape) {
        if (dimension < 0) {
            throw std::runtime_error("dynamic or negative tensor dimension is unsupported");
        }
        const std::size_t value = static_cast<std::size_t>(dimension);
        if (value != 0 && result > std::numeric_limits<std::size_t>::max() / value) {
            throw std::overflow_error("tensor element count overflow");
        }
        result *= value;
    }
    return result;
}

std::vector<std::uint8_t> read_exact(const std::string& path, std::size_t expected) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("failed to open input: " + path);
    }
    const std::streamoff length = input.tellg();
    if (length < 0 || static_cast<std::uint64_t>(length) != expected) {
        throw std::runtime_error("input byte count mismatch: expected=" + std::to_string(expected) +
                                 " actual=" + std::to_string(length));
    }
    std::vector<std::uint8_t> bytes(expected);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) {
        throw std::runtime_error("failed to read input: " + path);
    }
    return bytes;
}

void write_bytes(const std::string& path, const void* data, std::size_t bytes) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open output: " + path);
    }
    output.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!output) {
        throw std::runtime_error("failed to write output: " + path);
    }
}

std::string shape_string(const std::vector<std::int64_t>& shape) {
    std::string result;
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) result += 'x';
        result += std::to_string(shape[index]);
    }
    return result.empty() ? "scalar" : result;
}

std::uint64_t fnv1a64(const void* data, std::size_t bytes) {
    const auto* source = static_cast<const std::uint8_t*>(data);
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t index = 0; index < bytes; ++index) {
        hash ^= source[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

double process_cpu_us() {
    timespec value {};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) != 0) {
        throw std::runtime_error("CLOCK_PROCESS_CPUTIME_ID failed");
    }
    return static_cast<double>(value.tv_sec) * 1.0e6 + static_cast<double>(value.tv_nsec) / 1.0e3;
}

double percentile(const std::vector<double>& sorted, double quantile) {
    const double position = quantile * static_cast<double>(sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    if (lower == upper) return sorted[lower];
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

Stats calculate_stats(const std::vector<double>& values) {
    Stats stats;
    stats.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    stats.min = *std::min_element(values.begin(), values.end());
    stats.max = *std::max_element(values.begin(), values.end());
    double sum_sq = 0.0;
    for (double value : values) {
        const double delta = value - stats.mean;
        sum_sq += delta * delta;
    }
    stats.stddev = values.size() > 1 ? std::sqrt(sum_sq / static_cast<double>(values.size() - 1)) : 0.0;
    stats.cv_pct = stats.mean != 0.0 ? 100.0 * stats.stddev / stats.mean : 0.0;
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    stats.median = percentile(sorted, 0.50);
    stats.p90 = percentile(sorted, 0.90);
    stats.p95 = percentile(sorted, 0.95);
    return stats;
}

void print_stats(const char* metric, const Stats& stats) {
    std::cout << std::fixed << std::setprecision(6)
              << "stage46_stats metric=" << metric
              << " mean_us=" << stats.mean
              << " stddev_us=" << stats.stddev
              << " cv_pct=" << stats.cv_pct
              << " min_us=" << stats.min
              << " max_us=" << stats.max
              << " median_us=" << stats.median
              << " p90_us=" << stats.p90
              << " p95_us=" << stats.p95 << '\n';
}

void print_plugin_counter(const Options& options) {
    if (options.custom_op_library.empty() || options.plugin_counter_symbol.empty()) return;
    void* handle = dlopen(options.custom_op_library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        throw std::runtime_error(std::string("dlopen plugin counter failed: ") + dlerror());
    }
    using CounterFn = std::uint64_t (*)();
    dlerror();
    auto* function = reinterpret_cast<CounterFn>(dlsym(handle, options.plugin_counter_symbol.c_str()));
    const char* error = dlerror();
    if (error != nullptr || function == nullptr) {
        dlclose(handle);
        throw std::runtime_error(std::string("plugin counter symbol unavailable: ") + (error ? error : "null"));
    }
    std::cout << "stage46_plugin_counter symbol=" << options.plugin_counter_symbol
              << " value=" << function() << '\n';
    dlclose(handle);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::cout << "stage46_runtime"
                  << " ort_version=" << OrtGetApiBase()->GetVersionString()
                  << " ort_build_info=" << Ort::GetApi().GetBuildInfoString()
                  << " ort_api_version=" << ORT_API_VERSION
                  << " spacemit_ep_header_version=" << SpaceMITPROVIDER_VERSION
                  << " spacemit_ep_header_build_date=" << SpaceMITPROVIDER_BUILD_DATE
                  << " provider=" << options.provider
                  << " opt_level=" << options.optimization
                  << " execution_mode=" << options.execution_mode
                  << " intra_threads=" << options.intra_threads
                  << " inter_threads=" << options.inter_threads
                  << " memory_pattern=" << options.memory_pattern
                  << " cpu_arena=" << options.cpu_arena
                  << " thread_spinning=" << options.thread_spinning << '\n';
        for (const std::string& provider : Ort::GetAvailableProviders()) {
            std::cout << "stage46_provider registered=1 name=" << provider << '\n';
        }

        Ort::Env env(static_cast<OrtLoggingLevel>(options.log_severity), "stage46_runtime");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(options.intra_threads);
        session_options.SetInterOpNumThreads(options.inter_threads);
        session_options.SetGraphOptimizationLevel(graph_optimization_level(options.optimization));
        session_options.SetExecutionMode(options.execution_mode == "sequential" ? ORT_SEQUENTIAL : ORT_PARALLEL);
        session_options.SetLogId("stage46_runtime");
        session_options.SetLogSeverityLevel(options.log_severity);
        Ort::ThrowOnError(
            Ort::GetApi().SetSessionLogVerbosityLevel(session_options, options.log_verbosity));
        if (options.memory_pattern != 0) session_options.EnableMemPattern();
        else session_options.DisableMemPattern();
        if (options.cpu_arena != 0) session_options.EnableCpuMemArena();
        else session_options.DisableCpuMemArena();
        session_options.AddConfigEntry("session.intra_op.allow_spinning", options.thread_spinning != 0 ? "1" : "0");
        session_options.AddConfigEntry("session.inter_op.allow_spinning", options.thread_spinning != 0 ? "1" : "0");
        if (!options.profile_prefix.empty()) session_options.EnableProfiling(options.profile_prefix.c_str());
        if (!options.custom_op_library.empty()) {
            session_options.RegisterCustomOpsLibrary(options.custom_op_library.c_str());
            std::cout << "stage46_custom_op_library registered=1 path=" << options.custom_op_library << '\n';
        }
        if (options.provider == "spacemit") {
            std::unordered_map<std::string, std::string> provider_options;
            for (const auto& [key, value] : options.provider_options) provider_options[key] = value;
            for (const char* key : {
                     "SPACEMIT_EP_DISABLE_PASSES_FILTER",
                     "SPACEMIT_EP_DISABLE_OP_TYPE_FILTER",
                     "SPACEMIT_EP_DISABLE_OP_NAME_FILTER",
                     "SPACEMIT_EP_DISABLE_FLOAT16_EPILOGUE",
                     "SPACEMIT_EP_DUMP_SUBGRAPHS",
                     "SPACEMIT_EP_DEBUG_PROFILE",
                     "SPACEMIT_EP_DUMP_TENSORS",
                     "SPACEMIT_EP_PLUGIN_LIB",
                 }) {
                if (provider_options.contains(key)) continue;
                const char* value = std::getenv(key);
                if (value != nullptr && *value != '\0') provider_options[key] = value;
            }
            if (!options.custom_op_library.empty() && !provider_options.contains("SPACEMIT_EP_PLUGIN_LIB")) {
                provider_options["SPACEMIT_EP_PLUGIN_LIB"] = options.custom_op_library;
            }
            if (!provider_options.contains("SPACEMIT_EP_INTRA_THREAD_NUM")) {
                provider_options["SPACEMIT_EP_INTRA_THREAD_NUM"] = std::to_string(options.intra_threads);
            }
            for (const auto& [key, value] : provider_options) {
                std::cout << "stage46_provider_option key=" << key << " value=" << value << '\n';
            }
            Ort::ThrowOnError(Ort::SessionOptionsSpaceMITEnvInit(session_options, provider_options));
            std::cout << "stage46_provider appended=1 name=SpaceMITExecutionProvider\n";
        } else {
            std::cout << "stage46_provider appended=0 name=SpaceMITExecutionProvider\n";
        }

        const auto session_begin = Clock::now();
        Ort::Session session(env, options.model.c_str(), session_options);
        const auto session_end = Clock::now();
        const double session_create_us =
            std::chrono::duration<double, std::micro>(session_end - session_begin).count();
        std::cout << std::fixed << std::setprecision(6)
                  << "stage46_session status=created create_us=" << session_create_us << '\n';

        Ort::AllocatorWithDefaultOptions allocator;
        auto discovered_input_name = session.GetInputNameAllocated(0, allocator);
        auto discovered_output_name = session.GetOutputNameAllocated(0, allocator);
        const std::string input_name = options.input_name.empty() ? discovered_input_name.get() : options.input_name;
        const std::string output_name = options.output_name.empty() ? discovered_output_name.get() : options.output_name;
        const auto input_type_info = session.GetInputTypeInfo(0);
        const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
        const std::vector<std::int64_t> input_shape = input_info.GetShape();
        const ONNXTensorElementDataType input_type = input_info.GetElementType();
        const std::size_t input_element_size = element_size(input_type);
        std::cout << "stage46_input_metadata name=" << input_name
                  << " dtype=" << static_cast<int>(input_type)
                  << " shape=" << shape_string(input_shape)
                  << " element_size=" << input_element_size << '\n';
        if (input_element_size == 0) throw std::runtime_error("unsupported input dtype");
        const std::size_t input_elements = checked_element_count(input_shape);
        if (input_elements > std::numeric_limits<std::size_t>::max() / input_element_size) {
            throw std::overflow_error("input byte count overflow");
        }
        std::vector<std::uint8_t> input_bytes = read_exact(options.input, input_elements * input_element_size);
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor(
            memory, input_bytes.data(), input_bytes.size(), input_shape.data(), input_shape.size(), input_type);
        const char* input_names[] = {input_name.c_str()};
        const char* output_names[] = {output_name.c_str()};
        std::vector<Ort::Value> output_values;
        auto run_once = [&]() {
            output_values = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        };

        const auto first_begin = Clock::now();
        run_once();
        const auto first_end = Clock::now();
        const double first_run_us = std::chrono::duration<double, std::micro>(first_end - first_begin).count();
        for (int warmup = 0; warmup < options.warmup; ++warmup) run_once();

        std::vector<double> wall_repeat_us;
        std::vector<double> process_repeat_us;
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            const auto wall_begin = Clock::now();
            const double cpu_begin = process_cpu_us();
            for (int run = 0; run < options.runs; ++run) run_once();
            const double cpu_end = process_cpu_us();
            const auto wall_end = Clock::now();
            const double wall_us = std::chrono::duration<double, std::micro>(wall_end - wall_begin).count() /
                                   static_cast<double>(options.runs);
            const double cpu_us = (cpu_end - cpu_begin) / static_cast<double>(options.runs);
            wall_repeat_us.push_back(wall_us);
            process_repeat_us.push_back(cpu_us);
            std::cout << std::fixed << std::setprecision(6)
                      << "stage46_repeat index=" << repeat
                      << " runs=" << options.runs
                      << " wall_mean_us=" << wall_us
                      << " process_cpu_mean_us=" << cpu_us << '\n';
        }

        Ort::Value& output_value = output_values.at(0);
        const auto output_info = output_value.GetTensorTypeAndShapeInfo();
        const std::vector<std::int64_t> output_shape = output_info.GetShape();
        const ONNXTensorElementDataType output_type = output_info.GetElementType();
        const std::size_t output_element_size = element_size(output_type);
        if (output_element_size == 0) throw std::runtime_error("unsupported output dtype");
        const std::size_t output_elements = checked_element_count(output_shape);
        if (output_elements > std::numeric_limits<std::size_t>::max() / output_element_size) {
            throw std::overflow_error("output byte count overflow");
        }
        void* output_data = nullptr;
        Ort::ThrowOnError(Ort::GetApi().GetTensorMutableData(output_value, &output_data));
        const std::size_t output_bytes = output_elements * output_element_size;
        write_bytes(options.output, output_data, output_bytes);
        std::cout << "stage46_tensor"
                  << " input_name=" << input_name
                  << " input_shape=" << shape_string(input_shape)
                  << " input_dtype=" << static_cast<int>(input_type)
                  << " input_bytes=" << input_bytes.size()
                  << " output_name=" << output_name
                  << " output_shape=" << shape_string(output_shape)
                  << " output_dtype=" << static_cast<int>(output_type)
                  << " output_bytes=" << output_bytes
                  << " output_fnv1a64=" << fnv1a64(output_data, output_bytes) << '\n';
        std::cout << std::fixed << std::setprecision(6)
                  << "stage46_first_run first_run_us=" << first_run_us << '\n';
        print_stats("wall", calculate_stats(wall_repeat_us));
        print_stats("process_cpu", calculate_stats(process_repeat_us));
        print_plugin_counter(options);
        if (!options.profile_prefix.empty()) {
            auto profile_path = session.EndProfilingAllocated(allocator);
            std::cout << "stage46_profile path=" << profile_path.get() << '\n';
        }
        std::cout << "stage46_result status=pass\n";
        return 0;
    } catch (const Ort::Exception& exception) {
        std::cerr << "stage46_result status=ort_exception code=" << exception.GetOrtErrorCode()
                  << " message=" << exception.what() << '\n';
        return 10;
    } catch (const std::exception& exception) {
        std::cerr << "stage46_result status=exception message=" << exception.what() << '\n';
        return 11;
    }
}
