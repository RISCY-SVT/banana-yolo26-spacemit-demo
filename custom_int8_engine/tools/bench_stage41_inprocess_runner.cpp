#define Y26_STAGE16_NO_TEST_MAIN 1
#include "../tests/test_stage16_model4_c2f_runner.cpp"

#include "y26_stage42_runner_support.h"

#include <onnxruntime_c_api.h>

#include <algorithm>
#include <array>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr const char* kInputName = "images";
constexpr const char* kFinalOutputName = "output0";
constexpr const char* kModel4InputName = "/model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output";
constexpr const char* kModel4OutputName = "/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output";

using Clock = std::chrono::steady_clock;

struct Options {
    std::string mode = "validate";
    std::string model_path;
    std::string cut_dir;
    std::string input_npy;
    std::string model4_input_npy;
    std::string expected_output_npy;
    std::string expected_model4_output_npy;
    std::string profile_cuts_tsv;
    std::string ort_input_name;
    std::string ort_output_name;
    std::string dump_custom_model4_nhwc;
    std::string dump_custom_model4_nchw;
    std::string dump_ort_model4_nchw;
    std::string dump_model4_input_nchw;
    std::string dump_final_output;
    std::string custom_mode = "ime_threaded";
    std::string ort_opt_level = "disable";
    std::string ort_execution_mode = "sequential";
    std::string profile_prefix = "stage42_ort_profile";
    int warmup = 3;
    int runs = 10;
    int repeats = 3;
    int ort_intra_threads = 1;
    int ort_inter_threads = 1;
    int ort_enable_profiling = 0;
    int ort_log_severity = 2;
    int ort_log_verbosity = 0;
    int ort_memory_pattern = 0;
    int ort_cpu_arena = 0;
    int ort_thread_spinning = 0;
    int thread_branch0 = 4;
    int thread_branch1 = 4;
    int thread_model4_cv2 = 4;
};

struct Tensor {
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::vector<std::int64_t> shape;
    std::vector<std::uint8_t> bytes;

    std::size_t element_size() const {
        switch (type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                return sizeof(float);
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                return sizeof(std::uint8_t);
            default:
                return 0;
        }
    }

    std::size_t element_count() const {
        return y26_stage42::checked_element_count(shape);
    }

    y26_stage42::ElementType support_type() const {
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            return y26_stage42::ElementType::FLOAT32;
        }
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            return y26_stage42::ElementType::UINT8;
        }
        throw std::runtime_error("unsupported tensor element type");
    }

    y26_stage42::TensorView view() const {
        return y26_stage42::TensorView{support_type(), shape, bytes.data(), bytes.size()};
    }
};

struct MetricStats {
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;
    double median = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double cv_pct = 0.0;
};

using CompareResult = y26_stage42::Comparison;

struct PipelineTiming {
    double prefix_us = 0.0;
    double custom_model4_us = 0.0;
    double suffix_us = 0.0;
    double layout_in_us = 0.0;
    double layout_out_us = 0.0;
    double total_us = 0.0;
};

struct Summary {
    MetricStats total;
    PipelineTiming mean_timing;
    CompareResult final_compare;
    CompareResult model4_compare;
    unsigned long long final_checksum = 0;
    unsigned long long model4_checksum = 0;
    int affinity_ok = 1;
    std::vector<PipelineTiming> repeat_timings;
};

const OrtApi* g_ort = nullptr;

void check_status(OrtStatus* status, const char* context) {
    if (status == nullptr) {
        return;
    }
    const char* msg = g_ort->GetErrorMessage(status);
    std::string text = std::string(context) + ": " + (msg ? msg : "unknown ORT error");
    g_ort->ReleaseStatus(status);
    throw std::runtime_error(text);
}

std::string require_value(int& i, int argc, char** argv, const char* name) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
    }
    return argv[++i];
}

Options parse_options(int argc, char** argv) {
    Options options {};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mode") {
            options.mode = require_value(i, argc, argv, "--mode");
        } else if (arg == "--model") {
            options.model_path = require_value(i, argc, argv, "--model");
        } else if (arg == "--cut-dir") {
            options.cut_dir = require_value(i, argc, argv, "--cut-dir");
        } else if (arg == "--input-npy") {
            options.input_npy = require_value(i, argc, argv, "--input-npy");
        } else if (arg == "--model4-input-npy") {
            options.model4_input_npy = require_value(i, argc, argv, "--model4-input-npy");
        } else if (arg == "--expected-output-npy") {
            options.expected_output_npy = require_value(i, argc, argv, "--expected-output-npy");
        } else if (arg == "--expected-model4-output-npy") {
            options.expected_model4_output_npy = require_value(i, argc, argv, "--expected-model4-output-npy");
        } else if (arg == "--profile-cuts-tsv") {
            options.profile_cuts_tsv = require_value(i, argc, argv, "--profile-cuts-tsv");
        } else if (arg == "--ort-input-name") {
            options.ort_input_name = require_value(i, argc, argv, "--ort-input-name");
        } else if (arg == "--ort-output-name") {
            options.ort_output_name = require_value(i, argc, argv, "--ort-output-name");
        } else if (arg == "--dump-custom-model4-nhwc") {
            options.dump_custom_model4_nhwc = require_value(i, argc, argv, "--dump-custom-model4-nhwc");
        } else if (arg == "--dump-custom-model4-nchw") {
            options.dump_custom_model4_nchw = require_value(i, argc, argv, "--dump-custom-model4-nchw");
        } else if (arg == "--dump-ort-model4-nchw") {
            options.dump_ort_model4_nchw = require_value(i, argc, argv, "--dump-ort-model4-nchw");
        } else if (arg == "--dump-model4-input-nchw") {
            options.dump_model4_input_nchw = require_value(i, argc, argv, "--dump-model4-input-nchw");
        } else if (arg == "--dump-final-output") {
            options.dump_final_output = require_value(i, argc, argv, "--dump-final-output");
        } else if (arg == "--custom-mode") {
            options.custom_mode = require_value(i, argc, argv, "--custom-mode");
        } else if (arg == "--ort-opt-level") {
            options.ort_opt_level = require_value(i, argc, argv, "--ort-opt-level");
        } else if (arg == "--ort-execution-mode") {
            options.ort_execution_mode = require_value(i, argc, argv, "--ort-execution-mode");
        } else if (arg == "--ort-intra-threads") {
            options.ort_intra_threads = std::max(1, std::stoi(require_value(i, argc, argv, "--ort-intra-threads")));
        } else if (arg == "--ort-inter-threads") {
            options.ort_inter_threads = std::max(1, std::stoi(require_value(i, argc, argv, "--ort-inter-threads")));
        } else if (arg == "--ort-enable-profiling") {
            options.ort_enable_profiling = std::stoi(require_value(i, argc, argv, "--ort-enable-profiling"));
        } else if (arg == "--ort-profile-prefix") {
            options.profile_prefix = require_value(i, argc, argv, "--ort-profile-prefix");
        } else if (arg == "--ort-log-severity") {
            options.ort_log_severity = std::stoi(require_value(i, argc, argv, "--ort-log-severity"));
        } else if (arg == "--ort-log-verbosity") {
            options.ort_log_verbosity = std::stoi(require_value(i, argc, argv, "--ort-log-verbosity"));
        } else if (arg == "--ort-memory-pattern") {
            options.ort_memory_pattern = std::stoi(require_value(i, argc, argv, "--ort-memory-pattern"));
        } else if (arg == "--ort-cpu-arena") {
            options.ort_cpu_arena = std::stoi(require_value(i, argc, argv, "--ort-cpu-arena"));
        } else if (arg == "--ort-thread-spinning") {
            options.ort_thread_spinning = std::stoi(require_value(i, argc, argv, "--ort-thread-spinning"));
        } else if (arg == "--warmup") {
            options.warmup = std::max(0, std::stoi(require_value(i, argc, argv, "--warmup")));
        } else if (arg == "--runs") {
            options.runs = std::max(1, std::stoi(require_value(i, argc, argv, "--runs")));
        } else if (arg == "--repeats") {
            options.repeats = std::max(1, std::stoi(require_value(i, argc, argv, "--repeats")));
        } else if (arg == "--thread-branch0") {
            options.thread_branch0 = std::max(1, std::stoi(require_value(i, argc, argv, "--thread-branch0")));
        } else if (arg == "--thread-branch1") {
            options.thread_branch1 = std::max(0, std::stoi(require_value(i, argc, argv, "--thread-branch1")));
        } else if (arg == "--thread-model4-cv2") {
            options.thread_model4_cv2 =
                std::max(0, std::stoi(require_value(i, argc, argv, "--thread-model4-cv2")));
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    const std::array<std::string, 5> valid_modes = {"validate", "benchmark", "profile", "same-input-model4", "ort-only"};
    if (std::find(valid_modes.begin(), valid_modes.end(), options.mode) == valid_modes.end()) {
        throw std::runtime_error("--mode must be validate, benchmark, profile, same-input-model4, or ort-only");
    }
    if (options.custom_mode != "ime_threaded" && options.custom_mode != "scalar") {
        throw std::runtime_error("--custom-mode must be ime_threaded or scalar");
    }
    (void)y26_stage42::parse_ort_optimization_level(options.ort_opt_level);
    (void)y26_stage42::parse_ort_execution_mode(options.ort_execution_mode);
    for (const auto value : {options.ort_enable_profiling, options.ort_memory_pattern,
                             options.ort_cpu_arena, options.ort_thread_spinning}) {
        if (value != 0 && value != 1) {
            throw std::runtime_error("ORT boolean options must be 0 or 1");
        }
    }
    if (options.model_path.empty()) {
        throw std::runtime_error("--model is required");
    }
    if (options.mode == "same-input-model4") {
        if (options.cut_dir.empty() || options.model4_input_npy.empty()) {
            throw std::runtime_error("same-input-model4 requires --cut-dir and --model4-input-npy");
        }
    } else if (options.mode == "ort-only") {
        if (options.input_npy.empty() || options.ort_input_name.empty() || options.ort_output_name.empty()) {
            throw std::runtime_error("ort-only requires --input-npy, --ort-input-name, and --ort-output-name");
        }
    } else if (options.cut_dir.empty() || options.input_npy.empty()) {
        throw std::runtime_error("validate/benchmark/profile require --cut-dir and --input-npy");
    }
    return options;
}

std::vector<char> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open " + path);
    }
    return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string parse_header_string(const std::vector<char>& data, std::size_t& offset) {
    if (data.size() < 10 || std::memcmp(data.data(), "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("unsupported npy magic");
    }
    const unsigned major = static_cast<unsigned char>(data[6]);
    offset = 8;
    std::uint32_t header_len = 0;
    if (major == 1) {
        if (offset + 2 > data.size()) {
            throw std::runtime_error("truncated npy v1 header length");
        }
        header_len = static_cast<std::uint16_t>(static_cast<unsigned char>(data[offset]) |
                                                (static_cast<unsigned char>(data[offset + 1]) << 8));
        offset += 2;
    } else if (major == 2 || major == 3) {
        if (offset + 4 > data.size()) {
            throw std::runtime_error("truncated npy v2/v3 header length");
        }
        header_len = static_cast<std::uint32_t>(static_cast<unsigned char>(data[offset]) |
                                                (static_cast<unsigned char>(data[offset + 1]) << 8) |
                                                (static_cast<unsigned char>(data[offset + 2]) << 16) |
                                                (static_cast<unsigned char>(data[offset + 3]) << 24));
        offset += 4;
    } else {
        throw std::runtime_error("unsupported npy version");
    }
    if (offset + header_len > data.size()) {
        throw std::runtime_error("truncated npy header");
    }
    std::string header(data.data() + offset, data.data() + offset + header_len);
    offset += header_len;
    return header;
}

std::vector<std::int64_t> parse_shape(const std::string& header) {
    const std::size_t shape_key = header.find("shape");
    const std::size_t begin = header.find('(', shape_key);
    const std::size_t end = header.find(')', begin);
    if (shape_key == std::string::npos || begin == std::string::npos || end == std::string::npos) {
        throw std::runtime_error("npy shape not found");
    }
    std::vector<std::int64_t> shape;
    std::string number;
    for (std::size_t i = begin + 1; i < end; ++i) {
        const char c = header[i];
        if ((c >= '0' && c <= '9') || (c == '-' && number.empty())) {
            number.push_back(c);
        } else if (!number.empty()) {
            shape.push_back(std::stoll(number));
            number.clear();
        }
    }
    if (!number.empty()) {
        shape.push_back(std::stoll(number));
    }
    (void)y26_stage42::checked_element_count(shape);
    return shape;
}

Tensor load_npy(const std::string& path) {
    const std::vector<char> data = read_file(path);
    std::size_t payload_offset = 0;
    const std::string header = parse_header_string(data, payload_offset);
    if (header.find("fortran_order") == std::string::npos || header.find("True") != std::string::npos) {
        throw std::runtime_error("Fortran-order npy is not supported: " + path);
    }
    Tensor tensor;
    const bool is_f32 = header.find("'<f4'") != std::string::npos || header.find("\"<f4\"") != std::string::npos;
    const bool is_u8 = header.find("'|u1'") != std::string::npos || header.find("\"|u1\"") != std::string::npos ||
                       header.find("'<u1'") != std::string::npos || header.find("\"<u1\"") != std::string::npos;
    if (is_f32) {
        tensor.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    } else if (is_u8) {
        tensor.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    } else {
        throw std::runtime_error("unsupported npy dtype or endianness: " + path);
    }
    tensor.shape = parse_shape(header);
    const std::size_t expected = y26_stage42::checked_byte_count(tensor.shape, tensor.support_type());
    if (payload_offset + expected != data.size()) {
        throw std::runtime_error("npy payload size mismatch: " + path);
    }
    tensor.bytes.assign(reinterpret_cast<const std::uint8_t*>(data.data() + payload_offset),
                        reinterpret_cast<const std::uint8_t*>(data.data() + payload_offset + expected));
    return tensor;
}

bool write_file(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    if (path.empty()) {
        return true;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(out);
}

unsigned long long checksum_bytes(const std::vector<std::uint8_t>& bytes) {
    unsigned long long sum = 0;
    for (std::uint8_t b : bytes) {
        sum += b;
    }
    return sum;
}

MetricStats stats_from_values(const std::vector<double>& values) {
    MetricStats stats {};
    if (values.empty()) {
        return stats;
    }
    stats.min = *std::min_element(values.begin(), values.end());
    stats.max = *std::max_element(values.begin(), values.end());
    stats.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    double sum_sq = 0.0;
    for (double value : values) {
        const double delta = value - stats.mean;
        sum_sq += delta * delta;
    }
    stats.stddev = values.size() > 1 ? std::sqrt(sum_sq / static_cast<double>(values.size() - 1)) : 0.0;
    stats.cv_pct = stats.mean > 0.0 ? 100.0 * stats.stddev / stats.mean : 0.0;
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    auto percentile = [&](double quantile) {
        const double position = quantile * static_cast<double>(sorted.size() - 1U);
        const std::size_t lower = static_cast<std::size_t>(std::floor(position));
        const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
        if (lower == upper) {
            return sorted[lower];
        }
        const double fraction = position - static_cast<double>(lower);
        return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
    };
    stats.median = percentile(0.50);
    stats.p90 = percentile(0.90);
    stats.p95 = percentile(0.95);
    return stats;
}

CompareResult compare_tensors(const Tensor& actual, const Tensor& expected) {
    return y26_stage42::compare_tensors(actual.view(), expected.view());
}

std::string shape_string(const Tensor& tensor) {
    std::ostringstream out;
    for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
        if (i != 0) {
            out << 'x';
        }
        out << tensor.shape[i];
    }
    return out.str();
}

std::string histogram_string(const std::map<int, std::size_t>& histogram) {
    std::ostringstream out;
    bool first = true;
    for (const auto& [diff, count] : histogram) {
        if (!first) {
            out << ',';
        }
        first = false;
        out << diff << ':' << count;
    }
    return out.str();
}

void print_comparison(const char* label,
                      const Tensor& actual,
                      const Tensor& expected,
                      const CompareResult& result) {
    std::cout << std::fixed << std::setprecision(9)
              << "stage42_compare"
              << " label=" << label
              << " structural_valid=" << (result.structurally_valid ? 1 : 0)
              << " structural_error=" << (result.structural_error.empty() ? "none" : result.structural_error)
              << " shape=" << shape_string(actual)
              << " expected_shape=" << shape_string(expected)
              << " dtype=" << static_cast<int>(actual.type)
              << " expected_dtype=" << static_cast<int>(expected.type)
              << " elements=" << result.element_count
              << " mismatches=" << result.mismatch_count
              << " mismatch_ratio=" << result.mismatch_ratio
              << " max_abs_diff=" << result.max_abs_diff
              << " mean_abs_diff=" << result.mean_abs_diff
              << " rmse=" << result.rmse
              << " first_mismatch="
              << (result.first_mismatch_index.has_value() ? std::to_string(*result.first_mismatch_index) : "none")
              << " p50_abs_diff=" << result.p50_abs_diff
              << " p90_abs_diff=" << result.p90_abs_diff
              << " p95_abs_diff=" << result.p95_abs_diff
              << " p99_abs_diff=" << result.p99_abs_diff
              << " p999_abs_diff=" << result.p999_abs_diff
              << " lhs_min=" << result.lhs.min
              << " lhs_max=" << result.lhs.max
              << " lhs_mean=" << result.lhs.mean
              << " lhs_sum=" << result.lhs.sum
              << " lhs_nonfinite=" << result.lhs.nonfinite_count
              << " rhs_min=" << result.rhs.min
              << " rhs_max=" << result.rhs.max
              << " rhs_mean=" << result.rhs.mean
              << " rhs_sum=" << result.rhs.sum
              << " rhs_nonfinite=" << result.rhs.nonfinite_count
              << " byte_equal=" << (result.byte_equal ? 1 : 0)
              << " signed_diff_hist=" << (result.signed_difference_histogram.empty()
                                                   ? "not-applicable"
                                                   : histogram_string(result.signed_difference_histogram))
              << " float_policy=exact-finite_same-nan_same-signed-inf"
              << "\n";
}

struct OrtSessionHolder {
    OrtSession* session = nullptr;
    OrtSessionHolder() = default;
    OrtSessionHolder(const OrtSessionHolder&) = delete;
    OrtSessionHolder& operator=(const OrtSessionHolder&) = delete;
    OrtSessionHolder(OrtSessionHolder&& other) noexcept : session(std::exchange(other.session, nullptr)) {}
    OrtSessionHolder& operator=(OrtSessionHolder&& other) noexcept {
        if (this != &other) {
            if (session != nullptr) {
                g_ort->ReleaseSession(session);
            }
            session = std::exchange(other.session, nullptr);
        }
        return *this;
    }
    ~OrtSessionHolder() {
        if (session != nullptr) {
            g_ort->ReleaseSession(session);
        }
    }
};

struct OrtMemoryInfoHolder {
    OrtMemoryInfo* info = nullptr;
    ~OrtMemoryInfoHolder() {
        if (info != nullptr) {
            g_ort->ReleaseMemoryInfo(info);
        }
    }
};

struct OrtEnvHolder {
    OrtEnv* env = nullptr;
    ~OrtEnvHolder() {
        if (env != nullptr && g_ort != nullptr) {
            g_ort->ReleaseEnv(env);
        }
    }
};

struct OrtValueHolder {
    OrtValue* value = nullptr;
    OrtValueHolder() = default;
    explicit OrtValueHolder(OrtValue* ptr) : value(ptr) {}
    OrtValueHolder(const OrtValueHolder&) = delete;
    OrtValueHolder& operator=(const OrtValueHolder&) = delete;
    OrtValueHolder(OrtValueHolder&& other) noexcept : value(std::exchange(other.value, nullptr)) {}
    ~OrtValueHolder() {
        if (value != nullptr) {
            g_ort->ReleaseValue(value);
        }
    }
};

GraphOptimizationLevel graph_optimization_level(const Options& options) {
    switch (y26_stage42::parse_ort_optimization_level(options.ort_opt_level)) {
        case y26_stage42::OrtOptimizationLevel::DISABLE:
            return ORT_DISABLE_ALL;
        case y26_stage42::OrtOptimizationLevel::BASIC:
            return ORT_ENABLE_BASIC;
        case y26_stage42::OrtOptimizationLevel::EXTENDED:
            return ORT_ENABLE_EXTENDED;
        case y26_stage42::OrtOptimizationLevel::ALL:
            return ORT_ENABLE_ALL;
    }
    throw std::runtime_error("unsupported graph optimization level");
}

ExecutionMode execution_mode(const Options& options) {
    return y26_stage42::parse_ort_execution_mode(options.ort_execution_mode) ==
                   y26_stage42::OrtExecutionMode::SEQUENTIAL
               ? ORT_SEQUENTIAL
               : ORT_PARALLEL;
}

OrtSessionHolder create_session(OrtEnv* env,
                                const std::string& path,
                                const Options& options,
                                const std::string& session_label) {
    OrtSessionOptions* opts = nullptr;
    check_status(g_ort->CreateSessionOptions(&opts), "CreateSessionOptions");
    auto release_options = [&]() {
        if (opts != nullptr) {
            g_ort->ReleaseSessionOptions(opts);
            opts = nullptr;
        }
    };
    try {
        check_status(g_ort->SetIntraOpNumThreads(opts, options.ort_intra_threads), "SetIntraOpNumThreads");
        check_status(g_ort->SetInterOpNumThreads(opts, options.ort_inter_threads), "SetInterOpNumThreads");
        check_status(g_ort->SetSessionGraphOptimizationLevel(opts, graph_optimization_level(options)),
                     "SetSessionGraphOptimizationLevel");
        check_status(g_ort->SetSessionExecutionMode(opts, execution_mode(options)), "SetSessionExecutionMode");
        check_status(g_ort->SetSessionLogId(opts, session_label.c_str()), "SetSessionLogId");
        check_status(g_ort->SetSessionLogSeverityLevel(opts, options.ort_log_severity),
                     "SetSessionLogSeverityLevel");
        check_status(g_ort->SetSessionLogVerbosityLevel(opts, options.ort_log_verbosity),
                     "SetSessionLogVerbosityLevel");
        if (options.ort_memory_pattern != 0) {
            check_status(g_ort->EnableMemPattern(opts), "EnableMemPattern");
        } else {
            check_status(g_ort->DisableMemPattern(opts), "DisableMemPattern");
        }
        if (options.ort_cpu_arena != 0) {
            check_status(g_ort->EnableCpuMemArena(opts), "EnableCpuMemArena");
        } else {
            check_status(g_ort->DisableCpuMemArena(opts), "DisableCpuMemArena");
        }
        const char* spinning = options.ort_thread_spinning != 0 ? "1" : "0";
        check_status(g_ort->AddSessionConfigEntry(opts, "session.intra_op.allow_spinning", spinning),
                     "AddSessionConfigEntry intra spinning");
        check_status(g_ort->AddSessionConfigEntry(opts, "session.inter_op.allow_spinning", spinning),
                     "AddSessionConfigEntry inter spinning");
        if (options.ort_enable_profiling != 0) {
            const std::string profile_path = options.profile_prefix + "_" + session_label;
            check_status(g_ort->EnableProfiling(opts, profile_path.c_str()), "EnableProfiling");
        }
    } catch (...) {
        release_options();
        throw;
    }
    OrtSessionHolder holder;
    OrtStatus* status = g_ort->CreateSession(env, path.c_str(), opts, &holder.session);
    release_options();
    check_status(status, ("CreateSession " + path).c_str());
    return holder;
}

OrtValue* make_input_value(const Tensor& tensor, OrtMemoryInfo* memory_info) {
    const std::size_t expected_bytes = y26_stage42::checked_byte_count(tensor.shape, tensor.support_type());
    if (tensor.bytes.size() != expected_bytes) {
        throw std::runtime_error("input tensor shape/type does not match byte size");
    }
    OrtValue* value = nullptr;
    check_status(g_ort->CreateTensorWithDataAsOrtValue(memory_info,
                                                       const_cast<std::uint8_t*>(tensor.bytes.data()),
                                                       tensor.bytes.size(),
                                                       tensor.shape.data(),
                                                       tensor.shape.size(),
                                                       tensor.type,
                                                       &value),
                 "CreateTensorWithDataAsOrtValue");
    return value;
}

Tensor copy_output_tensor(OrtValue* output) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    check_status(g_ort->GetTensorTypeAndShape(output, &info), "GetTensorTypeAndShape");
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    check_status(g_ort->GetTensorElementType(info, &type), "GetTensorElementType");
    std::size_t rank = 0;
    check_status(g_ort->GetDimensionsCount(info, &rank), "GetDimensionsCount");
    Tensor tensor;
    tensor.type = type;
    tensor.shape.resize(rank);
    check_status(g_ort->GetDimensions(info, tensor.shape.data(), rank), "GetDimensions");
    const std::size_t checked_count = tensor.element_count();
    std::size_t element_count = 0;
    check_status(g_ort->GetTensorShapeElementCount(info, &element_count), "GetTensorShapeElementCount");
    g_ort->ReleaseTensorTypeAndShapeInfo(info);
    if (checked_count != element_count) {
        throw std::runtime_error("ORT output element count disagrees with concrete dimensions");
    }
    const std::size_t element_size = tensor.element_size();
    if (element_size == 0) {
        throw std::runtime_error("unsupported output tensor element type");
    }
    void* data = nullptr;
    check_status(g_ort->GetTensorMutableData(output, &data), "GetTensorMutableData");
    const auto* first = static_cast<const std::uint8_t*>(data);
    const std::size_t byte_count = y26_stage42::checked_byte_count(tensor.shape, tensor.support_type());
    tensor.bytes.assign(first, first + byte_count);
    return tensor;
}

Tensor run_one_output(OrtSession* session,
                      OrtMemoryInfo* memory_info,
                      const char* input_name,
                      const Tensor& input,
                      const char* output_name) {
    OrtValueHolder input_value(make_input_value(input, memory_info));
    OrtValueHolder output_value;
    const char* input_names[] = {input_name};
    const char* output_names[] = {output_name};
    OrtStatus* status =
        g_ort->Run(session, nullptr, input_names, &input_value.value, 1, output_names, 1, &output_value.value);
    check_status(status, "OrtRun");
    return copy_output_tensor(output_value.value);
}

void nchw_u8_to_nhwc(const Tensor& tensor, std::vector<std::uint8_t>& out) {
    if (tensor.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 || tensor.shape.size() != 4) {
        throw std::runtime_error("expected uint8 NCHW tensor");
    }
    const std::size_t count = tensor.element_count();
    if (tensor.bytes.size() != count || out.size() != count) {
        throw std::runtime_error("NCHW to NHWC buffer size mismatch");
    }
    const int n = static_cast<int>(tensor.shape[0]);
    const int c = static_cast<int>(tensor.shape[1]);
    const int h = static_cast<int>(tensor.shape[2]);
    const int w = static_cast<int>(tensor.shape[3]);
    for (int bn = 0; bn < n; ++bn) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                for (int ch = 0; ch < c; ++ch) {
                    const std::size_t src = ((static_cast<std::size_t>(bn) * c + ch) * h + y) * w + x;
                    const std::size_t dst = ((static_cast<std::size_t>(bn) * h + y) * w + x) * c + ch;
                    out[dst] = tensor.bytes[src];
                }
            }
        }
    }
}

void nhwc_u8_to_nchw(const std::vector<std::uint8_t>& bytes, Tensor& tensor) {
    const std::vector<std::int64_t>& shape_nchw = tensor.shape;
    if (shape_nchw.size() != 4) {
        throw std::runtime_error("expected 4D NCHW shape");
    }
    const std::size_t count = y26_stage42::checked_element_count(shape_nchw);
    if (bytes.size() != count || tensor.bytes.size() != count ||
        tensor.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
        throw std::runtime_error("NHWC to NCHW buffer size/type mismatch");
    }
    const int n = static_cast<int>(shape_nchw[0]);
    const int c = static_cast<int>(shape_nchw[1]);
    const int h = static_cast<int>(shape_nchw[2]);
    const int w = static_cast<int>(shape_nchw[3]);
    for (int bn = 0; bn < n; ++bn) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                for (int ch = 0; ch < c; ++ch) {
                    const std::size_t src = ((static_cast<std::size_t>(bn) * h + y) * w + x) * c + ch;
                    const std::size_t dst = ((static_cast<std::size_t>(bn) * c + ch) * h + y) * w + x;
                    tensor.bytes[dst] = bytes[src];
                }
            }
        }
    }
}

Y26Stage16Model4C2fConfig fullshape_config(int activation_mode) {
    const auto& fixture = y26_stage16_model4_c2f_fixture::kSyntheticSeededFixture;
    Y26Stage16Model4C2fConfig cfg = stage16_config_from_fixture(
        fixture, activation_mode, Y26_STAGE16_MERGE_MODE_STAGE39_BRANCH3X3_FAST_PACK);
    constexpr int kFullH = 80;
    constexpr int kFullW = 80;
    cfg.stage15.stage14.model4_cv1.params.input_h = kFullH;
    cfg.stage15.stage14.model4_cv1.params.input_w = kFullW;
    cfg.stage15.branch0.params.input_h = kFullH;
    cfg.stage15.branch0.params.input_w = kFullW;
    cfg.branch1.params.input_h = kFullH;
    cfg.branch1.params.input_w = kFullW;
    cfg.model4_cv2.params.input_h = kFullH;
    cfg.model4_cv2.params.input_w = kFullW;
    return cfg;
}

class Model4Runner {
public:
    Model4Runner(const Options& options)
        : use_ime_(options.custom_mode == "ime_threaded" ? 1 : 0),
          use_threaded_(options.custom_mode == "ime_threaded" ? 1 : 0),
          output_quantize_mode_(options.custom_mode == "ime_threaded"
                                    ? Y26_STAGE16_OUTPUT_QUANTIZE_STAGE38_RVV_DIRECT_STORE
                                    : Y26_STAGE16_OUTPUT_QUANTIZE_SCALAR),
          cfg_(fullshape_config(options.custom_mode == "ime_threaded" ? Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT
                                                                       : Y26_ACTIVATION_MODE_INT8_LUT)) {
        int status = y26_stage16_model4_c2f_prepare_cut(&cfg_, &ws_);
        if (status != Y26_CONV_STATUS_SUCCESS) {
            throw std::runtime_error("model4 prepare_cut failed");
        }
        if (use_threaded_ != 0) {
            status = y26_stage16_model4_c2f_prepare_cut_threaded_branch0(&cfg_, &ws_, options.thread_branch0);
            if (status != Y26_CONV_STATUS_SUCCESS) {
                throw std::runtime_error("model4 prepare branch0 threaded failed");
            }
            status = y26_stage16_model4_c2f_prepare_cut_threaded_branch1(&cfg_, &ws_, options.thread_branch1);
            if (status != Y26_CONV_STATUS_SUCCESS) {
                throw std::runtime_error("model4 prepare branch1 threaded failed");
            }
            status = y26_stage16_model4_c2f_prepare_cut_threaded_model4_cv2(&cfg_, &ws_, options.thread_model4_cv2);
            if (status != Y26_CONV_STATUS_SUCCESS) {
                throw std::runtime_error("model4 prepare cv2 threaded failed");
            }
        }
        input_nhwc_.resize(1U * 80U * 80U * 64U);
        output_nhwc_.resize(y26_stage16_model4_c2f_output_count(&cfg_));
        output_nchw_.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        output_nchw_.shape = {1, 128, 80, 80};
        output_nchw_.bytes.resize(output_nhwc_.size());
    }

    ~Model4Runner() {
        y26_stage16_model4_c2f_release(&ws_);
    }

    const Tensor& run(const Tensor& model4_input_nchw,
                      double* layout_in_us,
                      double* layout_out_us,
                      double* custom_us) {
        const auto layout_begin = Clock::now();
        nchw_u8_to_nhwc(model4_input_nchw, input_nhwc_);
        const auto layout_mid = Clock::now();
        Y26Stage16TimingUs timing {};
        const int status = y26_stage16_model4_c2f_run_cut_u8_output(
            &cfg_,
            &ws_,
            input_nhwc_.data(),
            output_nhwc_.data(),
            use_ime_,
            use_threaded_,
            output_quantize_mode_,
            &timing);
        if (status != Y26_CONV_STATUS_SUCCESS) {
            throw std::runtime_error("model4 custom run failed");
        }
        const auto custom_end = Clock::now();
        nhwc_u8_to_nchw(output_nhwc_, output_nchw_);
        const auto layout_end = Clock::now();
        if (layout_in_us != nullptr) {
            *layout_in_us += std::chrono::duration<double, std::micro>(layout_mid - layout_begin).count();
        }
        if (layout_out_us != nullptr) {
            *layout_out_us += std::chrono::duration<double, std::micro>(layout_end - custom_end).count();
        }
        if (custom_us != nullptr) {
            *custom_us += timing.total_us;
        }
        return output_nchw_;
    }

    int affinity_ok() const {
        return use_threaded_ != 0 ? y26_stage16_model4_c2f_threaded_worker_affinity_ok(&ws_) : 1;
    }

    const std::vector<std::uint8_t>& last_output_nhwc() const {
        return output_nhwc_;
    }

private:
    int use_ime_ = 1;
    int use_threaded_ = 1;
    int output_quantize_mode_ = Y26_STAGE16_OUTPUT_QUANTIZE_STAGE38_RVV_DIRECT_STORE;
    Y26Stage16Model4C2fConfig cfg_ {};
    Y26Stage16Model4C2fWorkspace ws_ {};
    std::vector<std::uint8_t> input_nhwc_;
    std::vector<std::uint8_t> output_nhwc_;
    Tensor output_nchw_;
};

struct ProfileCut {
    std::string block_id;
    std::string path;
    std::string input_name;
    std::string output_name;
};

std::vector<std::string> split_tsv_line(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : line) {
        if (c == '\t') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

std::vector<ProfileCut> read_profile_cuts(const std::string& path) {
    std::vector<ProfileCut> cuts;
    if (path.empty()) {
        return cuts;
    }
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open profile cuts tsv");
    }
    std::string line;
    if (!std::getline(in, line)) {
        return cuts;
    }
    const auto header = split_tsv_line(line);
    auto index_of = [&](const char* name) -> int {
        const auto it = std::find(header.begin(), header.end(), name);
        return it == header.end() ? -1 : static_cast<int>(it - header.begin());
    };
    const int block_idx = index_of("block_id");
    const int path_idx = index_of("cut_path");
    const int input_idx = index_of("input_name");
    const int output_idx = index_of("output_name");
    while (std::getline(in, line)) {
        const auto parts = split_tsv_line(line);
        if (block_idx < 0 || path_idx < 0 || input_idx < 0 || output_idx < 0 ||
            parts.size() <= static_cast<std::size_t>(std::max({block_idx, path_idx, input_idx, output_idx}))) {
            continue;
        }
        cuts.push_back(ProfileCut{parts[block_idx], parts[path_idx], parts[input_idx], parts[output_idx]});
    }
    return cuts;
}

Summary run_pipeline_benchmark(OrtSession* prefix,
                               OrtSession* suffix,
                               OrtMemoryInfo* memory,
                               Model4Runner& custom_model4,
                               const Tensor& input,
                               const Options& options,
                               Tensor& last_final,
                               Tensor& last_custom_model4,
                               Tensor& last_model4_input) {
    auto run_once = [&](PipelineTiming* timing, bool retain_outputs) {
        const auto total_begin = Clock::now();
        const auto prefix_begin = Clock::now();
        Tensor model4_input = run_one_output(prefix, memory, kInputName, input, kModel4InputName);
        const auto prefix_end = Clock::now();
        double layout_in_us = 0.0;
        double layout_out_us = 0.0;
        double custom_us = 0.0;
        const Tensor& custom_output =
            custom_model4.run(model4_input, &layout_in_us, &layout_out_us, &custom_us);
        const auto suffix_begin = Clock::now();
        Tensor final_output = run_one_output(suffix, memory, kModel4OutputName, custom_output, kFinalOutputName);
        const auto suffix_end = Clock::now();
        if (timing != nullptr) {
            timing->prefix_us += std::chrono::duration<double, std::micro>(prefix_end - prefix_begin).count();
            timing->custom_model4_us += custom_us;
            timing->layout_in_us += layout_in_us;
            timing->layout_out_us += layout_out_us;
            timing->suffix_us += std::chrono::duration<double, std::micro>(suffix_end - suffix_begin).count();
            timing->total_us += std::chrono::duration<double, std::micro>(suffix_end - total_begin).count();
        }
        if (retain_outputs) {
            last_model4_input = std::move(model4_input);
            last_custom_model4 = custom_output;
            last_final = std::move(final_output);
        }
    };

    for (int i = 0; i < options.warmup; ++i) {
        run_once(nullptr, false);
    }

    std::vector<double> repeat_totals;
    PipelineTiming timing_sum {};
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        PipelineTiming repeat_timing {};
        for (int run = 0; run < options.runs; ++run) {
            run_once(&repeat_timing, false);
        }
        const double denom = static_cast<double>(options.runs);
        repeat_timing.prefix_us /= denom;
        repeat_timing.custom_model4_us /= denom;
        repeat_timing.suffix_us /= denom;
        repeat_timing.layout_in_us /= denom;
        repeat_timing.layout_out_us /= denom;
        repeat_timing.total_us /= denom;
        repeat_totals.push_back(repeat_timing.total_us);
        timing_sum.prefix_us += repeat_timing.prefix_us;
        timing_sum.custom_model4_us += repeat_timing.custom_model4_us;
        timing_sum.suffix_us += repeat_timing.suffix_us;
        timing_sum.layout_in_us += repeat_timing.layout_in_us;
        timing_sum.layout_out_us += repeat_timing.layout_out_us;
        timing_sum.total_us += repeat_timing.total_us;
        std::cout << std::fixed << std::setprecision(6)
                  << "stage42_pipeline_repeat"
                  << " repeat=" << repeat
                  << " prefix_us=" << repeat_timing.prefix_us
                  << " layout_in_us=" << repeat_timing.layout_in_us
                  << " custom_model4_us=" << repeat_timing.custom_model4_us
                  << " layout_out_us=" << repeat_timing.layout_out_us
                  << " suffix_us=" << repeat_timing.suffix_us
                  << " total_us=" << repeat_timing.total_us
                  << "\n";
    }
    const double repeats = static_cast<double>(options.repeats);
    timing_sum.prefix_us /= repeats;
    timing_sum.custom_model4_us /= repeats;
    timing_sum.suffix_us /= repeats;
    timing_sum.layout_in_us /= repeats;
    timing_sum.layout_out_us /= repeats;
    timing_sum.total_us /= repeats;
    run_once(nullptr, true);

    Summary summary {};
    summary.total = stats_from_values(repeat_totals);
    summary.mean_timing = timing_sum;
    summary.final_checksum = checksum_bytes(last_final.bytes);
    summary.model4_checksum = checksum_bytes(last_custom_model4.bytes);
    summary.affinity_ok = custom_model4.affinity_ok();
    return summary;
}

MetricStats profile_cut(OrtEnv* env,
                        OrtMemoryInfo* memory,
                        const ProfileCut& cut,
                        const Tensor& model4_output,
                        const Options& options,
                        int warmup,
                        int runs,
                        int repeats) {
    OrtSessionHolder session = create_session(env, cut.path, options, "profile_" + cut.block_id);
    auto run_once = [&]() -> Tensor {
        return run_one_output(session.session, memory, cut.input_name.c_str(), model4_output, cut.output_name.c_str());
    };
    for (int i = 0; i < warmup; ++i) {
        (void)run_once();
    }
    std::vector<double> repeat_values;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto begin = Clock::now();
        Tensor last;
        for (int run = 0; run < runs; ++run) {
            last = run_once();
        }
        const auto end = Clock::now();
        (void)last;
        repeat_values.push_back(std::chrono::duration<double, std::micro>(end - begin).count() /
                                static_cast<double>(runs));
    }
    return stats_from_values(repeat_values);
}

bool comparison_passes(const CompareResult& result) {
    return result.structurally_valid && result.mismatch_count == 0;
}

void print_runtime_contract(const OrtApiBase* base, const Options& options) {
    std::cout << "stage42_runtime_contract"
              << " ort_version=" << (base->GetVersionString != nullptr ? base->GetVersionString() : "unavailable")
              << " ort_build_info=" << (g_ort->GetBuildInfoString != nullptr ? g_ort->GetBuildInfoString() : "unavailable")
              << " ort_api_version=" << ORT_API_VERSION
              << " opt_level=" << options.ort_opt_level
              << " execution_mode=" << options.ort_execution_mode
              << " intra_threads=" << options.ort_intra_threads
              << " inter_threads=" << options.ort_inter_threads
              << " memory_pattern=" << options.ort_memory_pattern
              << " cpu_arena=" << options.ort_cpu_arena
              << " thread_spinning=" << options.ort_thread_spinning
              << " profiling=" << options.ort_enable_profiling
              << " log_severity=" << options.ort_log_severity
              << " log_verbosity=" << options.ort_log_verbosity
              << "\n";

    char** providers = nullptr;
    int provider_count = 0;
    check_status(g_ort->GetAvailableProviders(&providers, &provider_count), "GetAvailableProviders");
    for (int i = 0; i < provider_count; ++i) {
        std::cout << "stage42_provider_inventory"
                  << " provider=" << providers[i]
                  << " registered=1 appended=0 priority=default"
                  << " node_assignment=not-observable-from-registration"
                  << "\n";
    }
    check_status(g_ort->ReleaseAvailableProviders(providers, provider_count), "ReleaseAvailableProviders");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const OrtApiBase* base = OrtGetApiBase();
        if (base == nullptr || base->GetApi == nullptr) {
            throw std::runtime_error("OrtGetApiBase returned no API base");
        }
        g_ort = base->GetApi(ORT_API_VERSION);
        if (g_ort == nullptr) {
            throw std::runtime_error("ORT API version is unavailable from linked runtime");
        }
        const Options options = parse_options(argc, argv);
        print_runtime_contract(base, options);
        OrtEnvHolder env;
        check_status(g_ort->CreateEnv(static_cast<OrtLoggingLevel>(options.ort_log_severity), "stage42", &env.env),
                     "CreateEnv");
        OrtMemoryInfoHolder memory;
        check_status(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory.info),
                     "CreateCpuMemoryInfo");

        if (options.mode == "ort-only") {
            OrtSessionHolder session = create_session(env.env, options.model_path, options, "ort_only");
            Tensor input = load_npy(options.input_npy);
            auto run_once = [&]() {
                return run_one_output(session.session,
                                      memory.info,
                                      options.ort_input_name.c_str(),
                                      input,
                                      options.ort_output_name.c_str());
            };
            for (int i = 0; i < options.warmup; ++i) {
                (void)run_once();
            }
            Tensor output;
            std::vector<double> repeat_us;
            for (int repeat = 0; repeat < options.repeats; ++repeat) {
                const auto begin = Clock::now();
                for (int run = 0; run < options.runs; ++run) {
                    output = run_once();
                }
                const auto end = Clock::now();
                const double mean_us =
                    std::chrono::duration<double, std::micro>(end - begin).count() /
                    static_cast<double>(options.runs);
                repeat_us.push_back(mean_us);
                std::cout << std::fixed << std::setprecision(6)
                          << "stage42_ort_only_repeat"
                          << " repeat=" << repeat
                          << " mean_us=" << mean_us
                          << "\n";
            }
            const MetricStats ort_timing = stats_from_values(repeat_us);
            bool pass = true;
            if (!options.expected_output_npy.empty()) {
                const Tensor expected = load_npy(options.expected_output_npy);
                const CompareResult comparison = compare_tensors(output, expected);
                print_comparison("ort_only_vs_expected", output, expected, comparison);
                pass = comparison_passes(comparison);
            }
            const bool dumped = write_file(options.dump_final_output, output.bytes);
            std::cout << "stage42_ort_only"
                      << " output_name=" << options.ort_output_name
                      << " shape=" << shape_string(output)
                      << " dtype=" << static_cast<int>(output.type)
                      << " checksum=" << checksum_bytes(output.bytes)
                      << " dump_ok=" << (dumped ? 1 : 0)
                      << "\n";
            std::cout << std::fixed << std::setprecision(6)
                      << "stage42_ort_only_benchmark"
                      << " warmup=" << options.warmup
                      << " runs=" << options.runs
                      << " repeats=" << options.repeats
                      << " reference_comparison_in_hot_loop=0"
                      << " file_io_in_hot_loop=0"
                      << " mean_us=" << ort_timing.mean
                      << " stddev_us=" << ort_timing.stddev
                      << " cv_pct=" << ort_timing.cv_pct
                      << " min_us=" << ort_timing.min
                      << " max_us=" << ort_timing.max
                      << " median_us=" << ort_timing.median
                      << " p90_us=" << ort_timing.p90
                      << " p95_us=" << ort_timing.p95
                      << "\n";
            return pass && dumped ? 0 : 1;
        }

        const std::string prefix_path = options.cut_dir + "/prefix_images_to_model4_input.onnx";
        const std::string model4_path = options.cut_dir + "/model4_input_to_model4_output.onnx";
        const std::string suffix_path = options.cut_dir + "/suffix_model4_output_to_output0.onnx";
        Model4Runner custom_model4(options);

        if (options.mode == "same-input-model4") {
            OrtSessionHolder model4_ort = create_session(env.env, model4_path, options, "model4_same_input");
            const Tensor model4_input = load_npy(options.model4_input_npy);
            if (model4_input.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
                model4_input.shape != std::vector<std::int64_t>({1, 64, 80, 80})) {
                throw std::runtime_error("same-input model4 tensor must be uint8 NCHW 1x64x80x80");
            }
            const Tensor ort_output =
                run_one_output(model4_ort.session, memory.info, kModel4InputName, model4_input, kModel4OutputName);
            const int fenv_before = std::fegetround();
            double layout_in_us = 0.0;
            double layout_out_us = 0.0;
            double custom_us = 0.0;
            const Tensor custom_output =
                custom_model4.run(model4_input, &layout_in_us, &layout_out_us, &custom_us);
            const int fenv_after = std::fegetround();
            const CompareResult custom_vs_ort = compare_tensors(custom_output, ort_output);
            print_comparison("custom_model4_vs_local_ort", custom_output, ort_output, custom_vs_ort);

            for (int i = 0; i < options.warmup; ++i) {
                double warmup_layout_in = 0.0;
                double warmup_layout_out = 0.0;
                double warmup_custom = 0.0;
                (void)custom_model4.run(model4_input,
                                        &warmup_layout_in,
                                        &warmup_layout_out,
                                        &warmup_custom);
            }
            std::vector<double> custom_repeat_us;
            std::vector<double> layout_in_repeat_us;
            std::vector<double> layout_out_repeat_us;
            for (int repeat = 0; repeat < options.repeats; ++repeat) {
                double repeat_layout_in = 0.0;
                double repeat_layout_out = 0.0;
                double repeat_custom = 0.0;
                for (int run = 0; run < options.runs; ++run) {
                    double run_layout_in = 0.0;
                    double run_layout_out = 0.0;
                    double run_custom = 0.0;
                    (void)custom_model4.run(model4_input,
                                            &run_layout_in,
                                            &run_layout_out,
                                            &run_custom);
                    repeat_layout_in += run_layout_in;
                    repeat_layout_out += run_layout_out;
                    repeat_custom += run_custom;
                }
                const double denom = static_cast<double>(options.runs);
                repeat_layout_in /= denom;
                repeat_layout_out /= denom;
                repeat_custom /= denom;
                layout_in_repeat_us.push_back(repeat_layout_in);
                layout_out_repeat_us.push_back(repeat_layout_out);
                custom_repeat_us.push_back(repeat_custom);
                std::cout << std::fixed << std::setprecision(6)
                          << "stage42_model4_custom_repeat"
                          << " repeat=" << repeat
                          << " layout_in_us=" << repeat_layout_in
                          << " custom_model4_us=" << repeat_custom
                          << " layout_out_us=" << repeat_layout_out
                          << "\n";
            }
            const MetricStats custom_timing = stats_from_values(custom_repeat_us);
            const MetricStats layout_in_timing = stats_from_values(layout_in_repeat_us);
            const MetricStats layout_out_timing = stats_from_values(layout_out_repeat_us);

            bool custom_expected_pass = true;
            bool ort_expected_pass = true;
            if (!options.expected_model4_output_npy.empty()) {
                const Tensor expected = load_npy(options.expected_model4_output_npy);
                const CompareResult custom_vs_expected = compare_tensors(custom_output, expected);
                const CompareResult ort_vs_expected = compare_tensors(ort_output, expected);
                print_comparison("custom_model4_vs_fixed_expected", custom_output, expected, custom_vs_expected);
                print_comparison("local_ort_model4_vs_fixed_expected", ort_output, expected, ort_vs_expected);
                custom_expected_pass = comparison_passes(custom_vs_expected);
                ort_expected_pass = comparison_passes(ort_vs_expected);
            }
            const bool dump_input = write_file(options.dump_model4_input_nchw, model4_input.bytes);
            const bool dump_ort = write_file(options.dump_ort_model4_nchw, ort_output.bytes);
            const bool dump_custom_nchw = write_file(options.dump_custom_model4_nchw, custom_output.bytes);
            const bool dump_custom_nhwc =
                write_file(options.dump_custom_model4_nhwc, custom_model4.last_output_nhwc());
            std::cout << std::fixed << std::setprecision(6)
                      << "stage42_model4_same_input"
                      << " custom_mode=" << options.custom_mode
                      << " local_ort_vs_fixed_expected_pass=" << (ort_expected_pass ? 1 : 0)
                      << " custom_vs_fixed_expected_pass=" << (custom_expected_pass ? 1 : 0)
                      << " custom_vs_local_ort_pass=" << (comparison_passes(custom_vs_ort) ? 1 : 0)
                      << " fenv_before=" << fenv_before
                      << " fenv_after=" << fenv_after
                      << " layout_in_us=" << layout_in_us
                      << " custom_model4_us=" << custom_us
                      << " layout_out_us=" << layout_out_us
                      << " affinity_ok=" << custom_model4.affinity_ok()
                      << " input_checksum=" << checksum_bytes(model4_input.bytes)
                      << " ort_checksum=" << checksum_bytes(ort_output.bytes)
                      << " custom_checksum=" << checksum_bytes(custom_output.bytes)
                      << " dump_input_ok=" << (dump_input ? 1 : 0)
                      << " dump_ort_ok=" << (dump_ort ? 1 : 0)
                      << " dump_custom_nchw_ok=" << (dump_custom_nchw ? 1 : 0)
                      << " dump_custom_nhwc_ok=" << (dump_custom_nhwc ? 1 : 0)
                      << "\n";
            std::cout << std::fixed << std::setprecision(6)
                      << "stage42_model4_custom_benchmark"
                      << " custom_mode=" << options.custom_mode
                      << " warmup=" << options.warmup
                      << " runs=" << options.runs
                      << " repeats=" << options.repeats
                      << " ort_reference_in_hot_loop=0"
                      << " file_io_in_hot_loop=0"
                      << " mean_custom_model4_us=" << custom_timing.mean
                      << " stddev_custom_model4_us=" << custom_timing.stddev
                      << " cv_custom_model4_pct=" << custom_timing.cv_pct
                      << " min_custom_model4_us=" << custom_timing.min
                      << " max_custom_model4_us=" << custom_timing.max
                      << " median_custom_model4_us=" << custom_timing.median
                      << " p90_custom_model4_us=" << custom_timing.p90
                      << " p95_custom_model4_us=" << custom_timing.p95
                      << " mean_layout_in_us=" << layout_in_timing.mean
                      << " mean_layout_out_us=" << layout_out_timing.mean
                      << "\n";
            return custom_expected_pass && custom_model4.affinity_ok() == 1 && fenv_before == fenv_after &&
                           dump_input && dump_ort && dump_custom_nchw && dump_custom_nhwc
                       ? 0
                       : 1;
        }

        const Tensor input = load_npy(options.input_npy);
        if (input.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            input.shape != std::vector<std::int64_t>({1, 3, 640, 640})) {
            throw std::runtime_error("full input must be float32 NCHW 1x3x640x640");
        }

        if (options.mode == "benchmark") {
            OrtSessionHolder prefix = create_session(env.env, prefix_path, options, "benchmark_prefix");
            OrtSessionHolder suffix = create_session(env.env, suffix_path, options, "benchmark_suffix");
            Tensor last_final;
            Tensor last_custom_model4;
            Tensor last_model4_input;
            Summary summary = run_pipeline_benchmark(prefix.session,
                                                     suffix.session,
                                                     memory.info,
                                                     custom_model4,
                                                     input,
                                                     options,
                                                     last_final,
                                                     last_custom_model4,
                                                     last_model4_input);
            const bool dump_input = write_file(options.dump_model4_input_nchw, last_model4_input.bytes);
            const bool dump_model4 = write_file(options.dump_custom_model4_nhwc, custom_model4.last_output_nhwc());
            const bool dump_model4_nchw = write_file(options.dump_custom_model4_nchw, last_custom_model4.bytes);
            const bool dump_final = write_file(options.dump_final_output, last_final.bytes);
            const double attributed = summary.mean_timing.prefix_us + summary.mean_timing.layout_in_us +
                                      summary.mean_timing.custom_model4_us + summary.mean_timing.layout_out_us +
                                      summary.mean_timing.suffix_us;
            const double attr_pct = summary.mean_timing.total_us > 0.0
                                        ? 100.0 * attributed / summary.mean_timing.total_us
                                        : 0.0;
            std::cout << std::fixed << std::setprecision(6)
                      << "stage42_pipeline_benchmark"
                      << " reference_in_hot_loop=0"
                      << " model4_ort_session_created=0"
                      << " file_io_in_hot_loop=0"
                      << " local_layout_allocation_in_hot_loop=0"
                      << " ort_output_allocation=runtime_managed_unavoidable"
                      << " warmup=" << options.warmup
                      << " runs=" << options.runs
                      << " repeats=" << options.repeats
                      << " custom_mode=" << options.custom_mode
                      << " affinity_ok=" << summary.affinity_ok
                      << " mean_total_us=" << summary.total.mean
                      << " stddev_total_us=" << summary.total.stddev
                      << " min_total_us=" << summary.total.min
                      << " max_total_us=" << summary.total.max
                      << " median_total_us=" << summary.total.median
                      << " p90_total_us=" << summary.total.p90
                      << " p95_total_us=" << summary.total.p95
                      << " cv_total_pct=" << summary.total.cv_pct
                      << " mean_prefix_us=" << summary.mean_timing.prefix_us
                      << " mean_layout_in_us=" << summary.mean_timing.layout_in_us
                      << " mean_custom_model4_us=" << summary.mean_timing.custom_model4_us
                      << " mean_layout_out_us=" << summary.mean_timing.layout_out_us
                      << " mean_suffix_us=" << summary.mean_timing.suffix_us
                      << " mean_attribution_pct=" << attr_pct
                      << " final_checksum=" << summary.final_checksum
                      << " model4_checksum=" << summary.model4_checksum
                      << " dump_input_ok=" << (dump_input ? 1 : 0)
                      << " dump_model4_ok=" << (dump_model4 ? 1 : 0)
                      << " dump_model4_nchw_ok=" << (dump_model4_nchw ? 1 : 0)
                      << " dump_final_ok=" << (dump_final ? 1 : 0)
                      << " note=selected_pipeline_timing_not_model_fps"
                      << "\n";
            return summary.affinity_ok == 1 && dump_input && dump_model4 && dump_model4_nchw && dump_final ? 0 : 1;
        }

        OrtSessionHolder full = create_session(env.env, options.model_path, options, "validate_full");
        OrtSessionHolder prefix = create_session(env.env, prefix_path, options, "validate_prefix");
        OrtSessionHolder model4_ort = create_session(env.env, model4_path, options, "validate_model4");
        OrtSessionHolder suffix = create_session(env.env, suffix_path, options, "validate_suffix");
        const Tensor full_output = run_one_output(full.session, memory.info, kInputName, input, kFinalOutputName);
        const Tensor expected = options.expected_output_npy.empty() ? full_output : load_npy(options.expected_output_npy);
        const Tensor model4_input = run_one_output(prefix.session, memory.info, kInputName, input, kModel4InputName);
        const Tensor ort_model4 =
            run_one_output(model4_ort.session, memory.info, kModel4InputName, model4_input, kModel4OutputName);
        const int fenv_before = std::fegetround();
        double layout_in_us = 0.0;
        double layout_out_us = 0.0;
        double custom_us = 0.0;
        const Tensor custom_model4_output =
            custom_model4.run(model4_input, &layout_in_us, &layout_out_us, &custom_us);
        const int fenv_after = std::fegetround();
        const Tensor custom_final =
            run_one_output(suffix.session, memory.info, kModel4OutputName, custom_model4_output, kFinalOutputName);

        const CompareResult full_vs_expected = compare_tensors(full_output, expected);
        const CompareResult custom_vs_ort = compare_tensors(custom_model4_output, ort_model4);
        const CompareResult custom_final_vs_full = compare_tensors(custom_final, full_output);
        print_comparison("full_ort_vs_expected", full_output, expected, full_vs_expected);
        print_comparison("custom_model4_vs_ort_model4", custom_model4_output, ort_model4, custom_vs_ort);
        print_comparison("custom_through_suffix_vs_full_ort", custom_final, full_output, custom_final_vs_full);

        const bool dump_input = write_file(options.dump_model4_input_nchw, model4_input.bytes);
        const bool dump_ort = write_file(options.dump_ort_model4_nchw, ort_model4.bytes);
        const bool dump_custom_nchw = write_file(options.dump_custom_model4_nchw, custom_model4_output.bytes);
        const bool dump_custom_nhwc =
            write_file(options.dump_custom_model4_nhwc, custom_model4.last_output_nhwc());
        const bool dump_final = write_file(options.dump_final_output, custom_final.bytes);
        std::cout << std::fixed << std::setprecision(6)
                  << "stage42_validate"
                  << " custom_mode=" << options.custom_mode
                  << " full_exact=" << (comparison_passes(full_vs_expected) ? 1 : 0)
                  << " model4_exact=" << (comparison_passes(custom_vs_ort) ? 1 : 0)
                  << " final_exact=" << (comparison_passes(custom_final_vs_full) ? 1 : 0)
                  << " fenv_before=" << fenv_before
                  << " fenv_after=" << fenv_after
                  << " layout_in_us=" << layout_in_us
                  << " custom_model4_us=" << custom_us
                  << " layout_out_us=" << layout_out_us
                  << " affinity_ok=" << custom_model4.affinity_ok()
                  << " full_checksum=" << checksum_bytes(full_output.bytes)
                  << " model4_ort_checksum=" << checksum_bytes(ort_model4.bytes)
                  << " model4_custom_checksum=" << checksum_bytes(custom_model4_output.bytes)
                  << " final_custom_checksum=" << checksum_bytes(custom_final.bytes)
                  << " dump_input_ok=" << (dump_input ? 1 : 0)
                  << " dump_ort_ok=" << (dump_ort ? 1 : 0)
                  << " dump_custom_nchw_ok=" << (dump_custom_nchw ? 1 : 0)
                  << " dump_custom_nhwc_ok=" << (dump_custom_nhwc ? 1 : 0)
                  << " dump_final_ok=" << (dump_final ? 1 : 0)
                  << "\n";

        if (options.mode == "profile") {
            const std::vector<ProfileCut> cuts = read_profile_cuts(options.profile_cuts_tsv);
            for (const ProfileCut& cut : cuts) {
                const MetricStats stats = profile_cut(env.env,
                                                      memory.info,
                                                      cut,
                                                      custom_model4_output,
                                                      options,
                                                      std::min(options.warmup, 2),
                                                      std::min(options.runs, 20),
                                                      std::min(options.repeats, 3));
                std::cout << "stage42_suffix_profile"
                          << " block_id=" << cut.block_id
                          << " output_name=" << cut.output_name
                          << " mean_us=" << stats.mean
                          << " stddev_us=" << stats.stddev
                          << " min_us=" << stats.min
                          << " max_us=" << stats.max
                          << " median_us=" << stats.median
                          << " p90_us=" << stats.p90
                          << " p95_us=" << stats.p95
                          << " cv_pct=" << stats.cv_pct
                          << " cumulative_session_diagnostic=1"
                          << "\n";
            }
        }

        return comparison_passes(full_vs_expected) && comparison_passes(custom_vs_ort) &&
                       comparison_passes(custom_final_vs_full) && custom_model4.affinity_ok() == 1 &&
                       fenv_before == fenv_after && dump_input && dump_ort && dump_custom_nchw &&
                       dump_custom_nhwc && dump_final
                   ? 0
                   : 1;
    } catch (const std::exception& exc) {
        std::cerr << "stage42_inprocess_runner_error: " << exc.what() << "\n";
        return 1;
    }
}
