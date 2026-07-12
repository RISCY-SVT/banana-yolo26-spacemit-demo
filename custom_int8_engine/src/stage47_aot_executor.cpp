#include "y26_k1x_stage47_aot.h"

#include "y26_k1x_activation.h"
#include "y26_k1x_conv_kernels.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace y26::stage47 {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

std::vector<std::string> split_tsv(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t begin = 0;
    for (;;) {
        const std::size_t end = line.find('\t', begin);
        fields.push_back(line.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return fields;
}

std::vector<std::unordered_map<std::string, std::string>> read_tsv(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open TSV: " + path.string());
    }
    std::string line;
    if (!std::getline(stream, line)) {
        throw std::runtime_error("empty TSV: " + path.string());
    }
    const std::vector<std::string> header = split_tsv(line);
    std::vector<std::unordered_map<std::string, std::string>> rows;
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> values = split_tsv(line);
        if (values.size() != header.size()) {
            throw std::runtime_error("malformed TSV row: " + path.string());
        }
        auto& row = rows.emplace_back();
        for (std::size_t index = 0; index < header.size(); ++index) {
            row.emplace(header[index], values[index]);
        }
    }
    return rows;
}

int integer(const std::unordered_map<std::string, std::string>& row, const char* key) {
    const auto found = row.find(key);
    if (found == row.end()) {
        throw std::runtime_error(std::string("missing integer field: ") + key);
    }
    int result = 0;
    const char* begin = found->second.data();
    const char* end = begin + found->second.size();
    const auto parsed = std::from_chars(begin, end, result);
    if (parsed.ec != std::errc() || parsed.ptr != end) {
        throw std::runtime_error(std::string("invalid integer field: ") + key);
    }
    return result;
}

std::size_t size_value(const std::unordered_map<std::string, std::string>& row, const char* key) {
    const auto found = row.find(key);
    if (found == row.end()) {
        throw std::runtime_error(std::string("missing size field: ") + key);
    }
    std::size_t result = 0;
    const char* begin = found->second.data();
    const char* end = begin + found->second.size();
    const auto parsed = std::from_chars(begin, end, result);
    if (parsed.ec != std::errc() || parsed.ptr != end) {
        throw std::runtime_error(std::string("invalid size field: ") + key);
    }
    return result;
}

float floating(const std::unordered_map<std::string, std::string>& row, const char* key) {
    const auto found = row.find(key);
    if (found == row.end()) {
        throw std::runtime_error(std::string("missing float field: ") + key);
    }
    std::size_t consumed = 0;
    const float result = std::stof(found->second, &consumed);
    if (consumed != found->second.size() || !std::isfinite(result)) {
        throw std::runtime_error(std::string("invalid float field: ") + key);
    }
    return result;
}

const std::string& text_field(const std::unordered_map<std::string, std::string>& row, const char* key) {
    const auto found = row.find(key);
    if (found == row.end()) {
        throw std::runtime_error(std::string("missing text field: ") + key);
    }
    return found->second;
}

template <typename T>
std::vector<T> read_binary(const std::filesystem::path& path, std::size_t expected_count) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open binary: " + path.string());
    }
    const std::streamsize bytes = stream.tellg();
    const std::size_t expected_bytes = expected_count * sizeof(T);
    if (bytes < 0 || static_cast<std::size_t>(bytes) != expected_bytes) {
        throw std::runtime_error("binary size mismatch: " + path.string());
    }
    stream.seekg(0);
    std::vector<T> values(expected_count);
    if (expected_bytes != 0 && !stream.read(reinterpret_cast<char*>(values.data()), bytes)) {
        throw std::runtime_error("binary read failed: " + path.string());
    }
    return values;
}

std::int8_t signed_storage(std::uint8_t code) {
    return static_cast<std::int8_t>(static_cast<int>(code) - 128);
}

std::uint8_t unsigned_code(std::int8_t storage) {
    return static_cast<std::uint8_t>(static_cast<int>(storage) + 128);
}

std::int8_t quantize_s8(float value, const TensorSpec& output) {
    return signed_storage(y26_quantize_u8_nearest_even_f32(value, output.scale, output.zero_point_u8));
}

float silu(float value) {
    return value / (1.0f + std::exp(-value));
}

int extract_json_integer(const std::filesystem::path& path, const std::string& key) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open JSON: " + path.string());
    }
    const std::string contents((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    const std::string quoted = "\"" + key + "\"";
    const std::size_t key_position = contents.find(quoted);
    if (key_position == std::string::npos) {
        throw std::runtime_error("missing JSON key: " + key);
    }
    const std::size_t colon = contents.find(':', key_position + quoted.size());
    const std::size_t begin = contents.find_first_of("-0123456789", colon + 1);
    const std::size_t end = contents.find_first_not_of("0123456789-", begin);
    if (colon == std::string::npos || begin == std::string::npos) {
        throw std::runtime_error("invalid JSON integer: " + key);
    }
    return std::stoi(contents.substr(begin, end - begin));
}

}  // namespace

struct AotExecutor::Impl {
    struct TensorEntry {
        int id = -1;
        std::string name;
        TensorSpec spec;
        std::size_t offset = 0;
        std::size_t bytes = 0;
    };

    struct Operation {
        int index = -1;
        std::string kind;
        std::string name;
        std::array<int, 3> inputs {-1, -1, -1};
        std::array<int, 2> outputs {-1, -1};
        IntegratedConv conv;
        std::array<std::int8_t, 256> lut {};
        std::array<std::array<std::int8_t, 256>, 3> concat_luts {};
        std::vector<std::int8_t> add_lut;
    };

    std::filesystem::path package;
    std::vector<TensorEntry> tensors;
    std::vector<Operation> operations;
    std::vector<std::int8_t> arena;
    std::vector<std::int8_t> saved_input;
    std::unique_ptr<WorkerPool> pool;
    int input_id = -1;
    int output_id = -1;
    std::size_t total_packed_weight_bytes = 0;
    std::string last_error;
    bool ready = false;

    const TensorEntry& tensor(int id) const {
        if (id < 0 || static_cast<std::size_t>(id) >= tensors.size() || tensors[static_cast<std::size_t>(id)].id != id) {
            throw std::runtime_error("invalid tensor id");
        }
        return tensors[static_cast<std::size_t>(id)];
    }

    std::int8_t* data(int id) {
        const TensorEntry& value = tensor(id);
        return arena.data() + value.offset;
    }

    const std::int8_t* data(int id) const {
        const TensorEntry& value = tensor(id);
        return arena.data() + value.offset;
    }
};

namespace {

void build_lut_operation(AotExecutor::Impl& executor, AotExecutor::Impl::Operation& operation) {
    const auto& input = executor.tensor(operation.inputs[0]).spec;
    const auto& output = executor.tensor(operation.outputs[0]).spec;
    for (int code = 0; code < 256; ++code) {
        const float value = static_cast<float>(code - input.zero_point_u8) * input.scale;
        operation.lut[static_cast<std::size_t>(code)] = quantize_s8(silu(value), output);
    }
}

void build_add_operation(AotExecutor::Impl& executor, AotExecutor::Impl::Operation& operation) {
    const auto& left = executor.tensor(operation.inputs[0]).spec;
    const auto& right = executor.tensor(operation.inputs[1]).spec;
    const auto& output = executor.tensor(operation.outputs[0]).spec;
    if (left.h != right.h || left.w != right.w || left.c != right.c ||
        left.h != output.h || left.w != output.w || left.c != output.c) {
        throw std::runtime_error("add tensor shape mismatch");
    }
    operation.add_lut.resize(256U * 256U);
    for (int left_code = 0; left_code < 256; ++left_code) {
        const float left_value = static_cast<float>(left_code - left.zero_point_u8) * left.scale;
        for (int right_code = 0; right_code < 256; ++right_code) {
            const float right_value = static_cast<float>(right_code - right.zero_point_u8) * right.scale;
            operation.add_lut[static_cast<std::size_t>(left_code) * 256U + right_code] =
                quantize_s8(left_value + silu(right_value), output);
        }
    }
}

void build_concat_operation(AotExecutor::Impl& executor, AotExecutor::Impl::Operation& operation) {
    const auto& output = executor.tensor(operation.outputs[0]).spec;
    int channels = 0;
    for (std::size_t input_index = 0; input_index < operation.inputs.size(); ++input_index) {
        if (operation.inputs[input_index] < 0) {
            continue;
        }
        const auto& input = executor.tensor(operation.inputs[input_index]).spec;
        if (input.h != output.h || input.w != output.w) {
            throw std::runtime_error("concat spatial shape mismatch");
        }
        channels += input.c;
        for (int code = 0; code < 256; ++code) {
            const float value = static_cast<float>(code - input.zero_point_u8) * input.scale;
            operation.concat_luts[input_index][static_cast<std::size_t>(code)] = quantize_s8(value, output);
        }
    }
    if (channels != output.c) {
        throw std::runtime_error("concat channel mismatch");
    }
}

void run_lut(AotExecutor::Impl& executor, const AotExecutor::Impl::Operation& operation) {
    const auto& output = executor.tensor(operation.outputs[0]);
    const std::int8_t* source = executor.data(operation.inputs[0]);
    std::int8_t* destination = executor.data(operation.outputs[0]);
    for (std::size_t index = 0; index < output.bytes; ++index) {
        destination[index] = operation.lut[unsigned_code(source[index])];
    }
}

void run_add(AotExecutor::Impl& executor, const AotExecutor::Impl::Operation& operation) {
    const auto& output = executor.tensor(operation.outputs[0]);
    const std::int8_t* left = executor.data(operation.inputs[0]);
    const std::int8_t* right = executor.data(operation.inputs[1]);
    std::int8_t* destination = executor.data(operation.outputs[0]);
    for (std::size_t index = 0; index < output.bytes; ++index) {
        destination[index] = operation.add_lut[static_cast<std::size_t>(unsigned_code(left[index])) * 256U +
                                               unsigned_code(right[index])];
    }
}

void run_concat(AotExecutor::Impl& executor, const AotExecutor::Impl::Operation& operation) {
    const auto& output = executor.tensor(operation.outputs[0]);
    std::int8_t* destination = executor.data(operation.outputs[0]);
    const std::size_t pixels = static_cast<std::size_t>(output.spec.h) * output.spec.w;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        int output_channel = 0;
        for (std::size_t input_index = 0; input_index < operation.inputs.size(); ++input_index) {
            const int tensor_id = operation.inputs[input_index];
            if (tensor_id < 0) {
                continue;
            }
            const auto& input = executor.tensor(tensor_id);
            const std::int8_t* source = executor.data(tensor_id) + pixel * input.spec.c;
            for (int channel = 0; channel < input.spec.c; ++channel) {
                destination[pixel * output.spec.c + output_channel++] =
                    operation.concat_luts[input_index][unsigned_code(source[channel])];
            }
        }
    }
}

}  // namespace

AotExecutor::AotExecutor() : impl_(std::make_unique<Impl>()) {}
AotExecutor::~AotExecutor() = default;
AotExecutor::AotExecutor(AotExecutor&&) noexcept = default;
AotExecutor& AotExecutor::operator=(AotExecutor&&) noexcept = default;

int AotExecutor::prepare(const std::filesystem::path& package_dir, int worker_capacity) {
    if (worker_capacity < 1 || worker_capacity > 4) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    impl_->last_error.clear();
    try {
        Impl prepared;
        prepared.package = std::filesystem::canonical(package_dir);
        const auto tensor_rows = read_tsv(prepared.package / "tensors.tsv");
        prepared.tensors.reserve(tensor_rows.size());
        std::size_t required_arena = 0;
        for (const auto& row : tensor_rows) {
            Impl::TensorEntry tensor;
            tensor.id = integer(row, "id");
            tensor.name = text_field(row, "logical_name");
            tensor.spec.h = integer(row, "h");
            tensor.spec.w = integer(row, "w");
            tensor.spec.c = integer(row, "c");
            tensor.spec.scale = floating(row, "scale");
            tensor.spec.zero_point_u8 = integer(row, "zero_point");
            tensor.offset = size_value(row, "arena_offset");
            tensor.bytes = size_value(row, "bytes");
            const std::size_t expected = static_cast<std::size_t>(tensor.spec.h) * tensor.spec.w * tensor.spec.c;
            if (tensor.id != static_cast<int>(prepared.tensors.size()) || tensor.bytes != expected ||
                tensor.offset > std::numeric_limits<std::size_t>::max() - tensor.bytes) {
                throw std::runtime_error("invalid tensor descriptor");
            }
            required_arena = std::max(required_arena, tensor.offset + tensor.bytes);
            prepared.tensors.push_back(std::move(tensor));
        }
        const int declared_arena = extract_json_integer(prepared.package / "package.json", "arena_bytes");
        prepared.input_id = extract_json_integer(prepared.package / "package.json", "input_tensor_id");
        prepared.output_id = extract_json_integer(prepared.package / "package.json", "output_tensor_id");
        if (declared_arena < 0 || static_cast<std::size_t>(declared_arena) < required_arena) {
            throw std::runtime_error("arena descriptor too small");
        }
        prepared.arena.resize(static_cast<std::size_t>(declared_arena));
        prepared.pool = std::make_unique<WorkerPool>(worker_capacity);

        const auto operation_rows = read_tsv(prepared.package / "ops.tsv");
        prepared.operations.reserve(operation_rows.size());
        for (const auto& row : operation_rows) {
            Impl::Operation operation;
            operation.index = integer(row, "index");
            operation.kind = text_field(row, "kind");
            operation.name = text_field(row, "name");
            operation.inputs = {integer(row, "input0"), integer(row, "input1"), integer(row, "input2")};
            operation.outputs = {integer(row, "output0"), integer(row, "output1")};
            if (operation.index != static_cast<int>(prepared.operations.size())) {
                throw std::runtime_error("operation index is not static/contiguous");
            }
            if (operation.kind == "lut") {
                build_lut_operation(prepared, operation);
            } else if (operation.kind == "add_silu") {
                build_add_operation(prepared, operation);
            } else if (operation.kind == "concat") {
                build_concat_operation(prepared, operation);
            } else if (operation.kind == "conv") {
                const auto& input = prepared.tensor(operation.inputs[0]);
                const auto& output0 = prepared.tensor(operation.outputs[0]);
                ConvSpec spec;
                spec.input = input.spec;
                spec.output_h = output0.spec.h;
                spec.output_w = output0.spec.w;
                spec.output_c = integer(row, "segment0_channel_count") + integer(row, "segment1_channel_count");
                spec.kernel_h = integer(row, "kernel_h");
                spec.kernel_w = integer(row, "kernel_w");
                spec.stride_h = integer(row, "stride_h");
                spec.stride_w = integer(row, "stride_w");
                spec.pad_h = integer(row, "pad_h");
                spec.pad_w = integer(row, "pad_w");
                spec.group = integer(row, "group");
                spec.conv_output_scale = floating(row, "conv_output_scale");
                spec.conv_output_zero_point_u8 = integer(row, "conv_output_zero_point");
                const std::size_t weight_count = size_value(row, "weight_count");
                auto weights = read_binary<std::int8_t>(prepared.package / text_field(row, "weights_file"), weight_count);
                auto scales = read_binary<float>(prepared.package / text_field(row, "weight_scales_file"),
                                                 static_cast<std::size_t>(spec.output_c));
                auto bias = read_binary<std::int32_t>(prepared.package / text_field(row, "bias_file"),
                                                      static_cast<std::size_t>(spec.output_c));
                spec.weights_ohwi_s8 = weights.data();
                spec.weight_count = weights.size();
                spec.weight_scales = scales.data();
                spec.weight_scale_count = scales.size();
                spec.bias_i32 = bias.data();
                spec.bias_count = bias.size();
                OutputSegmentSpec segment0;
                segment0.channel_begin = integer(row, "segment0_channel_begin");
                segment0.channel_count = integer(row, "segment0_channel_count");
                segment0.output = output0.spec;
                segment0.silu = text_field(row, "segment0_activation") == "silu";
                spec.segments.push_back(segment0);
                if (operation.outputs[1] >= 0) {
                    const auto& output1 = prepared.tensor(operation.outputs[1]);
                    OutputSegmentSpec segment1;
                    segment1.channel_begin = integer(row, "segment1_channel_begin");
                    segment1.channel_count = integer(row, "segment1_channel_count");
                    segment1.output = output1.spec;
                    segment1.silu = text_field(row, "segment1_activation") == "silu";
                    spec.segments.push_back(segment1);
                }
                const int status = operation.conv.prepare(spec);
                if (status != Y26_CONV_STATUS_SUCCESS) {
                    throw std::runtime_error("conv prepare failed: " + operation.name);
                }
                prepared.total_packed_weight_bytes += operation.conv.prepared_weight_bytes();
            } else {
                throw std::runtime_error("unsupported AOT operation: " + operation.kind);
            }
            prepared.operations.push_back(std::move(operation));
        }
        if (extract_json_integer(prepared.package / "package.json", "operation_count") !=
            static_cast<int>(prepared.operations.size())) {
            throw std::runtime_error("operation count mismatch");
        }
        prepared.ready = true;
        *impl_ = std::move(prepared);
        return Y26_CONV_STATUS_SUCCESS;
    } catch (const std::exception& error) {
        impl_->last_error = error.what();
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
}

int AotExecutor::set_input(const std::int8_t* source, std::size_t bytes) {
    if (!impl_ || !impl_->ready || source == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    const auto& input = impl_->tensor(impl_->input_id);
    if (bytes != input.bytes) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    impl_->saved_input.assign(source, source + bytes);
    std::memcpy(impl_->data(impl_->input_id), source, bytes);
    return Y26_CONV_STATUS_SUCCESS;
}

int AotExecutor::run(const std::int8_t* input_nhwc_s8,
                     std::int8_t* output_nhwc_s8,
                     const RunOptions& options,
                     ExecutorTiming* timing) {
    if (!impl_ || !impl_->ready || !impl_->pool || options.workers < 1 ||
        options.workers > impl_->pool->capacity()) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    if (input_nhwc_s8 != nullptr) {
        const auto& input = impl_->tensor(impl_->input_id);
        std::memcpy(impl_->data(impl_->input_id), input_nhwc_s8, input.bytes);
    } else if (!impl_->saved_input.empty()) {
        const auto& input = impl_->tensor(impl_->input_id);
        if (impl_->saved_input.size() != input.bytes) {
            return Y26_CONV_STATUS_INVALID_ARGUMENT;
        }
        std::memcpy(impl_->data(impl_->input_id), impl_->saved_input.data(), input.bytes);
    }
    if (timing != nullptr) {
        *timing = {};
        timing->operations.reserve(impl_->operations.size());
    }
    const int final_operation = options.stop_after_operation < 0
                                    ? static_cast<int>(impl_->operations.size()) - 1
                                    : std::min(options.stop_after_operation,
                                               static_cast<int>(impl_->operations.size()) - 1);
    if (final_operation < 0) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    const auto total_begin = Clock::now();
    for (int operation_index = 0; operation_index <= final_operation; ++operation_index) {
        Impl::Operation& operation = impl_->operations[static_cast<std::size_t>(operation_index)];
        const auto begin = Clock::now();
        IntegratedTiming conv_timing;
        int status = Y26_CONV_STATUS_SUCCESS;
        if (operation.kind == "conv") {
            std::array<std::int8_t*, 2> outputs {impl_->data(operation.outputs[0]), nullptr};
            std::size_t output_count = 1;
            if (operation.outputs[1] >= 0) {
                outputs[1] = impl_->data(operation.outputs[1]);
                output_count = 2;
            }
            status = operation.conv.run(*impl_->pool,
                                        impl_->data(operation.inputs[0]),
                                        outputs,
                                        output_count,
                                        options,
                                        timing != nullptr ? &conv_timing : nullptr);
        } else if (operation.kind == "lut") {
            run_lut(*impl_, operation);
        } else if (operation.kind == "add_silu") {
            run_add(*impl_, operation);
        } else if (operation.kind == "concat") {
            run_concat(*impl_, operation);
        }
        const double duration = elapsed_us(begin, Clock::now());
        if (status != Y26_CONV_STATUS_SUCCESS) {
            return status;
        }
        if (timing != nullptr) {
            OperationTiming row;
            row.operation_index = operation.index;
            row.name = operation.name;
            row.kind = operation.kind;
            row.total_us = duration;
            row.gather_pack_us = conv_timing.gather_pack_us;
            row.vmadot_us = conv_timing.vmadot_us;
            row.fused_epilogue_us = conv_timing.fused_epilogue_us;
            timing->operations.push_back(std::move(row));
            if (operation.kind == "conv") timing->conv_us += duration;
            else if (operation.kind == "lut") timing->lut_us += duration;
            else if (operation.kind == "add_silu") timing->add_us += duration;
            else if (operation.kind == "concat") timing->concat_us += duration;
        }
    }
    if (timing != nullptr) {
        timing->total_us = elapsed_us(total_begin, Clock::now());
    }
    if (output_nhwc_s8 != nullptr && final_operation == static_cast<int>(impl_->operations.size()) - 1) {
        const auto& output = impl_->tensor(impl_->output_id);
        std::memcpy(output_nhwc_s8, impl_->data(impl_->output_id), output.bytes);
    }
    return Y26_CONV_STATUS_SUCCESS;
}

int AotExecutor::copy_tensor(int tensor_id, std::int8_t* destination, std::size_t bytes) const {
    if (!impl_ || !impl_->ready || destination == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    try {
        const auto& tensor = impl_->tensor(tensor_id);
        if (bytes != tensor.bytes) {
            return Y26_CONV_STATUS_INVALID_ARGUMENT;
        }
        std::memcpy(destination, impl_->data(tensor_id), bytes);
        return Y26_CONV_STATUS_SUCCESS;
    } catch (const std::exception&) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
}

const TensorSpec* AotExecutor::tensor_spec(int tensor_id) const noexcept {
    if (!impl_ || tensor_id < 0 || static_cast<std::size_t>(tensor_id) >= impl_->tensors.size()) {
        return nullptr;
    }
    return &impl_->tensors[static_cast<std::size_t>(tensor_id)].spec;
}

std::size_t AotExecutor::tensor_bytes(int tensor_id) const noexcept {
    if (!impl_ || tensor_id < 0 || static_cast<std::size_t>(tensor_id) >= impl_->tensors.size()) {
        return 0;
    }
    return impl_->tensors[static_cast<std::size_t>(tensor_id)].bytes;
}

std::size_t AotExecutor::arena_bytes() const noexcept { return impl_ ? impl_->arena.size() : 0; }
std::size_t AotExecutor::packed_weight_bytes() const noexcept { return impl_ ? impl_->total_packed_weight_bytes : 0; }
int AotExecutor::operation_count() const noexcept { return impl_ ? static_cast<int>(impl_->operations.size()) : 0; }
int AotExecutor::input_tensor_id() const noexcept { return impl_ ? impl_->input_id : -1; }
int AotExecutor::output_tensor_id() const noexcept { return impl_ ? impl_->output_id : -1; }
bool AotExecutor::worker_affinity_ok() const noexcept { return impl_ && impl_->pool && impl_->pool->affinity_ok(); }
const std::string& AotExecutor::last_error() const noexcept {
    static const std::string empty;
    return impl_ ? impl_->last_error : empty;
}

}  // namespace y26::stage47
