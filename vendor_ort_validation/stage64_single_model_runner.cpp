#include <onnxruntime_cxx_api.h>
#include <spacemit_ort_env.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Options {
    std::string provider;
    std::string model;
    std::string input;
    std::string output;
};

std::string require_value(int& index, int argc, char** argv,
                          const char* option) {
    if (++index >= argc) {
        throw std::runtime_error(std::string("missing value for ") + option);
    }
    return argv[index];
}

Options parse_options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--provider") {
            result.provider = require_value(index, argc, argv, "--provider");
        } else if (argument == "--model") {
            result.model = require_value(index, argc, argv, "--model");
        } else if (argument == "--input") {
            result.input = require_value(index, argc, argv, "--input");
        } else if (argument == "--output") {
            result.output = require_value(index, argc, argv, "--output");
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: stage64_single_model_runner --provider cpu|spacemit "
                   "--model FILE --input FILE --output FILE\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (result.provider != "cpu" && result.provider != "spacemit") {
        throw std::runtime_error("--provider must be cpu or spacemit");
    }
    if (result.model.empty() || result.input.empty() || result.output.empty()) {
        throw std::runtime_error("--model, --input, and --output are required");
    }
    return result;
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
            throw std::runtime_error(
                "unsupported tensor element type: " +
                std::to_string(static_cast<int>(type)));
    }
}

std::size_t element_count(const std::vector<std::int64_t>& shape) {
    std::size_t result = 1;
    for (const std::int64_t dimension : shape) {
        if (dimension < 0) {
            throw std::runtime_error("dynamic tensor dimension is unsupported");
        }
        const auto value = static_cast<std::size_t>(dimension);
        if (value != 0 &&
            result > std::numeric_limits<std::size_t>::max() / value) {
            throw std::overflow_error("tensor element count overflow");
        }
        result *= value;
    }
    return result;
}

std::vector<std::uint8_t> read_exact(const std::string& path,
                                     std::size_t expected_bytes) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open input: " + path);
    }
    const std::streamoff length = stream.tellg();
    if (length < 0 || static_cast<std::size_t>(length) != expected_bytes) {
        throw std::runtime_error("input byte count mismatch");
    }
    std::vector<std::uint8_t> data(expected_bytes);
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(data.data()),
                static_cast<std::streamsize>(data.size()));
    if (!stream) {
        throw std::runtime_error("cannot read input: " + path);
    }
    return data;
}

void write_exact(const std::string& path, const void* data, std::size_t bytes) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open output: " + path);
    }
    stream.write(static_cast<const char*>(data),
                 static_cast<std::streamsize>(bytes));
    if (!stream) {
        throw std::runtime_error("cannot write output: " + path);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "stage64-minimal");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);
        if (options.provider == "spacemit") {
            std::unordered_map<std::string, std::string> provider_options;
            Ort::ThrowOnError(Ort::SessionOptionsSpaceMITEnvInit(
                session_options, provider_options));
        }

        Ort::Session session(environment, options.model.c_str(), session_options);
        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
            throw std::runtime_error("runner requires one input and one output");
        }

        Ort::AllocatorWithDefaultOptions allocator;
        const auto input_name = session.GetInputNameAllocated(0, allocator);
        const auto output_name = session.GetOutputNameAllocated(0, allocator);
        const auto input_type_info = session.GetInputTypeInfo(0);
        const auto input_tensor_info =
            input_type_info.GetTensorTypeAndShapeInfo();
        const auto input_shape = input_tensor_info.GetShape();
        const auto input_type = input_tensor_info.GetElementType();
        const std::size_t input_bytes =
            element_count(input_shape) * element_size(input_type);
        auto input_data = read_exact(options.input, input_bytes);

        const auto memory =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input = Ort::Value::CreateTensor(
            memory, input_data.data(), input_data.size(), input_shape.data(),
            input_shape.size(), input_type);
        const char* input_names[] = {input_name.get()};
        const char* output_names[] = {output_name.get()};
        auto outputs =
            session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1,
                        output_names, 1);

        const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
        const std::size_t output_bytes =
            output_info.GetElementCount() *
            element_size(output_info.GetElementType());
        write_exact(options.output, outputs[0].GetTensorRawData(), output_bytes);
        std::cout << "stage64_minimal_result status=pass provider="
                  << options.provider << " output_bytes=" << output_bytes
                  << '\n';
        return 0;
    } catch (const Ort::Exception& error) {
        std::cerr << "stage64_minimal_result status=ort-error code="
                  << static_cast<int>(error.GetOrtErrorCode())
                  << " message=" << error.what() << '\n';
        return 3;
    } catch (const std::exception& error) {
        std::cerr << "stage64_minimal_result status=error message="
                  << error.what() << '\n';
        return 2;
    }
}
