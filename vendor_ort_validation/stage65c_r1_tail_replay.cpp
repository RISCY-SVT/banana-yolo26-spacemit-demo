#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string tail_model;
  std::string boundary_dir;
  std::string output;
  int threads = 4;
  int runs = 1;
};

std::string value(int &index, int argc, char **argv, const char *option) {
  if (++index >= argc)
    throw std::runtime_error(std::string("missing value for ") + option);
  return argv[index];
}

Options parse(int argc, char **argv) {
  Options result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--tail-model")
      result.tail_model = value(index, argc, argv, "--tail-model");
    else if (argument == "--boundary-dir")
      result.boundary_dir = value(index, argc, argv, "--boundary-dir");
    else if (argument == "--output")
      result.output = value(index, argc, argv, "--output");
    else if (argument == "--threads")
      result.threads = std::max(1, std::stoi(value(index, argc, argv, "--threads")));
    else if (argument == "--runs")
      result.runs = std::max(1, std::stoi(value(index, argc, argv, "--runs")));
    else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: stage65c_r1_tail_replay --tail-model FILE "
                   "--boundary-dir DIR --output FILE [--threads 4] [--runs 1]\n";
      std::exit(0);
    } else
      throw std::runtime_error("unknown argument: " + argument);
  }
  if (result.tail_model.empty() || result.boundary_dir.empty() || result.output.empty())
    throw std::runtime_error("--tail-model, --boundary-dir and --output are required");
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
    throw std::runtime_error("unsupported tensor type: " +
                             std::to_string(static_cast<int>(type)));
  }
}

std::size_t element_count(const std::vector<std::int64_t> &shape) {
  std::size_t result = 1;
  for (const auto dimension : shape) {
    if (dimension < 0)
      throw std::runtime_error("dynamic tail input is unsupported");
    const auto value = static_cast<std::size_t>(dimension);
    if (value && result > std::numeric_limits<std::size_t>::max() / value)
      throw std::overflow_error("tensor size overflow");
    result *= value;
  }
  return result;
}

std::vector<std::uint8_t> read_exact(const std::string &path,
                                     std::size_t expected) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot open boundary: " + path);
  const auto length = stream.tellg();
  if (length < 0 || static_cast<std::size_t>(length) != expected)
    throw std::runtime_error("boundary byte count mismatch: " + path);
  std::vector<std::uint8_t> bytes(expected);
  stream.seekg(0);
  stream.read(reinterpret_cast<char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  if (!stream)
    throw std::runtime_error("cannot read boundary: " + path);
  return bytes;
}

void write_exact(const std::string &path, const void *data, std::size_t bytes) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("cannot create output: " + path);
  stream.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
  if (!stream)
    throw std::runtime_error("cannot write output: " + path);
}

std::uint64_t fnv1a64(const void *data, std::size_t bytes) {
  const auto *input = static_cast<const std::uint8_t *>(data);
  std::uint64_t result = 1469598103934665603ULL;
  for (std::size_t index = 0; index < bytes; ++index) {
    result ^= input[index];
    result *= 1099511628211ULL;
  }
  return result;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse(argc, argv);
    Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "stage65c-r1-tail");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    session_options.SetExecutionMode(ORT_SEQUENTIAL);
    session_options.SetIntraOpNumThreads(options.threads);
    session_options.SetInterOpNumThreads(1);
    session_options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    session_options.AddConfigEntry("session.inter_op.allow_spinning", "0");
    Ort::Session session(environment, options.tail_model.c_str(), session_options);
    if (session.GetOutputCount() != 1 || session.GetInputCount() != 6)
      throw std::runtime_error("tail must expose six inputs and one output");

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<std::string> names;
    std::vector<const char *> name_pointers;
    std::vector<std::vector<std::uint8_t>> storage;
    std::vector<Ort::Value> tensors;
    names.reserve(6);
    name_pointers.reserve(6);
    storage.reserve(6);
    tensors.reserve(6);
    for (std::size_t index = 0; index < 6; ++index) {
      auto name = session.GetInputNameAllocated(index, allocator);
      names.emplace_back(name.get());
      const auto type_info = session.GetInputTypeInfo(index);
      const auto info = type_info.GetTensorTypeAndShapeInfo();
      const auto shape = info.GetShape();
      const auto type = info.GetElementType();
      const auto bytes = element_count(shape) * element_size(type);
      storage.push_back(read_exact(options.boundary_dir + "/boundary-" +
                                       std::to_string(index) + ".bin",
                                   bytes));
      tensors.push_back(Ort::Value::CreateTensor(
          memory, storage.back().data(), storage.back().size(), shape.data(),
          shape.size(), type));
      std::cout << "stage65c_r1_tail_input index=" << index << " name="
                << names.back() << " bytes=" << bytes << '\n';
    }
    for (const auto &name : names)
      name_pointers.push_back(name.c_str());
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    const char *output_names[] = {output_name.get()};
    std::vector<Ort::Value> output;
    std::uint64_t expected_hash = 0;
    for (int run = 0; run < options.runs; ++run) {
      output = session.Run(Ort::RunOptions{nullptr}, name_pointers.data(),
                           tensors.data(), tensors.size(), output_names, 1);
      const auto info = output[0].GetTensorTypeAndShapeInfo();
      const auto bytes = info.GetElementCount() * element_size(info.GetElementType());
      const auto hash = fnv1a64(output[0].GetTensorRawData(), bytes);
      if (run && hash != expected_hash)
        throw std::runtime_error("tail output changed across repeated runs");
      expected_hash = hash;
      std::cout << "stage65c_r1_tail_run index=" << run << " fnv1a64=0x"
                << std::hex << hash << std::dec << '\n';
    }
    const auto info = output[0].GetTensorTypeAndShapeInfo();
    const auto bytes = info.GetElementCount() * element_size(info.GetElementType());
    write_exact(options.output, output[0].GetTensorRawData(), bytes);
    std::cout << "stage65c_r1_tail_result status=pass runs=" << options.runs
              << " output_bytes=" << bytes << "\n";
    return 0;
  } catch (const Ort::Exception &error) {
    std::cerr << "stage65c_r1_tail_result status=ort-error code="
              << static_cast<int>(error.GetOrtErrorCode())
              << " message=" << error.what() << '\n';
    return 3;
  } catch (const std::exception &error) {
    std::cerr << "stage65c_r1_tail_result status=error message=" << error.what()
              << '\n';
    return 2;
  }
}
