#include <onnxruntime_cxx_api.h>
#include <spacemit_ort_env.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
  std::string inference_model;
  std::string tail_model;
  std::string input;
  std::string output;
  std::string boundary_output_dir;
  std::string profile_prefix;
  std::string samples_output;
  int intra_threads = 4;
  int inter_threads = 1;
  int log_severity = 2;
  int log_verbosity = 0;
  int warmup = 0;
  int runs = 1;
  int repeats = 1;
  std::vector<std::pair<std::string, std::string>> provider_options;
};

struct Sample {
  double inference_us = 0.0;
  double tail_us = 0.0;
  double total_us = 0.0;
  std::uint64_t output_fnv1a64 = 0;
};

std::string require_value(int &index, int argc, char **argv,
                          const char *option) {
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + option);
  }
  return argv[++index];
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--provider") {
      options.provider = require_value(index, argc, argv, "--provider");
    } else if (argument == "--inference-model") {
      options.inference_model =
          require_value(index, argc, argv, "--inference-model");
    } else if (argument == "--tail-model") {
      options.tail_model = require_value(index, argc, argv, "--tail-model");
    } else if (argument == "--input") {
      options.input = require_value(index, argc, argv, "--input");
    } else if (argument == "--output") {
      options.output = require_value(index, argc, argv, "--output");
    } else if (argument == "--boundary-output-dir") {
      options.boundary_output_dir =
          require_value(index, argc, argv, "--boundary-output-dir");
    } else if (argument == "--profile-prefix") {
      options.profile_prefix =
          require_value(index, argc, argv, "--profile-prefix");
    } else if (argument == "--samples-output") {
      options.samples_output =
          require_value(index, argc, argv, "--samples-output");
    } else if (argument == "--intra-threads") {
      options.intra_threads = std::max(
          1, std::stoi(require_value(index, argc, argv, "--intra-threads")));
    } else if (argument == "--inter-threads") {
      options.inter_threads = std::max(
          1, std::stoi(require_value(index, argc, argv, "--inter-threads")));
    } else if (argument == "--log-severity") {
      options.log_severity =
          std::stoi(require_value(index, argc, argv, "--log-severity"));
    } else if (argument == "--log-verbosity") {
      options.log_verbosity =
          std::stoi(require_value(index, argc, argv, "--log-verbosity"));
    } else if (argument == "--warmup") {
      options.warmup =
          std::max(0, std::stoi(require_value(index, argc, argv, "--warmup")));
    } else if (argument == "--runs") {
      options.runs =
          std::max(1, std::stoi(require_value(index, argc, argv, "--runs")));
    } else if (argument == "--repeats") {
      options.repeats =
          std::max(1, std::stoi(require_value(index, argc, argv, "--repeats")));
    } else if (argument == "--provider-option") {
      const std::string value =
          require_value(index, argc, argv, "--provider-option");
      const std::size_t separator = value.find('=');
      if (separator == std::string::npos || separator == 0) {
        throw std::runtime_error("--provider-option requires key=value");
      }
      options.provider_options.emplace_back(value.substr(0, separator),
                                            value.substr(separator + 1));
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (options.provider != "cpu" && options.provider != "spacemit") {
    throw std::runtime_error("--provider must be cpu or spacemit");
  }
  if (options.inference_model.empty() || options.tail_model.empty() ||
      options.input.empty() || options.output.empty()) {
    throw std::runtime_error(
        "--inference-model, --tail-model, --input, and --output are required");
  }
  return options;
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

std::size_t element_count(const std::vector<std::int64_t> &shape) {
  std::size_t result = 1;
  for (std::int64_t dimension : shape) {
    if (dimension < 0) {
      throw std::runtime_error("dynamic input shape is unsupported");
    }
    const std::size_t value = static_cast<std::size_t>(dimension);
    if (value != 0 &&
        result > std::numeric_limits<std::size_t>::max() / value) {
      throw std::overflow_error("input element count overflow");
    }
    result *= value;
  }
  return result;
}

std::vector<std::uint8_t> read_exact(const std::string &path,
                                     std::size_t expected) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("failed to open input: " + path);
  }
  const std::streamoff actual = input.tellg();
  if (actual < 0 || static_cast<std::uint64_t>(actual) != expected) {
    throw std::runtime_error(
        "input byte count mismatch expected=" + std::to_string(expected) +
        " actual=" + std::to_string(actual));
  }
  std::vector<std::uint8_t> bytes(expected);
  input.seekg(0);
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input) {
    throw std::runtime_error("failed to read input: " + path);
  }
  return bytes;
}

void write_bytes(const std::string &path, const void *data, std::size_t bytes) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to open output: " + path);
  }
  output.write(static_cast<const char *>(data),
               static_cast<std::streamsize>(bytes));
  if (!output) {
    throw std::runtime_error("failed to write output: " + path);
  }
}

std::uint64_t fnv1a64(const void *data, std::size_t bytes) {
  const auto *source = static_cast<const std::uint8_t *>(data);
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::size_t index = 0; index < bytes; ++index) {
    hash ^= source[index];
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::string shape_string(const std::vector<std::int64_t> &shape) {
  std::string result;
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0)
      result += 'x';
    result += std::to_string(shape[index]);
  }
  return result.empty() ? "scalar" : result;
}

double percentile(std::vector<double> values, double quantile) {
  std::sort(values.begin(), values.end());
  const double position = quantile * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
  if (lower == upper)
    return values[lower];
  const double fraction = position - static_cast<double>(lower);
  return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

void print_stats(const char *metric, const std::vector<double> &values) {
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) /
                      static_cast<double>(values.size());
  double square_sum = 0.0;
  for (const double value : values) {
    const double delta = value - mean;
    square_sum += delta * delta;
  }
  const double stddev =
      values.size() > 1
          ? std::sqrt(square_sum / static_cast<double>(values.size() - 1))
          : 0.0;
  std::cout << std::fixed << std::setprecision(6)
            << "stage64_stats metric=" << metric << " samples=" << values.size()
            << " mean_us=" << mean << " stddev_us=" << stddev
            << " cv_pct=" << (mean == 0.0 ? 0.0 : 100.0 * stddev / mean)
            << " min_us=" << *std::min_element(values.begin(), values.end())
            << " max_us=" << *std::max_element(values.begin(), values.end())
            << " median_us=" << percentile(values, 0.50)
            << " p95_us=" << percentile(values, 0.95)
            << " p99_us=" << percentile(values, 0.99)
            << " p999_us=" << percentile(values, 0.999) << '\n';
}

std::vector<std::string> names(Ort::Session &session, bool inputs,
                               Ort::AllocatorWithDefaultOptions &allocator) {
  const std::size_t count =
      inputs ? session.GetInputCount() : session.GetOutputCount();
  std::vector<std::string> result;
  result.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    auto name = inputs ? session.GetInputNameAllocated(index, allocator)
                       : session.GetOutputNameAllocated(index, allocator);
    result.emplace_back(name.get());
  }
  return result;
}

Ort::SessionOptions make_options(const Options &options,
                                 bool inference_session) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(options.intra_threads);
  session_options.SetInterOpNumThreads(options.inter_threads);
  session_options.SetExecutionMode(ORT_SEQUENTIAL);
  session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
  session_options.SetLogSeverityLevel(options.log_severity);
  Ort::ThrowOnError(Ort::GetApi().SetSessionLogVerbosityLevel(
      session_options, options.log_verbosity));
  session_options.AddConfigEntry("session.intra_op.allow_spinning", "0");
  session_options.AddConfigEntry("session.inter_op.allow_spinning", "0");
  if (inference_session && !options.profile_prefix.empty()) {
    session_options.EnableProfiling(options.profile_prefix.c_str());
  }
  if (inference_session && options.provider == "spacemit") {
    std::unordered_map<std::string, std::string> provider_options;
    for (const auto &[key, value] : options.provider_options) {
      provider_options[key] = value;
    }
    if (!provider_options.contains("SPACEMIT_EP_INTRA_THREAD_NUM")) {
      provider_options["SPACEMIT_EP_INTRA_THREAD_NUM"] =
          std::to_string(options.intra_threads);
    }
    for (const auto &[key, value] : provider_options) {
      std::cout << "stage64_provider_option key=" << key << " value=" << value
                << '\n';
    }
    Ort::ThrowOnError(
        Ort::SessionOptionsSpaceMITEnvInit(session_options, provider_options));
    std::cout << "stage64_provider appended=1 name=SpaceMITExecutionProvider\n";
  }
  return session_options;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::cout << "stage64_runtime"
              << " ort_version=" << OrtGetApiBase()->GetVersionString()
              << " ort_build_info=" << Ort::GetApi().GetBuildInfoString()
              << " spacemit_ep_header_version=" << SpaceMITPROVIDER_VERSION
              << " provider=" << options.provider << " inference_provider="
              << (options.provider == "spacemit" ? "SpaceMITExecutionProvider"
                                                 : "CPUExecutionProvider")
              << " tail_provider=CPUExecutionProvider"
              << " intra_threads=" << options.intra_threads
              << " inter_threads=" << options.inter_threads << '\n';

    Ort::Env env(static_cast<OrtLoggingLevel>(options.log_severity),
                 "stage64_two_stage");
    Ort::SessionOptions inference_options = make_options(options, true);
    Ort::SessionOptions tail_options = make_options(options, false);

    const auto inference_create_begin = Clock::now();
    Ort::Session inference_session(env, options.inference_model.c_str(),
                                   inference_options);
    const auto inference_create_end = Clock::now();
    const auto tail_create_begin = Clock::now();
    Ort::Session tail_session(env, options.tail_model.c_str(), tail_options);
    const auto tail_create_end = Clock::now();
    std::cout << std::fixed << std::setprecision(6)
              << "stage64_session inference_create_us="
              << std::chrono::duration<double, std::micro>(
                     inference_create_end - inference_create_begin)
                     .count()
              << " tail_create_us="
              << std::chrono::duration<double, std::micro>(tail_create_end -
                                                           tail_create_begin)
                     .count()
              << '\n';

    Ort::AllocatorWithDefaultOptions allocator;
    const std::vector<std::string> inference_inputs =
        names(inference_session, true, allocator);
    const std::vector<std::string> inference_outputs =
        names(inference_session, false, allocator);
    const std::vector<std::string> tail_inputs =
        names(tail_session, true, allocator);
    const std::vector<std::string> tail_outputs =
        names(tail_session, false, allocator);
    if (inference_inputs.size() != 1 || tail_outputs.size() != 1) {
      throw std::runtime_error("unsupported model input/output count");
    }
    for (const std::string &name : tail_inputs) {
      if (std::find(inference_outputs.begin(), inference_outputs.end(), name) ==
          inference_outputs.end()) {
        throw std::runtime_error("tail input missing from inference outputs: " +
                                 name);
      }
    }
    std::cout << "stage64_contract inference_outputs="
              << inference_outputs.size()
              << " tail_inputs=" << tail_inputs.size() << " pairing=by-name\n";
    for (std::size_t index = 0; index < tail_inputs.size(); ++index) {
      std::cout << "stage64_boundary index=" << index
                << " name=" << tail_inputs[index] << '\n';
    }

    const auto input_type_info = inference_session.GetInputTypeInfo(0);
    const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
    const std::vector<std::int64_t> input_shape = input_info.GetShape();
    const ONNXTensorElementDataType input_type = input_info.GetElementType();
    const std::size_t input_element_size = element_size(input_type);
    const std::size_t input_bytes_count =
        element_count(input_shape) * input_element_size;
    std::cout << "stage64_input name=" << inference_inputs[0]
              << " shape=" << shape_string(input_shape)
              << " type=" << static_cast<int>(input_type)
              << " element_size=" << input_element_size
              << " bytes=" << input_bytes_count << '\n';
    std::vector<std::uint8_t> input_bytes =
        read_exact(options.input, input_bytes_count);
    Ort::MemoryInfo memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor(
        memory, input_bytes.data(), input_bytes.size(), input_shape.data(),
        input_shape.size(), input_type);

    std::vector<const char *> inference_input_names = {
        inference_inputs[0].c_str()};
    std::vector<const char *> inference_output_names;
    std::vector<const char *> tail_input_names;
    for (const std::string &name : tail_inputs) {
      inference_output_names.push_back(name.c_str());
      tail_input_names.push_back(name.c_str());
    }
    std::vector<const char *> tail_output_names = {tail_outputs[0].c_str()};
    std::vector<Ort::Value> boundaries;
    std::vector<Ort::Value> final_output;
    auto run_once = [&]() -> Sample {
      const auto begin = Clock::now();
      boundaries = inference_session.Run(
          Ort::RunOptions{nullptr}, inference_input_names.data(), &input_tensor,
          1, inference_output_names.data(), inference_output_names.size());
      const auto middle = Clock::now();
      final_output = tail_session.Run(
          Ort::RunOptions{nullptr}, tail_input_names.data(), boundaries.data(),
          boundaries.size(), tail_output_names.data(), 1);
      const auto end = Clock::now();
      const auto final_info = final_output[0].GetTensorTypeAndShapeInfo();
      const std::vector<std::int64_t> final_shape = final_info.GetShape();
      const std::size_t final_bytes = element_count(final_shape) *
                                      element_size(final_info.GetElementType());
      const std::uint64_t output_hash =
          fnv1a64(final_output[0].GetTensorRawData(), final_bytes);
      return {
          std::chrono::duration<double, std::micro>(middle - begin).count(),
          std::chrono::duration<double, std::micro>(end - middle).count(),
          std::chrono::duration<double, std::micro>(end - begin).count(),
          output_hash,
      };
    };

    const Sample first = run_once();
    std::cout << std::fixed << std::setprecision(6)
              << "stage64_first inference_us=" << first.inference_us
              << " tail_us=" << first.tail_us << " total_us=" << first.total_us
              << '\n';
    for (int index = 0; index < options.warmup; ++index) {
      run_once();
    }
    std::vector<double> inference_samples;
    std::vector<double> tail_samples;
    std::vector<double> total_samples;
    std::ofstream sample_output;
    if (!options.samples_output.empty()) {
      sample_output.open(options.samples_output);
      if (!sample_output) {
        throw std::runtime_error("failed to open samples output: " +
                                 options.samples_output);
      }
      sample_output << "repeat\trun\tinference_us\ttail_us\ttotal_us"
                       "\toutput_fnv1a64\n";
      sample_output << std::fixed << std::setprecision(6);
    }
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
      double inference_sum = 0.0;
      double tail_sum = 0.0;
      double total_sum = 0.0;
      for (int run = 0; run < options.runs; ++run) {
        const Sample sample = run_once();
        inference_samples.push_back(sample.inference_us);
        tail_samples.push_back(sample.tail_us);
        total_samples.push_back(sample.total_us);
        inference_sum += sample.inference_us;
        tail_sum += sample.tail_us;
        total_sum += sample.total_us;
        if (sample_output) {
          sample_output << repeat << '\t' << run << '\t' << sample.inference_us
                        << '\t' << sample.tail_us << '\t' << sample.total_us
                        << "\t0x" << std::hex << sample.output_fnv1a64
                        << std::dec << '\n';
        }
      }
      std::cout << std::fixed << std::setprecision(6)
                << "stage64_repeat index=" << repeat << " runs=" << options.runs
                << " inference_mean_us=" << inference_sum / options.runs
                << " tail_mean_us=" << tail_sum / options.runs
                << " total_mean_us=" << total_sum / options.runs << '\n';
    }
    print_stats("inference", inference_samples);
    print_stats("tail", tail_samples);
    print_stats("two_stage_total", total_samples);

    if (!options.boundary_output_dir.empty()) {
      std::filesystem::create_directories(options.boundary_output_dir);
      for (std::size_t index = 0; index < boundaries.size(); ++index) {
        Ort::Value &boundary = boundaries[index];
        const auto boundary_info = boundary.GetTensorTypeAndShapeInfo();
        const std::vector<std::int64_t> boundary_shape =
            boundary_info.GetShape();
        const ONNXTensorElementDataType boundary_type =
            boundary_info.GetElementType();
        const std::size_t boundary_bytes =
            element_count(boundary_shape) * element_size(boundary_type);
        const void *boundary_data = boundary.GetTensorRawData();
        const std::string boundary_path = options.boundary_output_dir +
                                          "/boundary-" + std::to_string(index) +
                                          ".bin";
        write_bytes(boundary_path, boundary_data, boundary_bytes);
        std::cout << "stage64_boundary_output index=" << index
                  << " name=" << tail_inputs[index]
                  << " shape=" << shape_string(boundary_shape)
                  << " type=" << static_cast<int>(boundary_type)
                  << " bytes=" << boundary_bytes << " fnv1a64=0x" << std::hex
                  << fnv1a64(boundary_data, boundary_bytes) << std::dec
                  << " path=" << boundary_path << '\n';
      }
    }

    Ort::Value &output_value = final_output.at(0);
    const auto output_info = output_value.GetTensorTypeAndShapeInfo();
    const std::vector<std::int64_t> output_shape = output_info.GetShape();
    const ONNXTensorElementDataType output_type = output_info.GetElementType();
    const std::size_t output_bytes =
        element_count(output_shape) * element_size(output_type);
    const void *output_data = output_value.GetTensorRawData();
    write_bytes(options.output, output_data, output_bytes);
    std::cout << "stage64_result status=pass"
              << " output_shape=" << shape_string(output_shape)
              << " output_type=" << static_cast<int>(output_type)
              << " output_bytes=" << output_bytes << " output_fnv1a64=0x"
              << std::hex << fnv1a64(output_data, output_bytes) << std::dec
              << '\n';
    return 0;
  } catch (const Ort::Exception &exception) {
    std::cerr << "stage64_result status=ort_exception"
              << " code=" << static_cast<int>(exception.GetOrtErrorCode())
              << " message=" << exception.what() << '\n';
    return 10;
  } catch (const std::exception &exception) {
    std::cerr << "stage64_result status=exception"
              << " message=" << exception.what() << '\n';
    return 11;
  }
}
