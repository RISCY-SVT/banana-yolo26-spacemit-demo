/**
 * Stage64 COCO predictor for a quantized six-output inference graph followed
 * by an explicitly CPU-owned floating-point post-processing graph.
 */

#include <onnxruntime_cxx_api.h>
#include <spacemit_ort_env.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
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
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr std::array<int, 80> kCocoCategoryIds = {
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
};

struct Options {
  std::string inference_model;
  std::string tail_model;
  std::string images;
  std::string predictions;
  std::string timing;
  std::string provider = "spacemit";
  int threads = 4;
  int limit = 0;
  int log_every = 100;
  float confidence = 0.001F;
};

struct Letterbox {
  float scale = 1.0F;
  float pad_x = 0.0F;
  float pad_y = 0.0F;
  int source_width = 0;
  int source_height = 0;
};

struct Detection {
  float x1 = 0.0F;
  float y1 = 0.0F;
  float x2 = 0.0F;
  float y2 = 0.0F;
  float confidence = 0.0F;
  int class_id = -1;
};

std::string value(int &index, int argc, char **argv, const char *option) {
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + option);
  }
  return argv[++index];
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--inference-model") {
      options.inference_model = value(index, argc, argv, "--inference-model");
    } else if (argument == "--tail-model") {
      options.tail_model = value(index, argc, argv, "--tail-model");
    } else if (argument == "--images") {
      options.images = value(index, argc, argv, "--images");
    } else if (argument == "--output") {
      options.predictions = value(index, argc, argv, "--output");
    } else if (argument == "--timing-tsv") {
      options.timing = value(index, argc, argv, "--timing-tsv");
    } else if (argument == "--provider") {
      options.provider = value(index, argc, argv, "--provider");
    } else if (argument == "--threads") {
      options.threads =
          std::max(1, std::stoi(value(index, argc, argv, "--threads")));
    } else if (argument == "--limit") {
      options.limit =
          std::max(0, std::stoi(value(index, argc, argv, "--limit")));
    } else if (argument == "--log-every") {
      options.log_every =
          std::max(0, std::stoi(value(index, argc, argv, "--log-every")));
    } else if (argument == "--conf") {
      options.confidence = std::stof(value(index, argc, argv, "--conf"));
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: stage64_two_stage_coco --inference-model FILE "
                   "--tail-model FILE --images DIR --output FILE "
                   "--timing-tsv FILE [--provider cpu|spacemit] "
                   "[--threads 4] [--conf 0.001] [--limit N]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (options.inference_model.empty() || options.tail_model.empty() ||
      options.images.empty() || options.predictions.empty() ||
      options.timing.empty()) {
    throw std::runtime_error(
        "--inference-model, --tail-model, --images, --output, and "
        "--timing-tsv are required");
  }
  if (options.provider != "cpu" && options.provider != "spacemit") {
    throw std::runtime_error("--provider must be cpu or spacemit");
  }
  return options;
}

std::vector<fs::path> list_images(const fs::path &directory, int limit) {
  std::vector<fs::path> result;
  for (const fs::directory_entry &entry : fs::directory_iterator(directory)) {
    if (!entry.is_regular_file())
      continue;
    std::string extension = entry.path().extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char character) {
                     return static_cast<char>(std::tolower(character));
                   });
    if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") {
      result.push_back(entry.path());
    }
  }
  std::sort(result.begin(), result.end());
  if (limit > 0 && static_cast<std::size_t>(limit) < result.size()) {
    result.resize(static_cast<std::size_t>(limit));
  }
  return result;
}

int image_id(const fs::path &path) {
  const std::string stem = path.stem().string();
  std::size_t first = stem.find_first_not_of('0');
  if (first == std::string::npos)
    return 0;
  return std::stoi(stem.substr(first));
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

Ort::SessionOptions session_options(const Options &options, bool inference) {
  Ort::SessionOptions result;
  result.SetIntraOpNumThreads(options.threads);
  result.SetInterOpNumThreads(1);
  result.SetExecutionMode(ORT_SEQUENTIAL);
  result.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
  result.SetLogSeverityLevel(2);
  result.AddConfigEntry("session.intra_op.allow_spinning", "0");
  result.AddConfigEntry("session.inter_op.allow_spinning", "0");
  if (inference && options.provider == "spacemit") {
    std::unordered_map<std::string, std::string> provider_options = {
        {"SPACEMIT_EP_INTRA_THREAD_NUM", std::to_string(options.threads)},
    };
    Ort::ThrowOnError(
        Ort::SessionOptionsSpaceMITEnvInit(result, provider_options));
  }
  return result;
}

std::vector<float> preprocess(const cv::Mat &bgr, Letterbox &letterbox) {
  constexpr int kInput = 640;
  if (bgr.empty())
    throw std::runtime_error("empty image");
  letterbox.source_width = bgr.cols;
  letterbox.source_height = bgr.rows;
  letterbox.scale =
      std::min(static_cast<float>(kInput) / static_cast<float>(bgr.cols),
               static_cast<float>(kInput) / static_cast<float>(bgr.rows));
  const int resized_width = static_cast<int>(
      std::nearbyint(static_cast<double>(bgr.cols) * letterbox.scale));
  const int resized_height = static_cast<int>(
      std::nearbyint(static_cast<double>(bgr.rows) * letterbox.scale));
  letterbox.pad_x = static_cast<float>(kInput - resized_width) / 2.0F;
  letterbox.pad_y = static_cast<float>(kInput - resized_height) / 2.0F;

  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(resized_width, resized_height), 0.0, 0.0,
             cv::INTER_LINEAR);
  const int left = static_cast<int>(std::nearbyint(letterbox.pad_x - 0.1F));
  const int right = static_cast<int>(std::nearbyint(letterbox.pad_x + 0.1F));
  const int top = static_cast<int>(std::nearbyint(letterbox.pad_y - 0.1F));
  const int bottom = static_cast<int>(std::nearbyint(letterbox.pad_y + 0.1F));
  cv::Mat padded;
  cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                     cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  if (padded.rows != kInput || padded.cols != kInput) {
    throw std::runtime_error("letterbox output is not 640x640");
  }
  cv::Mat rgb;
  cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
  cv::Mat fp32;
  rgb.convertTo(fp32, CV_32FC3, 1.0 / 255.0);
  const std::size_t plane =
      static_cast<std::size_t>(kInput) * static_cast<std::size_t>(kInput);
  std::vector<float> nchw(3 * plane);
  for (int channel = 0; channel < 3; ++channel) {
    float *destination =
        nchw.data() + static_cast<std::size_t>(channel) * plane;
    for (int row = 0; row < kInput; ++row) {
      const cv::Vec3f *source = fp32.ptr<cv::Vec3f>(row);
      for (int column = 0; column < kInput; ++column) {
        *destination++ = source[column][channel];
      }
    }
  }
  return nchw;
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

std::vector<Detection> decode(const float *output, std::size_t count,
                              const Letterbox &letterbox, float threshold,
                              std::size_t &non_finite) {
  constexpr std::size_t kColumns = 6;
  if (count != 300 * kColumns) {
    throw std::runtime_error("expected output element count 1800, got " +
                             std::to_string(count));
  }
  std::vector<Detection> detections;
  for (std::size_t row = 0; row < 300; ++row) {
    const float *item = output + row * kColumns;
    bool finite = true;
    for (std::size_t column = 0; column < kColumns; ++column) {
      finite &= std::isfinite(item[column]);
    }
    if (!finite) {
      ++non_finite;
      continue;
    }
    if (item[4] < threshold)
      continue;
    const float rounded_class = std::nearbyint(item[5]);
    if (std::abs(item[5] - rounded_class) > 1.0e-4F)
      continue;
    const int class_id = static_cast<int>(rounded_class);
    if (class_id < 0 || class_id >= 80)
      continue;

    const float x1 = std::clamp(item[0], 0.0F, 640.0F);
    const float y1 = std::clamp(item[1], 0.0F, 640.0F);
    const float x2 = std::clamp(item[2], 0.0F, 640.0F);
    const float y2 = std::clamp(item[3], 0.0F, 640.0F);
    Detection detection;
    detection.x1 = std::clamp((x1 - letterbox.pad_x) / letterbox.scale, 0.0F,
                              static_cast<float>(letterbox.source_width));
    detection.y1 = std::clamp((y1 - letterbox.pad_y) / letterbox.scale, 0.0F,
                              static_cast<float>(letterbox.source_height));
    detection.x2 = std::clamp((x2 - letterbox.pad_x) / letterbox.scale, 0.0F,
                              static_cast<float>(letterbox.source_width));
    detection.y2 = std::clamp((y2 - letterbox.pad_y) / letterbox.scale, 0.0F,
                              static_cast<float>(letterbox.source_height));
    detection.confidence = item[4];
    detection.class_id = class_id;
    if (detection.x2 > detection.x1 && detection.y2 > detection.y1) {
      detections.push_back(detection);
    }
  }
  return detections;
}

void json_number(std::ostream &output, float value) {
  output << std::fixed << std::setprecision(6)
         << (std::isfinite(value) ? value : 0.0F);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "stage64_coco");
    Ort::SessionOptions inference_options = session_options(options, true);
    Ort::SessionOptions tail_options = session_options(options, false);
    Ort::Session inference(environment, options.inference_model.c_str(),
                           inference_options);
    Ort::Session tail(environment, options.tail_model.c_str(), tail_options);
    Ort::AllocatorWithDefaultOptions allocator;
    const std::vector<std::string> inference_inputs =
        names(inference, true, allocator);
    const std::vector<std::string> inference_outputs =
        names(inference, false, allocator);
    const std::vector<std::string> tail_inputs = names(tail, true, allocator);
    const std::vector<std::string> tail_outputs = names(tail, false, allocator);
    if (inference_inputs.size() != 1 || tail_outputs.size() != 1 ||
        inference_outputs.size() != tail_inputs.size()) {
      throw std::runtime_error("unexpected split graph contract");
    }

    std::vector<const char *> inference_output_names;
    std::vector<const char *> tail_input_names;
    for (const std::string &name : tail_inputs) {
      const auto match =
          std::find(inference_outputs.begin(), inference_outputs.end(), name);
      if (match == inference_outputs.end()) {
        throw std::runtime_error(
            "tail input is absent from inference outputs: " + name);
      }
      inference_output_names.push_back(name.c_str());
      tail_input_names.push_back(name.c_str());
    }
    const char *inference_input_name = inference_inputs[0].c_str();
    const char *tail_output_name = tail_outputs[0].c_str();
    const std::vector<fs::path> images =
        list_images(options.images, options.limit);
    const fs::path prediction_parent =
        fs::path(options.predictions).parent_path();
    const fs::path timing_parent = fs::path(options.timing).parent_path();
    if (!prediction_parent.empty()) {
      fs::create_directories(prediction_parent);
    }
    if (!timing_parent.empty()) {
      fs::create_directories(timing_parent);
    }
    std::ofstream predictions(options.predictions);
    std::ofstream timing(options.timing);
    if (!predictions || !timing) {
      throw std::runtime_error("failed to create output files");
    }
    timing << "image_id\tpath\tobjects\tpreprocess_ms\tinference_ms\t"
              "tail_ms\tdecode_ms\ttotal_ms\toutput_fnv1a64\t"
              "non_finite_rows\n";
    predictions << "[\n";
    bool first_prediction = true;
    std::size_t completed = 0;
    std::size_t failures = 0;
    std::size_t prediction_count = 0;
    std::size_t non_finite_total = 0;
    Ort::MemoryInfo memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    constexpr std::array<std::int64_t, 4> kInputShape = {1, 3, 640, 640};

    std::cout << "stage64_coco runtime=" << OrtGetApiBase()->GetVersionString()
              << " inference_provider="
              << (options.provider == "spacemit" ? "SpaceMITExecutionProvider"
                                                 : "CPUExecutionProvider")
              << " tail_provider=CPUExecutionProvider"
              << " images=" << images.size() << '\n';

    for (std::size_t index = 0; index < images.size(); ++index) {
      try {
        const Clock::time_point begin = Clock::now();
        cv::Mat image = cv::imread(images[index].string(), cv::IMREAD_COLOR);
        Letterbox letterbox;
        std::vector<float> input = preprocess(image, letterbox);
        const Clock::time_point preprocessed = Clock::now();
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory, input.data(), input.size(), kInputShape.data(),
            kInputShape.size());
        std::vector<Ort::Value> boundaries = inference.Run(
            Ort::RunOptions{nullptr}, &inference_input_name, &input_tensor, 1,
            inference_output_names.data(), inference_output_names.size());
        const Clock::time_point inferred = Clock::now();
        std::vector<Ort::Value> final = tail.Run(
            Ort::RunOptions{nullptr}, tail_input_names.data(),
            boundaries.data(), boundaries.size(), &tail_output_name, 1);
        const Clock::time_point tailed = Clock::now();
        const auto output_info = final[0].GetTensorTypeAndShapeInfo();
        if (output_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          throw std::runtime_error("tail output is not float32");
        }
        const std::size_t output_count = output_info.GetElementCount();
        const float *output = final[0].GetTensorData<float>();
        std::size_t non_finite = 0;
        const std::vector<Detection> detections = decode(
            output, output_count, letterbox, options.confidence, non_finite);
        const Clock::time_point decoded = Clock::now();
        non_finite_total += non_finite;
        prediction_count += detections.size();
        const int id = image_id(images[index]);
        for (const Detection &detection : detections) {
          if (!first_prediction)
            predictions << ",\n";
          first_prediction = false;
          predictions
              << "  {\"image_id\":" << id << ",\"category_id\":"
              << kCocoCategoryIds[static_cast<std::size_t>(detection.class_id)]
              << ",\"bbox\":[";
          json_number(predictions, detection.x1);
          predictions << ',';
          json_number(predictions, detection.y1);
          predictions << ',';
          json_number(predictions, detection.x2 - detection.x1);
          predictions << ',';
          json_number(predictions, detection.y2 - detection.y1);
          predictions << "],\"score\":";
          json_number(predictions, detection.confidence);
          predictions << '}';
        }
        const double preprocess_ms =
            std::chrono::duration<double, std::milli>(preprocessed - begin)
                .count();
        const double inference_ms =
            std::chrono::duration<double, std::milli>(inferred - preprocessed)
                .count();
        const double tail_ms =
            std::chrono::duration<double, std::milli>(tailed - inferred)
                .count();
        const double decode_ms =
            std::chrono::duration<double, std::milli>(decoded - tailed).count();
        const double total_ms =
            std::chrono::duration<double, std::milli>(decoded - begin).count();
        timing << id << '\t' << images[index].filename().string() << '\t'
               << detections.size() << '\t' << std::fixed
               << std::setprecision(6) << preprocess_ms << '\t' << inference_ms
               << '\t' << tail_ms << '\t' << decode_ms << '\t' << total_ms
               << "\t0x" << std::hex
               << fnv1a64(output, output_count * sizeof(float)) << std::dec
               << '\t' << non_finite << '\n';
        ++completed;
      } catch (const std::exception &exception) {
        ++failures;
        std::cerr << "stage64_coco_image_failure path="
                  << images[index].filename().string()
                  << " message=" << exception.what() << '\n';
      }
      if (options.log_every > 0 &&
          ((index + 1) % static_cast<std::size_t>(options.log_every) == 0 ||
           index + 1 == images.size())) {
        std::cout << "stage64_coco_progress completed=" << completed
                  << " attempted=" << index + 1 << " failures=" << failures
                  << '\n';
      }
    }
    predictions << "\n]\n";
    std::cout << "stage64_coco_result completed=" << completed
              << " failures=" << failures << " predictions=" << prediction_count
              << " non_finite_rows=" << non_finite_total << '\n';
    return failures == 0 && completed == images.size() && non_finite_total == 0
               ? 0
               : 12;
  } catch (const Ort::Exception &exception) {
    std::cerr << "stage64_coco_result status=ort_exception code="
              << static_cast<int>(exception.GetOrtErrorCode())
              << " message=" << exception.what() << '\n';
    return 10;
  } catch (const std::exception &exception) {
    std::cerr << "stage64_coco_result status=exception message="
              << exception.what() << '\n';
    return 11;
  }
}
