/**
 * @file yolo11_detector.cpp
 * @brief ONNX Runtime backed YOLO11 preprocessing, execution, and decode logic.
 */

#include "banana_demo/infer/detector.h"

#include "banana_demo/render/renderer.h"
#include "banana_demo/util/sha256.h"

#include "spacemit_ort_env.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>

namespace banana_demo {

namespace {

/** @brief Collect provider options that the public vendor runtime reads from the environment. */
std::unordered_map<std::string, std::string> BuildProviderOptionsFromEnv()
{
    static const char* kKeys[] = {
        "SPACEMIT_EP_DISABLE_OP_TYPE_FILTER",
        "SPACEMIT_EP_DISABLE_OP_NAME_FILTER",
        "SPACEMIT_EP_DISABLE_FLOAT16_EPILOGUE",
        "SPACEMIT_EP_DUMP_SUBGRAPHS",
        "SPACEMIT_EP_DEBUG_PROFILE",
        "SPACEMIT_EP_DUMP_TENSORS",
    };

    std::unordered_map<std::string, std::string> provider_options;
    for (const char* key : kKeys)
    {
        const char* value = std::getenv(key);
        if (value && *value)
            provider_options[key] = value;
    }
    return provider_options;
}

/** @brief Enable verbose preprocess/decode diagnostics for forensic runs. */
bool DebugDecodeEnabled()
{
    const char* value = std::getenv("BANANA_DEMO_DEBUG_DECODE");
    return value && *value && std::string(value) != "0";
}

/** @brief Hash a raw float output tensor for reproducibility reports. */
std::string HashOutputTensor(const std::vector<float>& data)
{
    const auto* ptr = reinterpret_cast<const uint8_t*>(data.data());
    return Sha256Hex(ptr, data.size() * sizeof(float));
}

/** @brief Hash decoded detections after converting them to a stable byte layout. */
std::string HashDetections(const std::vector<Detection>& detections)
{
    std::vector<uint8_t> bytes;
    bytes.reserve(detections.size() * (sizeof(float) * 5 + sizeof(int)));
    for (const auto& det : detections)
    {
        const auto* class_ptr = reinterpret_cast<const uint8_t*>(&det.class_id);
        bytes.insert(bytes.end(), class_ptr, class_ptr + sizeof(det.class_id));
        for (float v : {det.score, det.x1, det.y1, det.x2, det.y2})
        {
            const auto* value_ptr = reinterpret_cast<const uint8_t*>(&v);
            bytes.insert(bytes.end(), value_ptr, value_ptr + sizeof(v));
        }
    }
    return Sha256Hex(bytes);
}

/** @brief Render ONNX tensor element types in compact log form. */
const char* TensorElementTypeName(ONNXTensorElementDataType type)
{
    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return "bfloat16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return "int8";
    default:
        return "other";
    }
}

/** @brief Return whether the path matches one repo-generated FP16 keep-IO model. */
bool IsGeneratedFp16KeepIoModel(const std::string& model_path)
{
    const std::string name = std::filesystem::path(model_path).filename().string();
    return name == "yolov11n_320x320.fp16_iop32.onnx" || name == "yolov11n_640x640.fp16_iop32.onnx";
}

/** @brief Return whether the path matches one repo-generated full-I/O FP16 model. */
bool IsGeneratedFp16FullIoModel(const std::string& model_path)
{
    const std::string name = std::filesystem::path(model_path).filename().string();
    return name == "yolov11n_320x320.fp16.onnx" || name == "yolov11n_640x640.fp16.onnx";
}

/** @brief Return whether the path matches one YOLO26 R&D FP16 model with float32 input. */
bool IsYolo26Fp16FloatInputModel(const std::string& model_path)
{
    const std::string name = std::filesystem::path(model_path).filename().string();
    return name == "yolo26n_640_e2e_native_fp16_body_headfp32_keep_io.onnx" ||
           name == "yolo26n_640_e2e_native_fp16_keep_io.onnx" ||
           name == "yolo26n_640_e2e_xslim_fp16.onnx";
}

/** @brief Return whether the path matches one YOLO26 R&D FP16 model with float16 input. */
bool IsYolo26Fp16FloatInputOutputModel(const std::string& model_path)
{
    const std::string name = std::filesystem::path(model_path).filename().string();
    return name == "yolo26n_640_e2e_native_fp16_body_headfp32_full_io.onnx" ||
           name == "yolo26n_640_e2e_native_fp16_full_io.onnx";
}

/** @brief Return one known fixed output tensor shape for the repo-managed FP16 models. */
std::vector<int64_t> KnownFp16OutputShape(const std::string& model_path, int input_size)
{
    if (IsYolo26Fp16FloatInputModel(model_path) || IsYolo26Fp16FloatInputOutputModel(model_path))
        return {1, 300, 6};
    if (input_size == 320)
        return {1, 84, 2100};
    if (input_size == 640)
        return {1, 84, 8400};
    return {};
}

/** @brief Convert one host float buffer to ORT float16 storage. */
std::vector<Ort::Float16_t> ToFloat16Buffer(const std::vector<float>& values)
{
    std::vector<Ort::Float16_t> fp16(values.size());
    std::transform(values.begin(), values.end(), fp16.begin(), [](float value) {
        return Ort::Float16_t(value);
    });
    return fp16;
}

/** @brief Convert one ORT float16 tensor buffer back to float32 host values. */
std::vector<float> FromFloat16Buffer(const Ort::Float16_t* values, size_t count)
{
    std::vector<float> fp32(count);
    std::transform(values, values + count, fp32.begin(), [](Ort::Float16_t value) {
        return static_cast<float>(value);
    });
    return fp32;
}

/** @brief Compute intersection-over-union between two boxes. */
float Iou(const Detection& a, const Detection& b)
{
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float w = std::max(0.f, x2 - x1);
    const float h = std::max(0.f, y2 - y1);
    const float inter = w * h;
    const float area_a = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
    const float area_b = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
    const float uni = area_a + area_b - inter;
    return uni <= 0.f ? 0.f : inter / uni;
}

/** @brief Perform per-class greedy NMS. */
std::vector<Detection> NmsClasswise(const std::vector<Detection>& detections, float iou_threshold)
{
    std::vector<Detection> out;
    if (detections.empty())
        return out;

    std::set<int> classes;
    for (const auto& det : detections)
        classes.insert(det.class_id);

    for (int class_id : classes)
    {
        std::vector<Detection> one;
        for (const auto& det : detections)
        {
            if (det.class_id == class_id)
                one.push_back(det);
        }

        std::sort(one.begin(), one.end(), [](const Detection& a, const Detection& b) {
            if (a.score != b.score)
                return a.score > b.score;
            if (a.x1 != b.x1)
                return a.x1 < b.x1;
            if (a.y1 != b.y1)
                return a.y1 < b.y1;
            if (a.x2 != b.x2)
                return a.x2 < b.x2;
            return a.y2 < b.y2;
        });

        std::vector<char> removed(one.size(), 0);
        for (size_t i = 0; i < one.size(); ++i)
        {
            if (removed[i])
                continue;
            out.push_back(one[i]);
            for (size_t j = i + 1; j < one.size(); ++j)
            {
                if (!removed[j] && Iou(one[i], one[j]) >= iou_threshold)
                    removed[j] = 1;
            }
        }
    }

    return out;
}

}  // namespace

Yolo11Detector::Yolo11Detector(const AppOptions& options)
    : options_(options), env_(ORT_LOGGING_LEVEL_WARNING, "banana_yolo11_demo")
{
    BuildSession();
    ResolveInputShape();
    LoadLabels();
}

void Yolo11Detector::LoadLabels()
{
    labels_.clear();
    std::ifstream ifs(options_.labels);
    std::string line;
    while (std::getline(ifs, line))
    {
        if (!line.empty())
            labels_.push_back(line);
    }
}

void Yolo11Detector::ResolveInputShape()
{
    metadata_fallback_applied_ = false;
    metadata_fallback_reason_.clear();
    input_shape_.clear();

    try
    {
        auto input_info = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        input_shape_ = input_info.GetShape();
        input_element_type_ = input_info.GetElementType();
    }
    catch (const std::exception& ex)
    {
        metadata_fallback_applied_ = true;
        metadata_fallback_reason_ += std::string("input-metadata-exception=") + ex.what();
        input_element_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }

    try
    {
        output_element_type_ = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
    }
    catch (const std::exception& ex)
    {
        metadata_fallback_applied_ = true;
        if (!metadata_fallback_reason_.empty())
            metadata_fallback_reason_ += ';';
        metadata_fallback_reason_ += std::string("output-metadata-exception=") + ex.what();
        output_element_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }

    if (input_shape_.size() == 4)
    {
        input_height_ = input_shape_[2] > 0 ? static_cast<int>(input_shape_[2]) : options_.input_size;
        input_width_ = input_shape_[3] > 0 ? static_cast<int>(input_shape_[3]) : options_.input_size;
    }
    else
    {
        input_height_ = options_.input_size;
        input_width_ = options_.input_size;
        if (IsGeneratedFp16KeepIoModel(options_.model) ||
            IsGeneratedFp16FullIoModel(options_.model) ||
            IsYolo26Fp16FloatInputModel(options_.model) ||
            IsYolo26Fp16FloatInputOutputModel(options_.model))
        {
            input_shape_ = {1, 3, options_.input_size, options_.input_size};
            metadata_fallback_applied_ = true;
            if (!metadata_fallback_reason_.empty())
                metadata_fallback_reason_ += ';';
            metadata_fallback_reason_ += "input-shape-fallback";
        }
    }

    if (IsGeneratedFp16KeepIoModel(options_.model))
    {
        if (input_element_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        {
            input_element_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
            metadata_fallback_applied_ = true;
            if (!metadata_fallback_reason_.empty())
                metadata_fallback_reason_ += ';';
            metadata_fallback_reason_ += "input-dtype=fp32-by-model-contract";
        }
        if (output_element_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        {
            output_element_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
            metadata_fallback_applied_ = true;
            if (!metadata_fallback_reason_.empty())
                metadata_fallback_reason_ += ';';
            metadata_fallback_reason_ += "output-dtype=fp32-by-model-contract";
        }
    }
    else if (IsGeneratedFp16FullIoModel(options_.model))
    {
        if (input_element_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        {
            input_element_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
            metadata_fallback_applied_ = true;
            if (!metadata_fallback_reason_.empty())
                metadata_fallback_reason_ += ';';
            metadata_fallback_reason_ += "input-dtype=fp16-by-model-contract";
        }
        if (output_element_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        {
            output_element_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
            metadata_fallback_applied_ = true;
            if (!metadata_fallback_reason_.empty())
                metadata_fallback_reason_ += ';';
            metadata_fallback_reason_ += "output-dtype=fp16-by-model-contract";
        }
    }
    else if (IsYolo26Fp16FloatInputModel(options_.model))
    {
        if (input_element_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        {
            input_element_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
            metadata_fallback_applied_ = true;
            if (!metadata_fallback_reason_.empty())
                metadata_fallback_reason_ += ';';
            metadata_fallback_reason_ += "input-dtype=fp32-by-yolo26-fp16-contract";
        }
        if (output_element_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        {
            output_element_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
            metadata_fallback_applied_ = true;
            if (!metadata_fallback_reason_.empty())
                metadata_fallback_reason_ += ';';
            metadata_fallback_reason_ += "output-dtype=fp16-by-yolo26-fp16-contract";
        }
    }
    else if (IsYolo26Fp16FloatInputOutputModel(options_.model))
    {
        if (input_element_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        {
            input_element_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
            metadata_fallback_applied_ = true;
            if (!metadata_fallback_reason_.empty())
                metadata_fallback_reason_ += ';';
            metadata_fallback_reason_ += "input-dtype=fp16-by-yolo26-full-io-contract";
        }
        if (output_element_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        {
            output_element_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
            metadata_fallback_applied_ = true;
            if (!metadata_fallback_reason_.empty())
                metadata_fallback_reason_ += ';';
            metadata_fallback_reason_ += "output-dtype=fp16-by-yolo26-full-io-contract";
        }
    }
    else
    {
        if (input_element_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
            input_element_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        {
            input_element_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
            metadata_fallback_applied_ = true;
            if (!metadata_fallback_reason_.empty())
                metadata_fallback_reason_ += ';';
            metadata_fallback_reason_ += "input-dtype=fp32-generic-fallback";
        }
        if (output_element_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
            output_element_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        {
            output_element_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
            metadata_fallback_applied_ = true;
            if (!metadata_fallback_reason_.empty())
                metadata_fallback_reason_ += ';';
            metadata_fallback_reason_ += "output-dtype=fp32-generic-fallback";
        }
        if (input_shape_.empty() && options_.input_size > 0)
        {
            input_shape_ = {1, 3, options_.input_size, options_.input_size};
            metadata_fallback_applied_ = true;
            if (!metadata_fallback_reason_.empty())
                metadata_fallback_reason_ += ';';
            metadata_fallback_reason_ += "input-shape=generic-fallback";
        }
    }
}

void Yolo11Detector::BuildSession()
{
    // Keep the runtime configuration deterministic so board-side repros stay comparable.
    session_options_.SetIntraOpNumThreads(options_.threads);
    session_options_.SetInterOpNumThreads(1);
    session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (options_.disable_cpu_fallback)
        session_options_.AddConfigEntry("session.disable_cpu_ep_fallback", "1");

    if (options_.provider == "spacemit")
    {
        auto provider_options = BuildProviderOptionsFromEnv();
        Ort::Status status = Ort::SessionOptionsSpaceMITEnvInit(session_options_, provider_options);
        if (status)
            throw Ort::Exception(status.GetErrorMessage(), OrtErrorCode::ORT_FAIL);
    }

    session_ = std::make_unique<Ort::Session>(env_, options_.model.c_str(), session_options_);
    auto input_name = session_->GetInputNameAllocated(0, allocator_);
    input_name_ = input_name.get();

    const size_t output_count = session_->GetOutputCount();
    output_names_.clear();
    output_name_ptrs_.clear();
    for (size_t i = 0; i < output_count; ++i)
    {
        auto output_name = session_->GetOutputNameAllocated(i, allocator_);
        output_names_.emplace_back(output_name.get());
    }
    for (const auto& name : output_names_)
        output_name_ptrs_.push_back(name.c_str());
}

Yolo11Detector::PreprocessMode Yolo11Detector::ResolvePreprocessMode() const
{
    if (options_.preprocess_mode == "letterbox")
        return PreprocessMode::kLetterbox;
    if (options_.preprocess_mode == "resize")
        return PreprocessMode::kResize;
    return PreprocessMode::kLetterbox;
}

bool Yolo11Detector::PreprocessToNchw(const cv::Mat& bgr, std::vector<float>& nchw, PreprocessInfo& info) const
{
    if (bgr.empty())
        return false;

    info.src_w = bgr.cols;
    info.src_h = bgr.rows;
    info.dst_w = input_width_;
    info.dst_h = input_height_;
    info.scale_x = static_cast<float>(info.src_w) / static_cast<float>(input_width_);
    info.scale_y = static_cast<float>(info.src_h) / static_cast<float>(input_height_);
    info.pad_x = 0.f;
    info.pad_y = 0.f;
    info.mode = ResolvePreprocessMode();

    cv::Mat preprocessed;
    if (info.mode == PreprocessMode::kResize)
    {
        // Resize mode is mainly a debugging path kept for vendor contract investigations.
        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(resized, preprocessed, cv::COLOR_BGR2RGB);
    }
    else
    {
        // Letterbox is the repository's default truth source for vendor320 visual and dynamic640.
        const float ratio = std::min(static_cast<float>(input_width_) / static_cast<float>(bgr.cols),
                                     static_cast<float>(input_height_) / static_cast<float>(bgr.rows));
        const int new_w = static_cast<int>(std::round(static_cast<float>(bgr.cols) * ratio));
        const int new_h = static_cast<int>(std::round(static_cast<float>(bgr.rows) * ratio));
        info.scale_x = 1.f / ratio;
        info.scale_y = 1.f / ratio;
        info.pad_x = (static_cast<float>(input_width_ - new_w)) / 2.f;
        info.pad_y = (static_cast<float>(input_height_ - new_h)) / 2.f;

        cv::Mat resized;
        if (new_w != bgr.cols || new_h != bgr.rows)
            cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
        else
            resized = bgr;

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        const int top = static_cast<int>(std::round(info.pad_y - 0.1f));
        const int bottom = static_cast<int>(std::round(info.pad_y + 0.1f));
        const int left = static_cast<int>(std::round(info.pad_x - 0.1f));
        const int right = static_cast<int>(std::round(info.pad_x + 0.1f));
        cv::copyMakeBorder(rgb, preprocessed, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }

    cv::Mat float_rgb;
    preprocessed.convertTo(float_rgb, CV_32FC3, 1.f / 255.f);

    const size_t plane_size = static_cast<size_t>(input_width_) * static_cast<size_t>(input_height_);
    nchw.resize(plane_size * 3u);
    for (int c = 0; c < 3; ++c)
    {
        float* out = nchw.data() + plane_size * static_cast<size_t>(c);
        for (int y = 0; y < input_height_; ++y)
        {
            const cv::Vec3f* row = float_rgb.ptr<cv::Vec3f>(y);
            for (int x = 0; x < input_width_; ++x)
                *out++ = row[x][c];
        }
    }

    if (DebugDecodeEnabled())
    {
        const cv::Vec3b bgr00 = bgr.at<cv::Vec3b>(0, 0);
        const cv::Vec3b rgb00 = preprocessed.at<cv::Vec3b>(0, 0);
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(nchw.data());
        std::cerr << "[preprocess] mode=" << (info.mode == PreprocessMode::kResize ? "resize" : "letterbox")
                  << " src=" << info.src_w << 'x' << info.src_h
                  << " dst=" << info.dst_w << 'x' << info.dst_h
                  << " bgr00=[" << static_cast<int>(bgr00[0]) << ',' << static_cast<int>(bgr00[1]) << ','
                  << static_cast<int>(bgr00[2]) << ']'
                  << " rgb00=[" << static_cast<int>(rgb00[0]) << ',' << static_cast<int>(rgb00[1]) << ','
                  << static_cast<int>(rgb00[2]) << ']'
                  << " input_sha256=" << Sha256Hex(ptr, nchw.size() * sizeof(float))
                  << " first_values=";
        const size_t sample_count = std::min<size_t>(12, nchw.size());
        for (size_t i = 0; i < sample_count; ++i)
        {
            if (i)
                std::cerr << ',';
            std::cerr << nchw[i];
        }
        std::cerr << '\n';
    }

    return true;
}

Yolo11Detector::OutputTensor Yolo11Detector::RunSingle(const std::vector<float>& nchw) const
{
    const std::array<int64_t, 4> dims = {1, 3, input_height_, input_width_};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor{nullptr};
    std::vector<Ort::Float16_t> nchw_fp16;
    if (input_element_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
        input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, const_cast<float*>(nchw.data()), nchw.size(), dims.data(), dims.size());
    }
    else if (input_element_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
    {
        nchw_fp16 = ToFloat16Buffer(nchw);
        input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
            mem_info, nchw_fp16.data(), nchw_fp16.size(), dims.data(), dims.size());
    }
    else
    {
        throw std::runtime_error("unsupported input tensor dtype: " + std::string(TensorElementTypeName(input_element_type_)));
    }
    const char* input_name = input_name_.c_str();

    std::vector<Ort::Value> outputs = session_->Run(Ort::RunOptions{nullptr},
                                                    &input_name, &input_tensor, 1,
                                                    output_name_ptrs_.data(), output_name_ptrs_.size());
    if (outputs.empty())
        throw std::runtime_error("ORT run returned no outputs");

    OutputTensor output;
    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    try
    {
        output.shape = info.GetShape();
    }
    catch (const std::exception&)
    {
        output.shape = KnownFp16OutputShape(options_.model, options_.input_size);
    }
    if (output.shape.empty())
        output.shape = KnownFp16OutputShape(options_.model, options_.input_size);
    size_t count = 0;
    try
    {
        count = info.GetElementCount();
    }
    catch (const std::exception&)
    {
        for (int64_t dim : output.shape)
            count = count == 0 ? static_cast<size_t>(dim) : count * static_cast<size_t>(dim);
    }
    if (output_element_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
        const float* ptr = outputs[0].GetTensorData<float>();
        output.data.assign(ptr, ptr + count);
    }
    else if (output_element_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
    {
        const auto* ptr = outputs[0].GetTensorData<Ort::Float16_t>();
        output.data = FromFloat16Buffer(ptr, count);
    }
    else
    {
        throw std::runtime_error("unsupported output tensor dtype: " + std::string(TensorElementTypeName(output_element_type_)));
    }
    return output;
}

std::vector<Detection> Yolo11Detector::Decode(const OutputTensor& output, const PreprocessInfo& info) const
{
    std::vector<Detection> detections;
    if (output.shape.size() != 3)
    {
        if (DebugDecodeEnabled())
            std::cerr << "[decode] unsupported output rank=" << output.shape.size() << '\n';
        return detections;
    }

    const int64_t dim1 = output.shape[1];
    const int64_t dim2 = output.shape[2];
    enum class Layout
    {
        kUnknown,
        kChannelsFirst,
        kAnchorsFirst,
        kBoxesLast,
    };

    Layout layout = Layout::kUnknown;
    int64_t channels = 0;
    int64_t anchors = 0;

    if (dim2 == 6 && dim1 > 6)
    {
        layout = Layout::kBoxesLast;
        anchors = dim1;
        channels = dim2;
    }
    else if (dim1 > 0 && dim1 <= dim2)
    {
        layout = Layout::kChannelsFirst;
        channels = dim1;
        anchors = dim2;
    }
    else if (dim2 > 0 && dim2 < dim1)
    {
        layout = Layout::kAnchorsFirst;
        anchors = dim1;
        channels = dim2;
    }

    auto access = [&](int64_t anchor_idx, int64_t channel_idx) -> float {
        if (layout == Layout::kChannelsFirst)
            return output.data[static_cast<size_t>(channel_idx * anchors + anchor_idx)];
        return output.data[static_cast<size_t>(anchor_idx * channels + channel_idx)];
    };

    if (layout == Layout::kUnknown)
    {
        if (DebugDecodeEnabled())
            std::cerr << "[decode] unknown layout for shape=[" << output.shape[0] << ','
                      << output.shape[1] << ',' << output.shape[2] << "]\n";
        return detections;
    }

    int kept_before_nms = 0;
    std::vector<std::string> top_samples;

    for (int64_t anchor = 0; anchor < anchors; ++anchor)
    {
        Detection det{};

        if (channels == 6)
        {
            // End-to-end `[N,6]` outputs are xyxy + score + class in model-input coordinates.
            // Undo the letterbox transform so YOLO26/YOLOv10-style boxes align with source images.
            det.x1 = (access(anchor, 0) - info.pad_x) * info.scale_x;
            det.y1 = (access(anchor, 1) - info.pad_y) * info.scale_y;
            det.x2 = (access(anchor, 2) - info.pad_x) * info.scale_x;
            det.y2 = (access(anchor, 3) - info.pad_y) * info.scale_y;
            det.score = access(anchor, 4);
            det.class_id = static_cast<int>(access(anchor, 5));
            if (det.score <= options_.conf_threshold)
                continue;
        }
        else
        {
            // Ultralytics-style heads keep box center/size plus per-class scores in tensor space.
            float max_score = -1.f;
            int max_class = -1;
            for (int64_t channel = 4; channel < channels; ++channel)
            {
                const float score = access(anchor, channel);
                if (score > max_score)
                {
                    max_score = score;
                    max_class = static_cast<int>(channel - 4);
                }
            }

            if (max_score <= options_.conf_threshold)
                continue;

            const float cx = access(anchor, 0);
            const float cy = access(anchor, 1);
            const float w = access(anchor, 2);
            const float h = access(anchor, 3);
            det.x1 = (cx - w * 0.5f - info.pad_x) * info.scale_x;
            det.y1 = (cy - h * 0.5f - info.pad_y) * info.scale_y;
            det.x2 = (cx + w * 0.5f - info.pad_x) * info.scale_x;
            det.y2 = (cy + h * 0.5f - info.pad_y) * info.scale_y;
            det.class_id = max_class;
            det.score = max_score;
        }

        det.x1 = std::clamp(det.x1, 0.f, static_cast<float>(info.src_w - 1));
        det.y1 = std::clamp(det.y1, 0.f, static_cast<float>(info.src_h - 1));
        det.x2 = std::clamp(det.x2, 0.f, static_cast<float>(info.src_w - 1));
        det.y2 = std::clamp(det.y2, 0.f, static_cast<float>(info.src_h - 1));
        detections.push_back(det);
        kept_before_nms += 1;
        if (DebugDecodeEnabled() && top_samples.size() < 8)
        {
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss.precision(3);
            oss << "anchor=" << anchor
                << " cls=" << det.class_id
                << " score=" << det.score
                << " box=[" << det.x1 << ',' << det.y1 << ',' << det.x2 << ',' << det.y2 << ']';
            top_samples.push_back(oss.str());
        }
    }

    std::vector<Detection> final_detections = NmsClasswise(detections, options_.iou_threshold);
    if (DebugDecodeEnabled())
    {
        std::cerr << "[decode] shape=[" << output.shape[0] << ',' << output.shape[1] << ','
                  << output.shape[2] << "] layout=";
        if (layout == Layout::kChannelsFirst)
            std::cerr << "channels_first";
        else if (layout == Layout::kAnchorsFirst)
            std::cerr << "anchors_first";
        else
            std::cerr << "boxes_last";
        std::cerr << " channels=" << channels
                  << " anchors=" << anchors
                  << " conf=" << options_.conf_threshold
                  << " pre_nms=" << kept_before_nms
                  << " post_nms=" << final_detections.size()
                  << " preprocess=" << (info.mode == PreprocessMode::kResize ? "resize" : "letterbox")
                  << '\n';
        for (const auto& sample : top_samples)
            std::cerr << "[decode] " << sample << '\n';
    }
    return final_detections;
}

InferenceResult Yolo11Detector::ProcessImage(const cv::Mat& bgr, bool render_output)
{
    (void)render_output;
    InferenceResult result;
    PreprocessInfo info;
    std::vector<float> nchw;

    const auto t0 = std::chrono::steady_clock::now();
    const auto pre_begin = std::chrono::steady_clock::now();
    if (!PreprocessToNchw(bgr, nchw, info))
        throw std::runtime_error("preprocess failed");
    const auto pre_end = std::chrono::steady_clock::now();

    const auto inf_begin = std::chrono::steady_clock::now();
    OutputTensor output = RunSingle(nchw);
    const auto inf_end = std::chrono::steady_clock::now();

    const auto post_begin = std::chrono::steady_clock::now();
    result.detections = Decode(output, info);
    const auto post_end = std::chrono::steady_clock::now();

    result.metrics.preprocess_ms = std::chrono::duration<double, std::milli>(pre_end - pre_begin).count();
    result.metrics.inference_ms = std::chrono::duration<double, std::milli>(inf_end - inf_begin).count();
    result.metrics.postprocess_ms = std::chrono::duration<double, std::milli>(post_end - post_begin).count();
    result.metrics.total_ms = std::chrono::duration<double, std::milli>(post_end - t0).count();
    result.metrics.objects = static_cast<int>(result.detections.size());

    result.output_sha256 = HashOutputTensor(output.data);
    result.detections_sha256 = HashDetections(result.detections);
    return result;
}

BenchmarkSummary Yolo11Detector::BenchmarkImage(const cv::Mat& bgr)
{
    if (bgr.empty())
        throw std::runtime_error("benchmark image is empty");

    BenchmarkSummary summary;
    std::vector<double> repeat_means_ms;
    repeat_means_ms.reserve(static_cast<size_t>(options_.repeats));

    std::vector<float> cached_nchw;
    PreprocessInfo cached_info;
    if (options_.benchmark_mode == "forward" && !PreprocessToNchw(bgr, cached_nchw, cached_info))
        throw std::runtime_error("preprocess failed before forward benchmark");

    OutputTensor last_output;
    std::vector<Detection> last_detections;

    for (int repeat = 0; repeat < options_.repeats; ++repeat)
    {
        double sum_ms = 0.0;

        auto run_once = [&](bool measure, bool save_last) {
            std::vector<float> nchw;
            PreprocessInfo info;

            auto t0 = std::chrono::steady_clock::time_point{};
            if (measure && options_.benchmark_mode == "full")
                t0 = std::chrono::steady_clock::now();

            if (options_.benchmark_mode == "full")
            {
                // Full mode measures preprocess + inference + decode to match application-level timing.
                if (!PreprocessToNchw(bgr, nchw, info))
                    throw std::runtime_error("preprocess failed inside full benchmark");
            }
            else
            {
                // Forward mode reuses a cached input tensor so the result is comparable to perf_test.
                nchw = cached_nchw;
                info = cached_info;
            }

            if (measure && options_.benchmark_mode == "forward")
                t0 = std::chrono::steady_clock::now();

            OutputTensor output = RunSingle(nchw);
            if (options_.benchmark_mode == "full")
                last_detections = Decode(output, info);

            const auto t1 = std::chrono::steady_clock::now();
            if (measure)
                sum_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (save_last)
                last_output = std::move(output);
        };

        for (int i = 0; i < options_.warmup; ++i)
            run_once(false, false);

        for (int i = 0; i < options_.runs; ++i)
            run_once(true, repeat == options_.repeats - 1 && i == options_.runs - 1);

        repeat_means_ms.push_back(sum_ms / static_cast<double>(options_.runs));
    }

    summary.mean_ms = std::accumulate(repeat_means_ms.begin(), repeat_means_ms.end(), 0.0) /
                      static_cast<double>(repeat_means_ms.size());
    if (repeat_means_ms.size() > 1)
    {
        double var = 0.0;
        for (double v : repeat_means_ms)
            var += (v - summary.mean_ms) * (v - summary.mean_ms);
        var /= static_cast<double>(repeat_means_ms.size());
        summary.std_ms = std::sqrt(var);
    }
    summary.fps = summary.mean_ms > 0.0 ? 1000.0 / summary.mean_ms : 0.0;
    summary.output_sha256 = HashOutputTensor(last_output.data);
    summary.detections_sha256 = HashDetections(last_detections);

    if (!options_.dump_out.empty() && !last_output.data.empty())
    {
        std::ofstream ofs(options_.dump_out, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(last_output.data.data()),
                  static_cast<std::streamsize>(last_output.data.size() * sizeof(float)));
    }

    return summary;
}

const std::vector<std::string>& Yolo11Detector::Labels() const
{
    return labels_;
}

int Yolo11Detector::InputWidth() const
{
    return input_width_;
}

int Yolo11Detector::InputHeight() const
{
    return input_height_;
}

const std::string& Yolo11Detector::InputName() const
{
    return input_name_;
}

std::string Yolo11Detector::ProviderSummary() const
{
    std::ostringstream oss;
    oss << "provider=" << options_.provider
        << " input=" << input_name_
        << " shape=[";
    for (size_t i = 0; i < input_shape_.size(); ++i)
    {
        if (i)
            oss << ",";
        oss << input_shape_[i];
    }
    oss << "]"
        << " input_dtype=" << TensorElementTypeName(input_element_type_)
        << " output0_dtype=" << TensorElementTypeName(output_element_type_)
        << " preprocess=" << (ResolvePreprocessMode() == PreprocessMode::kResize ? "resize" : "letterbox");
    if (metadata_fallback_applied_)
        oss << " metadata_fallback=" << metadata_fallback_reason_;
    return oss.str();
}

}  // namespace banana_demo
