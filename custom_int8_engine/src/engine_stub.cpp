#include "y26_k1x_engine.h"

namespace y26_k1x {

Engine::Engine(EngineOptions options) : options_(options) {}

StatusCode Engine::load_model(std::span<const std::byte> model_blob) {
    if (model_blob.size() < sizeof(ModelHeaderV0)) {
        return StatusCode::kInvalidModel;
    }

    const auto* header = reinterpret_cast<const ModelHeaderV0*>(model_blob.data());
    const StatusCode status = validate_model_header(*header);
    if (status != StatusCode::kOk) {
        return status;
    }

    model_loaded_ = true;
    return StatusCode::kOk;
}

StatusCode Engine::infer(std::span<const std::int8_t> input_nchw, std::span<Detection> output_detections) {
    (void)input_nchw;
    (void)output_detections;
    if (!model_loaded_) {
        return StatusCode::kInvalidModel;
    }
    return StatusCode::kNotImplemented;
}

}  // namespace y26_k1x
